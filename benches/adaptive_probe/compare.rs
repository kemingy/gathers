//! Full-source experiment for per-round, pilot-calibrated hierarchy probing.
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Result, ensure};
use argh::FromArgs;
use gathers::distance::{Distance, neg_dot_product, squared_euclidean};
use gathers::kmeans::{KMeansConfig, base_assign_parallel, update_centroids};
use gathers::reduction::{PCA, Reduction};
use gathers::sampling::sample_indices;
use gathers::utils::try_normalize_rows;
use gathers::{distance, kmeans, simd, utils};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rayon::slice::ParallelSlice;
use serde_json::json;

#[allow(dead_code, unused_imports)]
#[path = "../../crates/cli/src/fvecs.rs"]
mod fvecs;
#[allow(dead_code, unused_imports)]
#[path = "../../src/kmeans/hierarchy.rs"]
mod hierarchy;
#[allow(dead_code, unused_imports)]
#[path = "../../src/linalg.rs"]
mod linalg;
#[allow(dead_code, unused_imports)]
#[path = "../../src/kmeans/matrix.rs"]
mod matrix;

use hierarchy::{Hierarchy, ProbeOptions};
use matrix::MatrixAssignmentWorkspace;

/// Test adaptive probing with full source rows and exact final audits.
#[derive(FromArgs)]
struct Args {
    /// original full source fvecs
    #[argh(positional)]
    source: PathBuf,
    /// source distance: l2 or cos
    #[argh(positional)]
    metric: Distance,
    /// cache of full-source-fitted PCA128 rows; created if absent
    #[argh(option)]
    prepared: PathBuf,
    /// only prepare the full projection cache
    #[argh(switch)]
    prepare_only: bool,
    /// exact, adaptive, or fixed (PR65 default budget with the same incumbent policy)
    #[argh(option, default = "String::from(\"adaptive\")")]
    method: String,
    /// calibration rows; no effect on training row count
    #[argh(option, default = "256")]
    pilots: usize,
    /// exact final rounds, included in iteration limit
    #[argh(option, default = "2")]
    exact_final_rounds: usize,
    /// total Lloyd rounds
    #[argh(option, default = "10")]
    iterations: usize,
    /// worker count
    #[argh(option, default = "16")]
    threads: usize,
    /// random seed for initialization, calibration and audits
    #[argh(option, default = "42")]
    seed: u64,
    /// omit previous-assignment incumbent (ablation)
    #[argh(switch)]
    no_incumbent: bool,
}

fn gather(vectors: &[f32], dim: usize, indices: &[usize]) -> Vec<f32> {
    indices
        .iter()
        .flat_map(|&i| vectors[i * dim..(i + 1) * dim].iter().copied())
        .collect()
}

#[allow(unused_imports)]
mod policy;
use policy::budget;

fn exact(vectors: &[f32], centroids: &[f32], dim: usize, distance: Distance, labels: &mut [u32]) {
    if let Some(mut workspace) =
        MatrixAssignmentWorkspace::try_new(vectors, centroids.len() / dim, dim, distance)
    {
        workspace.assign(vectors, centroids, dim, labels);
    } else {
        base_assign_parallel(vectors, centroids, dim, distance, labels);
    }
}

fn objective(
    vectors: &[f32],
    centroids: &[f32],
    dim: usize,
    distance: Distance,
    labels: &[u32],
) -> f64 {
    vectors
        .par_chunks_exact(dim)
        .zip(labels.par_iter())
        .map(|(v, &label)| {
            let c = &centroids[label as usize * dim..(label as usize + 1) * dim];
            match distance {
                Distance::SquaredEuclidean => f64::from(squared_euclidean(v, c)),
                _ => 1.0 + f64::from(neg_dot_product(v, c)),
            }
        })
        .sum::<f64>()
        / labels.len() as f64
}

fn empty_count(labels: &[u32], count: usize) -> usize {
    let mut sizes = vec![0usize; count];
    for &label in labels {
        sizes[label as usize] += 1;
    }
    sizes.iter().filter(|&&n| n == 0).count()
}

fn calibrate(h: &Hierarchy, pilots: &[f32], distance: Distance) -> (usize, Vec<usize>, Vec<u32>) {
    let mut required = vec![0; pilots.len() / 128];
    let mut labels = vec![0; required.len()];
    h.probe_parallel(
        pilots,
        &mut labels,
        &mut required,
        ProbeOptions {
            distance,
            budget: usize::MAX,
            incumbent_centroids: None,
        },
    );
    (budget(&required), required, labels)
}

fn main() -> Result<()> {
    let args: Args = argh::from_env();
    ensure!(
        matches!(args.metric, Distance::SquaredEuclidean | Distance::Cosine),
        "use l2 or cos"
    );
    ensure!(
        matches!(args.method.as_str(), "exact" | "adaptive" | "fixed"),
        "invalid method"
    );
    ensure!(
        (256..=1024).contains(&args.pilots),
        "pilots must be in 256..=1024"
    );
    ensure!(
        args.threads > 0 && args.iterations > 0 && args.exact_final_rounds <= args.iterations,
        "invalid worker/iteration count"
    );
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()?;
    let source = fvecs::Reader::open(&args.source)?;
    let rows = source.rows;
    let original_dim = source.dim;
    let count = KMeansConfig::default()
        .resolve(rows, original_dim)?
        .n_clusters
        .unwrap() as usize;
    if !args.prepared.exists() {
        let started = Instant::now();
        let mut raw = fvecs::read(&args.source, None)?;
        if args.metric == Distance::Cosine {
            try_normalize_rows(&mut raw.data, raw.dim, false)?;
        }
        let pca = PCA::fit(&raw.data, raw.dim, 128)?;
        let mut projected = pca.transform(&raw.data)?;
        drop(raw);
        if args.metric == Distance::Cosine {
            try_normalize_rows(&mut projected, 128, true)?;
        }
        fvecs::write(&args.prepared, &projected, 128)?;
        println!(
            "{}",
            json!({"phase":"prepare", "rows": rows, "original_dim": original_dim, "dim":128, "seconds":started.elapsed().as_secs_f64(), "pca_fit_rows":rows})
        );
    }
    if args.prepare_only {
        return Ok(());
    }
    let prepared = fvecs::read(&args.prepared, None)?;
    ensure!(
        prepared.len() == rows && prepared.dim == 128,
        "cache must contain every source row at PCA128"
    );
    let vectors = prepared.data;
    let dim = 128;
    let distance = if args.metric == Distance::Cosine {
        Distance::NegativeDotProduct
    } else {
        args.metric
    };
    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut centroids = gather(&vectors, dim, &sample_indices(rows, count, &mut rng));
    if distance == Distance::NegativeDotProduct {
        try_normalize_rows(&mut centroids, dim, true)?;
    }
    let mut audit_rng = StdRng::seed_from_u64(args.seed.wrapping_add(10_000));
    let audit_indices = sample_indices(rows, 16_384.min(rows), &mut audit_rng);
    let audit = gather(&vectors, dim, &audit_indices);
    let mut labels = vec![0; rows];
    let mut training_s = 0.0;
    let mut diagnostic_s = 0.0;
    let wall = Instant::now();
    let mut completed = 0;
    let mut stopped_empty = 0;
    for iteration in 0..args.iterations {
        let iteration_start = Instant::now();
        let approximate =
            args.method != "exact" && iteration < args.iterations - args.exact_final_rounds;
        let mut build_s = 0.0;
        let mut calibration_s = 0.0;
        let mut p98 = 0;
        let mut p100 = 0;
        let mut selected_budget = 0;
        let mut pilot_labels = Vec::new();
        let mut pilot_vectors = Vec::new();
        let h = if approximate {
            let start = Instant::now();
            let h = Hierarchy::build(
                &centroids,
                dim,
                Some(args.seed.wrapping_add(iteration as u64)),
            )
            .expect("dataset must support hierarchy");
            build_s = start.elapsed().as_secs_f64();
            if args.method == "adaptive" {
                let start = Instant::now();
                let mut pilot_rng =
                    StdRng::seed_from_u64(args.seed.wrapping_add(20_000 + iteration as u64));
                let indices: Vec<_> =
                    sample_indices(rows, 1024 + audit_indices.len(), &mut pilot_rng)
                        .into_iter()
                        .filter(|i| !audit_indices.contains(i))
                        .take(args.pilots)
                        .collect();
                pilot_vectors = gather(&vectors, dim, &indices);
                let (b, mut required, winners) = calibrate(&h, &pilot_vectors, distance);
                calibration_s = start.elapsed().as_secs_f64();
                required.sort_unstable();
                p98 = required[(98 * required.len()).div_ceil(100) - 1];
                p100 = required[required.len() - 1];
                selected_budget = b;
                pilot_labels = winners;
            } else {
                let parents = ((count as f64).sqrt() as usize).min(count / 196);
                selected_budget = ((parents as f64).sqrt().ceil() as usize).clamp(1, 10);
            }
            Some(h)
        } else {
            None
        };
        let start = Instant::now();
        if let Some(h) = &h {
            h.probe_parallel(
                &vectors,
                &mut labels,
                &mut [],
                ProbeOptions {
                    distance,
                    budget: selected_budget,
                    incumbent_centroids: (iteration > 0 && !args.no_incumbent)
                        .then_some(centroids.as_slice()),
                },
            );
        } else {
            exact(&vectors, &centroids, dim, distance, &mut labels);
        }
        let assignment_s = start.elapsed().as_secs_f64();
        let diagnostic_start = Instant::now();
        let assigned_audit: Vec<_> = audit_indices.iter().map(|&i| labels[i]).collect();
        let mut exact_audit = vec![0; audit_indices.len()];
        exact(&audit, &centroids, dim, distance, &mut exact_audit);
        let agreement = assigned_audit
            .iter()
            .zip(&exact_audit)
            .filter(|(a, b)| a == b)
            .count() as f64
            / exact_audit.len() as f64;
        if !pilot_labels.is_empty() {
            let mut reference = vec![0; args.pilots];
            base_assign_parallel(&pilot_vectors, &centroids, dim, distance, &mut reference);
            ensure!(
                reference == pilot_labels,
                "full-probe pilots must equal direct exact labels"
            );
        }
        let audit_objective = objective(&audit, &centroids, dim, distance, &exact_audit);
        let round_diagnostic_s = diagnostic_start.elapsed().as_secs_f64();
        diagnostic_s += round_diagnostic_s;
        let start = Instant::now();
        let empties = empty_count(&labels, count);
        let mut diff = 0.0;
        if empties == 0 {
            diff = update_centroids(&vectors, &mut centroids, dim, &labels)?;
            if distance == Distance::NegativeDotProduct {
                try_normalize_rows(&mut centroids, dim, false)?;
            }
        }
        let update_s = start.elapsed().as_secs_f64();
        let round_s = iteration_start.elapsed().as_secs_f64() - round_diagnostic_s;
        training_s += round_s;
        completed += 1;
        println!(
            "{}",
            json!({"phase":"round", "method":args.method, "pilots":args.pilots, "iteration":iteration+1, "approximate":approximate, "rows":rows, "clusters":count, "dim":dim, "budget":selected_budget, "p98":p98, "p100":p100, "build_s":build_s, "calibration_s":calibration_s, "assignment_s":assignment_s, "update_s":update_s, "round_s":round_s, "empty_clusters":empties, "shift":diff, "assignment_recall":agreement, "audit_exact_objective":audit_objective})
        );
        if empties != 0 {
            stopped_empty = empties;
            break;
        }
    }
    let start = Instant::now();
    let mut final_labels = vec![0; rows];
    exact(&vectors, &centroids, dim, distance, &mut final_labels);
    let final_exact_assignment_s = start.elapsed().as_secs_f64();
    let final_objective = objective(&vectors, &centroids, dim, distance, &final_labels);
    let final_empty = empty_count(&final_labels, count);
    println!(
        "{}",
        json!({"phase":"summary", "method":args.method, "pilots":args.pilots, "seed":args.seed, "threads":args.threads, "rows":rows, "clusters":count, "dim":dim, "original_dim":original_dim, "completed_rounds":completed, "stopped_empty":stopped_empty, "training_s":training_s, "diagnostic_s":diagnostic_s, "final_exact_assignment_s":final_exact_assignment_s, "final_exact_objective":final_objective, "final_empty_clusters":final_empty, "wall_with_diagnostics_s":wall.elapsed().as_secs_f64(), "incumbent":!args.no_incumbent, "exact_final_rounds":args.exact_final_rounds})
    );
    Ok(())
}
