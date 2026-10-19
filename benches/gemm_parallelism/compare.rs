//! Real-data comparison of assignment backends, row policies, and crossover thresholds.
//!
//! Compile the private implementation in this target instead of exporting benchmark APIs.
//! Paths are checked by Cargo and Clippy; there is no generated or patched Rust source.
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use argh::FromArgs;
use faer::{Accum, MatMut, MatRef, Par};
use gathers::distance::Distance;
use gathers::kmeans::{AssignmentPolicy, CentroidAssigner, base_assign_parallel};
use gathers::utils::try_normalize_rows;
// The private modules use these same library modules in production.
use gathers::{distance, kmeans, utils};

// Reused modules also contain test imports that a custom bench harness does not execute.
#[allow(dead_code, unused_imports)]
#[path = "../../crates/cli/src/fvecs.rs"]
mod fvecs;
#[allow(dead_code, unused_imports)]
#[path = "../../src/linalg.rs"]
mod linalg;
#[allow(dead_code, unused_imports)]
#[path = "../../src/kmeans/matrix.rs"]
mod matrix;

use matrix::MatrixAssignmentWorkspace;

/// Compare exact assignment on a complete pair of fvecs files.
#[derive(FromArgs)]
struct Args {
    /// full query fvecs file
    #[argh(positional)]
    queries: PathBuf,
    /// final centroid fvecs file
    #[argh(positional)]
    centroids: PathBuf,
    /// distance: l2, dot, or cos
    #[argh(positional)]
    metric: Distance,
    /// number of Rayon workers
    #[argh(option, default = "16")]
    threads: usize,
    /// queries per processing batch; includes workspace setup when supplied
    #[argh(option)]
    batch_rows: Option<usize>,
    /// restrict backend comparison to faer or accelerate
    #[argh(option)]
    backend: Option<Backend>,
    /// restrict row policy to fixed512 or adaptive
    #[argh(option)]
    row_policy: Option<RowPolicy>,
    /// restrict batch crossover comparison to previous or current
    #[argh(option)]
    crossover: Option<Crossover>,
    /// compare the public exact and automatic APIs, including validation and normalization
    #[argh(switch)]
    public_api: bool,
    /// measured calls after one warmup
    #[argh(option, default = "7")]
    repeats: usize,
}

#[derive(Clone, Copy, PartialEq)]
enum Backend {
    Portable,
    #[cfg(target_os = "macos")]
    Accelerate,
}

impl FromStr for Backend {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "faer" => Ok(Self::Portable),
            #[cfg(target_os = "macos")]
            "accelerate" => Ok(Self::Accelerate),
            _ => Err("expected faer or accelerate (macOS only)".into()),
        }
    }
}

impl Backend {
    fn name(self) -> &'static str {
        match self {
            Self::Portable => "faer",
            #[cfg(target_os = "macos")]
            Self::Accelerate => "accelerate",
        }
    }

    fn assign(
        self,
        workspace: &mut MatrixAssignmentWorkspace,
        vectors: &[f32],
        inputs: &Inputs,
        labels: &mut [u32],
        block_size: Option<usize>,
    ) {
        match self {
            Self::Portable => workspace.assign_with_kernel(
                vectors,
                &inputs.centroids.data,
                inputs.queries.dim,
                labels,
                block_size,
                portable_matmul,
            ),
            #[cfg(target_os = "macos")]
            Self::Accelerate => workspace.assign_with_kernel(
                vectors,
                &inputs.centroids.data,
                inputs.queries.dim,
                labels,
                block_size,
                linalg::matmul,
            ),
        }
    }
}

fn portable_matmul(
    dst: MatMut<'_, f32>,
    accum: Accum,
    lhs: MatRef<'_, f32>,
    rhs: MatRef<'_, f32>,
    alpha: f32,
    parallelism: Par,
) {
    faer::linalg::matmul::matmul(dst, accum, lhs, rhs, alpha, parallelism);
}

#[derive(Clone, Copy, PartialEq)]
enum RowPolicy {
    Fixed512,
    Adaptive,
}

impl FromStr for RowPolicy {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "fixed512" => Ok(Self::Fixed512),
            "adaptive" => Ok(Self::Adaptive),
            _ => Err("expected fixed512 or adaptive".into()),
        }
    }
}

impl RowPolicy {
    fn name(self) -> &'static str {
        match self {
            Self::Fixed512 => "fixed512",
            Self::Adaptive => "adaptive",
        }
    }

    fn block_size(self, rows: usize, centroids: usize) -> Option<usize> {
        // A bulk row count selects the largest tile allowed by the production scratch cap.
        let rows = match self {
            Self::Fixed512 => usize::MAX,
            Self::Adaptive => rows,
        };
        matrix::matmul_block_size(rows, centroids, rayon::current_num_threads())
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Crossover {
    Previous,
    Current,
}

impl FromStr for Crossover {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "previous" => Ok(Self::Previous),
            "current" => Ok(Self::Current),
            _ => Err("expected previous or current".into()),
        }
    }
}

impl Crossover {
    fn name(self) -> &'static str {
        match self {
            Self::Previous => "previous",
            Self::Current => "current",
        }
    }
}

struct Inputs {
    queries: fvecs::Vectors,
    centroids: fvecs::Vectors,
    distance: Distance,
}

impl Inputs {
    fn workspace(
        &self,
        vectors: &[f32],
        crossover: Crossover,
    ) -> Option<MatrixAssignmentWorkspace> {
        let dim = self.queries.dim;
        match crossover {
            Crossover::Current => MatrixAssignmentWorkspace::try_new(
                vectors,
                self.centroids.len(),
                dim,
                self.distance,
            ),
            // Historical policy, kept here solely as a benchmark comparison.
            Crossover::Previous => MatrixAssignmentWorkspace::try_new_with_threshold(
                vectors,
                self.centroids.len(),
                dim,
                self.distance,
                if dim < 64 { 1 << 22 } else { 1 << 19 },
            ),
        }
    }

    fn direct(&self, vectors: &[f32], labels: &mut [u32]) {
        base_assign_parallel(
            vectors,
            &self.centroids.data,
            self.queries.dim,
            self.distance,
            labels,
        );
    }
}

fn measure(
    repeats: usize,
    labels: &mut [u32],
    oracle: &[u32],
    mut assign: impl FnMut(&mut [u32]) -> usize,
) -> Result<(Vec<f64>, usize)> {
    let matrix_batches = assign(labels);
    ensure!(
        labels == oracle,
        "warmup labels differ from direct assignment"
    );
    let mut times = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        let actual_batches = assign(labels);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        ensure!(labels == oracle, "labels differ from direct assignment");
        ensure!(
            actual_batches == matrix_batches,
            "assignment dispatch changed"
        );
        times.push(elapsed);
    }
    Ok((times, matrix_batches))
}

fn median_ms(times: &[f64]) -> f64 {
    let mut sorted = times.to_vec();
    sorted.sort_by(f64::total_cmp);
    if sorted.len().is_multiple_of(2) {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) * 0.5
    } else {
        sorted[sorted.len() / 2]
    }
}

fn compare(
    args: &Args,
    inputs: &Inputs,
    oracle: &[u32],
    backend: Backend,
    rows: RowPolicy,
    crossover: Crossover,
) -> Result<()> {
    let mut labels = vec![0; inputs.queries.len()];
    let (times, matrix_batches) = if let Some(batch_rows) = args.batch_rows {
        let batch_values = batch_rows
            .checked_mul(inputs.queries.dim)
            .context("batch shape overflow")?;
        measure(args.repeats, &mut labels, oracle, |labels| {
            let mut matrix_batches = 0;
            for (vectors, labels) in inputs
                .queries
                .data
                .chunks(batch_values)
                .zip(labels.chunks_mut(batch_rows))
            {
                if let Some(mut workspace) = inputs.workspace(vectors, crossover) {
                    let block_size = rows.block_size(labels.len(), inputs.centroids.len());
                    backend.assign(&mut workspace, vectors, inputs, labels, block_size);
                    matrix_batches += 1;
                } else {
                    inputs.direct(vectors, labels);
                }
            }
            matrix_batches
        })?
    } else {
        let mut workspace = inputs.workspace(&inputs.queries.data, crossover);
        measure(args.repeats, &mut labels, oracle, |labels| {
            if let Some(workspace) = &mut workspace {
                let block_size = rows.block_size(labels.len(), inputs.centroids.len());
                backend.assign(workspace, &inputs.queries.data, inputs, labels, block_size);
                1
            } else {
                inputs.direct(&inputs.queries.data, labels);
                0
            }
        })?
    };
    let median = median_ms(&times);
    println!(
        "{}",
        serde_json::json!({
            "backend": backend.name(),
            "row_policy": rows.name(),
            "crossover": crossover.name(),
            "batch_rows": args.batch_rows,
            "workspace_setup_included": args.batch_rows.is_some(),
            "matrix_batches": matrix_batches,
            "threads": args.threads,
            "queries": inputs.queries.len(),
            "centroids": inputs.centroids.len(),
            "dim": inputs.queries.dim,
            "metric": format!("{:?}", args.metric),
            "median_ms": median,
            "times_ms": times,
            "same_labels": true,
        })
    );
    Ok(())
}

fn compare_public(args: &Args, inputs: &Inputs) -> Result<()> {
    let mut oracle = vec![0; inputs.queries.len()];
    let mut labels = oracle.clone();
    for policy in [AssignmentPolicy::Exact, AssignmentPolicy::Auto] {
        let start = Instant::now();
        let mut assigner = CentroidAssigner::new(
            &inputs.centroids.data,
            inputs.centroids.dim,
            inputs.distance,
            policy,
            Some(42),
        )?;
        let build_ms = start.elapsed().as_secs_f64() * 1000.0;
        assigner.assign(&inputs.queries.data, &mut labels)?;
        if policy == AssignmentPolicy::Exact {
            oracle.copy_from_slice(&labels);
        }
        let mut times_ms = Vec::with_capacity(args.repeats);
        let mut agreement = Vec::with_capacity(args.repeats);
        for _ in 0..args.repeats {
            let start = Instant::now();
            assigner.assign(&inputs.queries.data, &mut labels)?;
            times_ms.push(start.elapsed().as_secs_f64() * 1000.0);
            let matching = labels.iter().zip(&oracle).filter(|(a, b)| a == b).count();
            if policy == AssignmentPolicy::Exact {
                ensure!(
                    matching == labels.len(),
                    "exact labels changed between calls"
                );
            }
            agreement.push(matching as f64 / labels.len() as f64);
        }
        let median = median_ms(&times_ms);
        println!(
            "{}",
            serde_json::json!({
                "api": "CentroidAssigner",
                "policy": format!("{policy:?}"),
                "queries": inputs.queries.len(),
                "centroids": inputs.centroids.len(),
                "dim": inputs.queries.dim,
                "metric": format!("{:?}", inputs.distance),
                "threads": args.threads,
                "seed": 42,
                "build_ms": build_ms,
                "times_ms": times_ms,
                "median_ms": median,
                "exact_agreement": agreement,
                "validation_and_normalization_included": true,
                "assigner_reused": true,
            })
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    // Cargo adds --bench to custom harnesses. With no datasets, print help so
    // cargo test --all-targets can check the target without requiring local data.
    let argv: Vec<_> = std::env::args().filter(|arg| arg != "--bench").collect();
    let values: Vec<_> = argv.iter().skip(1).map(String::as_str).collect();
    let values: &[&str] = if values.is_empty() {
        &["--help"]
    } else {
        &values
    };
    let args: Args = match Args::from_args(&[&argv[0]], values) {
        Ok(args) => args,
        Err(exit) => {
            print!("{}", exit.output);
            std::process::exit(if exit.status.is_ok() { 0 } else { 2 });
        }
    };
    ensure!(args.threads > 0, "threads must be positive");
    ensure!(args.repeats > 0, "repeats must be positive");
    ensure!(args.batch_rows != Some(0), "batch rows must be positive");
    ensure!(
        args.crossover.is_none() || args.batch_rows.is_some(),
        "--crossover requires --batch-rows"
    );
    ensure!(
        args.row_policy.is_none() || args.batch_rows.is_none(),
        "--row-policy is for backend comparison; batch mode uses adaptive rows"
    );
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()?;
    let mut inputs = Inputs {
        queries: fvecs::read(&args.queries, None)?,
        centroids: fvecs::read(&args.centroids, None)?,
        distance: args.metric,
    };
    ensure!(
        inputs.queries.dim == inputs.centroids.dim,
        "dimension mismatch"
    );
    if args.public_api {
        ensure!(
            args.backend.is_none()
                && args.row_policy.is_none()
                && args.crossover.is_none()
                && args.batch_rows.is_none(),
            "--public-api cannot be combined with private backend, row, crossover or batch \
             overrides"
        );
        return compare_public(&args, &inputs);
    }
    if inputs.distance == Distance::Cosine {
        try_normalize_rows(&mut inputs.queries.data, inputs.queries.dim, false)?;
        try_normalize_rows(&mut inputs.centroids.data, inputs.centroids.dim, false)?;
        inputs.distance = Distance::NegativeDotProduct;
    }
    let mut oracle = vec![0; inputs.queries.len()];
    inputs.direct(&inputs.queries.data, &mut oracle);
    if args.batch_rows.is_some() {
        #[cfg(target_os = "macos")]
        let backend = Backend::Accelerate;
        #[cfg(not(target_os = "macos"))]
        let backend = Backend::Portable;
        let backend = args.backend.unwrap_or(backend);
        for crossover in [Crossover::Previous, Crossover::Current] {
            if args.crossover.is_some_and(|selected| selected != crossover) {
                continue;
            }
            compare(
                &args,
                &inputs,
                &oracle,
                backend,
                RowPolicy::Adaptive,
                crossover,
            )?;
        }
    } else {
        let backends = [
            Backend::Portable,
            #[cfg(target_os = "macos")]
            Backend::Accelerate,
        ];
        for backend in backends {
            if args.backend.is_some_and(|selected| selected != backend) {
                continue;
            }
            for rows in [RowPolicy::Fixed512, RowPolicy::Adaptive] {
                if args.row_policy.is_some_and(|selected| selected != rows) {
                    continue;
                }
                compare(&args, &inputs, &oracle, backend, rows, Crossover::Current)?;
            }
        }
    }
    Ok(())
}
