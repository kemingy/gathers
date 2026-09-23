use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use argh::FromArgs;
use gathers::distance::Distance;
use gathers::kmeans::KMeans;
use gathers::sampling::sample_indices;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::{fvecs, report, wait_for_profiler};

#[derive(FromArgs)]
#[argh(subcommand, name = "kmeans")]
/// Train squared-Euclidean K-means and save centroids as fvecs.
pub(crate) struct Args {
    /// input fvecs file
    #[argh(option, short = 'i')]
    input: PathBuf,
    /// output centroid fvecs file
    #[argh(option, short = 'o')]
    output: PathBuf,
    /// number of clusters; defaults to max(1, floor(rows^0.8 / 16))
    #[argh(option, short = 'n')]
    n_cluster: Option<u32>,
    /// maximum number of iterations
    #[argh(option, short = 'm', default = "25")]
    max_iter: u32,
    /// exact training sample size; defaults to min(input rows, 256 * clusters)
    #[argh(option)]
    training_samples: Option<usize>,
    /// rows per input batch
    #[argh(option, default = "4096")]
    batch_rows: usize,
    /// scan and validate every source row instead of reading only the selected sample
    #[argh(switch)]
    validate_all: bool,
    /// preflight memory budget in decimal GB; includes conservative workspace headroom
    #[argh(option, default = "48")]
    memory_budget_gb: u64,
}

fn cluster_count(rows: usize, explicit: Option<u32>) -> Result<u32> {
    let count = explicit
        .map(f64::from)
        .unwrap_or_else(|| ((rows as f64).powf(0.8) / 16.0).floor().max(1.0));
    ensure!(
        count >= 1.0 && count <= u32::MAX as f64,
        "cluster count must fit a positive u32"
    );
    let count = count as u32;
    ensure!(
        rows / count as usize >= 39,
        "at least 39 input rows per cluster are required"
    );
    Ok(count)
}

// A conservative admission estimate, not an OS-enforced RSS limit. Keep ample slack for
// allocator overhead, Rayon stacks, faer packing, and runtime allocations. Never silently
// reduce the sample when the user's requested workload does not fit.
fn check_memory(
    samples: usize,
    dim: usize,
    clusters: u32,
    batch_rows: usize,
    budget_gb: u64,
) -> Result<u64> {
    ensure!(
        batch_rows > 0 && budget_gb > 0,
        "batch-rows and memory-budget-gb must be positive"
    );
    let padded_dim = (dim as u128).div_ceil(64) * 64;
    let threads = rayon::current_num_threads() as u128;
    let sample_bytes = samples as u128 * dim as u128 * 4;
    let batch_bytes = batch_rows as u128 * (dim as u128 + 1) * 4;
    let row_metadata = samples as u128 * (size_of::<usize>() as u128 + 8);
    let centroid_workspace = clusters as u128 * padded_dim * 4 * 32;
    let worker_workspace = threads * 64 * (padded_dim + clusters as u128) * 4;
    let training_bytes =
        sample_bytes + batch_bytes + row_metadata + centroid_workspace + worker_workspace;
    // Sampling finishes before vector allocation. Allow for rand's temporary u32 source
    // array (in-place sampler), or hash table and output (rejection sampler), in addition
    // to the final usize indices. This exceeds their current per-sample scratch needs;
    // the fixed runtime margin also covers the small-sample algorithm variants.
    let sampling_bytes = samples as u128 * 256;
    let estimated = training_bytes.max(sampling_bytes) + (4_u128 << 30);
    ensure!(
        estimated <= budget_gb as u128 * 1_000_000_000,
        "estimated memory {:.2} GB exceeds budget {budget_gb} GB; reduce --training-samples or \
         --n-cluster, or prepare lower-dimensional data",
        estimated as f64 / 1e9
    );
    u64::try_from(estimated).context("memory estimate exceeds supported address space")
}

pub(crate) fn run(args: &Args, common: &crate::Args) -> Result<()> {
    ensure!(args.max_iter > 0, "max-iter must be positive");
    let mut reader = fvecs::Reader::open(&args.input)?;
    let num_vectors = reader.rows;
    let dim = reader.dim;
    let num_clusters = cluster_count(num_vectors, args.n_cluster)?;
    let kmeans = KMeans::new(
        num_clusters,
        args.max_iter,
        0.01,
        Distance::SquaredEuclidean,
        false,
    )
    .seed(common.seed);
    let training_rows = args
        .training_samples
        .unwrap_or_else(|| kmeans.training_sample_size(num_vectors));
    ensure!(
        training_rows > 0 && training_rows <= num_vectors,
        "training-samples must be positive and no larger than the source"
    );
    ensure!(
        training_rows / num_clusters as usize >= 39,
        "training-samples requires at least 39 rows per cluster; increase the sample or reduce \
         --n-cluster"
    );
    let estimated_memory_bytes = check_memory(
        training_rows,
        dim,
        num_clusters,
        args.batch_rows,
        args.memory_budget_gb,
    )?;
    let prepare_start = Instant::now();
    let mut indices = sample_indices(
        num_vectors,
        training_rows,
        &mut StdRng::seed_from_u64(common.seed),
    );
    let index_sample_ms = prepare_start.elapsed().as_secs_f64() * 1_000.0;
    let start = Instant::now();
    indices.sort_unstable();
    let index_sort_ms = start.elapsed().as_secs_f64() * 1_000.0;
    let start = Instant::now();
    let vectors = reader
        .read_sample(&indices, args.batch_rows, args.validate_all)
        .with_context(|| format!("cannot sample fvecs from {}", args.input.display()))?;
    let sample_read_ms = start.elapsed().as_secs_f64() * 1_000.0;
    drop(indices);
    drop(reader);
    let prepare_ms = prepare_start.elapsed().as_secs_f64() * 1_000.0;
    if common.wait_for_profiler {
        wait_for_profiler()?;
    }
    let start = Instant::now();
    let centroids = kmeans.fit_sample(vectors.data, dim);
    let fit_ms = start.elapsed().as_secs_f64() * 1_000.0;
    fvecs::write(&args.output, &centroids, dim)?;
    report(serde_json::json!({
        "command": "kmeans",
        "input": args.input,
        "output": args.output,
        "num_vectors": num_vectors,
        "training_rows": training_rows,
        "batch_rows": args.batch_rows,
        "validate_all": args.validate_all,
        "memory_budget_gb": args.memory_budget_gb,
        "estimated_memory_bytes": estimated_memory_bytes,
        "prepare_ms": prepare_ms,
        "index_sample_ms": index_sample_ms,
        "index_sort_ms": index_sort_ms,
        "sample_read_ms": sample_read_ms,
        "num_centroids": centroids.len() / dim,
        "dim": dim,
        "max_iter": args.max_iter,
        "seed": common.seed,
        "threads": rayon::current_num_threads(),
        "cpu": common.cpu,
        "arch": std::env::consts::ARCH,
        "os": std::env::consts::OS,
        "debug_assertions": cfg!(debug_assertions),
        "fit_ms": fit_ms,
    }))
}

#[cfg(test)]
mod tests {
    use gathers::distance::Distance;
    use gathers::kmeans::KMeans;

    use super::{check_memory, cluster_count};

    #[test]
    fn plan_uses_source_rows_and_checks_memory_before_loading() {
        let clusters = cluster_count(10_000_000, None).unwrap();
        assert_eq!(clusters, 24_881);
        let model = KMeans::new(clusters, 1, 0.01, Distance::SquaredEuclidean, false);
        let samples = model.training_sample_size(10_000_000);
        assert_eq!(samples, 6_369_536);
        assert!(check_memory(samples, 768, clusters, 4096, 48).unwrap() < 30_000_000_000);
        assert!(check_memory(samples, 4096, clusters, 4096, 48).is_err());
        assert!(check_memory(100, 2, 1, 0, 48).is_err());
        assert!(cluster_count(100, Some(0)).is_err());
        assert!(cluster_count(100, Some(3)).is_err());
        if let Ok(rows) = usize::try_from(10_000_000_000_u64) {
            let clusters = cluster_count(rows, None).unwrap();
            assert_eq!(clusters, 6_250_000);
            // Even the minimum permitted sample is too large; fast index selection
            // does not make this default clustering configuration fit in local RAM.
            assert!(check_memory(clusters as usize * 39, 768, clusters, 4096, 48).is_err());
        }
    }
}
