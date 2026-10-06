use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use argh::FromArgs;
use gathers::distance::Distance;
use gathers::kmeans::{KMeans, KMeansConfig, ReductionConfig};
use gathers::sampling::sample_indices;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::{fvecs, report, wait_for_profiler};

#[derive(FromArgs)]
#[argh(subcommand, name = "kmeans")]
/// Train K-means and save centroids as fvecs.
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
    #[argh(option, short = 'm', default = "10")]
    max_iter: u32,
    /// training rows per cluster, at least 39; ignored when training-samples is set
    #[argh(option, default = "256")]
    samples_per_cluster: usize,
    /// exact training sample size; overrides samples-per-cluster
    #[argh(option)]
    training_samples: Option<usize>,
    /// training distance: l2, cos, or dot
    #[argh(option, default = "String::from(\"l2\")")]
    distance: String,
    /// dimensionality reduction: auto (default), raw, pca, or srht
    #[argh(option, default = "String::from(\"auto\")")]
    reduction: String,
    /// projected dimension for explicit pca/srht; defaults to 128
    #[argh(option)]
    reduced_dim: Option<usize>,
    /// rows used to fit PCA; defaults to min(training rows, 100 * input dimension)
    #[argh(option)]
    projection_training_samples: Option<usize>,
    /// rows per input batch
    #[argh(option, default = "4096")]
    batch_rows: usize,
    /// scan and validate every source row instead of reading only the selected sample
    #[argh(switch)]
    validate_all: bool,
    /// optional preflight memory limit in decimal GB; includes conservative workspace
    /// headroom. Unset means no limit
    #[argh(option)]
    memory_limit_gb: Option<u64>,
}

// A conservative admission estimate, not an OS-enforced RSS limit. Keep ample slack for
// allocator overhead, Rayon stacks, faer packing, and runtime allocations. Never silently
// reduce the sample when the user's requested workload does not fit.
fn check_memory(
    samples: usize,
    dim: usize,
    clusters: u32,
    batch_rows: usize,
    reduced_dim: Option<usize>,
    projection_training_samples: usize,
    limit_gb: Option<u64>,
) -> Result<u64> {
    ensure!(batch_rows > 0, "batch-rows must be positive");
    if let Some(limit_gb) = limit_gb {
        ensure!(limit_gb > 0, "memory-limit-gb must be positive");
    }
    let padded_dim = (dim as u128).div_ceil(64) * 64;
    let threads = rayon::current_num_threads() as u128;
    let sample_bytes = samples as u128 * dim as u128 * 4;
    let batch_bytes = batch_rows as u128 * (dim as u128 + 1) * 4;
    let row_metadata = samples as u128 * (size_of::<usize>() as u128 + 8);
    let centroid_workspace = clusters as u128 * padded_dim * 4 * 32;
    let worker_workspace = threads * 64 * (padded_dim + clusters as u128) * 4;
    let training_bytes =
        sample_bytes + batch_bytes + row_metadata + centroid_workspace + worker_workspace;
    let projection_bytes = reduced_dim.map_or(0, |reduced_dim| {
        let projected_sample = samples as u128 * reduced_dim as u128 * 4;
        let projected_centroids = clusters as u128 * (dim as u128 + reduced_dim as u128) * 4;
        // Only PCA fits a projection: SRHT has no training copy or dim × dim covariance.
        let pca_workspace = if projection_training_samples > 0 {
            projection_training_samples as u128 * dim as u128 * 4 + dim as u128 * dim as u128 * 4
        } else {
            0
        };
        // PCA::transform holds a centered copy of the full sample alongside its input and output.
        let centered_sample = if projection_training_samples > 0 {
            sample_bytes
        } else {
            0
        };
        projected_sample + projected_centroids + pca_workspace + centered_sample
    });
    // Sampling finishes before vector allocation. Allow for rand's temporary u32 source
    // array (in-place sampler), or hash table and output (rejection sampler), in addition
    // to the final usize indices. This exceeds their current per-sample scratch needs;
    // the fixed runtime margin also covers the small-sample algorithm variants.
    let sampling_bytes = samples as u128 * 256;
    let estimated = (training_bytes + projection_bytes).max(sampling_bytes) + (4_u128 << 30);
    if let Some(limit_gb) = limit_gb {
        ensure!(
            estimated <= limit_gb as u128 * 1_000_000_000,
            "estimated memory {:.2} GB exceeds limit {limit_gb} GB; reduce --training-samples or \
             --n-cluster, or prepare lower-dimensional data",
            estimated as f64 / 1e9
        );
    }
    u64::try_from(estimated).context("memory estimate exceeds supported address space")
}

pub(crate) fn run(args: &Args, common: &crate::Args) -> Result<()> {
    let mut reader = fvecs::Reader::open(&args.input)?;
    let num_vectors = reader.rows;
    let dim = reader.dim;
    let distance = args.distance.parse::<Distance>()?;
    let reduction = match args.reduction.parse::<ReductionConfig>()? {
        ReductionConfig::PCA { output_dim, .. } => ReductionConfig::PCA {
            output_dim: args.reduced_dim.unwrap_or(output_dim),
            training_samples: args.projection_training_samples,
        },
        ReductionConfig::SRHT { output_dim } => {
            ensure!(
                args.projection_training_samples.is_none(),
                "projection-training-samples only applies to PCA"
            );
            ReductionConfig::SRHT {
                output_dim: args.reduced_dim.unwrap_or(output_dim),
            }
        }
        reduction => {
            ensure!(
                args.reduced_dim.is_none() && args.projection_training_samples.is_none(),
                "projection options require explicit pca or srht reduction"
            );
            reduction
        }
    };
    let config = KMeansConfig {
        n_clusters: args.n_cluster,
        max_iter: args.max_iter,
        tolerance: 0.01,
        distance,
        samples_per_cluster: args.samples_per_cluster,
        training_samples: args.training_samples,
        reduction,
        seed: Some(common.seed),
        ..Default::default()
    }
    .resolve(num_vectors, dim)?;
    let num_clusters = config.n_clusters.expect("resolved cluster count");
    let training_rows = config.training_samples.expect("resolved training sample");
    let (reduction_name, reduced_dim, projection_training_rows) = match config.reduction {
        ReductionConfig::None => ("raw", None, 0),
        ReductionConfig::PCA {
            output_dim,
            training_samples,
        } => (
            "pca",
            Some(output_dim),
            training_samples.expect("resolved PCA sample"),
        ),
        ReductionConfig::SRHT { output_dim } => ("srht", Some(output_dim), 0),
        ReductionConfig::Auto => unreachable!("automatic settings have been resolved"),
    };
    let kmeans = KMeans::new(config);
    let estimated_memory_bytes = check_memory(
        training_rows,
        dim,
        num_clusters,
        args.batch_rows,
        reduced_dim,
        projection_training_rows,
        args.memory_limit_gb,
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
    let input_prepare_ms = prepare_start.elapsed().as_secs_f64() * 1_000.0;
    if common.wait_for_profiler {
        wait_for_profiler()?;
    }
    let pipeline_start = Instant::now();
    let fit = kmeans.fit_sample(vectors.data, dim)?;
    let pipeline_ms = pipeline_start.elapsed().as_secs_f64() * 1_000.0;
    let normalization_ms = fit.normalization_time.as_secs_f64() * 1_000.0;
    let projection_fit_ms = fit.projection_fit_time.as_secs_f64() * 1_000.0;
    let projection_transform_ms = fit.projection_transform_time.as_secs_f64() * 1_000.0;
    let fit_ms = fit.fit_time.as_secs_f64() * 1_000.0;
    let reconstruction_ms = fit.reconstruction_time.as_secs_f64() * 1_000.0;
    // Include boundary validation and orchestration overhead, not only individually timed kernels.
    let prepare_ms = input_prepare_ms + pipeline_ms - fit_ms - reconstruction_ms;
    let reconstruction_empty_clusters = fit.empty_clusters;
    let preserved_variance = fit.preserved_variance;
    let training_dim = fit.training_dim;
    let centroids = fit.centroids;
    let num_centroids = centroids.len() / dim;
    fvecs::write(&args.output, &centroids, dim)?;
    report(serde_json::json!({
        "command": "kmeans",
        "input": args.input,
        "output": args.output,
        "num_vectors": num_vectors,
        "training_rows": training_rows,
        "samples_per_cluster": args.samples_per_cluster,
        "distance": args.distance,
        "normalization_ms": normalization_ms,
        "reduction": reduction_name,
        "requested_reduction": args.reduction,
        "training_dim": training_dim,
        "projection_training_rows": projection_training_rows,
        "projection_fit_ms": projection_fit_ms,
        "projection_transform_ms": projection_transform_ms,
        "reconstruction_ms": reconstruction_ms,
        "reconstruction_empty_clusters": reconstruction_empty_clusters,
        "preserved_variance": preserved_variance,
        "batch_rows": args.batch_rows,
        "validate_all": args.validate_all,
        "memory_limit_gb": args.memory_limit_gb,
        "estimated_memory_bytes": estimated_memory_bytes,
        "prepare_ms": prepare_ms,
        "index_sample_ms": index_sample_ms,
        "index_sort_ms": index_sort_ms,
        "sample_read_ms": sample_read_ms,
        "num_centroids": num_centroids,
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
    use gathers::kmeans::{KMeansConfig, ReductionConfig};

    use super::check_memory;

    #[test]
    fn reduction_method_names() {
        for (value, expected) in [
            ("auto", ReductionConfig::Auto),
            ("raw", ReductionConfig::None),
            (
                "pca",
                ReductionConfig::PCA {
                    output_dim: 128,
                    training_samples: None,
                },
            ),
            ("srht", ReductionConfig::SRHT { output_dim: 128 }),
        ] {
            assert_eq!(value.parse::<ReductionConfig>().unwrap(), expected);
        }
        assert_eq!(
            "svd".parse::<ReductionConfig>().unwrap_err().to_string(),
            "reduction must be auto, raw, pca, or srht"
        );
    }

    #[test]
    fn plan_uses_source_rows_and_checks_memory_before_loading() {
        let config = KMeansConfig::default().resolve(10_000_000, 768).unwrap();
        let clusters = config.n_clusters.unwrap();
        assert_eq!(clusters, 24_881);
        let samples = config.training_samples.unwrap();
        assert_eq!(samples, 6_369_536);
        assert!(
            check_memory(samples, 768, clusters, 4096, None, 0, Some(48)).unwrap() < 30_000_000_000
        );
        assert!(check_memory(samples, 768, clusters, 4096, Some(128), 76_800, Some(48)).is_err());
        // An SRHT run (no PCA training rows) is not charged for the covariance or the
        // centered training copy; only the PCA run pays for those buffers.
        let srht = check_memory(samples, 4096, clusters, 4096, Some(128), 0, None).unwrap();
        let pca = check_memory(samples, 4096, clusters, 4096, Some(128), 76_800, None).unwrap();
        let pca_only_bytes = (76_800 * 4096 + 4096 * 4096 + samples * 4096) as u64 * 4;
        assert_eq!(pca - srht, pca_only_bytes);
        assert!(check_memory(samples, 4096, clusters, 4096, None, 0, Some(48)).is_err());
        // No limit accepts the same workload that Some(48) rejects.
        assert!(check_memory(samples, 4096, clusters, 4096, None, 0, None).is_ok());
        assert!(check_memory(100, 2, 1, 0, None, 0, Some(48)).is_err());
        assert!(check_memory(100, 2, 1, 1, None, 0, Some(0)).is_err());
        assert!(
            KMeansConfig {
                n_clusters: Some(0),
                ..Default::default()
            }
            .resolve(100, 3)
            .is_err()
        );
        assert!(
            KMeansConfig {
                n_clusters: Some(3),
                ..Default::default()
            }
            .resolve(100, 3)
            .is_err()
        );
        if let Ok(rows) = usize::try_from(10_000_000_000_u64) {
            let clusters = KMeansConfig::default()
                .resolve(rows, 768)
                .unwrap()
                .n_clusters
                .unwrap();
            assert_eq!(clusters, 6_250_000);
            // Even the minimum permitted sample is too large; fast index selection
            // does not make this default clustering configuration fit in local RAM.
            assert!(
                check_memory(
                    clusters as usize * 39,
                    768,
                    clusters,
                    4096,
                    None,
                    0,
                    Some(48)
                )
                .is_err()
            );
        }
    }
}
