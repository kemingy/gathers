use std::borrow::Cow;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Instant;

use aligned_vec::AVec;
use anyhow::{Context, Result, bail, ensure};
use argh::FromArgs;
use gathers::distance::Distance;
use gathers::kmeans::KMeans;
use gathers::reduction::{PCA, SRHT};
use gathers::sampling::sample_indices;
use gathers::utils::try_normalize_rows;
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
    /// exact training sample size; defaults to min(input rows, 256 * clusters)
    #[argh(option)]
    training_samples: Option<usize>,
    /// training distance: l2, cos, or dot
    #[argh(option, default = "String::from(\"l2\")")]
    distance: String,
    /// dimensionality reduction: raw, pca, or srht
    #[argh(option, default = "String::from(\"raw\")")]
    reduction: String,
    /// projected dimension; required for pca and srht
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
        let pca_workspace =
            projection_training_samples as u128 * dim as u128 * 4 + dim as u128 * dim as u128 * 4;
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

#[allow(
    clippy::upper_case_acronyms,
    reason = "Match the public reduction type names"
)]
enum Projection {
    PCA(PCA),
    SRHT(SRHT),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(
    clippy::upper_case_acronyms,
    reason = "Match the public reduction type names"
)]
enum ReductionMethod {
    Raw,
    PCA,
    SRHT,
}

impl FromStr for ReductionMethod {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "raw" => Ok(Self::Raw),
            "pca" => Ok(Self::PCA),
            "srht" => Ok(Self::SRHT),
            _ => bail!("unknown reduction '{value}'; expected raw, pca, or srht"),
        }
    }
}

impl Projection {
    fn transform(&self, vectors: &[f32]) -> Result<AVec<f32>> {
        match self {
            Self::PCA(model) => Ok(model.transform(vectors)?),
            Self::SRHT(model) => Ok(model.transform(vectors)?),
        }
    }

    fn inverse_transform(&self, vectors: &[f32]) -> Result<AVec<f32>> {
        match self {
            Self::PCA(model) => Ok(model.inverse_transform(vectors)?),
            Self::SRHT(model) => Ok(model.inverse_transform(vectors)?),
        }
    }
}

fn reconstruct_original_centroids(
    projection: &Projection,
    original_vectors: &[f32],
    reduced_centroids: &[f32],
    labels: &[u32],
    input_dim: usize,
    output_dim: usize,
    distance: Distance,
) -> Result<(AVec<f32>, usize)> {
    let num_centroids = reduced_centroids.len() / output_dim;
    debug_assert_eq!(labels.len(), original_vectors.len() / input_dim);

    // An inverse projection cannot recover components in a projection's null space. Use it only
    // as a deterministic fallback for an empty final cluster; occupied centroids are the exact
    // original-space means of the rows assigned by reduced-space K-means, unless that mean is zero.
    let mut centroids = projection.inverse_transform(reduced_centroids)?;
    let mut counts = vec![0_usize; num_centroids];
    let mut representatives = if distance == Distance::SquaredEuclidean {
        Vec::new()
    } else {
        vec![None; num_centroids]
    };
    for (row, (&label, vector)) in labels
        .iter()
        .zip(original_vectors.chunks_exact(input_dim))
        .enumerate()
    {
        let index = label as usize;
        let centroid = &mut centroids[index * input_dim..(index + 1) * input_dim];
        if counts[index] == 0 {
            centroid.fill(0.0);
        }
        if distance != Distance::SquaredEuclidean
            && representatives[index].is_none()
            && vector.iter().any(|&value| value != 0.0)
        {
            representatives[index] = Some(row);
        }
        counts[index] += 1;
        centroid
            .iter_mut()
            .zip(vector)
            .for_each(|(sum, &value)| *sum += value);
    }
    for (index, &count) in counts.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let inverse = (count as f32).recip();
        for value in &mut centroids[index * input_dim..(index + 1) * input_dim] {
            *value *= inverse;
        }
    }
    let empty_clusters = counts.iter().filter(|&&count| count == 0).count();
    if distance != Distance::SquaredEuclidean {
        for (index, centroid) in centroids.chunks_exact_mut(input_dim).enumerate() {
            if centroid.iter().all(|&value| value == 0.0) {
                // A zero mean has no preferred dot-product direction. For cosine output,
                // preserve a unit centroid using an assigned row (or any row if empty).
                let representative =
                    representatives[index].or_else(|| (distance == Distance::Cosine).then_some(0));
                if let Some(row) = representative {
                    centroid
                        .copy_from_slice(&original_vectors[row * input_dim..(row + 1) * input_dim]);
                }
            }
        }
        try_normalize_rows(
            &mut centroids,
            input_dim,
            distance == Distance::NegativeDotProduct,
        )?;
    }
    Ok((centroids, empty_clusters))
}

fn projection_training_data<'a>(
    vectors: &'a [f32],
    dim: usize,
    rows: usize,
    seed: u64,
) -> Cow<'a, [f32]> {
    if rows == vectors.len() / dim {
        return Cow::Borrowed(vectors);
    }
    let mut indices = sample_indices(
        vectors.len() / dim,
        rows,
        &mut StdRng::seed_from_u64(seed ^ 0x5043_415f_5341_4d50),
    );
    indices.sort_unstable();
    let mut sample = Vec::with_capacity(rows * dim);
    for index in indices {
        sample.extend_from_slice(&vectors[index * dim..(index + 1) * dim]);
    }
    Cow::Owned(sample)
}

pub(crate) fn run(args: &Args, common: &crate::Args) -> Result<()> {
    ensure!(args.max_iter > 0, "max-iter must be positive");
    let mut reader = fvecs::Reader::open(&args.input)?;
    let num_vectors = reader.rows;
    let dim = reader.dim;
    let num_clusters = cluster_count(num_vectors, args.n_cluster)?;
    let distance = args.distance.parse::<Distance>()?;
    let reduction = args.reduction.parse::<ReductionMethod>()?;
    // Projected zero rows have no preferred direction. Normalize nonzero projections below,
    // then use dot training, which permits zero rows; original cosine rows are still validated.
    let training_distance = if distance == Distance::Cosine && reduction != ReductionMethod::Raw {
        Distance::NegativeDotProduct
    } else {
        distance
    };
    let kmeans =
        KMeans::new(num_clusters, args.max_iter, 0.01, training_distance, false).seed(common.seed);
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
    let reduced_dim = match reduction {
        ReductionMethod::Raw => {
            ensure!(
                args.reduced_dim.is_none() && args.projection_training_samples.is_none(),
                "raw reduction does not accept projection options"
            );
            None
        }
        ReductionMethod::PCA | ReductionMethod::SRHT => {
            let reduced_dim = args
                .reduced_dim
                .context("--reduced-dim is required for pca and srht")?;
            ensure!(
                reduced_dim > 0 && reduced_dim < dim,
                "reduced-dim must be positive and smaller than input dimension"
            );
            Some(reduced_dim)
        }
    };
    ensure!(
        reduction == ReductionMethod::PCA || args.projection_training_samples.is_none(),
        "projection-training-samples only applies to PCA"
    );
    let projection_training_rows = if reduction == ReductionMethod::PCA {
        args.projection_training_samples
            .unwrap_or_else(|| training_rows.min(100 * dim))
    } else {
        0
    };
    if reduction == ReductionMethod::PCA {
        ensure!(
            (2..=training_rows).contains(&projection_training_rows),
            "projection-training-samples must be between 2 and training-samples"
        );
    }
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
    let mut vectors = reader
        .read_sample(&indices, args.batch_rows, args.validate_all)
        .with_context(|| format!("cannot sample fvecs from {}", args.input.display()))?;
    let sample_read_ms = start.elapsed().as_secs_f64() * 1_000.0;
    drop(indices);
    drop(reader);
    let mut normalization_ms = 0.0;
    if distance == Distance::Cosine {
        let start = Instant::now();
        try_normalize_rows(&mut vectors.data, dim, false)?;
        normalization_ms += start.elapsed().as_secs_f64() * 1_000.0;
    }
    let projection_fit_start = Instant::now();
    let projection = match reduction {
        ReductionMethod::Raw => None,
        ReductionMethod::PCA => {
            let training =
                projection_training_data(&vectors.data, dim, projection_training_rows, common.seed);
            Some(Projection::PCA(PCA::fit(
                &training,
                dim,
                reduced_dim.unwrap(),
            )?))
        }
        ReductionMethod::SRHT => Some(Projection::SRHT(SRHT::new(
            dim,
            reduced_dim.unwrap(),
            common.seed,
        )?)),
    };
    let projection_fit_ms = projection_fit_start.elapsed().as_secs_f64() * 1_000.0;
    let preserved_variance = match &projection {
        Some(Projection::PCA(model)) => Some(model.preserved_variance()),
        _ => None,
    };
    let training_dim = reduced_dim.unwrap_or(dim);
    let projection_transform_start = Instant::now();
    let (mut training_vectors, original_vectors) = if let Some(projection) = &projection {
        let transformed = projection.transform(&vectors.data)?;
        (transformed, Some(vectors.data))
    } else {
        (vectors.data, None)
    };
    let projection_transform_ms = projection_transform_start.elapsed().as_secs_f64() * 1_000.0;
    if distance == Distance::Cosine && projection.is_some() {
        let start = Instant::now();
        try_normalize_rows(&mut training_vectors, training_dim, true)?;
        normalization_ms += start.elapsed().as_secs_f64() * 1_000.0;
    }
    let prepare_ms = prepare_start.elapsed().as_secs_f64() * 1_000.0;
    if common.wait_for_profiler {
        wait_for_profiler()?;
    }
    let start = Instant::now();
    let (reduced_centroids, labels) = kmeans.fit_sample(training_vectors, training_dim);
    let fit_ms = start.elapsed().as_secs_f64() * 1_000.0;
    let num_centroids = reduced_centroids.len() / training_dim;
    let reconstruction_start = Instant::now();
    let (centroids, reconstruction_empty_clusters) = match (original_vectors, projection.as_ref()) {
        (Some(original_vectors), Some(projection)) => reconstruct_original_centroids(
            projection,
            &original_vectors,
            &reduced_centroids,
            &labels,
            dim,
            training_dim,
            distance,
        )?,
        (None, None) => (reduced_centroids, 0),
        _ => unreachable!(),
    };
    let reconstruction_ms = reconstruction_start.elapsed().as_secs_f64() * 1_000.0;
    fvecs::write(&args.output, &centroids, dim)?;
    report(serde_json::json!({
        "command": "kmeans",
        "input": args.input,
        "output": args.output,
        "num_vectors": num_vectors,
        "training_rows": training_rows,
        "distance": args.distance,
        "normalization_ms": normalization_ms,
        "reduction": args.reduction,
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
    use gathers::distance::Distance;
    use gathers::kmeans::KMeans;
    use gathers::reduction::SRHT;

    use super::{
        Projection, ReductionMethod, check_memory, cluster_count, reconstruct_original_centroids,
    };

    #[test]
    fn reduction_method_names() {
        for (value, expected) in [
            ("raw", ReductionMethod::Raw),
            ("pca", ReductionMethod::PCA),
            ("srht", ReductionMethod::SRHT),
        ] {
            assert_eq!(value.parse::<ReductionMethod>().unwrap(), expected);
        }
        assert_eq!(
            "svd".parse::<ReductionMethod>().unwrap_err().to_string(),
            "unknown reduction 'svd'; expected raw, pca, or srht"
        );
    }

    #[test]
    fn projected_assignments_reconstruct_original_space_means() {
        let projection = Projection::SRHT(SRHT::new(2, 1, 42).unwrap());
        let original = [-1.0, 100.0, -2.0, 200.0, 1.0, 300.0, 2.0, 400.0];
        let reduced_centroids = [-1.5, 1.5, 7.0];
        let fallback = projection.inverse_transform(&reduced_centroids).unwrap();
        let labels = [0, 0, 1, 1];
        let (centroids, empty) = reconstruct_original_centroids(
            &projection,
            &original,
            &reduced_centroids,
            &labels,
            2,
            1,
            Distance::SquaredEuclidean,
        )
        .unwrap();
        assert_eq!(empty, 1);
        assert_eq!(&centroids[..4], &[-1.5, 150.0, 1.5, 350.0]);
        assert_eq!(&centroids[4..], &fallback[4..]);
    }

    #[test]
    fn projected_antipodal_cluster_uses_an_assigned_unit_direction() {
        let projection = Projection::SRHT(SRHT::new(2, 1, 42).unwrap());
        let original = [1.0, 0.0, -1.0, 0.0];
        let (centroids, empty) = reconstruct_original_centroids(
            &projection,
            &original,
            &[1.0, 0.0],
            &[0, 0],
            2,
            1,
            Distance::Cosine,
        )
        .unwrap();
        assert_eq!(empty, 1);
        assert_eq!(&*centroids, &[1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn projected_dot_training_can_preserve_zero_centroids() {
        let projection = Projection::SRHT(SRHT::new(2, 1, 42).unwrap());
        let (centroids, empty) = reconstruct_original_centroids(
            &projection,
            &[0.0, 0.0, 0.0, 0.0],
            &[0.0, 0.0],
            &[0, 0],
            2,
            1,
            Distance::NegativeDotProduct,
        )
        .unwrap();
        assert_eq!(empty, 1);
        assert_eq!(&*centroids, &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn plan_uses_source_rows_and_checks_memory_before_loading() {
        let clusters = cluster_count(10_000_000, None).unwrap();
        assert_eq!(clusters, 24_881);
        let model = KMeans::new(clusters, 1, 0.01, Distance::SquaredEuclidean, false);
        let samples = model.training_sample_size(10_000_000);
        assert_eq!(samples, 6_369_536);
        assert!(
            check_memory(samples, 768, clusters, 4096, None, 0, Some(48)).unwrap() < 30_000_000_000
        );
        assert!(check_memory(samples, 768, clusters, 4096, Some(128), 76_800, Some(48)).is_err());
        assert!(check_memory(samples, 4096, clusters, 4096, None, 0, Some(48)).is_err());
        // No limit accepts the same workload that Some(48) rejects.
        assert!(check_memory(samples, 4096, clusters, 4096, None, 0, None).is_ok());
        assert!(check_memory(100, 2, 1, 0, None, 0, Some(48)).is_err());
        assert!(check_memory(100, 2, 1, 1, None, 0, Some(0)).is_err());
        assert!(cluster_count(100, Some(0)).is_err());
        assert!(cluster_count(100, Some(3)).is_err());
        if let Ok(rows) = usize::try_from(10_000_000_000_u64) {
            let clusters = cluster_count(rows, None).unwrap();
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
