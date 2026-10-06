# Reduced-space K-means

`gathers kmeans` supports `--reduction auto|raw|pca|srht` (default `auto`). PCA and SRHT reduce the
dimension used for centroid assignment. The CLI writes centroids in the **original** dimension, so
the existing `assign` command needs no projection. The Rust transforms are also available as
`gathers::reduction::{PCA, SRHT}`. Import `gathers::reduction::Reduction` to use their shared
`input_dim`, `output_dim`, `transform`, and `inverse_transform` methods. Generic consumers can
accept `&dyn Reduction`; construction stays method-specific (`PCA::fit` or `SRHT::new`).
PCA's `preserved_variance()` reports the retained variance fraction; per-component variances
are not stored after fitting. The mean, components, and SRHT sampling details remain private
implementation state.

## Library and Python training

The library owns configuration resolution, sampling, normalization, reduction, fitting, and
original-space centroid reconstruction. Construct `KMeans::new(KMeansConfig { ... })`, then call
`fit` or `fit_sample`. Both return `Result`; there are no reduction-specific fitting methods.

Defaults derive cluster count as max(1, floor(rows^0.8 / 16)), bounded by at least 39 rows per
cluster, and retain min(source rows, samples_per_cluster * clusters) training rows. The factor
defaults to 256; use `KMeansConfig::samples_per_cluster`, `--samples-per-cluster`, or Python's
`samples_per_cluster` to choose another value such as 128 (minimum 39). An exact `training_samples`
count overrides the factor; the unused factor is ignored and is not validated. **Auto reduction selects PCA
to 128 dimensions when original source rows >= 1,000,000 and input dimension > 196; otherwise
it selects raw training.** This is a workload heuristic motivated by the GIST/Cohere comparison,
not a guarantee of retained recall. Use `ReductionConfig::None` or `--reduction raw` to opt out.
Explicit PCA/SRHT choices override the automatic threshold. PCA fitting defaults to
min(selected training rows, 100 * input dimension). The metric is not inferred; it defaults to L2.

`fit` selects its sample and returns original-dimensional centroids. `fit_sample` uses every
supplied row and returns `KMeansFit` with centroids, final training labels, PCA retained variance,
empty projected cluster count, and stage timings. Labels are not recomputed against the returned
original-dimensional centroids. Both sampling settings are ignored by `fit_sample`.

For streaming, call `config.resolve(original_rows, dim)?` before loading vectors. The returned
config contains explicit cluster count, sample size, and projection settings. Read only that
sample and pass the resolved config to `KMeans::new(...).fit_sample(...)`. This preserves the
original source's automatic decisions even if the loaded sample is below the PCA threshold.

```rust
use gathers::distance::Distance;
use gathers::kmeans::{KMeans, KMeansConfig, ReductionConfig};
use gathers::utils::as_continuous_vec;

let rows = vec![vec![1.0, 2.0, 3.0]; 40];
let kmeans = KMeans::new(KMeansConfig {
    n_clusters: Some(1),
    distance: Distance::Cosine,
    seed: Some(42),
    reduction: ReductionConfig::PCA { output_dim: 2, training_samples: None },
    ..Default::default()
});
let fit = kmeans.fit_sample(as_continuous_vec(&rows), 3)?;
assert_eq!(fit.centroids.len(), 3);
```

Python uses the same defaults and resolution. Cluster count can be omitted:

```python
centroids = gathers.fit(data, distance="cos", seed=42)  # auto clusters and reduction
labels = gathers.batch_assign(data, centroids, distance="cos")

# Reduce the training budget without fixing the cluster count.
centroids = gathers.fit(data, samples_per_cluster=128, seed=42)

# Explicit choices override auto.
centroids = gathers.fit(
    data, 10, reduction="pca", reduced_dim=128, distance="cos", seed=42,
    projection_training_samples=10_000,
)
```

Explicit Python/CLI PCA and SRHT default their output dimension to 128 when omitted. It must
remain smaller than the input dimension. Projection options require an explicit PCA/SRHT choice;
`projection_training_samples` applies only to PCA. Inputs remain unchanged; centroids always
have the original dimension.

The sampling factor controls the total uniformly selected training rows, not a quota for each
final cluster. Reducing it can save sample memory and fitting work, but may affect recall. PCA's
dimension-based fitting cap can stay unchanged when both training budgets exceed it.

## Choosing a method

- **Raw** avoids projection and gave the best low-probe recall in our GIST and Cohere runs.
- **PCA** learns high-variance directions from a sample. It retained more recall than SRHT at 128
  dimensions in our tests, but fitting and applying it cost time and memory.
- **SRHT** uses random signs and a fast Hadamard transform, with no fitting step or dense
  projection matrix. It is the faster reduction when some recall loss is acceptable.

This matches the direction of the [reduced-space clustering study](https://arxiv.org/pdf/2608.14648):
PCA preserved more IVF recall than a random JL projection at the same output dimension. Our SRHT
is a structured JL transform, not the dense orthogonal transform used in that study's
implementation. Matryoshka prefixes are another option only when the embedding model was trained
for them; GIST was not. Sparse JL targets sparse inputs, while these benchmarks use dense vectors.

We implemented PCA over the existing `faer` dependency rather than add a second linear-algebra
stack. The Rust PCA crates we reviewed did not match this combination of `f32` vectors, no
feature standardization, and bounded fitting data: `linfa-reduction` fits in `f64`, while
`smartcore` and `efficient_pca` had unsuitable copying or standardization behavior for this path.
`faer` supplies the eigendecomposition; we do not implement an eigensolver.

## CLI and data flow

```sh
gathers --threads 16 --seed 42 kmeans -i vectors.fvecs -o centroids.fvecs \
  --reduction pca --reduced-dim 128 --distance l2
gathers --threads 16 --seed 42 kmeans -i vectors.fvecs -o centroids.fvecs \
  --reduction srht --reduced-dim 128 --distance l2
```

`--reduced-dim` defaults to 128 for explicit PCA/SRHT and must be smaller than the input dimension.
`--projection-training-samples` applies only to PCA. Its default is the smaller of the K-means
training sample and `100 × input_dimension` rows. PCA uses covariance eigenvectors in descending
variance order; it does not whiten or standardize features. It computes the mean in `f64`, forms an
`f32` covariance matrix, and transforms rows with `faer` matrix multiplication. SRHT applies seeded
random signs, zero-pads to the next power of two, runs a fast Walsh-Hadamard transform, and samples
output coordinates.

Finite inputs can still overflow PCA's `f32` residuals or covariance, or either method's transforms.
Both return `ReductionError::NumericalOverflow` (Python `ValueError`) instead of accepting non-finite
intermediates or outputs, including inverse transforms. SRHT checks its unscaled Hadamard workspace,
so it can reject intermediate overflow even when final scaling would make the exact result finite.
Rescale unusually large input values before retrying. Non-finite PCA eigendecomposition results
return `ReductionError::DecompositionFailed`.

K-means retries overflowing centroid sums in `f64` and divides before casting back to `f32`,
for both raw training and original-space reconstruction. Squared centroid shifts also fall back
to `f64` on overflow. Unrepresentable empty-cluster perturbations, L2 residuals, or restored
centroids return `KMeansError::NumericalOverflow` (Python `ValueError`). Assignment and RaBitQ
rotation retain ordinary `f32` arithmetic without per-score overflow checks or magnitude-based
fallbacks. Finite inputs alone do not guarantee representable intermediate scores; rescale extreme
magnitudes before training or assignment. The public `update_centroids` now returns
`Result<f64, KMeansError>`.
Buffer dimensions that overflow addressable sizes return `ReductionError::SizeOverflow`; this
validation does not guarantee enough physical memory for otherwise addressable allocations.

The CLI samples row indices from the fvecs source before loading vectors, so it does not load the
whole corpus. It keeps the selected sample in RAM for K-means. A projected run temporarily holds
both original and reduced samples for centroid reconstruction; PCA transformation also creates a
centered copy. `--memory-limit-gb` estimates these buffers but is not an enforced RSS limit. See
the [profiling guide](./profiling.md) for sampling and memory details.

K-means returns its final reduced-space labels. The library averages the corresponding **original**
sample rows to produce each occupied full-dimensional centroid. Inverting a projected centroid
would lose its null-space component, particularly with SRHT. An inverse projection is used only
for an empty final cluster. SRHT reconstructs in padded space and discards padding; with a
non-power-of-two input dimension, projecting that approximation again may change its coordinates.
For `cos` and `dot`, a zero original-space mean uses an assigned nonzero row as a direction;
a dot cluster containing only zero rows may remain zero.

`--distance cos` requires finite nonzero original rows and normalizes them before projection.
Nonzero projected rows are normalized before dot training; rows that become zero through
centering or projection are preserved and contribute a tied dot score to every centroid.
The library normalizes original cosine rows once during preparation, for both raw and projected
training; CLI and Python do not repeat that normalization. `dot`
preserves input magnitudes.
Both use negative-dot-product assignment and unit nonzero centroids. For large samples, `l2`
may use approximate RaBitQ assignment; `cos` and `dot` use exact assignment. Do not compare their
timings as if only the metric changed.

The JSON report separates `projection_fit_ms`, `projection_transform_ms`, `fit_ms`, and
`reconstruction_ms`. `prepare_ms` includes sampling, reading, projection fitting, transformation,
and normalization/validation of original and projected rows. `normalization_ms` measures both
normalization steps in projected cosine runs, or the single library preparation step in raw cosine runs.
`fit_ms` measures only K-means fitting, not source I/O. The reported training-stage total is
`prepare_ms + fit_ms + reconstruction_ms`; it excludes writing the output fvecs file. For projected
runs, `--wait-for-profiler` pauses before the whole library pipeline, including normalization,
projection fitting and transformation. JSON `reduction` reports the resolved method; `requested_reduction`
records whether the user requested auto or an explicit method.
