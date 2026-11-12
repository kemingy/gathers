# Native Rust e-means

Gathers implements e-means in the Rust library alongside Lloyd k-means. It uses the existing
exact SIMD/matrix assignment kernels and Rayon. Python calls the same Rust trainer through
PyO3; neither the library, CLI, nor Python training option requires Faiss or SciPy.
Lloyd remains the default.

## Usage

```rust
use gathers::kmeans::{EmeansConfig, KMeans, KMeansConfig, TrainingAlgorithm};
use gathers::utils::as_continuous_vec;

let trainer = KMeans::new(KMeansConfig {
    n_clusters: Some(1),
    algorithm: TrainingAlgorithm::Emeans(EmeansConfig::default()),
    max_iter: Some(3),
    seed: Some(42),
    ..Default::default()
});
let fit = trainer.fit_sample(as_continuous_vec(&[[1.0, 2.0]; 40]), 2)?;
```

```python
from gathers import Gathers

centroids = Gathers().fit(data, 512, algorithm="emeans", max_iter=3, seed=42)
centroids = Gathers().fit(
    data, 512, algorithm="emeans", max_iter=3, distance="cos",
    reduction="pca", reduced_dim=128, seed=42,
)
```

```sh
gathers --threads 4 --seed 42 kmeans -i vectors.fvecs -o centroids.fvecs \
  -n 512 --algorithm emeans --max-iter 3 --reduction pca --reduced-dim 128
```

`max_iter` / `--max-iter` controls the training budget for both algorithms. Omission selects
three complete shuffled passes for e-means or ten full-batch rounds for Lloyd. Explicit
positive values override either default, including ten passes for e-means. Lloyd can stop
early using its squared-shift tolerance; e-means completes its configured passes. Two passes
favor throughput; three were the more consistent quality choice in the reference study below.
Rust callers select `TrainingAlgorithm::Emeans(EmeansConfig)` and use `max_iter: None` for the
algorithm default or `Some(limit)` for an explicit budget. Python uses `max_iter=None` for
the same default resolution; positional iteration arguments remain supported.

Both algorithms use internal trainer structs with matching fitting interfaces. `KMeans` owns
the shared preparation and reconstruction pipeline; the e-means trainer resolves its recipe
automatically from the prepared training shape and pass budget. `KMeansFit.parameters` records
the resolved `TrainingParameters` for either algorithm. Reporting reads these values from the
completed fit.

Sampling, minimum rows per cluster, normalization, automatic reduction, and source-metadata
resolution follow the existing [training and reduction contracts](./reduction.md).
PCA/SRHT reconstruct occupied centroids by averaging original-space rows using the final
moving labels. Empty projected clusters retain the inverse-transform fallback. Returned
training labels describe the last moving pass, not a fresh assignment to final centroids.

## Recipe and implementation

The implementation follows Algorithm 1 and the CPU throughput recipe in the
[pinned upstream source](https://github.com/facebookresearch/faiss/tree/2b648f0605f2ef3d14c4705350f8a0b214463878/contrib/e_means)
for [e-means](https://e-means.github.io/):

- Resolve statistical batch size and initial EMA step from retained training rows, clusters,
  and pass budget, after projection. Keep remainder batches.
- Anneal the step with the upstream cosine schedule towards one tenth of the initial step.
  Updates use steps `0..total_steps`; the last executed step stays above that endpoint.
- Start cluster masses at `2^-16`; after every fifth update, revive low-mass clusters using
  distinct donors sampled in proportion to their masses and symmetric perturbations.
  The default relative threshold is 0.1, disabled at training dimensions at most two.
- During the last pass, accumulate unweighted closing means in shuffled order and store labels
  in source row order. Unassigned clusters keep their current positions.

Rust `EmeansConfig` additionally exposes `batch_size`, `alpha0`, and `revival_threshold` for
controlled experiments. `resolve(rows, training_dim, clusters, max_iter)` returns an `EmeansRecipe`
without allocating vector buffers. Changing batch size alone recomputes the coupled step law.
The configured step must remain positive when represented as FP32 throughout the schedule.
The CLI JSON records the recipe used by the trainer separately from its existing stage timings.

Logical batches can exceed the physical assignment buffer. Assignment chunks use at most
32 MiB of vector data, or one exceptionally wide row, with one unchanged centroid snapshot
for the entire logical batch. Permutations, labels, sums, counts, directions, and revival
scratch are reused. Matrix assignment computes source-vector norms once and gathers cached
norms in the same shuffled order as each chunk, preserving the assignment error bounds.
Independent row copies use Rayon; statistics still accumulate in shuffled order.
E-means uses exact assignment for all metrics; unlike Lloyd's large L2 path, it does not rebuild
a RaBitQ index after each batch. The `perf` feature uses the existing single-thread assignment
path rather than matrix/Rayon assignment.

EMA positions and ordinary batch sums use FP32. Overflowing batch sums are retried in FP64;
closing sums use FP64 to preserve finite means. Unrepresentable revival perturbations return
`KMeansError::NumericalOverflow`. Cosine uses normalized directions for assignment while retaining
unnormalized EMA positions. A cancelling final direction uses a nonzero sample representative,
consistent with gathers' cosine output contract; an all-zero projected sample is reconstructed
in the original space. Dot training preserves source magnitudes. These numerical choices and
Rust's random permutations mean results need not be bitwise identical to upstream NumPy.

## Validation and scope

Tests cover pinned upstream recipe/update fixtures, schedule endpoints, closing means, remainder
batches and source-order labels, assignment-chunk invariance, weighted revival, zero directions,
overflow fallback, and raw/PCA/SRHT integration. Python and CLI tests exercise the native option
and preserve the Lloyd default. The Python test also runs with Faiss import disabled.

## Native/reference comparison

The archived study compared ten-round gathers with pinned NumPy/Faiss and native e-means
at two and three passes. It used SIFT/GIST L2 and Cohere cosine, each in raw space and
with shared Rust PCA (SIFT to 64 dimensions; GIST/Cohere to 128). Each case retained 100,000
training rows and 10,000 disjoint held-out rows, 512 clusters, training seeds 11/22/33, and a
separate fixed sample/PCA seed. Cases ran sequentially on Apple M3 Max, macOS arm64, release
Rust with NEON/Accelerate, four Rayon/Faiss workers and one BLAS thread.

Quality used fresh exact assignment in original space and FP64 loss accumulation. Initial
centroids and permutations differ between Rust and NumPy, so paired seeds test practical
quality rather than identical stochastic trajectories. Fixed-batch upstream fixtures separately
check L2 and spherical update arithmetic. Raw timings were warm API fits that include native
validation/input ownership and the Python boundary, while excluding extension loading, dataset
I/O and evaluation. PCA timings exclude bridge I/O/input cloning; compute totals charge shared
preparation to every method and include original-space reconstruction. These are not full
production CLI wall times.

Held-out quality stayed close to upstream throughout; small differences went in both directions.
Three-pass native training was faster than upstream on every tested seed in all six cases.
Two-pass native training was also faster by median throughout; one raw Cohere seed was
essentially tied and slightly slower. PCA compute totals favored native training by median
after charging shared preparation and reconstruction. These local results support three passes
as the more consistent opt-in choice, with two passes available for throughput. Three passes
also trained faster with lower held-out loss than gathers10 on every tested seed. Two passes
had a small held-out regression against gathers10 on one GIST PCA seed. These trials do not
promise a universal speedup: upstream Python already delegates its main numerical work to
compiled Faiss/NumPy/SciPy, so Python overhead alone does not determine the outcome.

Profiling identified repeated source-norm scans and serial shuffled-row copies. Caching norms
and parallelizing only the copies reduced those costs without changing accumulation order.
Before/after checks preserved native centroids; separate private reports retain each variant.

Full-corpus quality, downstream ANN recall, representative x86 measurements, and large-codebook
tuning remain necessary before considering a default change. Numeric M3 Max results, profiles,
source snapshots, and reproducible commands stay in the private benchmark repository under
`gathers/m3-max/e-means-20261112/native-rust-v3/`.

The earlier reference ablation favored EMA over cumulative-count MacQueen updates and retained
revival, particularly for cosine quality. Closing means improved training loss; their held-out
effect was small. Nearby batch sizes and initial steps changed quality and runtime; the slower
quality batch law did not consistently beat the throughput law. Tiny initial masses behaved
similarly, with no consistent advantage from a unit prior. Extra passes did not improve every
seed. With PCA, original-space mean reconstruction was more reliable across datasets than
inverse-projecting every centroid. A final projected-space reassignment improved quality at the
cost of another assignment pass. The archived scripts retain the single-component controls,
batch/step/prior sensitivity checks, and shared-initialization Faiss comparison.

## Rust benchmarks

`benches/kmeans.rs` compares native Lloyd at ten rounds and e-means at two/three passes through
`KMeans::fit_sample`, using the same generated input and training seed. The `training_algorithm`
group covers L2 and cosine in raw space and with PCA to 128 dimensions. It retains 100,000 rows
of 256 dimensions and 512 clusters. Generation and input cloning happen outside timing; each
fit includes validation, normalization, optional PCA fitting/projection, training and
original-space reconstruction. Throughput counts input rows, since the algorithms perform
different numbers and kinds of updates. This synthetic benchmark tracks runtime regressions;
it does not replace held-out real-data quality or upstream-equivalence checks.

Use a fixed random seed and worker count for comparisons. Keep numeric results private on
M3 Max, including Criterion reports and console output:

```sh
benchmark_dir="$HOME/GitHub/benchmark/gathers/e-means"
mkdir -p "$benchmark_dir"
GATHERS_RANDOM_SEED=20261112 RAYON_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=1 \
  CARGO_TARGET_DIR="$benchmark_dir/target" \
  cargo bench --bench kmeans -- training_algorithm > "$benchmark_dir/criterion.log" 2>&1
```

For fvecs workloads, the release CLI accepts `--algorithm lloyd --max-iter 10` or
`--algorithm emeans --max-iter 2|3` with the same explicit sample, metric, projection and seed.
Its JSON separates preparation, fitting and reconstruction; see the [profiling guide](./profiling.md)
for timing scope. The production tests retain upstream L2/spherical fixtures and PCA/SRHT
integration checks. The temporary Python comparison drivers and PCA bridge are archived with
the private reports rather than shipped as a second benchmark framework.
