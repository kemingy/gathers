# Reduced-space K-means

`gathers kmeans` supports `--reduction raw|pca|srht` (default `raw`). PCA and SRHT reduce the
dimension used for centroid assignment. The CLI writes centroids in the **original** dimension, so
the existing `assign` command needs no projection. The Rust transforms are also available as
`gathers::reduction::{PCA, SRHT}`.

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

`--reduced-dim` is required for PCA and SRHT and must be smaller than the input dimension.
`--projection-training-samples` applies only to PCA. Its default is the smaller of the K-means
training sample and `100 × input_dimension` rows. PCA uses covariance eigenvectors in descending
variance order; it does not whiten or standardize features. It computes the mean in `f64`, forms an
`f32` covariance matrix, and transforms rows with `faer` matrix multiplication. SRHT applies seeded
random signs, zero-pads to the next power of two, runs a fast Walsh-Hadamard transform, and samples
output coordinates.

The CLI samples row indices from the fvecs source before loading vectors, so it does not load the
whole corpus. It keeps the selected sample in RAM for K-means. A projected run temporarily holds
both original and reduced samples for centroid reconstruction; PCA transformation also creates a
centered copy. `--memory-limit-gb` estimates these buffers but is not an enforced RSS limit. See
the [profiling guide](./profiling.md) for sampling and memory details.

K-means returns its final reduced-space labels. The CLI averages the corresponding **original**
sample rows to produce each occupied full-dimensional centroid. Inverting a projected centroid
would lose its null-space component, particularly with SRHT. An inverse projection is used only
for an empty final cluster. SRHT reconstructs in padded space and discards padding; with a
non-power-of-two input dimension, projecting that approximation again may change its coordinates.
For `cos` and `dot`, a zero original-space mean uses an assigned nonzero row as a direction;
a dot cluster containing only zero rows may remain zero.

`--distance cos` requires finite nonzero original rows and normalizes them before projection.
Nonzero projected rows are normalized before dot training; rows that become zero through
centering or projection are preserved and contribute a tied dot score to every centroid.
Raw cosine training normalizes rows inside K-means fitting. `dot` preserves input magnitudes.
Both use negative-dot-product assignment and unit nonzero centroids. For large samples, `l2`
may use approximate RaBitQ assignment; `cos` and `dot` use exact assignment. Do not compare their
timings as if only the metric changed.

The JSON report separates `projection_fit_ms`, `projection_transform_ms`, `fit_ms`, and
`reconstruction_ms`. `prepare_ms` includes sampling, reading, projection fitting, transformation,
and normalization of original and projected rows. `normalization_ms` measures both normalization
steps in projected cosine runs. `fit_ms` includes raw cosine's internal normalization, but not
source I/O. The reported training-stage total is `prepare_ms + fit_ms + reconstruction_ms`; it
excludes writing the output fvecs file.
