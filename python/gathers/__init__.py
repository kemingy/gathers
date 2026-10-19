from __future__ import annotations

from os import environ

import numpy as np

from .gatherspy import assign, batch_assign, kmeans_fit

__all__ = ["Gathers"]

MATRIX_SHAPE = 2


def _require_float32(name: str, array: np.ndarray, ndim: int) -> None:
    if array.dtype != np.float32:
        raise ValueError(f"{name} must contain float32 values, got {array.dtype}")
    if array.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}-D array, got {array.ndim}-D")


class Gathers:
    def __init__(self, verbose: bool = False):
        if verbose:
            environ["GATHERS_LOG"] = "debug"

    def assign(
        self, vec: np.ndarray, centroids: np.ndarray, *, distance: str = "l2"
    ) -> int:
        """
        Assign the vector to the nearest centroid.

        Uses exact direct SIMD scoring. Use batch_assign for throughput over many rows.
        """
        _require_float32("vec", vec, 1)
        _require_float32("centroids", centroids, MATRIX_SHAPE)
        if vec.shape[0] != centroids.shape[1]:
            raise ValueError(
                f"vec dimension {vec.shape[0]} must match centroids dimension "
                f"{centroids.shape[1]}"
            )
        return assign(
            np.ascontiguousarray(vec), np.ascontiguousarray(centroids), distance
        )

    def batch_assign(
        self,
        vecs: np.ndarray,
        centroids: np.ndarray,
        *,
        distance: str = "l2",
        assignment: str = "auto",
        seed: int | None = None,
    ) -> list[int]:
        """
        Assign vectors using the same policy as training, in the supplied space.

        Auto uses pilot-calibrated hierarchy probing for 3,136+ centroids, otherwise
        exact flat assignment. Probing is approximate for all metrics. Exact searches
        every centroid with guarded GEMM or SIMD. No projection is applied.
        Set seed for repeatable hierarchy and pilots within the same build/worker count.
        Inputs are unchanged; empty query batches return an empty list.

        Returns:
            list[int]: The list of the assigned labels.
        """
        if seed is not None and seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}")
        _require_float32("vecs", vecs, MATRIX_SHAPE)
        _require_float32("centroids", centroids, MATRIX_SHAPE)
        if vecs.shape[1] != centroids.shape[1]:
            raise ValueError(
                f"vecs dimension {vecs.shape[1]} must match centroids dimension "
                f"{centroids.shape[1]}"
            )
        return batch_assign(
            np.ascontiguousarray(vecs),
            np.ascontiguousarray(centroids),
            distance,
            assignment=assignment,
            seed=seed,
        )

    # Keep the training options explicit rather than introduce a configuration wrapper.
    def fit(  # noqa: PLR0913
        self,
        vecs: np.ndarray,
        n_cluster: int | None = None,
        max_iter: int = 10,
        *,
        distance: str = "l2",
        assignment: str = "auto",
        reduction: str = "auto",
        reduced_dim: int | None = None,
        samples_per_cluster: int = 256,
        training_samples: int | None = None,
        projection_training_samples: int | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Cluster vectors, choosing the cluster count automatically when n_cluster is None.

        Distance is l2, cos, or dot. Default auto reduction selects PCA to 128 dimensions
        when source rows >= 1,000,000 and input dimension > 196, otherwise raw training.
        Explicit raw/pca/srht overrides auto; pca/srht default reduced_dim to 128.
        K-means samples min(source rows, samples_per_cluster * clusters); the factor
        defaults to 256 and must be at least 39. An exact training_samples overrides it.
        PCA fitting defaults to
        at most 100 * input dimension rows, or projection_training_samples if supplied.
        Cosine rows are normalized before fitting PCA and before projected dot training.
        Assignment auto uses a rebuilt, pilot-calibrated hierarchy for 3,136+ clusters;
        this is approximate. Set assignment="exact" for flat Lloyd assignment.
        Input arrays are unchanged. Set seed for reproducibility within the same build.

        Returns:
            np.ndarray: Centroids in the original input dimension, even with reduction.
        """
        _require_float32("vecs", vecs, MATRIX_SHAPE)
        for name, value in [
            ("n_cluster", n_cluster),
            ("max_iter", max_iter),
            ("reduced_dim", reduced_dim),
            ("samples_per_cluster", samples_per_cluster),
            ("training_samples", training_samples),
            ("projection_training_samples", projection_training_samples),
            ("seed", seed),
        ]:
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        return kmeans_fit(
            np.ascontiguousarray(vecs),
            n_cluster,
            max_iter,
            distance=distance,
            assignment=assignment,
            reduction=reduction,
            reduced_dim=reduced_dim,
            samples_per_cluster=samples_per_cluster,
            training_samples=training_samples,
            projection_training_samples=projection_training_samples,
            seed=seed,
        )
