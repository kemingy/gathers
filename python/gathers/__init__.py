from __future__ import annotations

from os import environ

import numpy as np

from .gatherspy import assign, batch_assign, kmeans_fit

__all__ = ["Gathers"]

MATRIX_SHAPE = 2


class Gathers:
    def __init__(self, verbose: bool = False):
        if verbose:
            environ["GATHERS_LOG"] = "debug"

    def assign(
        self, vec: np.ndarray, centroids: np.ndarray, *, distance: str = "l2"
    ) -> int:
        """
        Assign the vector to the nearest centroid.

        This method is slower than :py:meth:`~Gathers.batch_assign`, but it's
        100% accurate.
        """
        assert (
            len(vec.shape) == 1
            and len(centroids.shape) == MATRIX_SHAPE
            and vec.shape[0] == centroids.shape[1]
        )
        return assign(
            np.ascontiguousarray(vec), np.ascontiguousarray(centroids), distance
        )

    def batch_assign(
        self, vecs: np.ndarray, centroids: np.ndarray, *, distance: str = "l2"
    ) -> list[int]:
        """
        Assign vectors with RaBitQ for l2, or exact normalized-dot/dot scoring for cos/dot.

        The l2 RaBitQ path is approximate and usually faster for large dimensions.
        Cosine and dot use exact assignment.

        Returns:
            list[int]: The list of the assigned labels.
        """
        assert (
            len(vecs.shape) == MATRIX_SHAPE
            and len(centroids.shape) == MATRIX_SHAPE
            and vecs.shape[1] == centroids.shape[1]
        )
        return batch_assign(
            np.ascontiguousarray(vecs), np.ascontiguousarray(centroids), distance
        )

    # Keep the training options explicit rather than introduce a configuration wrapper.
    def fit(  # noqa: PLR0913
        self,
        vecs: np.ndarray,
        n_cluster: int | None = None,
        max_iter: int = 10,
        *,
        distance: str = "l2",
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
        Input arrays are unchanged. Set seed for reproducibility within the same build.

        Returns:
            np.ndarray: Centroids in the original input dimension, even with reduction.
        """
        assert len(vecs.shape) == MATRIX_SHAPE
        return kmeans_fit(
            np.ascontiguousarray(vecs),
            n_cluster,
            max_iter,
            distance=distance,
            reduction=reduction,
            reduced_dim=reduced_dim,
            samples_per_cluster=samples_per_cluster,
            training_samples=training_samples,
            projection_training_samples=projection_training_samples,
            seed=seed,
        )
