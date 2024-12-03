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

    def assign(self, vec: np.ndarray, centroids: np.ndarray) -> int:
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
        return assign(vec, centroids)

    def batch_assign(self, vecs: np.ndarray, centroids: np.ndarray) -> list[int]:
        """
        Assign each vector to the nearest centroid with RaBitQ.

        This method does not guarantee it's 100% accurate, but it's usually faster,
        especially when the dim is large.

        Returns:
            list[int]: The list of the assigned labels.
        """
        assert (
            len(vecs.shape) == MATRIX_SHAPE
            and len(centroids.shape) == MATRIX_SHAPE
            and vecs.shape[1] == centroids.shape[1]
        )
        return batch_assign(vecs, centroids)

    def fit(self, vecs: np.ndarray, n_cluster: int, max_iter: int = 25) -> np.ndarray:
        """
        Cluster the vectors into `n_cluster` clusters with max iteration `max_iter`.

        Returns:
            np.ndarray: The centroids of the clusters.
        """
        assert len(vecs.shape) == MATRIX_SHAPE
        return kmeans_fit(vecs, n_cluster, max_iter)
