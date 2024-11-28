import numpy as np

from gathers import assign, kmeans_fit


def test_kmeans():
    arr = np.random.rand(1000, 8).astype(np.float32)
    c = kmeans_fit(arr, 10, 10)
    assert c.shape == (10, 8), c.shape

    for vec in arr:
        distances = np.linalg.norm(c - vec, axis=1)
        assert np.argmin(distances) == assign(vec, c)
