import numpy as np

from gathers import assign, batch_assign, kmeans_fit

NUM = 1000
CLUSTER = 10
DIM = 16
RABITQ_MATCH_RATE = 0.99


def test_kmeans():
    rng = np.random.default_rng()
    arr = rng.random((NUM, DIM), dtype=np.float32)
    c = kmeans_fit(arr, CLUSTER, 10)
    assert c.shape == (CLUSTER, DIM), c.shape

    for vec in arr:
        distances = np.linalg.norm(c - vec, axis=1)
        assert np.argmin(distances) == assign(vec, c)


def test_rabitq():
    rng = np.random.default_rng()
    arr = rng.random((NUM, DIM), dtype=np.float32)
    c = kmeans_fit(arr, CLUSTER, 10)
    assert c.shape == (CLUSTER, DIM), c.shape
    labels = batch_assign(arr, c)
    assert len(labels) == len(arr)
    expect = []

    for vec in arr:
        distances = np.linalg.norm(c - vec, axis=1)
        expect.append(np.argmin(distances))

    match_rate = np.sum(np.array(expect) == np.array(labels)) / NUM
    assert match_rate > RABITQ_MATCH_RATE, match_rate
