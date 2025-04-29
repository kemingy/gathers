import numpy as np

from gathers import Gathers

NUM = 1000
CLUSTER = 10
DIM = 64
RABITQ_MATCH_RATE = 0.99


def test_rabitq():
    gathers = Gathers(verbose=True)
    rng = np.random.default_rng()

    for i in range(100):
        arr = rng.random((NUM, DIM), dtype=np.float32)
        c = gathers.fit(arr, CLUSTER)
        assert c.shape == (CLUSTER, DIM), c.shape

        # test `assign`
        for vec in arr:
            distances = np.linalg.norm(c - vec, axis=1)
            assert np.argmin(distances) == gathers.assign(vec, c)

        # test `batch_assign`
        labels = gathers.batch_assign(arr, c)
        assert len(labels) == len(arr)
        expect = [np.argmin(np.linalg.norm(c - vec, axis=1)) for vec in arr]
        match_rate = np.sum(np.array(expect) == np.array(labels)) / NUM
        assert match_rate > RABITQ_MATCH_RATE, f"failed at {i} with rate {match_rate}"
