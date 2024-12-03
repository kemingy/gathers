from pathlib import Path
from struct import pack, unpack

import numpy as np


def read_vec(filepath: str, vec_type: np.dtype = np.float32):
    """Read vectors from a file. Support `fvecs`, `ivecs` and `bvecs` format.
    Args:
        filepath: The path of the file.
        vec_type: The type of the vectors.
    """
    size = np.dtype(vec_type).itemsize
    with open(filepath, "rb") as f:
        vecs = []
        while True:
            try:
                buf = f.read(4)
                if len(buf) == 0:
                    break
                dim = unpack("<i", buf)[0]
                vecs.append(np.frombuffer(f.read(dim * size), dtype=vec_type))
            except Exception as err:
                print(err)
                break
    return np.array(vecs)


def write_vec(filepath: str, vecs: np.ndarray, vec_type: np.dtype = np.float32):
    """Write vectors to a file. Support `fvecs`, `ivecs` and `bvecs` format."""
    with open(filepath, "wb") as f:
        for vec in vecs:
            f.write(pack("<i", len(vec)))
            f.write(vec.tobytes())


def generate_random(dim: int, num: int, query: int, topk: int, filepath: str):
    rng = np.random.default_rng()
    x = rng.integers(0, 256, (num, dim), dtype=np.uint8).astype(np.float32)
    mean = x.mean(axis=0)
    q = rng.integers(0, 256, (query, dim), dtype=np.uint8).astype(np.float32)
    # labels = [[np.argmin(np.linalg.norm(x - vec, axis=1))] for vec in q]
    labels = [
        np.argpartition(np.linalg.norm(x - vec, axis=1), topk)[:topk] for vec in q
    ]

    path = Path(filepath)
    path.mkdir(parents=True, exist_ok=True)
    write_vec(path / "base.fvecs", x)
    write_vec(path / "centroids.fvecs", np.array([mean]))
    write_vec(path / "query.fvecs", q)
    write_vec(path / "groundtruth.ivecs", np.array(labels, dtype=np.int32))


if __name__ == "__main__":
    generate_random(100, 1000, 10, 1, "dataset-random")
