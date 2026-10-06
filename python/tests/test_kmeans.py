import numpy as np
import pytest

from gathers import Gathers

NUM = 1000
CLUSTER = 10
DIM = 32
RABITQ_MATCH_RATE = 0.99


@pytest.mark.parametrize("reduction", ["raw", "pca"])
@pytest.mark.parametrize("distance", ["l2", "dot"])
def test_extreme_finite_means_do_not_overflow(reduction, distance):
    data = np.full((40, 2), np.finfo(np.float32).max, dtype=np.float32)
    options = {"reduction": reduction, "distance": distance, "seed": 42}
    if reduction == "pca":
        options["reduced_dim"] = 1
    centroids = Gathers().fit(data, 1, **options)
    expected = data[0] if distance == "l2" else np.full(2, np.sqrt(0.5))
    np.testing.assert_allclose(centroids[0], expected, rtol=1e-6)


def test_centroid_perturbation_overflow_raises_value_error():
    data = np.full((80, 2), np.finfo(np.float32).max, dtype=np.float32)
    with pytest.raises(ValueError, match="K-means arithmetic overflowed"):
        Gathers().fit(data, 2, reduction="raw", seed=42)


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


@pytest.mark.parametrize("reduction", ["raw", "pca", "srht"])
@pytest.mark.parametrize("distance", ["l2", "cos", "dot"])
def test_fit_reduction_returns_original_space_means(reduction, distance):
    # A strided input also verifies the thin Python preprocessing preserves input data.
    data = (np.arange(40 * 6, dtype=np.float32).reshape(40, 6) + 1)[:, ::2]
    original = data.copy()
    options = {"reduction": reduction, "distance": distance, "seed": 42}
    if reduction != "raw":
        options["reduced_dim"] = 1
    if reduction == "pca":
        options["projection_training_samples"] = 20
    gathers = Gathers()
    centroids = gathers.fit(data, 1, 2, **options)
    assert centroids.shape == (1, 3)
    assert centroids.dtype == np.float32
    np.testing.assert_array_equal(data, original)
    np.testing.assert_array_equal(centroids, gathers.fit(data, 1, 2, **options))

    prepared = original.astype(np.float64)
    if distance == "cos":
        prepared /= np.linalg.norm(prepared, axis=1, keepdims=True)
    expected = prepared.mean(axis=0)
    if distance != "l2":
        expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(centroids[0], expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("distance, expected", [("l2", 1), ("cos", 1), ("dot", 0)])
def test_assignment_distance(distance, expected):
    gathers = Gathers()
    vector = np.array([1, 2], dtype=np.float32)
    centroids = np.array([[100, 0], [1, 1]], dtype=np.float32)
    assert gathers.assign(vector, centroids, distance=distance) == expected
    assert gathers.batch_assign(vector[None, :], centroids, distance=distance) == [expected]


@pytest.mark.parametrize("reduction", ["raw", "pca", "srht"])
def test_cosine_original_zero_rows_are_rejected(reduction):
    data = np.ones((40, 3), dtype=np.float32)
    data[3] = 0
    options = {"distance": "cos", "reduction": reduction}
    if reduction != "raw":
        options["reduced_dim"] = 1
    with pytest.raises(ValueError, match="nonzero"):
        Gathers().fit(data, 1, **options)


@pytest.mark.parametrize("reduction", ["pca", "srht"])
def test_projected_zero_rows_and_dot_zero_inputs_are_supported(reduction):
    model = Gathers()
    zeros = np.zeros((40, 3), dtype=np.float32)
    dot = model.fit(zeros, 1, distance="dot", reduction=reduction, reduced_dim=1, seed=42)
    np.testing.assert_array_equal(dot, np.zeros((1, 3), dtype=np.float32))
    # PCA centering maps identical nonzero cosine rows to zero, which is valid after projection.
    if reduction == "pca":
        cosine = model.fit(
            np.ones((40, 3), dtype=np.float32),
            1,
            distance="cos",
            reduction=reduction,
            reduced_dim=1,
            seed=42,
        )
        np.testing.assert_allclose(np.linalg.norm(cosine, axis=1), 1, atol=1e-6)


@pytest.mark.parametrize(
    "options",
    [
        {"reduction": "unknown"},
        {"distance": "unknown"},
        {"reduction": "pca"},
        {"reduction": "srht", "reduced_dim": 3},
        {"reduced_dim": 1},
        {"projection_training_samples": 2},
        {"reduction": "pca", "reduced_dim": 1, "projection_training_samples": 0},
        {"reduction": "pca", "reduced_dim": 1, "projection_training_samples": 41},
    ],
)
def test_invalid_training_options_raise_value_error(options):
    with pytest.raises(ValueError):
        Gathers().fit(np.ones((40, 3), dtype=np.float32), 1, **options)


@pytest.mark.parametrize(
    "rows, dim, clusters, iterations",
    [(0, 3, 1, 1), (38, 3, 1, 1), (40, 0, 1, 1), (40, 3, 0, 1), (40, 3, 1, 0)],
)
def test_invalid_training_shape_and_counts(rows, dim, clusters, iterations):
    with pytest.raises(ValueError):
        Gathers().fit(np.ones((rows, dim), dtype=np.float32), clusters, iterations)


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
@pytest.mark.parametrize("reduction", ["raw", "pca", "srht"])
def test_nonfinite_training_rows_are_rejected(invalid, reduction):
    data = np.ones((40, 3), dtype=np.float32)
    data[2, 1] = invalid
    options = {"reduction": reduction}
    if reduction != "raw":
        options["reduced_dim"] = 1
    with pytest.raises(ValueError, match="finite"):
        Gathers().fit(data, 1, **options)


@pytest.mark.parametrize("reduction, scale", [
    ("pca", np.finfo(np.float32).max), ("pca", 1e20), ("srht", np.finfo(np.float32).max),
])
def test_reduction_arithmetic_overflow_raises_value_error(reduction, scale):
    data = np.zeros((40, 2), dtype=np.float32)
    data[:, 0] = -scale
    data[0, 0] = scale
    data[:, 1] = data[:, 0]
    with pytest.raises(ValueError, match="reduction arithmetic overflowed"):
        Gathers().fit(data, 1, reduction=reduction, reduced_dim=1, seed=42)


def test_auto_defaults_cluster_count_and_small_inputs_stay_raw():
    data = np.arange(1000 * 3, dtype=np.float32).reshape(1000, 3) + 1
    model = Gathers()
    expected_clusters = int(len(data) ** 0.8 / 16)
    automatic = model.fit(data, seed=42)
    assert automatic.shape == (expected_clusters, 3)
    np.testing.assert_array_equal(
        automatic, model.fit(data, expected_clusters, reduction="raw", seed=42)
    )


def test_explicit_pca_defaults_to_128_dimensions():
    data = np.arange(40 * 197, dtype=np.float32).reshape(40, 197)
    model = Gathers()
    default_dimension = model.fit(data, 1, reduction="pca", seed=42)
    assert default_dimension.shape == (1, 197)
    np.testing.assert_array_equal(
        default_dimension,
        model.fit(data, 1, reduction="pca", reduced_dim=128, seed=42),
    )


@pytest.mark.parametrize("factor", [39, 128, 256, 512])
@pytest.mark.parametrize("reduction", ["raw", "pca", "srht"])
def test_sampling_factor_matches_exact_total(factor, reduction):
    data = np.arange(1000 * 3, dtype=np.float32).reshape(1000, 3) + 1
    options = {"reduction": reduction, "seed": 42}
    if reduction != "raw":
        options["reduced_dim"] = 1
    model = Gathers()
    factored = model.fit(data, 2, 2, samples_per_cluster=factor, **options)
    explicit = model.fit(
        data, 2, 2, samples_per_cluster=0,
        training_samples=min(len(data), factor * 2), **options,
    )
    np.testing.assert_array_equal(factored, explicit)


def test_sampling_factor_with_automatic_cluster_count():
    data = np.arange(1000 * 3, dtype=np.float32).reshape(1000, 3) + 1
    model = Gathers()
    automatic = model.fit(data, samples_per_cluster=39, seed=42)
    clusters = int(len(data) ** 0.8 / 16)
    np.testing.assert_array_equal(
        automatic, model.fit(data, clusters, training_samples=39 * clusters, seed=42)
    )


@pytest.mark.parametrize("options", [
    {"samples_per_cluster": 0}, {"samples_per_cluster": 38},
    {"training_samples": 0}, {"training_samples": 77}, {"training_samples": 1001},
])
def test_invalid_sampling_options_raise_value_error(options):
    with pytest.raises(ValueError):
        Gathers().fit(np.ones((1000, 3), dtype=np.float32), 2, **options)
