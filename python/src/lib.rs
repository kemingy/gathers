use gathers::distance::Distance;
use gathers::kmeans::{
    KMeans, KMeansConfig, ReductionConfig, base_assign, base_assign_parallel,
    rabitq_assign_parallel,
};
use gathers::utils::{as_continuous_vec, as_matrix};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyModule, PyModuleMethods};
use pyo3::{Bound, PyResult, pyfunction, pymodule, wrap_pyfunction};

fn validate_rows(values: &[f32], dim: usize, distance: Distance) -> PyResult<()> {
    if !values.iter().all(|value| value.is_finite()) {
        return Err(PyValueError::new_err(
            "vectors must contain finite coordinates",
        ));
    }
    if distance == Distance::Cosine
        && values
            .chunks_exact(dim)
            .any(|row| row.iter().all(|&value| value == 0.0))
    {
        return Err(PyValueError::new_err(
            "cosine rows must have a finite nonzero L2 norm",
        ));
    }
    Ok(())
}

/// Assign a vector to its nearest centroid with l2, cos, or dot.
#[pyfunction]
#[pyo3(signature = (vec, centroids, distance = "l2"))]
fn assign<'py>(
    vec: PyReadonlyArray1<'py, f32>,
    centroids: PyReadonlyArray2<'py, f32>,
    distance: &str,
) -> PyResult<u32> {
    let distance = distance
        .parse::<Distance>()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let vector = vec
        .as_slice()
        .map_err(|_| PyValueError::new_err("vector must be contiguous"))?;
    let matrix = centroids.as_array();
    if vector.is_empty() || matrix.nrows() == 0 || matrix.ncols() != vector.len() {
        return Err(PyValueError::new_err(
            "centroids must be nonempty and match the vector dimension",
        ));
    }
    let centroids = matrix
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("centroids must be C-contiguous"))?;
    validate_rows(vector, vector.len(), distance)?;
    validate_rows(centroids, vector.len(), distance)?;
    let mut labels = [0];
    base_assign(vector, centroids, vector.len(), distance, &mut labels);
    Ok(labels[0])
}

/// assign batch of vectors to the nearest centroid.
#[pyfunction]
#[pyo3(signature = (vecs, centroids, distance = "l2"))]
fn batch_assign<'py>(
    vecs: PyReadonlyArray2<'py, f32>,
    centroids: PyReadonlyArray2<'py, f32>,
    distance: &str,
) -> PyResult<Vec<u32>> {
    let distance = distance
        .parse::<Distance>()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let vectors = vecs.as_array();
    let matrix = centroids.as_array();
    let dim = vectors.ncols();
    if dim == 0 || matrix.nrows() == 0 || matrix.ncols() != dim {
        return Err(PyValueError::new_err(
            "centroids must be nonempty and match the vector dimension",
        ));
    }
    let data = vectors
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("vectors must be C-contiguous"))?;
    let centroids = matrix
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("centroids must be C-contiguous"))?;
    validate_rows(data, dim, distance)?;
    validate_rows(centroids, dim, distance)?;
    let mut labels = vec![0; vectors.nrows()];
    if distance == Distance::SquaredEuclidean {
        rabitq_assign_parallel(data, centroids, dim, &mut labels);
    } else {
        base_assign_parallel(data, centroids, dim, distance, &mut labels);
    }
    Ok(labels)
}

/// Train a K-means and return the centroids.
#[pyfunction]
#[pyo3(signature = (source, n_cluster = None, max_iter = 10, *, distance = "l2", reduction = "auto", reduced_dim = None, samples_per_cluster = 256, training_samples = None, projection_training_samples = None, seed = None))]
#[allow(
    clippy::too_many_arguments,
    reason = "Translate the flat Python keyword interface directly"
)]
fn kmeans_fit<'py>(
    source: PyReadonlyArray2<'py, f32>,
    n_cluster: Option<u32>,
    max_iter: u32,
    distance: &str,
    reduction: &str,
    reduced_dim: Option<usize>,
    samples_per_cluster: usize,
    training_samples: Option<usize>,
    projection_training_samples: Option<usize>,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let vecs = source.as_array();
    let dim = vecs.ncols();
    let distance = distance
        .parse::<Distance>()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let reduction = reduction
        .parse::<ReductionConfig>()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let reduction = match reduction {
        ReductionConfig::PCA { output_dim, .. } => ReductionConfig::PCA {
            output_dim: reduced_dim.unwrap_or(output_dim),
            training_samples: projection_training_samples,
        },
        ReductionConfig::SRHT { output_dim } => {
            if projection_training_samples.is_some() {
                return Err(PyValueError::new_err(
                    "projection_training_samples only applies to PCA",
                ));
            }
            ReductionConfig::SRHT {
                output_dim: reduced_dim.unwrap_or(output_dim),
            }
        }
        reduction => {
            if reduced_dim.is_some() || projection_training_samples.is_some() {
                return Err(PyValueError::new_err(
                    "projection options require explicit pca or srht reduction",
                ));
            }
            reduction
        }
    };
    let config = KMeansConfig {
        n_clusters: n_cluster,
        max_iter,
        distance,
        samples_per_cluster,
        training_samples,
        reduction,
        seed,
        ..Default::default()
    }
    .resolve(vecs.nrows(), dim)
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let data = vecs
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("source must be C-contiguous"))?;
    let centroids = KMeans::new(config)
        .fit(as_continuous_vec(&[data]), dim)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let matrix = as_matrix(&centroids, dim);
    Ok(PyArray2::from_vec2(source.py(), &matrix)?)
}

/// A Python module implemented in Rust.
#[pymodule]
fn gatherspy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(kmeans_fit, m)?)?;
    m.add_function(wrap_pyfunction!(assign, m)?)?;
    m.add_function(wrap_pyfunction!(batch_assign, m)?)?;
    Ok(())
}
