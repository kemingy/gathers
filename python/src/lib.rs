use core::f32;

use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use gathers::distance::{Distance, argmin, squared_euclidean};
use gathers::kmeans::{KMeans, rabitq_assign};
use gathers::utils::as_matrix;

/// assign the vector to the nearest centroid.
#[pyfunction]
#[pyo3(signature = (vec, centroids))]
fn assign<'py>(
    vec: PyReadonlyArray1<'py, f32>,
    centroids: PyReadonlyArray2<'py, f32>,
) -> u32 {
    let v = vec.as_array();
    let c = centroids.as_array();
    let num = c.nrows();
    let mut distances = vec![f32::MAX; num];

    for (i, centroid) in c.rows().into_iter().enumerate() {
        distances[i] = squared_euclidean(v.as_slice().unwrap(), centroid.as_slice().unwrap());
    }

    argmin(&distances) as u32
}

/// assign batch of vectors to the nearest centroid.
#[pyfunction]
#[pyo3(signature = (vecs, centroids))]
fn batch_assign<'py>(
    vecs: PyReadonlyArray2<'py, f32>,
    centroids: PyReadonlyArray2<'py, f32>,
) -> Vec<u32> {
    let vectors = vecs.as_array();
    let mut labels = vec![0; vectors.nrows()];
    rabitq_assign(
        vectors.as_slice().expect("failed to get the vecs slice"),
        centroids.as_array().as_slice().expect("failed to get the centroids slice"), vectors.ncols(), &mut labels);
    labels
}

/// Train a K-means and return the centroids.
#[pyfunction]
#[pyo3(signature = (source, n_cluster, max_iter = 25))]
fn kmeans_fit<'py>(
    source: PyReadonlyArray2<'py, f32>,
    n_cluster: u32,
    max_iter: u32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let vecs = source.as_array();
    let dim = vecs.ncols();
    let kmeans = KMeans::new(n_cluster, max_iter, 1e-4, Distance::SquaredEuclidean, false);
    let centroids = kmeans.fit(
        vecs.as_slice()
            .expect("failed to get the inner array")
            .to_owned(),
        dim,
    );
    let matrix = as_matrix(&centroids, dim);
    Ok(PyArray2::from_vec2_bound(source.py(), &matrix)?)
}

/// A Python module implemented in Rust.
#[pymodule]
fn gatherspy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(kmeans_fit, m)?)?;
    m.add_function(wrap_pyfunction!(assign, m)?)?;
    m.add_function(wrap_pyfunction!(batch_assign, m)?)?;
    Ok(())
}
