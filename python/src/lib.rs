use core::f32;

use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use gathers::distance::{Distance,squared_euclidean, argmin};
use gathers::kmeans::KMeans;
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
    Ok(())
}
