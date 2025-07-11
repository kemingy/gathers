use std::path::Path;

use argh::FromArgs;
use gathers::distance::Distance;
use gathers::kmeans::KMeans;
use gathers::utils::{as_continuous_vec, as_matrix, read_vecs, write_vecs};
use log::debug;
use logforth::append;
use logforth::filter::EnvFilter;

#[derive(FromArgs, Debug)]
/// gathers CLI args
struct Args {
    /// input file path
    #[argh(option, short = 'i')]
    input: String,
    /// output file path
    #[argh(option, short = 'o')]
    output: String,
    /// number of clusters
    #[argh(option, short = 'n', default = "4096")]
    n_cluster: u32,
    /// max number of iterations
    #[argh(option, short = 'm', default = "25")]
    max_iter: u32,
}

fn main() {
    let args: Args = argh::from_env();

    let env_filter = EnvFilter::from_env_or("GATHERS_LOG", "DEBUG");
    logforth::builder()
        .dispatch(|d| d.filter(env_filter).append(append::Stderr::default()))
        .apply();
    debug!("{args:?}");

    let vecs = read_vecs::<f32>(Path::new(&args.input)).expect("failed to read vecs");
    let dim = vecs[0].len();
    let kmeans = KMeans::new(
        args.n_cluster,
        args.max_iter,
        0.01,
        Distance::SquaredEuclidean,
        false,
    );
    let centroids = kmeans.fit(as_continuous_vec(&vecs), dim);
    let centroids_mat = as_matrix(&centroids, dim);
    write_vecs(Path::new(&args.output), &centroids_mat).expect("failed to write centroids");
}
