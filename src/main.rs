use std::env;
use std::path::Path;

use env_logger::Env;
use gathers::distance::Distance;
use gathers::kmeans::KMeans;
use gathers::utils::{as_continuous_vec, as_matrix, read_vecs, write_vecs};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        println!("requires a fvecs input file path & a fvecs output file path");
        return;
    }
    let input = Path::new(&args[1]);
    let output = Path::new(&args[2]);

    let env = Env::default().filter_or("RUST_LOG", "debug");
    env_logger::init_from_env(env);

    let vecs = read_vecs::<f32>(input).expect("failed to read vecs");
    let dim = vecs[0].len();
    let kmeans = KMeans::new(4096, 3, 0.01, Distance::SquaredEuclidean, false);
    let centroids = kmeans.fit(as_continuous_vec(&vecs), dim);
    let centroids_mat = as_matrix(&centroids, dim);
    write_vecs(output, centroids_mat).expect("failed to write centroids");
}
