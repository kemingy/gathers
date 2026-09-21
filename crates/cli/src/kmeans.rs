use std::io;
use std::path::PathBuf;
use std::time::Instant;

use argh::FromArgs;
use gathers::distance::Distance;
use gathers::kmeans::KMeans;

use crate::{fvecs, invalid, ready, report};

#[derive(FromArgs)]
#[argh(subcommand, name = "kmeans")]
/// Train squared-Euclidean K-means and save centroids as fvecs.
pub(crate) struct Args {
    /// input fvecs file
    #[argh(option, short = 'i')]
    input: PathBuf,
    /// output centroid fvecs file
    #[argh(option, short = 'o')]
    output: PathBuf,
    /// number of clusters
    #[argh(option, short = 'n', default = "4096")]
    n_cluster: u32,
    /// maximum number of iterations
    #[argh(option, short = 'm', default = "25")]
    max_iter: u32,
}

pub(crate) fn run(args: &Args, common: &crate::Args) -> io::Result<()> {
    if args.n_cluster == 0 || args.max_iter == 0 {
        return Err(invalid("n-cluster and max-iter must be positive"));
    }
    let vectors = fvecs::read(&args.input, None)?;
    let num_vectors = vectors.len();
    let dim = vectors.dim;
    let kmeans = KMeans::new(
        args.n_cluster,
        args.max_iter,
        0.01,
        Distance::SquaredEuclidean,
        false,
    )
    .seed(common.seed);
    ready(common.wait_for_profiler)?;
    let start = Instant::now();
    let centroids = kmeans.fit(vectors.data, dim);
    let fit_ms = start.elapsed().as_secs_f64() * 1_000.0;
    fvecs::write(&args.output, &centroids, dim)?;
    report(serde_json::json!({
        "command": "kmeans",
        "input": args.input,
        "output": args.output,
        "num_vectors": num_vectors,
        "num_centroids": centroids.len() / dim,
        "dim": dim,
        "max_iter": args.max_iter,
        "seed": common.seed,
        "threads": rayon::current_num_threads(),
        "cpu": common.cpu,
        "arch": std::env::consts::ARCH,
        "os": std::env::consts::OS,
        "debug_assertions": cfg!(debug_assertions),
        "fit_ms": fit_ms,
    }))
}
