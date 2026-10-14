//! Real-data comparison of GEMM backends and fixed versus adaptive row blocks.
#[allow(dead_code)]
#[path = "../src/fvecs.rs"]
mod fvecs;
use std::path::Path;
use std::time::Instant;

use anyhow::{Result, ensure};
use gathers::distance::Distance;
use gathers::kmeans::{ReviewMatrix, review_set_matrix_block, review_use_accelerate};
use gathers::utils::try_normalize_rows;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(args.len() == 5, "queries centroids metric threads");
    let threads: usize = args[4].parse()?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    let mut q = fvecs::read(Path::new(&args[1]), None)?;
    let mut c = fvecs::read(Path::new(&args[2]), None)?;
    ensure!(q.dim == c.dim, "dimension mismatch");
    let mut distance = args[3].parse::<Distance>()?;
    if distance == Distance::Cosine {
        try_normalize_rows(&mut q.data, q.dim, false)?;
        try_normalize_rows(&mut c.data, c.dim, false)?;
        distance = Distance::NegativeDotProduct;
    }
    let mut workspace = ReviewMatrix::try_new(&q.data, c.len(), q.dim, distance).unwrap();
    let mut labels = vec![0; q.len()];
    review_use_accelerate(false);
    review_set_matrix_block(512);
    workspace.assign(&q.data, &c.data, q.dim, &mut labels);
    let oracle = labels.clone();
    for (accelerate, policy) in [(false, 512), (false, 1), (true, 512), (true, 1)] {
        review_use_accelerate(accelerate);
        review_set_matrix_block(policy);
        workspace.assign(&q.data, &c.data, q.dim, &mut labels);
        let times: Vec<_> = (0..7)
            .map(|_| {
                let start = Instant::now();
                workspace.assign(&q.data, &c.data, q.dim, &mut labels);
                start.elapsed().as_secs_f64() * 1000.
            })
            .collect();
        ensure!(labels == oracle, "labels changed with policy {policy}");
        let mut sorted = times.clone();
        sorted.sort_by(f64::total_cmp);
        println!(
            "{}",
            serde_json::json!({"backend":if accelerate && cfg!(target_os = "macos") {"accelerate"} else {"faer"},"policy":policy,"threads":threads,"queries":q.len(),"centroids":c.len(),"dim":q.dim,"median_ms":sorted[3],"times_ms":times,"same_labels":true})
        );
    }
    Ok(())
}
