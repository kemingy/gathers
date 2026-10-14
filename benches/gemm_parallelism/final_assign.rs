//! Full-corpus post-training assignment benchmark; no input sampling.
#[allow(dead_code)]
#[path = "../src/fvecs.rs"]
mod fvecs;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::{Duration, Instant};

use aligned_vec::AVec;
use anyhow::{Result, ensure};
use gathers::distance::Distance;
use gathers::kmeans::{ReviewMatrix, base_assign_parallel, review_set_matrix_block};
use gathers::utils::try_normalize_rows;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 8,
        "source centroids labels method metric threads batch_rows"
    );
    let overall = Instant::now();
    let threads: usize = args[6].parse()?;
    let batch_rows: usize = args[7].parse()?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    let distance: Distance = args[5].parse()?;
    let start = Instant::now();
    let mut centroids = fvecs::read(Path::new(&args[2]), None)?;
    let dim = centroids.dim;
    let k = centroids.len();
    let centroid_read = start.elapsed();
    let mut normalize = Duration::ZERO;
    let mut workspace_setup = Duration::ZERO;
    review_set_matrix_block(
        std::env::var("GEMM_POLICY")
            .unwrap_or_else(|_| "1".into())
            .parse()?,
    );
    let assignment_distance = if args[4] == "flat-matrix" && distance == Distance::Cosine {
        let start = Instant::now();
        try_normalize_rows(&mut centroids.data, dim, false)?;
        normalize += start.elapsed();
        Distance::NegativeDotProduct
    } else {
        distance
    };
    let mut reader = fvecs::Reader::open(Path::new(&args[1]))?;
    ensure!(reader.dim == dim, "dimension mismatch");
    let expected_rows = reader.rows;
    let mut output = BufWriter::new(File::create(&args[3])?);
    let mut vectors = AVec::new(64);
    let mut labels = Vec::new();
    let mut counts = vec![0_u64; k];
    let mut read = Duration::ZERO;
    let mut assign = Duration::ZERO;
    let mut write = Duration::ZERO;
    let mut rows = 0;
    loop {
        let start = Instant::now();
        let n = reader.read_batch(batch_rows, &mut vectors)?;
        read += start.elapsed();
        if n == 0 {
            break;
        }
        labels.resize(n, 0);
        if args[4] == "flat-matrix" && distance == Distance::Cosine {
            let start = Instant::now();
            try_normalize_rows(&mut vectors, dim, false)?;
            normalize += start.elapsed();
        }
        let start = Instant::now();
        let mut matrix = if args[4] == "flat-matrix" {
            Some(
                ReviewMatrix::try_new(&vectors, k, dim, assignment_distance)
                    .expect("eligible matrix workload"),
            )
        } else {
            None
        };
        workspace_setup += start.elapsed();
        let start = Instant::now();
        if let Some(workspace) = &mut matrix {
            workspace.assign(&vectors, &centroids.data, dim, &mut labels);
        } else {
            base_assign_parallel(&vectors, &centroids.data, dim, distance, &mut labels);
        }
        assign += start.elapsed();
        let start = Instant::now();
        for &label in &labels {
            counts[label as usize] += 1;
        }
        // Raw little-endian u32 memberships in source row order, no header.
        for &label in &labels {
            output.write_all(&label.to_le_bytes())?;
        }
        write += start.elapsed();
        rows += n;
        eprintln!(
            "FINAL_PROGRESS {} {} / {} assignment_s {:.3}",
            args[4],
            rows,
            expected_rows,
            assign.as_secs_f64()
        );
    }
    let start = Instant::now();
    output.flush()?;
    write += start.elapsed();
    ensure!(rows == expected_rows, "incomplete corpus assignment");
    println!(
        "{}",
        serde_json::json!({
            "method": args[4], "distance": args[5], "rows": rows, "dim": dim, "clusters": k,
            "threads": threads, "batch_rows": batch_rows, "centroid_read_s": centroid_read.as_secs_f64(),
            "read_s": read.as_secs_f64(),
            "normalize_s": normalize.as_secs_f64(), "workspace_setup_s": workspace_setup.as_secs_f64(),
            "assign_s": assign.as_secs_f64(), "label_count_write_s": write.as_secs_f64(),
            "empty_clusters": counts.iter().filter(|&&count| count == 0).count(),
            "wall_s": overall.elapsed().as_secs_f64(), "labels": args[3]
        })
    );
    Ok(())
}
