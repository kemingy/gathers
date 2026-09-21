//! Repeatable RaBitQ assignment timing and sampling with external fvecs data.

use std::hint::black_box;
use std::io;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use argh::FromArgs;
use gathers::rabitq::RaBitQ;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::{fvecs, invalid, ready, report};

#[derive(FromArgs)]
#[argh(subcommand, name = "assign")]
/// Profile repeated RaBitQ assignment against fixed centroids.
pub(crate) struct Args {
    /// query vector fvecs file
    #[argh(option)]
    vectors: PathBuf,
    /// centroid fvecs file (use trained centroids for representative results)
    #[argh(option)]
    centroids: PathBuf,
    /// query prefix length; default: all rows
    #[argh(option)]
    num_vectors: Option<usize>,
    /// centroid prefix length; default: all rows
    #[argh(option)]
    num_centroids: Option<usize>,
    /// untimed warm-up assignments
    #[argh(option, default = "1")]
    warmup: usize,
    /// minimum number of timed assignments
    #[argh(option, default = "5")]
    repeats: usize,
    /// minimum query-phase duration in seconds, useful for sampling
    #[argh(option, default = "0.0")]
    min_seconds: f64,
}

fn median_ms(times: &[Duration]) -> f64 {
    let mut sorted = times.to_vec();
    sorted.sort_unstable();
    let middle = sorted.len() / 2;
    let seconds = if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1].as_secs_f64() + sorted[middle].as_secs_f64()) / 2.0
    } else {
        sorted[middle].as_secs_f64()
    };
    seconds * 1_000.0
}

pub(crate) fn run(args: &Args, common: &crate::Args) -> io::Result<()> {
    if args.repeats == 0 {
        return Err(invalid("repeats must be positive"));
    }
    let min_duration = Duration::try_from_secs_f64(args.min_seconds)
        .map_err(|_| invalid("min-seconds must be finite and nonnegative"))?;
    let vectors = fvecs::read(&args.vectors, args.num_vectors)?;
    let centroids = fvecs::read(&args.centroids, args.num_centroids)?;
    if vectors.dim != centroids.dim {
        return Err(invalid("query and centroid dimensions must match"));
    }
    let dim = vectors.dim;
    let num_vectors = vectors.len();
    let num_centroids = centroids.len();
    if num_centroids > u32::MAX as usize {
        return Err(invalid("centroid count exceeds the label representation"));
    }
    let mut labels = vec![0; num_vectors];
    let mut rng = StdRng::seed_from_u64(common.seed);
    let start = Instant::now();
    let index = RaBitQ::new_with_rng(&centroids.data, dim, &mut rng);
    let build_ms = start.elapsed().as_secs_f64() * 1_000.0;
    for _ in 0..args.warmup {
        index.retrieve_top_one_batch(&vectors.data, dim, &mut labels);
    }
    ready(common.wait_for_profiler)?;
    let mut times = Vec::with_capacity(args.repeats);
    let phase_start = Instant::now();
    while times.len() < args.repeats || phase_start.elapsed() < min_duration {
        let start = Instant::now();
        index.retrieve_top_one_batch(black_box(&vectors.data), dim, black_box(&mut labels));
        times.push(start.elapsed());
    }
    let total_seconds: f64 = times.iter().map(Duration::as_secs_f64).sum();
    let label_hash = labels.iter().fold(0xcbf29ce484222325_u64, |hash, &label| {
        label.to_le_bytes().iter().fold(hash, |hash, &byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
        })
    });
    report(serde_json::json!({
        "command": "assign",
        "vectors": args.vectors,
        "centroids": args.centroids,
        "num_vectors": num_vectors,
        "num_centroids": num_centroids,
        "dim": dim,
        "padded_dim": index.dim(),
        "seed": common.seed,
        "threads": rayon::current_num_threads(),
        "cpu": common.cpu,
        "arch": std::env::consts::ARCH,
        "os": std::env::consts::OS,
        "debug_assertions": cfg!(debug_assertions),
        "warmup": args.warmup,
        "repeats": times.len(),
        "min_seconds": args.min_seconds,
        "build_ms": build_ms,
        "query_median_ms": median_ms(&times),
        "query_ms": times.iter().map(|time| time.as_secs_f64() * 1_000.0).collect::<Vec<_>>(),
        "queries_per_second": num_vectors as f64 * times.len() as f64 / total_seconds,
        "label_hash": format!("{label_hash:016x}"),
        "metrics_scope": "warmup_and_timed_queries",
        "metrics": index.metrics().to_string(),
    }))
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::median_ms;

    #[test]
    fn median_handles_even_and_odd_samples() {
        assert_eq!(median_ms(&[Duration::from_millis(7)]), 7.0);
        assert_eq!(
            median_ms(&[Duration::from_millis(9), Duration::from_millis(3)]),
            6.0
        );
        assert_eq!(
            median_ms(&[
                Duration::from_millis(9),
                Duration::from_millis(3),
                Duration::from_millis(6)
            ]),
            6.0
        );
    }
}
