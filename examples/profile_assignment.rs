//! Repeatable RaBitQ assignment timing and sampling with external float32 data.

use std::fs::File;
use std::hint::black_box;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use argh::FromArgs;
use gathers::rabitq::RaBitQ;
use rand::SeedableRng;
use rand::rngs::StdRng;

#[derive(FromArgs)]
/// Profile repeated RaBitQ assignment on raw row-major little-endian f32 files.
struct Args {
    /// query vector file (no header)
    #[argh(option)]
    vectors: PathBuf,
    /// centroid file in the same format (use trained centroids for representative results)
    #[argh(option)]
    centroids: PathBuf,
    /// coordinates per vector
    #[argh(option)]
    dim: usize,
    /// query prefix length; default: all rows
    #[argh(option)]
    num_vectors: Option<usize>,
    /// centroid prefix length; default: all rows
    #[argh(option)]
    num_centroids: Option<usize>,
    /// rayon worker count; zero selects the Rayon default
    #[argh(option, default = "0")]
    threads: usize,
    /// random rotation seed
    #[argh(option, default = "42")]
    seed: u64,
    /// untimed warm-up assignments
    #[argh(option, default = "1")]
    warmup: usize,
    /// minimum number of timed assignments
    #[argh(option, default = "5")]
    repeats: usize,
    /// minimum query-phase duration in seconds, useful for sampling
    #[argh(option, default = "0.0")]
    min_seconds: f64,
    /// wait for Enter after warm-up, before timed retrieval
    #[argh(switch)]
    wait_for_profiler: bool,
    /// machine/CPU description to include in the report
    #[argh(option, default = "String::from(\"unspecified\")")]
    cpu: String,
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

fn load_vectors(path: &Path, dim: usize, limit: Option<usize>) -> io::Result<Vec<f32>> {
    let mut file = File::open(path)?;
    let row_bytes = dim
        .checked_mul(size_of::<f32>())
        .filter(|&bytes| bytes > 0)
        .ok_or_else(|| invalid("dimension must be positive and fit in memory"))?;
    let bytes = file.metadata()?.len();
    if !bytes.is_multiple_of(row_bytes as u64) {
        return Err(invalid("input contains an incomplete vector"));
    }
    let available = usize::try_from(bytes / row_bytes as u64)
        .map_err(|_| invalid("input has too many rows"))?;
    let rows = limit.unwrap_or(available);
    if rows == 0 || rows > available {
        return Err(invalid(
            "requested row count must be positive and fit in the input",
        ));
    }
    let values = rows
        .checked_mul(dim)
        .ok_or_else(|| invalid("input shape overflow"))?;
    let mut data = Vec::new();
    data.try_reserve_exact(values).map_err(io::Error::other)?;
    data.resize(values, 0.0_f32);
    file.read_exact(bytemuck::cast_slice_mut(&mut data))?;
    for value in &mut data {
        *value = f32::from_bits(u32::from_le(value.to_bits()));
        if !value.is_finite() {
            return Err(invalid("vectors must contain only finite values"));
        }
    }
    Ok(data)
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();
    if cfg!(debug_assertions) {
        return Err(invalid("build with --profile perf or --release").into());
    }
    if args.repeats == 0 {
        return Err(invalid("repeats must be positive").into());
    }
    let min_duration = Duration::try_from_secs_f64(args.min_seconds)
        .map_err(|_| invalid("min-seconds must be finite and nonnegative"))?;
    let vectors = load_vectors(&args.vectors, args.dim, args.num_vectors)?;
    let centroids = load_vectors(&args.centroids, args.dim, args.num_centroids)?;
    let num_vectors = vectors.len() / args.dim;
    let num_centroids = centroids.len() / args.dim;
    if num_centroids > u32::MAX as usize {
        return Err(invalid("centroid count exceeds the label representation").into());
    }
    let mut labels = vec![0; num_vectors];
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build()?;
    pool.install(|| -> io::Result<()> {
        let mut rng = StdRng::seed_from_u64(args.seed);
        let start = Instant::now();
        let index = RaBitQ::new_with_rng(&centroids, args.dim, &mut rng);
        let build_ms = start.elapsed().as_secs_f64() * 1_000.0;
        for _ in 0..args.warmup {
            index.retrieve_top_one_batch(&vectors, args.dim, &mut labels);
        }
        eprintln!(
            "ready: pid={} N={} K={} D={} threads={}",
            std::process::id(),
            num_vectors,
            num_centroids,
            args.dim,
            rayon::current_num_threads(),
        );
        if args.wait_for_profiler {
            eprintln!("Attach the sampler, then press Enter to start retrieval.");
            if io::stdin().read_line(&mut String::new())? == 0 {
                return Err(invalid("stdin closed while waiting for the profiler"));
            }
        }
        let mut times = Vec::with_capacity(args.repeats);
        let phase_start = Instant::now();
        while times.len() < args.repeats || phase_start.elapsed() < min_duration {
            let start = Instant::now();
            index.retrieve_top_one_batch(black_box(&vectors), args.dim, black_box(&mut labels));
            times.push(start.elapsed());
        }
        let total_seconds: f64 = times.iter().map(Duration::as_secs_f64).sum();
        let label_hash = labels.iter().fold(0xcbf29ce484222325_u64, |hash, &label| {
            label.to_le_bytes().iter().fold(hash, |hash, &byte| {
                (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
            })
        });
        let report = serde_json::json!({
            "vectors": args.vectors,
            "centroids": args.centroids,
            "num_vectors": num_vectors,
            "num_centroids": num_centroids,
            "dim": args.dim,
            "padded_dim": index.dim(),
            "seed": args.seed,
            "threads": rayon::current_num_threads(),
            "cpu": args.cpu,
            "arch": std::env::consts::ARCH,
            "os": std::env::consts::OS,
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
        });
        let mut stdout = io::stdout().lock();
        serde_json::to_writer(&mut stdout, &report)?;
        writeln!(stdout)?;
        Ok(())
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Write;
    use std::time::Duration;

    use super::{load_vectors, median_ms};

    #[test]
    fn loader_validates_shape_prefix_and_values() {
        let path = std::env::temp_dir().join(format!(
            "gathers-profile-input-{}-{}.f32",
            std::process::id(),
            rand::random::<u64>(),
        ));
        let mut file = fs::File::create_new(&path).unwrap();
        for value in [1.0_f32, -2.0, 3.5, 4.0] {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
        assert_eq!(load_vectors(&path, 2, None).unwrap(), [1.0, -2.0, 3.5, 4.0]);
        assert_eq!(load_vectors(&path, 2, Some(1)).unwrap(), [1.0, -2.0]);
        for (dim, limit) in [(0, None), (3, None), (2, Some(0)), (2, Some(3))] {
            assert!(load_vectors(&path, dim, limit).is_err());
        }
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            fs::write(&path, value.to_le_bytes()).unwrap();
            assert!(load_vectors(&path, 1, None).is_err());
        }
        fs::write(&path, [0_u8; 3]).unwrap();
        assert!(load_vectors(&path, 1, None).is_err());
        fs::write(&path, []).unwrap();
        assert!(load_vectors(&path, 1, None).is_err());
        drop(file);
        fs::remove_file(path).unwrap();
    }

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
