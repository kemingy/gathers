//! Clustering and assignment profiling with fvecs inputs.

mod assign;
mod fvecs;
mod kmeans;

use std::io::{self, Write};

use anyhow::{Result, ensure};
use argh::FromArgs;
use logforth::append;
use logforth::filter::rustlog::RustLogFilterBuilder;

#[derive(FromArgs)]
/// Train clusters or profile assignment using fvecs files.
struct Args {
    /// rayon workers; zero selects the Rayon default
    #[argh(option, default = "0")]
    threads: usize,
    /// random seed (training: sampling/rotations/repair; assignment: rotation)
    #[argh(option, default = "42")]
    seed: u64,
    /// machine/CPU description included in the report
    #[argh(option, default = "String::from(\"unspecified\")")]
    cpu: String,
    /// wait for Enter before training or timed assignment
    #[argh(switch)]
    wait_for_profiler: bool,
    #[argh(subcommand)]
    command: Command,
}

#[derive(FromArgs)]
#[argh(subcommand)]
enum Command {
    Kmeans(kmeans::Args),
    Assign(assign::Args),
}

fn wait_for_profiler() -> Result<()> {
    eprintln!("ready: pid={}", std::process::id());
    eprintln!("Attach the sampler, then press Enter to start.");
    ensure!(
        io::stdin().read_line(&mut String::new())? != 0,
        "stdin closed while waiting for the profiler"
    );
    Ok(())
}

fn report(value: serde_json::Value) -> Result<()> {
    let mut stdout = io::stdout().lock();
    serde_json::to_writer(&mut stdout, &value)?;
    writeln!(stdout)?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Args = argh::from_env();
    if cfg!(debug_assertions) {
        eprintln!(
            "warning: debug build; use --profile perf or --release for performance measurements"
        );
    }
    let filter = RustLogFilterBuilder::from_env_or("GATHERS_LOG", "INFO").build();
    logforth::starter_log::builder()
        .dispatch(|d| d.filter(filter).append(append::Stderr::default()))
        .apply();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build()?;
    pool.install(|| match &args.command {
        Command::Kmeans(command) => kmeans::run(command, &args),
        Command::Assign(command) => assign::run(command, &args),
    })?;
    Ok(())
}
