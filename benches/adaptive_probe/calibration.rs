//! Pilot calibration and its paired backend measurements, shared across training rounds.

use std::time::Instant;

use anyhow::{Result, ensure};
use gathers::distance::Distance;
use serde_json::json;

use crate::TRAINING_DIM;
use crate::kmeans::hierarchy::policy::budget;
use crate::kmeans::hierarchy::{Hierarchy, ProbeOptions};

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Calibration {
    pub(crate) budget: usize,
    pub(crate) required: Vec<usize>,
    pub(crate) labels: Vec<u32>,
}

pub(crate) fn calibrate(
    h: &Hierarchy,
    pilots: &[f32],
    distance: Distance,
    percentile: usize,
    backend: &str,
) -> Calibration {
    let mut required = vec![0; pilots.len() / TRAINING_DIM];
    let mut labels = vec![0; required.len()];
    if backend == "gemm" {
        h.calibrate_batched(pilots, &mut labels, &mut required, distance);
    } else {
        h.probe_parallel(
            pilots,
            &mut labels,
            &mut required,
            ProbeOptions {
                distance,
                budget: usize::MAX,
                incumbent_centroids: None,
            },
        );
    }
    Calibration {
        budget: budget(&required, percentile),
        required,
        labels,
    }
}

/// Alternate backend order and verify every winner, required count and budget.
/// The caller places this diagnostic outside the training timer.
pub(crate) fn compare(
    hierarchy: &Hierarchy,
    pilots: &[f32],
    distance: Distance,
    percentile: usize,
    repeats: usize,
    iteration: usize,
) -> Result<()> {
    if repeats == 0 {
        return Ok(());
    }
    let mut simd_times = Vec::with_capacity(repeats);
    let mut gemm_times = Vec::with_capacity(repeats);
    for repeat in 0..repeats {
        let mut timed_calibration = |backend| {
            let start = Instant::now();
            let result = calibrate(hierarchy, pilots, distance, percentile, backend);
            let timings = if backend == "simd" {
                &mut simd_times
            } else {
                &mut gemm_times
            };
            timings.push(start.elapsed().as_secs_f64());
            result
        };
        // Alternate order to reduce systematic cache/thermal bias.
        let (first, second) = if repeat % 2 == 0 {
            ("simd", "gemm")
        } else {
            ("gemm", "simd")
        };
        let first = timed_calibration(first);
        let second = timed_calibration(second);
        ensure!(
            first == second,
            "pilot labels, ranks or budgets differ in round {iteration}"
        );
    }
    println!(
        "{}",
        json!({
            "phase":"calibration_comparison",
            "iteration":iteration,
            "simd_s":simd_times,
            "gemm_s":gemm_times,
            "mismatches":0,
        })
    );
    Ok(())
}
