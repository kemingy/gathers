//! Pilot budget policy shared by the experiment and its focused tests.

pub(crate) const DEFAULT_PILOTS: usize = 512;
pub(crate) const DEFAULT_PERCENTILE: usize = 99;

/// Nearest-rank percentile plus 10%, capped by the observed maximum.
pub(crate) fn budget(required: &[usize], percentile: usize) -> usize {
    assert!(!required.is_empty());
    assert!((1..=100).contains(&percentile));
    let mut sorted = required.to_vec();
    sorted.sort_unstable();
    let quantile = sorted[(percentile * sorted.len()).div_ceil(100) - 1];
    sorted[sorted.len() - 1]
        .min((110 * quantile).div_ceil(100))
        .max(1)
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_PERCENTILE, DEFAULT_PILOTS, budget};

    #[test]
    fn nearest_rank_margin_and_cap_at_requested_pilot_sizes() {
        assert_eq!((DEFAULT_PILOTS, DEFAULT_PERCENTILE), (512, 99));
        for size in [256usize, 512, 1024] {
            for percentile in [98usize, 99] {
                assert_eq!(budget(&vec![1; size], percentile), 1);
                let mut required = vec![10; size];
                required[size - 1] = 20;
                assert_eq!(budget(&required, percentile), 11);
                // Include exactly the nearest-rank boundary in the difficult tail.
                let tail_start = (percentile * size).div_ceil(100) - 1;
                required[tail_start..].fill(20);
                assert_eq!(budget(&required, percentile), 20);
                required.reverse();
                assert_eq!(budget(&required, percentile), 20);
            }
            // Distinguish the new p99 policy from the historical p98 baseline.
            let mut required = vec![10; size];
            required[(99 * size).div_ceil(100) - 1..].fill(20);
            assert_eq!(budget(&required, 98), 11);
            assert_eq!(budget(&required, 99), 20);
        }
    }
}
