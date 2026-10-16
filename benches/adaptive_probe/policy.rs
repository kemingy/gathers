//! Pilot budget policy shared by the experiment and its focused tests.

/// Nearest-rank percentile with ceil(1.10 * p98), capped by the observed maximum.
pub(crate) fn budget(required: &[usize]) -> usize {
    assert!(!required.is_empty());
    let mut sorted = required.to_vec();
    sorted.sort_unstable();
    let p98 = sorted[(98 * sorted.len()).div_ceil(100) - 1];
    sorted[sorted.len() - 1]
        .min((110 * p98).div_ceil(100))
        .max(1)
}

#[cfg(test)]
mod tests {
    use super::budget;

    #[test]
    fn nearest_rank_margin_and_cap_at_requested_pilot_sizes() {
        for size in [256usize, 512, 1024] {
            assert_eq!(budget(&vec![1; size]), 1);
            let mut required = vec![10; size];
            required[size - 1] = 20;
            assert_eq!(budget(&required), 11);
            // This many tail observations puts 20 at the nearest-rank p98.
            let tail_start = (98 * size).div_ceil(100) - 1;
            required[tail_start..].fill(20);
            assert_eq!(budget(&required), 20);
            required.reverse();
            assert_eq!(budget(&required), 20);
        }
    }
}
