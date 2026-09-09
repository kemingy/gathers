use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const RANDOM_SEED_ENV: &str = "GATHERS_RANDOM_SEED";

/// Creates a random generator from `GATHERS_RANDOM_SEED` or fresh entropy.
pub fn seeded_rng() -> StdRng {
    let seed = std::env::var(RANDOM_SEED_ENV).map_or_else(
        |_| rand::rng().random(),
        |value| {
            value
                .parse()
                .unwrap_or_else(|_| panic!("{RANDOM_SEED_ENV} must be an unsigned 64-bit integer"))
        },
    );
    eprintln!("random seed: {seed}; reproduce with {RANDOM_SEED_ENV}={seed}");
    StdRng::seed_from_u64(seed)
}
