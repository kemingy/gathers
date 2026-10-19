//! Validate the private PR65 experiment without exporting benchmark hooks.
use gathers::{distance, kmeans, simd, utils};

#[path = "../../../benches/adaptive_probe/cache.rs"]
mod cache;
#[allow(dead_code, unused_imports)]
#[path = "../../../src/kmeans/hierarchy.rs"]
mod hierarchy;
#[path = "../../../benches/adaptive_probe/policy.rs"]
mod policy;

#[allow(dead_code, unused_imports)]
#[path = "../../../src/linalg.rs"]
mod linalg;
#[allow(dead_code, unused_imports)]
#[path = "../../../src/kmeans/matrix.rs"]
mod matrix;
