//! Compile production modules in their library namespace for private ablations.
pub use gathers::kmeans::{
    AssignmentPolicy, KMeans, KMeansConfig, ReductionConfig, base_assign, base_assign_parallel,
};
#[path = "../../src/kmeans/hierarchy.rs"]
pub(crate) mod hierarchy;
#[path = "../../src/kmeans/matrix.rs"]
pub(crate) mod matrix;
