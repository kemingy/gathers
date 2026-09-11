//! SIMD kernels and runtime-dispatched interfaces.

mod interface;
pub(crate) mod native;

pub mod pulp;

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
pub mod aarch64;

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub mod x86;

pub use interface::{
    asymmetric_binary_dot_product, min_max_residual, scalar_quantize, vector_binarize_query,
};
pub use native::{
    binary_dot_product as binary_dot_product_native, min_max_residual as min_max_residual_native,
};
