//! Mathematical certificates from which Diffable's concrete geometry is built.
//!
//! The flat namespace contains the foundational algebra, tensor, chart, and
//! metric traits most generic code uses. Higher differentiation lives in
//! [`calculus`], global finite-cover topology in [`simplicial`], and the
//! feature-gated property-test suites in `testing`.

pub mod calculus;
pub mod simplicial;

mod algebra;
mod chart;
mod foundation;
mod vector;

pub use algebra::*;
pub use chart::*;
pub use foundation::*;
pub use vector::*;

#[cfg(feature = "testing")]
pub mod testing;
