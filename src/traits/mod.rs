//! Mathematical certificates from which Diffable's concrete geometry is built.
//!
//! The flat namespace contains the foundational algebra, tensor, chart, and
//! metric traits most generic code uses. Higher differentiation lives in
//! [`calculus`], global finite-cover topology in [`simplicial`], and the
//! feature-gated property-test suites in `testing`.

pub mod calculus;

#[cfg(feature = "simplicial")]
pub mod simplicial;

mod algebra;
mod category;
mod chart;
mod foundation;
mod topology;
mod vector;

pub use algebra::*;
pub use category::*;
pub use chart::*;
pub use foundation::*;
pub use topology::*;
pub use vector::*;

#[cfg(feature = "testing")]
pub mod testing;
