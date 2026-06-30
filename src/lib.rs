//! # Diffable
//!
//! A differential geometry framework for Rust, providing abstractions for
//! smooth manifolds, charts, Lie groups, and tangent bundles.
//!
//! ## Structure
//!
//! The library is organised around a hierarchy of traits that mirror the
//! mathematical structure of differential geometry:
//!
//! - [`traits::Point`] — a point on a smooth manifold
//! - [`traits::Chart`] — a coordinate chart; the space of all charts is an atlas
//! - [`traits::ExpMap`] — a chart with geodesic structure; straight lines map to geodesics
//! - [`traits::TangentBundle`] — a family of exp charts, one per point; the tangent bundle `TM`
//! - [`traits::LieGroup`] — a manifold with a smooth group structure, carrying an exponential
//!   map at the identity that automatically generates a full tangent bundle via left translation
//! - [`traits::Quotient`] — a quotient `G/H` of a Lie group by a central subgroup, automatically
//!   inheriting Lie group and tangent bundle structure from the parent group
//! - [`traits::Euclidean`] — flat Euclidean space Rⁿ; the canonical model for local coordinates,
//!   carrying a canonical inner product, norm, metric, and tangent bundle
//! - [`traits::InnerProduct`] — an inner product space, inducing a norm and metric
//! - [`traits::Metric`] — a notion of distance on a manifold
//!
//! ## Implementations
//!
//! - [`coords::Coords`] — the canonical Euclidean space Rⁿ
//! - [`hypersphere`] — hyperspheres S⁰, S¹, S³ as Lie groups with geodesic structure
//! - [`hypersphere::So3`] — the rotation group SO(3), as the quotient S³/{±1}
//!
//! ## Testing
//!
//! Enable the `testing` feature to access the `test_*` macros, which verify
//! that your implementations satisfy the mathematical invariants certified
//! by each trait:
//!
//! ```toml
//! [dev-dependencies]
//! diffable = { version = "...", features = ["testing"] }
//! ```
//!
//! ## Optional Features
//!
//! - `nalgebra` — interop with nalgebra's `SVector` and `UnitQuaternion`
//! - `testing` — property-testing macros for verifying trait implementations
//! - `all` — enables all features

pub mod coords;
pub mod hypersphere;
pub mod traits;

#[cfg(feature = "nalgebra")]
pub mod nalgebra;
