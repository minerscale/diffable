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
//! - [`traits::LieGroup`] — a manifold with a smooth group structure
//! - [`traits::Euclidean`] — flat Euclidean space R^N; the canonical model for local coordinates
//! - [`traits::Metric`] — a notion of distance on a manifold
//!
//! ## Implementations
//!
//! - [`coords::Coords`] — the canonical Euclidean space R^N
//! - [`hypersphere`] — hyperspheres S^0, S^1, S^3 as Lie groups with geodesic structure
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
