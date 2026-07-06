//! # Diffable
//!
//! A differential geometry framework for Rust. Implementing a trait certifies
//! that a type carries the corresponding mathematical structure — a group
//! operation, a metric, a smooth atlas — and the trait hierarchy encodes the
//! dependency order between those structures, so blanket implementations
//! derive the consequences automatically.
//!
//! ## Structure
//!
//! The library is organised around a hierarchy of traits that mirror
//! the mathematical structure of differential geometry:
//!
//! ### Foundation — sets, distances, inner products
//!
//! - [`traits::Point`] — an element of a carrier set; no topology, no
//!   smoothness, just the ability to hold and duplicate a value
//! - [`traits::Scalar`] — a real-number type for use as a coordinate field;
//!   see its doc comment for the library's stance on approximate equality
//! - [`traits::Metric`] — a distance function `d: M × M → R`, independent
//!   of any coordinate structure
//! - [`traits::InnerProduct`] — a bilinear inner product inducing a norm
//!   and metric
//!
//! ### Charts — local coordinate structure
//!
//! - [`traits::Chart`] — a coordinate chart mapping points of a manifold to
//!   a Euclidean coordinate space and back
//! - [`traits::ExpMap`] — a chart whose coordinate lines are geodesics and
//!   whose coordinate distances are arc lengths
//! - [`traits::TangentBundle`] — a family of `ExpMap` charts, one centred
//!   at each point of the manifold; the tangent bundle `TM`
//!
//! ### Smooth structure — self-charting manifolds
//!
//! - [`traits::Smooth`] — a manifold that charts itself: provides `exp` and
//!   `log` at every point, automatically generating `Chart`, `ExpMap`, and
//!   `TangentBundle` via blanket implementations; the only trait a
//!   self-charting manifold needs to implement
//!
//! ### Algebra — groups and Lie groups
//!
//! - [`traits::Group`] — an associative composition with identity and
//!   inverses, spelled with operator-agnostic named methods (`compose`,
//!   `inverse`) rather than `Add`/`Mul`, so it covers abelian and
//!   non-abelian groups alike; purely algebraic, no smoothness required.
//!   [`traits::CMonoid`]/[`traits::CGroup`] (`+`, commutative) and
//!   [`traits::Monoid`]/[`traits::MulGroup`] (`*`, not assumed commutative)
//!   are the two operator-flavoured presentations a concrete type may use,
//!   bridged to `Group` in one line via `impl_group_via_add!`/
//!   `impl_group_via_mul!`; [`traits::Rig`]/[`traits::Ring`] combine both
//!   for types with a compatible addition and multiplication
//! - [`traits::LieGroup`] — a group with a smooth exponential map at the
//!   identity; automatically derives `Smooth` (and therefore the full chart
//!   bundle) via left translation
//! - [`traits::Quotient`] — a quotient `G/H` of a Lie group by a subgroup,
//!   inheriting Lie group structure from the parent
//!
//! ### Euclidean — flat space
//!
//! - [`traits::Euclidean`] — the canonical flat space `Rⁿ`; simultaneously
//!   an inner-product space, its own tangent bundle, and an additive Lie
//!   group, with all three structures coinciding trivially
//!
//! ### Global topology — covers, nerves, fundamental groups
//!
//! - [`traits::Bounded`] — a `TangentBundle` chart with a bounded domain,
//!   expressed via a signed distance field
//! - [`traits::NerveComplex`] — a finite cover of a manifold by `Bounded`
//!   charts, whose overlap pattern forms a simplicial complex; computes
//!   global geodesic distance by graph search and recovers the fundamental
//!   group `π₁(M)` from the nerve
//! - [`traits::GroupPresentation`] — a group described by generators and
//!   relations; the output type of `NerveComplex::fundamental_group`
//!
//! ### Blanket chains
//!
//! The following chains fire automatically — implement the left-hand trait,
//! receive the right-hand traits for free:
//!
//! ```text
//! Smooth<V>             →  Chart<Self, V>, ExpMap<Self, V>, TangentBundle<Self, V>
//! LieGroup<V>           →  Smooth<V>  →  Chart, ExpMap, TangentBundle
//! Euclidean             →  Group, LieGroup<Self>  →  Smooth<Self>  →  Chart, ExpMap, TangentBundle
//! Quotient<G, H, V>     (via macro)  →  Group, LieGroup<V>  →  Smooth  →  ...
//! ```
//!
//! `Group` itself is reached via a one-line macro rather than a blanket
//! impl (`CMonoid`/`Monoid` can't both blanket-impl the same trait without
//! overlapping), so every `LieGroup` implementor pairs its `+`/`*`
//! structure with `impl_group_via_add!`/`impl_group_via_mul!` before
//! joining the chain above.
//!
//! ## Implementations
//!
//! - [`coords::Coords`] — the canonical Euclidean space `Rⁿ`, implemented
//!   as a fixed-size array of scalars
//! - [`hypersphere::Sphere`] — the unit hypersphere `Sⁿ` as a smooth
//!   manifold with geodesic structure for any dimension
//! - [`hypersphere::S0`], [`hypersphere::UnitComplex`],
//!   [`hypersphere::S3`] — the Lie group structures on `S⁰` (signs under
//!   multiplication), `S¹` (unit complex numbers), and `S³` (unit
//!   quaternions), as newtypes of `Sphere` that add group operations
//! - [`hypersphere::So3`] — the rotation group `SO(3)` as the quotient
//!   `S³/{±1}`, a newtype of `S3`
//! - [`hypersphere::Stereographic`] — stereographic projection charts for
//!   spheres, an external atlas independent of the geodesic self-charts
//! - [`discrete::Z`] — the integers, as the Grothendieck completion of the
//!   naturals [`discrete::N`]; also a degenerate 0-dimensional `LieGroup`
//! - [`flat::S1`] — the circle as the flat quotient `R/Z`; a more
//!   performant alternative model of `S¹` to `hypersphere::UnitComplex`
//! - [`flat::Torus`], [`flat::KleinBottle`] — flat surfaces built from two
//!   `flat::S1` coordinates, glued straight (a group) or with a
//!   fibre-flipping twist (not a group; the library's only non-orientable
//!   manifold)
//!
//! The newtype layering reflects the mathematical structure: `Sphere` is the
//! bare manifold (geometry only), `S3` adds the quaternion group operation,
//! and `So3` adds the antipodal identification. Each wrapper is zero-cost
//! and peelable — `.0` is the forgetful functor dropping one layer of
//! algebraic structure.
//!
//! ## Testing
//!
//! Enable the `testing` feature to access the `test_*` macros, which verify
//! that your implementations satisfy the mathematical invariants certified
//! by each trait. The scalar types [`R64`] and [`R32`] provide
//! tolerance-based equality suitable for property testing with floating
//! point.
//!
//! ```toml
//! [dev-dependencies]
//! diffable = { version = "...", features = ["testing"] }
//! ```
//!
//! ## Optional features
//!
//! - `testing` — property-testing macros and tolerance-based scalar types
//! - `all` — enables all features

pub mod coords;
pub mod hypersphere;
pub mod traits;

pub mod discrete;
pub mod epsilon_metric;
pub mod flat;
pub use epsilon_metric::{R32, R64};
