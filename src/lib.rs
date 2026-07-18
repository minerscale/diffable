//! # Diffable
//!
//! A differential geometry framework for Rust. Each trait represents a
//! mathematical structure—group, vector space, smooth atlas, metric, and
//! so on. Implementing a trait certifies that a type carries that structure,
//! while blanket implementations derive the structures that follow from it.
//! In practice, implementing a single high-level trait often gives
//! you the surrounding geometry for free.
//!
//! ### Structure
//!
//! The library is organised around a hierarchy of traits that mirror the
//! mathematical structure of differential geometry.
//!
//! #### Foundation — points, scalars, separation
//!
//! - [`traits::Point`] — The underlying set: an element of a manifold, group, or
//!   metric space. Anything can be a Point, in fact; anything that is `Clone` is
//!   a `Point`.
//! - [`traits::Field`] — The scalar field of a vector space. Follows the field axioms.
//! - [`traits::Real`] — An ordered real-number field, used as a
//!   coordinate scalar and as the target of intervals and metrics. See
//!   [`traits::ExactCmp`] for the strict order that convergence tests need
//!   when an implementor's equality is tolerance-based.
//! - [`traits::Interval`] — A *signed* squared interval `s²: M × M → R`
//!   (negative timelike, zero null, positive spacelike); the pseudo-metric
//!   base, claiming no metric-space axioms. `interval_squared` is the
//!   primitive, `interval` its signed square root, returning `Complex<R: Real>`
//! - [`traits::Metric`] — the *definite* refinement: a genuine non-negative
//!   distance `d = √(interval_squared)`. Independent of any coordinate
//!   structure
//!
//! #### Vector spaces and forms
//!
//! - [`traits::Vector`] — A finite-dimensional coordinate vector space over a
//!   `Field`. It is the local model a [`traits::Chart`] maps into and the
//!   tangent space of every manifold, and is its own additive
//!   [`traits::LieGroup`]. [`traits::Dual`] is the dual space `V*`.
//! - A bare `Vector` carries no metric. Scalar products are induced by
//!   progressively stronger traits:
//! 	- [`traits::Form`] gives a lowering map `♭: V → V*`
//!       and the induced pairing `dot`.
//! 	- [`traits::Nondegenerate`] makes `♭` invertible by adding `♯`.
//!     - [`traits::Sesquilinear`] specialises to Hermitian forms.
//!		- [`traits::Bilinear`] specialises further to symmetric bilinear forms.
//!     - [`traits::InnerProduct`] adds positive definiteness.
//!
//! #### Charts — local coordinate structure
//!
//! - [`traits::Chart`] — A coordinate chart mapping points of a manifold to a
//!   flat coordinate space and back.
//! - [`traits::ExpMap`] — A chart whose coordinate lines are geodesics and
//!   whose coordinate distances are (signed) arc lengths.
//! - [`traits::TangentBundle`] — The tangent bundle `TM`: an `ExpMap` chart
//!   centred at *every* point. This is the working surface of a smooth
//!   manifold — `exp` and `log`, geodesics, geodesic distance, and sectional
//!   curvature are all read off it, so most geometric computation is written
//!   against this trait.
//! - [`traits::PseudoRiemannian`] — Certifies that the exponential map and the
//!   tangent-space form agree: the geodesic interval equals `⟨v,v⟩`.
//!   Signature-agnostic; reduces to the usual `d(p, exp_p v) = ‖v‖` in the
//!   definite case.
//! - [`traits::Smooth`] — A helper trait for manifolds that charts themselves,
//!   providing `exp` and `log` at every point. Implement this one trait and the
//!   full chart bundle `Chart`, `ExpMap`, `TangentBundle` for free.
//!
//! #### Algebra — groups and Lie groups
//!
//! - [`traits::Group`] — an operator-agnostic group interface, using named
//!   methods rather than `+` or `*`.
//!
//!   [`traits::CMonoid`]/[`traits::CGroup`] and
//!   [`traits::Monoid`]/[`traits::MulGroup`] are the additive and multiplicative
//!   presentations used by concrete types. The helper macros `impl_group_via_add!`
//!   and `impl_group_via_mul!` connect them to `Group`.
//!
//!   [`traits::Rig`], [`traits::Ring`], [`traits::DivRing`],
//!   and `Field` combine both operations.
//! - [`traits::LieGroup`] — a group with a smooth exponential map at the
//!   identity; automatically derives `Smooth` (and therefore the whole chart
//!   bundle) via left translation
//! - [`traits::Quotient`] — a quotient `G/H` of a Lie group by a subgroup,
//!   inheriting Lie group structure from the parent
//!
//! #### (Pseudo-)Euclidean — flat space
//!
//! - [`traits::Quadratic`] — flat coordinate space of arbitrary signature
//!   (Minkowski included): a symmetric bilinear scalar product, its own
//!   tangent bundle, and an additive Lie group. The indefinite base, carrying
//!   no norm or distance
//! - [`traits::Euclidean`] — its positive-definite refinement: the canonical
//!   flat `Rⁿ` that is simultaneously an inner-product space, its own tangent
//!   bundle, and an additive Lie group.
//!
//! #### Global topology — covers, nerve complexes, fundamental groups
//!      and global geodesic minimisation
//!
//! - [`traits::Bounded`] — a `TangentBundle` chart with a bounded, open domain,
//!   expressed via a signed distance field.
//! - [`traits::NerveComplex`] — a finite cover of a manifold by `Bounded`
//!   charts whose overlap pattern forms a simplicial complex; computes global
//!   geodesic distance by graph search and recovers the fundamental group
//!   `π₁(M)` from the nerve. Since the cover finite and open,
//!   NerveComplex serves as a proof that the implemented manifold is compact.
//! - [`traits::GroupPresentation`] — a group described by generators and
//!   relations; the output of `NerveComplex::fundamental_group`. Group presentation
//!   does not implement `Group` because the
//!   [word problem](https://en.wikipedia.org/wiki/Word_problem_(mathematics))
//!   is uncomputable in general.
//!
//! #### Blanket chains
//!
//! Implement one trait; receive the these for free:
//!
//! ```text
//! Smooth<V>             →  Chart<Self, V>, ExpMap<Self, V>, TangentBundle<Self, V>
//! Vector                →  LieGroup<Self> → Smooth → Chart, ExpMap, TangentBundle
//! LieGroup<V>           →  Smooth<V> → Chart, ExpMap, TangentBundle
//! Quadratic/Euclidean   →  Group, LieGroup<Self> → Smooth → Chart, ExpMap, TangentBundle
//! Quotient<G, H, V>     →  Group, LieGroup<V> → Smooth → ...
//! Sesquilinear (real)   →  Bilinear
//! ```
//!
//! `Group` itself is reached via a one-line macro rather than a blanket impl
//! (`CMonoid`/`Monoid` can't both blanket-impl the same trait without
//! overlapping), so every `LieGroup` implementor pairs its `+`/`*` structure
//! with `impl_group_via_add!`/`impl_group_via_mul!` before joining the chain.
//!
//! ### Implementations
//!
//! - [`coords::Coords`] — the canonical flat space `R^(N−M, M)`, a fixed-size
//!   array parameterised by a signature `M` (the count of timelike
//!   directions). `M = 0` is ordinary Euclidean `Rⁿ` (with a norm and metric);
//!   `M > 0` is indefinite (`Coords<R, 4, 1>` is Minkowski spacetime),
//!   carrying only a `Bilinear` form
//! - [`complex::Complex`] — the complex numbers as a `Field`, with `conj` the
//!   Hermitian involution. [`traits::Symmetrized`] wraps a field to select its
//!   *bilinear* rather than Hermitian form
//! - [`matrix::Matrix`] — an `N×N` matrix, interpreted as the tensor
//!   `V ⊗ V*`, with variance encoded in the type so only variance-correct
//!   contractions typecheck. [`matrix::MatrixExponential`] provides `exp`/`log`.
//! - [`hypersphere::Sphere`] — the unit hypersphere `Sⁿ` as a smooth manifold
//!   with geodesic structure for any dimension
//! - [`hypersphere::S0`], [`hypersphere::UnitComplex`], [`hypersphere::S3`] —
//!   the Lie group structures on the three parallelizable spheres (signs, the
//!   unit complex numbers `U(1)`, the unit quaternions `SU(2)`), as newtypes
//!   of `Sphere` that add the group operation
//! - [`hypersphere::So3`] — the rotation group `SO(3)` as the quotient
//!   `S³/{±1}`, a newtype of `S3`
//! - [`hypersphere::Stereographic`] — stereographic projection charts, an
//!   external atlas independent of the geodesic self-charts
//! - [`spacetime::Minkowski`] — `Coords<R, 4, 1>`, spacetime with signature
//!   `(−,+,+,+)`; [`spacetime::Sl`]/[`spacetime::Sl2c`] the special linear
//!   group (`SL(2,ℂ)` double-covering the Lorentz group);
//!   [`spacetime::SlAlgebra`] its traceless Lie algebra with the Killing form;
//!   and [`spacetime::Lorentz`] the restricted Lorentz group `SO⁺(1,3)` as
//!   `SL(2,ℂ)/{±1}`
//! - [`discrete::Z`] — the integers, as the Grothendieck completion of the
//!   naturals [`discrete::N`]; also the covering lattice for `flat::S1`
//! - [`flat::S1`] — the circle as the flat quotient `R/Z`, a more performant
//!   model of `S¹` than `hypersphere::UnitComplex`;
//!   [`flat::Torus`]/[`flat::KleinBottle`] glue two circles straight (a group)
//!   or with a fibre-flipping twist (the library's only non-orientable
//!   manifold)
//!
//! The newtype layering reflects the mathematical structure: `Sphere` is the
//! bare manifold (geometry only), `S3` adds the quaternion group operation,
//! and `So3` adds the antipodal identification. Each wrapper is zero-cost and
//! peelable — `.0` is the forgetful functor dropping one layer of structure.
//!
//! ### Testing
//!
//! Diffable takes the philosophy that any axiom which is assumed true of a type
//! but not directly enforcable by the compiler should be emperically verified
//! via property testing. Enable the `testing` feature to access the `test_*`
//! macros, which verify that your implementations satisfy the mathematical
//! invariants certified by each trait. The `Real` types `R64` and `R32` provide
//! tolerance-based equality suitable for property testing with floating point,
//! since the library assumes that its real numbers are perfect.
//!
//! ```toml
//! [dev-dependencies]
//! diffable = { version = "...", features = ["testing"] }
//! ```
//!
//! ### Optional features
//!
//! - `testing` — property-testing macros and tolerance-based scalar types
//! - `all` — enables all features

pub mod coords;
pub mod hypersphere;
pub mod traits;

pub mod complex;
pub mod discrete;
pub mod epsilon_metric;
pub mod flat;
pub mod matrix;
pub mod spacetime;
pub use epsilon_metric::{R32, R64};
