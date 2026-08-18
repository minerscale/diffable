//! Diffable is an opinionated differential-geometry framework for Rust. Its
//! central idea is that mathematical structure should be executable:
//!
//! - a type supplies the underlying values;
//! - a trait implementation certifies what mathematical structure those values
//!   carry;
//! - blanket implementations encode theorems relating those structures; and
//! - property tests check the axioms Rust's type system cannot prove.
//!
//! Generic code can therefore ask for what its argument *is*, rather than for an
//! incidental collection of methods. A function that needs a nondegenerate form
//! can say so without requiring an inner product. A function that works in
//! spacetime can request a signed interval without pretending it has a metric.
//! And when one structure mathematically entails another, the consequence is
//! implemented once for every type.
//!
//! > **Traits are mathematical certificates. Blanket implementations are
//! > theorems.**
//!
//! Also, it is optionally `no-std`.
//!
//! ## Geometry from one local implementation
//!
//! A Lie group is homogeneous: its geometry at the identity can be transported
//! to every other point by left translation. Diffable writes that argument as a
//! blanket implementation:
//!
//! ```compile_fail,E0210
//! use diffable::prelude::*;
//!
//! impl<V: Vector, L: LieGroup<V>> Smooth<V> for L {
//!     fn exp(&self, coord: V) -> Self {
//!         self.compose(&Self::identity_exp(coord))
//!     }
//!
//!     fn log(&self, point: &Self) -> Option<V> {
//!         Self::identity_log(&self.inverse().compose(point))
//!     }
//! }
//! ```
//!
//! An implementor of [`LieGroup<V>`](traits::LieGroup) provides the group operation and the
//! exponential and logarithmic maps at the identity. Left translation then
//! supplies [`Smooth<V>`](traits::Smooth), from which the full chart bundle follows:
//!
//! ```text
//! LieGroup<V>
//!     ⇒ Smooth<V>
//!     ⇒ Chart<Self, V>
//!     ⇒ ExpMap<Self, V>
//!     ⇒ TangentBundle<Self, V>
//! ```
//!
//! This is the pattern throughout Diffable: implement the smallest structure
//! that characterises an object, then inherit its mathematical consequences.
//!
//! ## Handedness, duality, and geometry
//!
//! Diffable permits noncommutative scalar fields, so every
//! [`Vector`](traits::Vector) explicitly elects whether its field acts on the
//! left or on the right. Concrete coordinate spaces conventionally elect
//! [`Right`](traits::Right); [`Dual<V>`](traits::Dual) elects the opposite hand:
//!
//! ```text
//! V right-handed  ⇒ V* left-handed  ⇒ V** right-handed
//! V left-handed   ⇒ V* right-handed ⇒ V** left-handed
//! ```
//!
//! The ordinary `Mul<F>` operation always follows the elected hand. Thus `v * k`
//! means `vk` on a right module and `kv` on a left module; no separate dual
//! scalar API is needed.
//!
//! Canonical evaluation follows the same choice. For coordinates `vᵢ` and
//! `ωᵢ`,
//!
//! ```text
//! right-handed V:  ω(v) = Σ ωᵢvᵢ
//! left-handed  V:  ω(v) = Σ vᵢωᵢ
//! ```
//!
//! This order is invisible over the reals or complexes but observable over the
//! quaternions. [`Tensor::pairing`](traits::Tensor::pairing) selects it from
//! [`Tensor::Hand`](traits::Tensor::Hand), while `Dual<Dual<V>>` restores the
//! hand of `V`.
//!
//! The [`Dual<V>`](traits::Dual) wrapper is coordinate-identical to `V`, but raw
//! coordinates do not carry a geometric identification between the two spaces.
//! [`Dual::from_raw`](traits::Dual::from_raw) merely declares covector
//! coordinates. Finite dimensionality supplies only the evaluation isomorphism
//!
//! ```text
//! V** ≅ V
//! ```
//!
//! implemented by [`Dual<Dual<Tensor>>::collapse`](traits::Dual<traits::Dual<traits::Tensor>>::collapse).
//! Geometry enters when [`Form`](traits::Form) chooses a lowering map
//!
//! ```text
//! ♭ : V → V*
//! ```
//!
//! and defines `dot(a, b)` by evaluating `b♭` on `a`. A degenerate form may
//! collapse distinct vectors to the same covector; [`Nondegenerate`](traits::Nondegenerate)
//! certifies that `♭` is invertible and supplies the raising map
//!
//! ```text
//! ♯ : V* → V.
//! ```
//!
//! These are the musical isomorphisms. They are not coordinate
//! reinterpretations: they encode the space's chosen geometric relationship
//! with its dual. The dual space inherits the corresponding form through those
//! maps.
//!
//! ## Invariants are representation choices
//!
//! [`Sl<V, N>`](spacetime::Sl) represents the special linear group. Its matrix is private, and
//! there is no constructor from an arbitrary matrix. Values can be reached
//! through operations that preserve determinant one: identity, composition,
//! inverse, and exponentiation from the traceless Lie algebra.
//!
//! Likewise, [`SlAlgebra<F, N, D>`](spacetime::SlAlgebra) stores coordinates in a basis whose elements
//! are traceless. A non-traceless matrix is not an invalid value to be detected
//! later; it is not a value the representation can express.
//!
//! Consequently, exponentiation has the meaningful type
//!
//! ```text
//! exp : sl(N) → SL(N)
//! ```
//!
//! rather than returning an arbitrary matrix accompanied by a runtime claim that
//! it probably belongs to the group. Membership is a theorem about reachability.
//!
//! The same principle appears at smaller scales. [`NonZero<T>`](traits::NonZero) certifies that a
//! value lies in the multiplicative group, [`Dual<V>`](traits::Dual) distinguishes covectors
//! from vectors even when their coordinates coincide, and matrix variance is
//! encoded so that only variance-correct contractions typecheck.
//!
//! ## Constructions propagate structure
//!
//! Diffable's concrete spaces are deliberately built from reusable mathematical
//! constructions:
//!
//! ```text
//! S³ / {±1}       ⇒ SO(3)
//! SL(2, ℂ) / {±1} ⇒ SO⁺(1, 3)
//! ℝ / ℤ           ⇒ S¹
//! S¹ × S¹         ⇒ T²
//! ```
//!
//! The quotient machinery does not know about rotations or relativity. It knows
//! that a suitable quotient of a Lie group inherits Lie-group structure. The
//! same implementation therefore gives both `SO(3)` and the restricted Lorentz
//! group their group operations, exponential maps, charts, and tangent bundles.
//!
//! The torus and Klein bottle make the distinction equally clear. Both are made
//! by gluing two circles; straight gluing produces a Lie group, while twisted
//! gluing produces a smooth non-orientable manifold without falsely granting it
//! group structure.
//!
//! The type hierarchy records these differences instead of flattening every
//! space into coordinates and asking the programmer to remember what remains
//! valid.
//!
//! ## Automatic differentiation as typed programs
//!
//! [`d`](traits::calculus::d) turns an ordinary generic Rust function into a composable
//! differential program. Calling `d` does not evaluate anything: derivatives can
//! be nested, contracted with directions, and only then evaluated with `at`.
//!
//! ```rust
//! use diffable::{
//!     coords::Coords,
//!     traits::{calculus::d, Euclidean, Field, Tensor},
//! };
//!
//! fn cube<V: Euclidean>(x: V) -> V {
//!     x.map(|x| x.powi(3))
//! }
//!
//! // Full higher derivatives use the same operator recursively.
//! let third = d(d(d(cube))).at(Coords::from(-6.0));
//! assert_eq!(third[0], 6.0);
//!
//! // Contract a derivative slot before evaluation.
//! let directional = d(cube)
//!     .along(Coords::from(4.0))
//!     .at(Coords::from(7.0));
//! assert_eq!(directional[0], 588.0);
//!
//! // Differential programs remain differentiable, including their directions.
//! let diagonal = d(|v| d(cube).along(v).at(v))
//!     .at(Coords::from(7.0));
//! assert_eq!(diagonal[0], 441.0);
//! ```
//!
//! The last expression differentiates `v ↦ D(cube)ᵥ(v)`. This is deliberately
//! ordinary Rust syntax: there is no tracing macro, tape, boxed closure, or
//! type-erased expression graph. Internally, truncated Taylor presentations carry
//! coefficients through the existing field and tensor interfaces, while
//! [`JetMap`](traits::calculus::JetMap) interprets the program at each required nesting
//! depth. A type-level [`ConstantRoute`](traits::calculus::ConstantRoute) injects captured
//! base-field values through that jet tower.
//!
//! The full derivative of `f: U → V` is a
//! [`TangentMap`](traits::calculus::TangentMap), represented as `V ⊗ U*` in
//! output-by-input coordinate order. `along` contracts one input slot and returns
//! the directional derivative directly. If a composition cannot be evaluated,
//! the public `at` boundary reports that the function lacks the required jet
//! presentation, its tensor structure is incompatible, or a musical isomorphism
//! does not lift through the nested jets.
//!
//! This machinery is not restricted to flat coordinates. Implementing
//! [`TangentLift`](traits::calculus::TangentLift) tells Diffable how a manifold's tangent
//! charts act on jets. [`FormLift`](traits::calculus::FormLift) and
//! [`NondegenerateLift`](traits::calculus::NondegenerateLift) do the corresponding job for
//! the lowering and raising maps, allowing generic Euclidean code to remain valid
//! inside higher derivatives.
//!
//! ## Categories, contexts, and dependent structure
//!
//! Rust traits describe what a value *is* operationally. Diffable's category
//! machinery describes the corresponding mathematical theories and records how
//! particular mathematical objects depend on one another.
//!
//! It is useful to distinguish three levels:
//!
//! ```text
//! Rust trait       Field, Vector, Manifold
//!                  operational interface implemented by values
//!
//! category         𝐅𝐥𝐝, 𝐕𝐞𝐜𝐭, 𝐌𝐚𝐧
//!                  nominal mathematical theory describing required structure
//!
//! context          C![𝐅𝐥𝐝], C![𝐑𝐞𝐚𝐥], ...
//!                  finite type-level model witnessing a theory
//! ```
//!
//! The typography is deliberate. Ordinary Rust names denote executable traits
//! and types, while bold mathematical names denote theories in the category
//! language. `C![𝒞]` denotes the context of a theory `𝒞`.
//!
//! A category does not represent a runtime value. It describes the structure,
//! inherited properties, associated objects, and equations required by a
//! mathematical theory. Categories themselves form a refinement graph: for
//! example, `𝐑𝐞𝐚𝐥` carries field structure, while `𝐕𝐞𝐜𝐭` carries tensor and
//! group structure.
//!
//! A context is a finite model of such a theory for a particular Rust object.
//! The category says *what must hold*; the context records *how it holds here*.
//! In particular, a context may contain named dependent edges to other
//! mathematical objects, together with the exact contexts in which those child
//! objects are known.
//!
//! For example, a tensor has a scalar field. At the ordinary Rust level this is
//! simply an associated type:
//!
//! ```text
//! Tensor::F : Field
//! ```
//!
//! But the concrete scalar may carry more structure than `Tensor` requires. A
//! Euclidean coordinate space over `f64`, for example, may retain an edge of the
//! form
//!
//! ```text
//! tensor context
//!     |
//!     `-- tensor::F
//!           role    = Real
//!           value   = f64
//!           context = the Real context of f64
//! ```
//!
//! The tensor theory requires only that this child satisfy `Field`. Because
//! `Real` is a stronger role, the edge satisfies that weaker requirement without
//! losing the context it already stores. Thus a parent may be viewed through a
//! weaker theory while its dependent children retain stronger evidence.
//!
//! **Requirements may weaken; stored context does not.**
//!
//! Four operations form the core of this system:
//!
//! ```text
//! ι          include an ordinary Rust type in its distinguished richest context
//! Ⱶ<𝒞>       view an existing context as satisfying the weaker theory 𝒞
//! π<Name>    follow a named dependent edge and recover its stored child context
//! Model<𝒞,X> include X and then view its context as a model of 𝒞
//! ```
//!
//! A named binding is therefore richer than an associated-type equality.
//! Conceptually,
//!
//! ```text
//! Binds<Name, Role, Value, Context>
//! ```
//!
//! says that following `Name` reaches `Value`, that the child is known there
//! under `Role`, and that `Context` is the exact contextual evidence carried by
//! that edge. Applying `π<Name>` returns this stored information rather than
//! reconstructing a fresh interpretation from the Rust type.
//!
//! This gives Diffable a restricted form of dependent programming. Generic code
//! can reason not only about the type of an associated object, but about facts
//! which depend on the particular context in which that object was reached. A
//! manifold may therefore carry a tangent space together with its mathematical
//! structure, or a tensor may carry a scalar whose stronger field properties
//! remain available to later reasoning.
//!
//! Importantly, this machinery does **not** replace the ordinary trait hierarchy.
//! [`Field`](traits::Field), [`Tensor`](traits::Tensor),
//! [`Vector`](traits::Vector), [`Manifold`](traits::Manifold), and the rest remain
//! the normal computational API. Contexts are used only when code needs to retain
//! or inspect mathematical evidence which ordinary associated types cannot
//! express.
//!
//! The resulting picture is:
//!
//! ```text
//! ordinary Rust object
//!        |
//!        |  ι
//!        v
//! contextual object
//!        |
//!        |  π<Name>
//!        v
//! contextual child object
//! ```
//!
//! Most users need never spell these types explicitly. Their purpose is to let
//! Diffable's implementations preserve and derive mathematical facts without
//! forcing that proof machinery into ordinary numerical code.
//!
//! ### Stable negative bounds
//!
//! The same finite context language also provides a restricted form of stable
//! negative trait reasoning.
//!
//! Rust does not provide general stable negative trait bounds. In ordinary Rust,
//! it is difficult to write two coherent implementations distinguished only by
//!
//! ```text
//! T: Real
//! ```
//!
//! versus
//!
//! ```text
//! T: Field but not Real
//! ```
//!
//! because ordinary trait lookup is open-ended: the compiler cannot generally
//! treat the absence of an implementation as permanent mathematical evidence.
//!
//! A Diffable context is different. Its nominal theory graph is finite and
//! closed. Searching that graph for a property therefore has a total type-level
//! result: the property is either present or constructively absent. Diffable
//! represents these outcomes using witnesses such as `Present` and `Absent`.
//!
//! Thus
//!
//! ```text
//! C contains Real
//! ```
//!
//! and
//!
//! ```text
//! C contains Field
//! C does not contain Real
//! ```
//!
//! become disjoint regions of the context language. Implementations can dispatch
//! on those regions without unstable specialization or language-level negative
//! trait bounds.
//!
//! Automatic differentiation uses exactly this capability.
//!
//! A jet over a real scalar must itself support real operations so that ordinary
//! generic real-valued functions remain differentiable. A jet over a field which
//! is not real must remain merely field-valued. These implementations would
//! otherwise overlap precisely when a concrete real scalar is passed through
//! generic code which asks only for the weaker `Field` structure.
//!
//! [`JetRegion`](traits::calculus::JetRegion) resolves that ambiguity by
//! partitioning canonical scalar contexts into two disjoint cases:
//!
//! ```text
//! Real present
//!     ⇒ use the Real jet interpretation
//!
//! Field present, Real absent
//!     ⇒ use the Field jet interpretation
//! ```
//!
//! The distinction matters at the public API. The differentiated function itself
//! need not advertise that its concrete scalar will eventually be real:
//!
//! ```rust
//! use diffable::{
//!     coords::Coords,
//!     traits::{calculus::d, Vector},
//! };
//!
//! fn square<V: Vector>(x: V) -> V {
//!     V::from_iter([x[0] * x[0]])
//! }
//!
//! let first = d(square).at(Coords::from(3.0));
//! let second = d(d(square)).at(Coords::from(3.0));
//!
//! assert_eq!(first[0], 6.0);
//! assert_eq!(second[0], 2.0);
//! ```
//!
//! Here `square` requires only [`Vector`](traits::Vector), so its scalar is known
//! generically only through the weaker field theory. At evaluation, however,
//! `Coords<f64, 1>` supplies a concrete real scalar.
//!
//! At the context-free `at` boundary, `ι` selects that scalar's distinguished
//! richest context. `JetRegion` can then use both positive and negative facts
//! about the finite context to select exactly one jet interpretation. Nested
//! differentiation repeats the same process without requiring the caller to
//! choose a jet mode, specify a scalar theory, or disambiguate overlapping
//! implementations.
//!
//! Conceptually:
//!
//! ```text
//! ordinary Rust type
//!        |
//!        |  ι
//!        v
//! finite closed context
//!        |
//!        |  property search
//!        v
//!   Present / Absent
//!        |
//!        v
//! coherent implementation dispatch
//! ```
//!
//! This is intentionally weaker than general negative trait bounds. Diffable
//! cannot prove that an arbitrary Rust trait implementation will never exist.
//! It can prove absence only inside the finite nominal theory graph represented
//! by a context. That smaller closed-world proposition is nevertheless enough to
//! make otherwise-overlapping mathematical implementations coherently selectable
//! on stable Rust.
//!
//! ## Trait hierarchy
//!
//! The trait graph is intentionally fine-grained. Generic algorithms should
//! state the weakest honest assumptions their proofs require.
//!
//! - [`Field`](traits::Field) permits noncommutative division rings;
//!   [`CField`](traits::CField) adds
//!   commutativity.
//! - [`Form`](traits::Form) provides the lowering map `♭: V → V*`;
//!   [`Nondegenerate`](traits::Nondegenerate) adds its inverse `♯`.
//! - [`Sesquilinear`](traits::Sesquilinear) certifies a Hermitian form;
//!   [`Bilinear`](traits::Bilinear) is the fixed-field specialisation;
//!   [`InnerProduct`](traits::InnerProduct) adds positive definiteness.
//! - [`Interval`](traits::Interval) provides a signed squared separation and
//!   accommodates pseudo-Riemannian geometry; [`Metric`](traits::Metric) adds
//!   genuine metric-space distance.
//! - [`Chart`](traits::Chart) provides coordinates; [`ExpMap`](traits::ExpMap)
//!   says those coordinates are geodesic; [`TangentBundle`](traits::TangentBundle)
//!   supplies such a chart at every point.
//! - [`TangentLift`](traits::calculus::TangentLift) extends tangent charts through jet
//!   coordinates; [`FormLift`](traits::calculus::FormLift) and
//!   [`NondegenerateLift`](traits::calculus::NondegenerateLift) extend the musical maps.
//!
//! Degenerate and indefinite cases are not malformed approximations to
//! Euclidean geometry. They are first-class structures with precisely the
//! operations their axioms justify.
//!
//! ## Implementations
//!
//! ### Scalars, vectors, and tensors
//!
//! - [`coords::Coords`] is the canonical fixed-dimensional coordinate space
//!   `R^(N−M, M)`, parameterised by the number `M` of timelike directions.
//!   `M = 0` is Euclidean; `Coords<R, 4, 1>` is Minkowski spacetime.
//! - [`complex::Complex`] implements the complex numbers with conjugation as
//!   their elected involution. [`traits::Symmetrized`] elects the bilinear rather
//!   than Hermitian form.
//! - [`quaternion::Quaternion`] provides the quaternion division algebra.
//! - [`matrix::Matrix`] represents an endomorphism of `V`: as `V ⊗ V*` when
//!   `V` is right-handed and `V* ⊗ V` when it is left-handed. Tensor variance
//!   and handedness are carried by the types.
//!   [`matrix::MatrixExponential`] supplies matrix `exp` and `log`.
//! - [`traits::calculus::d`] and [`traits::calculus::Along`] implement forward automatic
//!   differentiation as typed programs. Internally, jettification preserves the existing
//!   field and tensor interfaces rather than introducing a separate public algebra.
//!
//! ### Manifolds and Lie groups
//!
//! - [`hypersphere::Sphere`] provides `Sⁿ` with its intrinsic geodesic structure.
//! - [`hypersphere::S0`], [`hypersphere::UnitComplex`], and [`hypersphere::S3`]
//!   add the Lie-group structures on
//!   the three group spheres: signs, unit complex numbers, and unit quaternions.
//! - [`hypersphere::So3`] constructs `SO(3)` as `S³/{±1}`.
//! - [`hypersphere::Stereographic`] provides an external stereographic atlas,
//!   independently of the sphere's intrinsic exponential charts.
//! - [`flat::S1`] constructs the circle as `R/Z`; [`flat::Torus`] and
//!   [`flat::KleinBottle`] provide straight and twisted gluings of two circles.
//! - [`spacetime::Sl`] and [`spacetime::SlAlgebra`] implement the special linear
//!   group and its traceless Lie algebra; [`spacetime::Lorentz`] constructs `SO⁺(1,3)` as
//!   `SL(2,C)/{±1}`.
//! - [`discrete::N`] and [`discrete::Z`] implement the naturals and their
//!   Grothendieck group completion; [`discrete::Z`] also supplies the lattice
//!   used by [`flat::S1`].
//!
//! The newtypes add mathematical meaning one layer at a time. `Sphere` is a
//! manifold, `S3` equips that manifold with quaternion multiplication, and
//! `So3` adds the antipodal quotient. Forgetting a wrapper drops structure
//! without changing the underlying object.
//!
//! ## Global geometry and topology
//!
//! [`Bounded`](traits::simplicial::Bounded) describes a bounded open exponential-chart
//! domain by a signed distance field. [`NerveComplex`](traits::simplicial::NerveComplex)
//! assembles a finite cover from those domains
//! and records their overlap as a simplicial complex.
//!
//! That finite global description supports:
//!
//! - certified global geodesic minimisation by graph search;
//! - recovery of the fundamental group from the nerve; and
//! - a compactness certificate for the implemented manifold.
//!
//! [`GroupPresentation`](traits::simplicial::GroupPresentation) represents the resulting
//! fundamental group by generators and relations. It deliberately does not
//! implement [`Group`](traits::Group): equality of words
//! in an arbitrary finite presentation is undecidable in general.
//!
//! ## Axioms are tested
//!
//! Rust can enforce that a [`Group`](traits::Group) has the required operations, but it cannot
//! prove that composition is associative. Diffable treats every such
//! unenforceable axiom as a property-testing obligation.
//!
//! Enable the `testing` feature to use the `test_*` macros for groups, fields,
//! forms, charts, tangent bundles, quotients, and the other certified
//! structures:
//!
//! ```toml
//! [dev-dependencies]
//! diffable = { version = "0.4", features = ["testing"] }
//! ```
//!
//! The testing module includes tolerance-aware [`R32`](epsilon_metric::R32) and
//! [`R64`](epsilon_metric::R64) scalar types so that floating-point implementations
//! can be tested against the exact mathematics they approximate.
//!
//! ## Trait map
//!
//! The principal derivation chains are:
//!
//! | Implement | Derived structure |
//! | --- | --- |
//! | `Smooth<V>` | `Chart<Self, V>`, `ExpMap<Self, V>`, `TangentBundle<Self, V>` |
//! | `LieGroup<V>` | `Smooth<V>` and the complete chart chain |
//! | `Vector` | additive `Group`, `LieGroup<Self>`, and flat tangent geometry |
//! | `Quotient<G, H, V>` via `impl_lie_group_via_quotient!` | quotient `Group`, `LieGroup<V>`, and the complete chart chain |
//! | `Sesquilinear<F = F::Fixed>` | `Bilinear` |
//!
//! `Group` is connected to additive or multiplicative operator syntax with
//! `impl_group_via_add!` and `impl_group_via_mul!`. These are one-line macros
//! rather than blanket implementations because the two blanket cases would
//! overlap under Rust's coherence rules.
//!
//! ## Status
//!
//! Diffable is an experimental library and an exploration of how faithfully
//! Rust's trait system can express differential geometry. The API is still
//! evolving, and the project currently prioritises structural correctness and
//! compositional design over broad algorithm coverage or compatibility
//! stability.
//!
//! Optional features:
//!
//! - `std` *(enabled by default)* — enables standard-library integration.
//! - `simplicial` *(enabled by default; implies `std`)* — enables finite
//!   covers, nerve complexes, fundamental-group recovery, and certified global
//!   geodesic search.
//! - `testing` *(implies `std`)* — enables the property-testing macros and
//!   tolerance-aware real scalars. It does not enable `simplicial`, so the core
//!   algebraic and differential hierarchy can be tested independently.
//! - `all` — enables `simplicial` and `testing`.
//!
//! For a core-only `no_std` build, disable default features:
//!
//! ```toml
//! diffable = { version = "0.4", default-features = false }
//! ```
//!
//! Licensed under either MIT or Apache-2.0.
//!
#![allow(
    clippy::needless_range_loop,
    clippy::type_complexity,
    confusable_idents,
    uncommon_codepoints,
    mixed_script_confusables
)]
#![no_std]
#[cfg(feature = "std")]
extern crate std;

pub mod complex;
pub mod coords;
pub mod discrete;
pub mod epsilon_metric;
pub mod flat;
pub mod hypersphere;
pub mod matrix;
pub mod quaternion;
pub mod spacetime;
pub mod traits;

/// Common carrier types and mathematical certificates for ordinary use.
///
/// Importing this module brings the differential operator
/// [`d`](crate::traits::calculus::d), canonical coordinates, scalar fields,
/// linear geometry, charts, groups, and their principal constructors into
/// scope. Lower-level tensor and topology machinery remains in
/// [`traits`] and [`traits::simplicial`].
pub mod prelude {
    // The derivative.
    pub use crate::traits::calculus::d;

    // Canonical carrier types.
    pub use crate::{
        complex::Complex,
        coords::Coords,
        traits::{Dual, NonZero},
    };

    // Scalar structure.
    pub use crate::traits::{CField, Field, FieldExp, Real};

    // Ordinary linear geometry.
    pub use crate::traits::{
        Bilinear, Euclidean, Form, InnerProduct, Nondegenerate, Sesquilinear, Vector,
    };

    // Metric and manifold structure.
    pub use crate::traits::{
        Chart, ExpMap, Interval, Metric, Point, PseudoRiemannian, Smooth, TangentBundle,
    };

    // Group structure.
    pub use crate::traits::{Group, LieGroup};

    pub use crate::epsilon_metric::{R32, R64};
}
