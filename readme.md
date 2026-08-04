# diffable

Diffable is a differential-geometry framework for Rust. Its
central idea is that mathematical structure should be executable:

- a type supplies the underlying values;
- a trait implementation certifies what mathematical structure those values
  carry;
- blanket implementations encode theorems relating those structures; and
- property tests check the axioms Rust's type system cannot prove.

Generic code can therefore ask for what its argument *is*, rather than for an
incidental collection of methods. A function that needs a nondegenerate form
can say so without requiring an inner product. A function that works in
spacetime can request a signed interval without pretending it has a metric.
And when one structure mathematically entails another, the consequence is
implemented once for every type.

> **Traits are mathematical certificates. Blanket implementations are
> theorems.**

### Geometry from one local implementation

A Lie group is homogeneous: its geometry at the identity can be transported
to every other point by left translation. Diffable writes that argument as a
blanket implementation:

```compile_fail,E0210
use diffable::prelude::*;

impl<V: Vector, L: LieGroup<V>> Smooth<V> for L {
    fn exp(&self, coord: V) -> Self {
        self.compose(&Self::identity_exp(coord))
    }

    fn log(&self, point: &Self) -> Option<V> {
        Self::identity_log(&self.inverse().compose(point))
    }
}
```

An implementor of [`LieGroup<V>`](traits::LieGroup) provides the group operation and the
exponential and logarithmic maps at the identity. Left translation then
supplies [`Smooth<V>`](traits::Smooth), from which the full chart bundle follows:

```
LieGroup<V>
    ⇒ Smooth<V>
    ⇒ Chart<Self, V>
    ⇒ ExpMap<Self, V>
    ⇒ TangentBundle<Self, V>
```

This is the pattern throughout Diffable: implement the smallest structure
that characterises an object, then inherit its mathematical consequences.

### Handedness, duality, and geometry

Diffable permits noncommutative scalar fields, so every
[`Vector`](traits::Vector) explicitly elects whether its field acts on the
left or on the right. Concrete coordinate spaces conventionally elect
[`Right`](traits::Right); [`Dual<V>`](traits::Dual) elects the opposite hand:

```
V right-handed  ⇒ V* left-handed  ⇒ V** right-handed
V left-handed   ⇒ V* right-handed ⇒ V** left-handed
```

The ordinary `Mul<F>` operation always follows the elected hand. Thus `v * k`
means `vk` on a right module and `kv` on a left module; no separate dual
scalar API is needed.

Canonical evaluation follows the same choice. For coordinates `vᵢ` and
`ωᵢ`,

```
right-handed V:  ω(v) = Σ ωᵢvᵢ
left-handed  V:  ω(v) = Σ vᵢωᵢ
```

This order is invisible over the reals or complexes but observable over the
quaternions. [`Tensor::pairing`](traits::Tensor::pairing) selects it from
[`Tensor::Hand`](traits::Tensor::Hand), while `Dual<Dual<V>>` restores the
hand of `V`.

The [`Dual<V>`](traits::Dual) wrapper is coordinate-identical to `V`, but raw
coordinates do not carry a geometric identification between the two spaces.
[`Dual::from_raw`](traits::Dual::from_raw) merely declares covector
coordinates. Finite dimensionality supplies only the evaluation isomorphism

```
V** ≅ V
```

implemented by [`Tensor::collapse`](traits::Tensor::collapse). Geometry enters when [`Form`](traits::Form)
chooses a lowering map

```
♭ : V → V*
```

and defines `dot(a, b)` by evaluating `b♭` on `a`. A degenerate form may
collapse distinct vectors to the same covector; [`Nondegenerate`](traits::Nondegenerate)
certifies that `♭` is invertible and supplies the raising map

```
♯ : V* → V.
```

These are the musical isomorphisms. They are not coordinate
reinterpretations: they encode the space's chosen geometric relationship
with its dual. The dual space inherits the corresponding form through those
maps.

### Invariants are representation choices

[`Sl<V, N>`](spacetime::Sl) represents the special linear group. Its matrix is private, and
there is no constructor from an arbitrary matrix. Values can be reached
through operations that preserve determinant one: identity, composition,
inverse, and exponentiation from the traceless Lie algebra.

Likewise, [`SlAlgebra<F, N, D>`](spacetime::SlAlgebra) stores coordinates in a basis whose elements
are traceless. A non-traceless matrix is not an invalid value to be detected
later; it is not a value the representation can express.

Consequently, exponentiation has the meaningful type

```
exp : sl(N) → SL(N)
```

rather than returning an arbitrary matrix accompanied by a runtime claim that
it probably belongs to the group. Membership is a theorem about reachability.

The same principle appears at smaller scales. [`NonZero<T>`](traits::NonZero) certifies that a
value lies in the multiplicative group, [`Dual<V>`](traits::Dual) distinguishes covectors
from vectors even when their coordinates coincide, and matrix variance is
encoded so that only variance-correct contractions typecheck.

### Constructions propagate structure

Diffable's concrete spaces are deliberately built from reusable mathematical
constructions:

```
S³ / {±1}       ⇒ SO(3)
SL(2, ℂ) / {±1} ⇒ SO⁺(1, 3)
ℝ / ℤ           ⇒ S¹
S¹ × S¹         ⇒ T²
```

The quotient machinery does not know about rotations or relativity. It knows
that a suitable quotient of a Lie group inherits Lie-group structure. The
same implementation therefore gives both `SO(3)` and the restricted Lorentz
group their group operations, exponential maps, charts, and tangent bundles.

The torus and Klein bottle make the distinction equally clear. Both are made
by gluing two circles; straight gluing produces a Lie group, while twisted
gluing produces a smooth non-orientable manifold without falsely granting it
group structure.

The type hierarchy records these differences instead of flattening every
space into coordinates and asking the programmer to remember what remains
valid.

### Automatic differentiation as typed programs

[`d`](traits::calculus::d) turns an ordinary generic Rust function into a composable
differential program. Calling `d` does not evaluate anything: derivatives can
be nested, contracted with directions, and only then evaluated with `at`.

```rust
use diffable::{
    coords::Coords,
    traits::{calculus::d, Euclidean, Field, Tensor},
};

fn cube<V: Euclidean>(x: V) -> V {
    x.map(|x| x.powi(3))
}

// Full higher derivatives use the same operator recursively.
let third = d(d(d(cube))).at(Coords::from(-6.0));
assert_eq!(third[0], 6.0);

// Contract a derivative slot before evaluation.
let directional = d(cube)
    .along(Coords::from(4.0))
    .at(Coords::from(7.0));
assert_eq!(directional[0], 588.0);

// Differential programs remain differentiable, including their directions.
let diagonal = d(|v| d(cube).along(v).at(v))
    .at(Coords::from(7.0));
assert_eq!(diagonal[0], 441.0);
```

The last expression differentiates `v ↦ D(cube)ᵥ(v)`. This is deliberately
ordinary Rust syntax: there is no tracing macro, tape, boxed closure, or
type-erased expression graph. [`Jet`](traits::calculus::Jet) supplies Taylor
coefficients, [`JetVector`](traits::calculus::JetVector) presents an existing tensor over
those coefficients, and [`JetMap`](traits::calculus::JetMap) interprets the program at
each required nesting depth. A type-level [`ConstantRoute`](traits::calculus::ConstantRoute)
injects captured base-field values through that jet tower.

The full derivative of `f: U → V` is a
[`TangentMap`](traits::calculus::TangentMap), represented as `V ⊗ U*` in
output-by-input coordinate order. `along` contracts one input slot and returns
the directional derivative directly. If a composition cannot be evaluated,
the public `at` boundary reports that the function lacks the required jet
presentation, its tensor structure is incompatible, or a musical isomorphism
does not lift through the nested jets.

This machinery is not restricted to flat coordinates. Implementing
[`TangentLift`](traits::calculus::TangentLift) tells Diffable how a manifold's tangent
charts act on jets. [`FormLift`](traits::calculus::FormLift) and
[`NondegenerateLift`](traits::calculus::NondegenerateLift) do the corresponding job for
the lowering and raising maps, allowing generic Euclidean code to remain valid
inside higher derivatives.

### Trait hierarchy

The trait graph is intentionally fine-grained. Generic algorithms should
state the weakest honest assumptions their proofs require.

- [`Field`](traits::Field) permits noncommutative division rings;
  [`CField`](traits::CField) adds
  commutativity.
- [`Form`](traits::Form) provides the lowering map `♭: V → V*`;
  [`Nondegenerate`](traits::Nondegenerate) adds its inverse `♯`.
- [`Sesquilinear`](traits::Sesquilinear) certifies a Hermitian form;
  [`Bilinear`](traits::Bilinear) is the fixed-field specialisation;
  [`InnerProduct`](traits::InnerProduct) adds positive definiteness.
- [`Interval`](traits::Interval) provides a signed squared separation and
  accommodates pseudo-Riemannian geometry; [`Metric`](traits::Metric) adds
  genuine metric-space distance.
- [`Chart`](traits::Chart) provides coordinates; [`ExpMap`](traits::ExpMap)
  says those coordinates are geodesic; [`TangentBundle`](traits::TangentBundle)
  supplies such a chart at every point.
- [`TangentLift`](traits::calculus::TangentLift) extends tangent charts through jet
  coordinates; [`FormLift`](traits::calculus::FormLift) and
  [`NondegenerateLift`](traits::calculus::NondegenerateLift) extend the musical maps.

Degenerate and indefinite cases are not malformed approximations to
Euclidean geometry. They are first-class structures with precisely the
operations their axioms justify.

### Implementations

#### Scalars, vectors, and tensors

- [`coords::Coords`] is the canonical fixed-dimensional coordinate space
  `R^(N−M, M)`, parameterised by the number `M` of timelike directions.
  `M = 0` is Euclidean; `Coords<R, 4, 1>` is Minkowski spacetime.
- [`complex::Complex`] implements the complex numbers with conjugation as
  their elected involution. [`traits::Symmetrized`] elects the bilinear rather
  than Hermitian form.
- [`quaternion::Quaternion`] provides the quaternion division algebra.
- [`matrix::Matrix`] represents an endomorphism of `V`: as `V ⊗ V*` when
  `V` is right-handed and `V* ⊗ V` when it is left-handed. Tensor variance
  and handedness are carried by the types.
  [`matrix::MatrixExponential`] supplies matrix `exp` and `log`.
- [`traits::calculus::Jet`] and [`traits::calculus::JetVector`] implement forward automatic
  differentiation without changing the logical tensor shape. [`traits::calculus::d`]
  and [`traits::calculus::Along`] compose full and directional derivatives into typed
  programs.

#### Manifolds and Lie groups

- [`hypersphere::Sphere`] provides `Sⁿ` with its intrinsic geodesic structure.
- [`hypersphere::S0`], [`hypersphere::UnitComplex`], and [`hypersphere::S3`]
  add the Lie-group structures on
  the three group spheres: signs, unit complex numbers, and unit quaternions.
- [`hypersphere::So3`] constructs `SO(3)` as `S³/{±1}`.
- [`hypersphere::Stereographic`] provides an external stereographic atlas,
  independently of the sphere's intrinsic exponential charts.
- [`flat::S1`] constructs the circle as `R/Z`; [`flat::Torus`] and
  [`flat::KleinBottle`] provide straight and twisted gluings of two circles.
- [`spacetime::Sl`] and [`spacetime::SlAlgebra`] implement the special linear
  group and its traceless Lie algebra; [`spacetime::Lorentz`] constructs `SO⁺(1,3)` as
  `SL(2,C)/{±1}`.
- [`discrete::N`] and [`discrete::Z`] implement the naturals and their
  Grothendieck group completion; [`discrete::Z`] also supplies the lattice
  used by [`flat::S1`].

The newtypes add mathematical meaning one layer at a time. `Sphere` is a
manifold, `S3` equips that manifold with quaternion multiplication, and
`So3` adds the antipodal quotient. Forgetting a wrapper drops structure
without changing the underlying object.

### Global geometry and topology

[`Bounded`](traits::simplicial::Bounded) describes a bounded open exponential-chart
domain by a signed distance field. [`NerveComplex`](traits::simplicial::NerveComplex)
assembles a finite cover from those domains
and records their overlap as a simplicial complex.

That finite global description supports:

- certified global geodesic minimisation by graph search;
- recovery of the fundamental group from the nerve; and
- a compactness certificate for the implemented manifold.

[`GroupPresentation`](traits::simplicial::GroupPresentation) represents the resulting
fundamental group by generators and relations. It deliberately does not
implement [`Group`](traits::Group): equality of words
in an arbitrary finite presentation is undecidable in general.

### Axioms are tested

Rust can enforce that a [`Group`](traits::Group) has the required operations, but it cannot
prove that composition is associative. Diffable treats every such
unenforceable axiom as a property-testing obligation.

Enable the `testing` feature to use the `test_*` macros for groups, fields,
forms, charts, tangent bundles, quotients, and the other certified
structures:

```toml
[dev-dependencies]
diffable = { version = "0.4", features = ["testing"] }
```

The testing module includes tolerance-aware [`R32`](epsilon_metric::R32) and
[`R64`](epsilon_metric::R64) scalar types so that floating-point implementations
can be tested against the exact mathematics they approximate.

### Trait map

The principal derivation chains are:

| Implement | Derived structure |
| --- | --- |
| `Smooth<V>` | `Chart<Self, V>`, `ExpMap<Self, V>`, `TangentBundle<Self, V>` |
| `LieGroup<V>` | `Smooth<V>` and the complete chart chain |
| `Vector` | additive `Group`, `LieGroup<Self>`, and flat tangent geometry |
| `Quotient<G, H, V>` via `impl_lie_group_via_quotient!` | quotient `Group`, `LieGroup<V>`, and the complete chart chain |
| `Sesquilinear<F = F::Fixed>` | `Bilinear` |

`Group` is connected to additive or multiplicative operator syntax with
`impl_group_via_add!` and `impl_group_via_mul!`. These are one-line macros
rather than blanket implementations because the two blanket cases would
overlap under Rust's coherence rules.

### Status

Diffable is an experimental library and an exploration of how faithfully
Rust's trait system can express differential geometry. The API is still
evolving, and the project currently prioritises structural correctness and
compositional design over broad algorithm coverage or compatibility
stability.

Optional features:

- `std` *(enabled by default)* — enables standard-library integration.
- `simplicial` *(enabled by default; implies `std`)* — enables finite
  covers, nerve complexes, fundamental-group recovery, and certified global
  geodesic search.
- `testing` *(implies `std`)* — enables the property-testing macros and
  tolerance-aware real scalars. It does not enable `simplicial`, so the core
  algebraic and differential hierarchy can be tested independently.
- `all` — enables `simplicial` and `testing`.

For a core-only `no_std` build, disable default features:

```toml
diffable = { version = "0.4", default-features = false }
```

Licensed under either MIT or Apache-2.0.


License: MIT OR Apache-2.0
