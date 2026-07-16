#[cfg(feature = "testing")]
use std::ops::{Add, Mul};

#[cfg(feature = "testing")]
use num_traits::Zero;

use crate::{
    complex::Complex, traits::{CGroup, Field, InvolutiveField, NonZero},
};
use num_traits::{Euclid, Inv};

/// An element of the carrier set of a manifold, group, or metric space.
///
/// The space of all values of a type `P: Point` is interpreted as a bare
/// set — the underlying collection of elements on which the library's other
/// traits impose structure. `Point` itself asserts *only* set membership:
/// the ability to hold and duplicate an element. It makes no claim of
/// topology, smoothness, or dimension; those arrive (if at all) through
/// [`Chart`], [`Metric`], [`Group`], and their refinements.
///
/// The name reflects the common case — points of a manifold — but nothing
/// requires a `Point` type to be a manifold. A group with no compatible
/// manifold structure (the p-adic integers ℤ_p, say) is a perfectly good
/// `Point` type that implements [`Group`] but not [`Chart`]: an element of
/// a set carrying algebraic but not differential structure.
///
/// Equality is the only *meaningful* operation on a bare element — whether
/// two elements are the same — and it is mathematically an [`Eq`] relation.
/// The library nonetheless does not require `Eq` (or even `PartialEq`) as a
/// bound, because for the scalar types in practical use that equality is not
/// computably decidable; see [`Real`]. Equality is required only in the
/// `#[cfg(feature = "testing")]` certification layer, never for use.
///
/// [`Chart`]: crate::traits::Chart
/// [`Group`]: crate::traits::Group
/// [`Real`]: crate::traits::Real
pub trait Point: Clone {}

impl<T: Clone> Point for T {}

/// A scalar field for use as the coordinate type of a Euclidean space.
///
/// Bundles the requirements that a scalar must satisfy to be usable
/// throughout diffable — real arithmetic and debuggability.
///
/// # Note on equality
/// Mathematically the scalars model the real numbers, which have genuine
/// equality. Computationally they do not: any finite representation that
/// is also fast (`f64`, `f32`) cannot satisfy the field axioms exactly,
/// and its `PartialEq` is therefore necessarily a *tolerance relation* —
/// see [`R64`]/[`R32`], which report equality up to a relative-or-absolute
/// epsilon. Such a relation is reflexive and symmetric but **not
/// transitive**: `a == b` and `b == c` do not imply `a == c`.
///
/// The library accommodates this rather than fighting it. Two consequences
/// an implementor should know:
///
/// - **The `check_*` invariants never chain equalities.** Every property
///   test performs a single comparison between a computed value and an
///   expected one; none relies on transitivity, so a tolerance-based
///   `PartialEq` is sound to use with them. Do not add checks that compare
///   `a` to `b`, then `b` to `c`, and infer `a` to `c` — that inference is
///   invalid for the scalars this library is designed to run on.
///
/// - **Exact scalars get exact semantics for free.** A symbolic real, an
///   arbitrary-precision rational, or any type whose `PartialEq` is true
///   equality satisfies everything above trivially (a transitive relation
///   is in particular a non-chained one), and runs the same invariants with
///   genuine equality. Approximation is a property of the scalar you choose,
///   not an assumption baked into the trait hierarchy.
///
/// This is why equality is required only where it is actually exercised —
/// in the `#[cfg(feature = "testing")]` invariants, via `PartialEq` bounds
/// on those methods — and is deliberately **not** a structural bound on
/// [`Point`]. Points have mathematical equality; the library declines to
/// require a *computable* witness of it, because for the reals no faithful
/// one exists.
///
/// [`R64`]: crate::epsilon_metric::R64
/// [`R32`]: crate::epsilon_metric::R32
pub trait RealNum: num_traits::real::Real + Euclid + std::fmt::Debug {}
impl<R: num_traits::real::Real + Euclid + std::fmt::Debug> RealNum for R {}
impl<R: RealNum> Field for R where NonZero<R>: Inv<Output = NonZero<R>> {}

pub trait Real: RealNum + Field {}
impl<R: RealNum + Field> Real for R {}

/// The genuine, transitive ordering on a real-number type, independent of
/// whatever tolerance its `PartialOrd` may carry for equality testing —
/// see [`Real`]'s doc comment on why that tolerance exists and why it is
/// deliberately not transitive. An iterative numerical algorithm's
/// convergence check needs the former: comparing against a
/// tolerance-relation order can report "not less than" forever once both
/// sides fall inside the tolerance band, regardless of which is truly
/// smaller.
///
/// Built entirely from operations `Real` already guarantees — `Sub` and
/// `is_sign_negative` (via `num_traits::Float`, already required through
/// `RealNum`) — with the same formula for every implementor, no
/// per-type override. `is_sign_negative` reads the sign bit directly
/// rather than going through `PartialEq`, exactly the same reasoning
/// `Complex::real_sqrt`'s branch relies on — so it never sees `R64`/`R32`'s
/// deliberately fuzzy comparison, and the blanket below is sound for
/// every `Real` type without exception, including any brought in from
/// outside this crate.
///
/// [`Real`]: crate::traits::Real
pub trait ExactCmp: Real {
    fn exact_lt(self, other: Self) -> bool {
        let d = self - other;
        d.is_sign_negative() && !d.is_zero()
    }

    fn exact_le(self, other: Self) -> bool {
        !other.exact_lt(self)
    }
}

impl<R: Real> ExactCmp for R {}

/// A notion of distance on a manifold.
///
/// The space of all values of a type `P: Metric<R>` is interpreted as
/// a metric space — a set `M` equipped with a distance function
/// `d: M × M → R` satisfying:
/// - **Non-negativity**: `d(a, b) >= 0`
/// - **Identity of indiscernibles**: `d(a, a) = 0`
/// - **Symmetry**: `d(a, b) = d(b, a)`
/// - **Triangle inequality**: `d(a, c) <= d(a, b) + d(b, c)`
///
/// These are not enforced by the type system but are certified by
/// implementing this trait. The first three are verified empirically by
/// the `test_metric!` macro; the triangle inequality is omitted from
/// automated testing since it is numerically fragile to check near-degenerate,
/// nearly-collinear triples without a carefully tuned tolerance.
///
/// A metric is independent of any coordinate structure — it requires
/// neither a [`Chart`] nor a [`Euclidean`] tangent space, only the ability
/// to measure distance between two points directly.
///
/// [`Chart`]: crate::traits::Chart
/// [`Euclidean`]: crate::traits::Euclidean
pub trait Metric<R: Real>: Interval<R> {
    fn distance(&self, other: &Self) -> R {
        let [a, _] = self.interval(other).into();

        a
    }

    #[cfg(feature = "testing")]
    fn check_self_distance_zero(a: Self) -> bool {
        a.distance(&a) == R::zero()
    }

    #[cfg(feature = "testing")]
    fn check_non_negative(a: Self, b: Self) -> bool {
        a.distance(&b) >= R::zero()
    }

    #[cfg(feature = "testing")]
    fn check_metric_symmetry(a: Self, b: Self) -> bool {
        a.distance(&b) == b.distance(&a)
    }
}

/// A signed interval on a manifold — the pseudo-Riemannian analogue of
/// [`Metric`]. Where `Metric` returns a non-negative distance, `Interval`
/// returns the *signed* squared interval s²(a,b): negative timelike,
/// zero null, positive spacelike (or your sign convention). No metric-space
/// axioms are claimed — this is not a distance, it is the value of the
/// line element between two points along the connecting geodesic.
pub trait Interval<R: Real>: Point {
    /// Interval between self and other. Real or imaginary
    /// carries causal character.
    fn interval(&self, other: &Self) -> Complex<R>;

    #[cfg(feature = "testing")]
    fn check_interval_symmetry(a: Self, b: Self) -> bool {
        a.interval(&b) == b.interval(&a)
    }

    #[cfg(feature = "testing")]
    fn check_self_interval_zero(a: Self) -> bool {
        a.interval(&a) == Complex::zero()
    }
}

/// A symmetric bilinear form on a vector space.
///
/// The space of all values of a type `P: Bilinear<R>` is interpreted as a
/// vector space equipped with a symmetric bilinear pairing
/// `⟨·,·⟩: P × P → R`. **No definiteness is assumed**: the induced quadratic
/// form `Q(v) = ⟨v,v⟩` may be positive, negative, or zero for `v ≠ 0`. This is
/// the structure of a pseudo-Euclidean (e.g. Minkowski) space as well as a
/// Euclidean one.
///
/// Because the form may be indefinite, `Bilinear` provides **no norm and no
/// distance**: `⟨v,v⟩` can be negative, so `sqrt(⟨v,v⟩)` need not be real, and
/// the induced "distance" fails the metric-space axioms (null vectors give
/// distinct points at separation zero; the triangle inequality reverses on
/// timelike triples). A norm and a [`Metric`] arise only once definiteness is
/// added — see [`InnerProduct`], which refines this trait with
/// positive-definiteness and is therefore the only branch that induces a
/// metric space.
///
/// `norm_squared` is provided as `⟨v,v⟩` and is **signed** — it is the value
/// of the quadratic form, not the square of a norm. Callers on indefinite
/// spaces should inspect its sign (causal character) rather than take its
/// square root.
///
/// The three certified invariants — symmetry, additivity, and scalar
/// linearity of the pairing — are signature-agnostic and hold in the
/// indefinite case exactly as in the definite one.
pub trait Bilinear<F: Field>: Point {
    fn dot(&self, other: &Self) -> F;

    fn self_dot(&self) -> F {
        self.dot(self)
    }

    #[cfg(feature = "testing")]
    fn check_symmetry(a: Self, b: Self) -> bool {
        a.dot(&b) == b.dot(&a)
    }

    #[cfg(feature = "testing")]
    fn check_additivity(a: Self, b: Self, c: Self) -> bool
    where
        Self: Add<Output = Self> + Clone,
    {
        let lhs = (a.clone() + b.clone()).dot(&c);
        let rhs = a.dot(&c) + b.dot(&c);
        lhs == rhs
    }

    #[cfg(feature = "testing")]
    fn check_scalar_linearity(a: Self, c: Self, k: F) -> bool
    where
        Self: Mul<F, Output = Self> + Clone,
    {
        let lhs = (a.clone() * k).dot(&c);
        let rhs = k * a.dot(&c);
        lhs == rhs
    }
}

/// A Hermitian (sesquilinear) form on a vector space.
///
/// The space of all values of a type `P: Sesquilinear<F>` is interpreted as a
/// vector space equipped with a Hermitian pairing
/// `⟨·,·⟩: P × P → F`, where `F` is an [`InvolutiveField`]. The pairing is
/// linear in its first argument and conjugate-linear in its second, satisfying
/// `⟨v,w⟩ = conj(⟨w,v⟩)`.
///
/// Unlike [`Bilinear`], the codomain may be a field with a nontrivial
/// involution, such as the complex numbers. Hermitian forms are the natural
/// analogue of symmetric bilinear forms over such fields.
///
/// No definiteness is assumed. The induced quadratic form
/// `Q(v) = ⟨v,v⟩` is always fixed by the involution (for example, real-valued
/// over `ℂ`), but it may still be positive, negative, or zero for `v ≠ 0`.
/// Consequently, this trait provides no norm or metric. A norm and the
/// associated [`Metric`] arise only once positive-definiteness is imposed
/// (see [`InnerProduct`] or the corresponding positive-definite Hermitian
/// refinement, if provided).
///
/// `self_dot` returns the value `⟨v,v⟩` in the fixed field `F::Fixed`. This is
/// the value of the quadratic form, not the square of a norm, and should not
/// be square-rooted unless positive-definiteness is known.
///
/// The certified invariants are Hermitian symmetry, additivity, and scalar
/// linearity in the first argument. Conjugate-linearity in the second argument
/// follows from these together with Hermitian symmetry.
pub trait Sesquilinear<F: InvolutiveField>: CGroup {
    fn hermitian(&self, other: &Self) -> F;
    fn norm_squared(&self) -> F::Fixed {
        self.hermitian(self).to_fixed()
    }

    // ⟨v,w⟩ = conj(⟨w,v⟩) — Hermitian symmetry, the sesquilinear analogue
    // of Bilinear::check_symmetry. Additivity and conjugate-linearity in
    // the second argument both follow from this plus linearity in the
    // first, and aren't separately checked for the same reason Bilinear
    // doesn't separately check them.
    #[cfg(feature = "testing")]
    fn check_hermitian_symmetry(a: Self, b: Self) -> bool {
        a.hermitian(&b) == b.hermitian(&a).conj()
    }

    #[cfg(feature = "testing")]
    fn check_additivity(a: Self, b: Self, c: Self) -> bool
    where
        Self: Add<Output = Self> + Clone,
    {
        (a.clone() + b.clone()).hermitian(&c) == a.hermitian(&c) + b.hermitian(&c)
    }

    #[cfg(feature = "testing")]
    fn check_scalar_linearity(a: Self, c: Self, k: F) -> bool
    where
        Self: Mul<F, Output = Self> + Clone,
    {
        (a.clone() * k).hermitian(&c) == k * a.hermitian(&c)
    }
}

/// An inner product structure on a vector space.
///
/// Refines [`Bilinear`] with **positive-definiteness**: `⟨v,v⟩ > 0` for all
/// `v ≠ 0`. This is exactly the property that makes the induced quantities
/// well-behaved — `norm(v) = sqrt(⟨v,v⟩)` is real and non-negative, and
/// `d(a,b) = ‖a - b‖` satisfies the metric-space axioms — which is why
/// `InnerProduct` is a refinement of [`Metric`], whereas the bare
/// [`Bilinear`] base is not.
///
/// Not every [`Metric`] is an `InnerProduct` — the sphere's geodesic distance
/// is a metric not arising from any inner product, since the sphere is not a
/// vector space. And not every [`Bilinear`] form is an `InnerProduct` — a
/// Minkowski scalar product is bilinear and symmetric but indefinite, so it
/// induces no metric at all.
pub trait InnerProduct<R: Real, F: InvolutiveField<Fixed = R> = R>: Sesquilinear<F> + Metric<R> {
    /// The norm `‖v‖ = sqrt(⟨v,v⟩)`. Well-defined and real because the form
    /// is positive-definite. On an indefinite [`Bilinear`] space this would
    /// not be real — which is why it lives here, not on the base.
    fn norm(&self) -> R {
        self.norm_squared().sqrt()
    }

    #[cfg(feature = "testing")]
    fn check_positive_definite(a: Self) -> bool
    where
        Self: Zero + PartialEq,
    {
        a == Self::zero() || a.norm() > R::zero()
    }

    #[cfg(feature="testing")]
    fn check_metric_compatibility(a: Self, b: Self) -> bool {
        a.sub(&b).norm_squared().sqrt() == a.distance(&b)
    }
}

impl<R: Real, F: InvolutiveField<Fixed = R>, P: Sesquilinear<F> + Metric<R>> InnerProduct<R, F> for P {}
