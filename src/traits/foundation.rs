#[cfg(feature = "testing")]
use num_traits::Zero;

use crate::{
    complex::Complex,
    traits::{Field, FieldExp, NatZero, NonZero},
};
use num_traits::{Euclid, Inv, real::Real as _};

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
impl<R: RealNum> Field for R
where
    NonZero<R>: Inv<Output = NonZero<R>>,
{
    type Fixed = Self;
    type Characteristic = NatZero;

    fn to_fixed(self) -> Self {
        self
    }

    fn from_fixed(x: Self) -> Self {
        x
    }

    fn conj(&self) -> Self {
        *self
    }
}

/// A real-number field: totally ordered, and its own involution fixed field
/// (`Fixed = Self`, so `conj = id`).
///
/// Implementors (`R64`, `R32`) carry a **tolerance** in their `PartialEq`/
/// `PartialOrd`: two values within a relative epsilon compare equal, so that
/// floating-point round-off doesn't fracture geometric equality. That tolerance
/// is deliberately **not transitive** (`a ≈ b` and `b ≈ c` does not give
/// `a ≈ c`), which is fine for equality testing but wrong for the strict,
/// transitive order an iterative algorithm needs to decide convergence — hence
/// [`ExactCmp`], which recovers the genuine order from the sign bit instead of
/// the tolerant comparison.
pub trait Real: RealNum + Field<Fixed = Self> {}
impl<R: RealNum + Field<Fixed = Self>> Real for R {}

impl<R: Real<Characteristic = NatZero> + Metric> FieldExp for R {
    fn exp(&self) -> Self {
        <Self as num_traits::real::Real>::exp(*self)
    }
}

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
pub trait Metric: Interval {
    fn distance(&self, other: &Self) -> Self::R {
        self.interval_squared(other).sqrt()
    }

    #[cfg(feature = "testing")]
    fn check_non_negative(a: Self, b: Self) -> bool {
        a.distance(&b) >= Self::R::zero()
    }

    #[cfg(feature = "testing")]
    fn check_distance_agrees_with_interval(a: Self, b: Self) -> bool {
        a.distance(&b) == a.interval_squared(&b).sqrt()
    }
}

pub trait FromReal: Interval {
    fn from_real(r: Self::R) -> Self;
}

impl<F: Field<Fixed: Real>> FromReal for F {
    fn from_real(r: Self::R) -> Self {
        Self::from_fixed(r)
    }
}

/// A signed interval on a manifold — the pseudo-Riemannian analogue of
/// [`Metric`]. Where `Metric` returns a non-negative distance, `Interval`
/// returns the *signed* squared interval s²(a,b): negative timelike,
/// zero null, positive spacelike (or your sign convention). No metric-space
/// axioms are claimed — this is not a distance, it is the value of the
/// line element between two points along the connecting geodesic.
pub trait Interval: Point {
    /// The ordered field the interval is valued in — the real field where
    /// magnitudes, distances, and convergence live. Distinct from a scalar
    /// field's involution `Fixed`: analysis happens here regardless of the
    /// algebraic involution.
    type R: Real;

    /// Interval between self and other. Real or imaginary
    /// carries causal character.
    fn interval(&self, other: &Self) -> Complex<Self::R> {
        Complex::real_sqrt(self.interval_squared(other))
    }

    /// The **signed** squared interval `s²(a, b)` — the primitive from which
    /// [`interval`](Interval::interval) (its signed square root) and
    /// [`distance`](Metric::distance) both derive.
    ///
    /// Signed carries causal character: negative timelike, zero null, positive
    /// spacelike (or your sign convention). It is the value of the line element,
    /// not a metric-space distance — no non-negativity or triangle inequality is
    /// claimed here. [`Metric`] is the refinement that additionally promises it
    /// is definite.
    fn interval_squared(&self, other: &Self) -> Self::R;

    #[cfg(feature = "testing")]
    fn check_interval_symmetry(a: Self, b: Self) -> bool {
        a.interval(&b) == b.interval(&a)
    }

    #[cfg(feature = "testing")]
    fn check_self_interval_zero(a: Self) -> bool {
        a.interval(&a) == Complex::zero()
    }

    #[cfg(feature = "testing")]
    fn check_interval_squared_agrees_with_interval(a: &Self, b: &Self) -> bool {
        Complex::real_sqrt(a.interval_squared(b)) == a.interval(b)
    }
}
