#[cfg(feature = "testing")]
use std::ops::{Add, Mul};

#[cfg(feature = "testing")]
use num_traits::Zero;
use num_traits::real::Real;

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
/// computably decidable; see [`Scalar`]. Equality is required only in the
/// `#[cfg(feature = "testing")]` certification layer, never for use.
///
/// [`Chart`]: crate::traits::Chart
/// [`Group`]: crate::traits::Group
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
pub trait Scalar: Real + std::fmt::Debug {}

impl<R: Real + std::fmt::Debug> Scalar for R {}

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
pub trait Metric<R: Real>: Point {
    fn distance(&self, other: &Self) -> R;

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

/// An inner product structure on a manifold.
///
/// The space of all values of a type `P: InnerProduct<R>` is interpreted
/// as an inner product space — a vector space equipped with a bilinear,
/// symmetric, positive-definite pairing `⟨·,·⟩: P × P → R`.
///
/// An inner product induces a norm, `|v| = sqrt(⟨v,v⟩)`, and that norm in
/// turn induces a metric, `d(a, b) = |a - b|` — which is why `InnerProduct`
/// is a refinement of [`Metric`] rather than an independent trait. `norm`
/// and `norm_squared` are provided as default methods derived purely from
/// `dot`.
///
/// Not every `Metric` is an `InnerProduct` — the sphere's geodesic distance,
/// for instance, is a perfectly good metric that does not arise from any
/// inner product, since the sphere is not a vector space.
pub trait InnerProduct<R: Real>: Metric<R> {
    fn dot(&self, other: &Self) -> R;

    fn norm(&self) -> R {
        self.norm_squared().sqrt()
    }

    fn norm_squared(&self) -> R {
        self.dot(self)
    }

    #[cfg(feature = "testing")]
    fn check_inner_product_symmetry(a: Self, b: Self) -> bool {
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
    fn check_scalar_linearity(a: Self, c: Self, k: R) -> bool
    where
        Self: Mul<R, Output = Self> + Clone,
    {
        let lhs = (a.clone() * k).dot(&c);
        let rhs = k * a.dot(&c);
        lhs == rhs
    }

    #[cfg(feature = "testing")]
    fn check_positive_definite(a: Self) -> bool
    where
        Self: Zero + PartialEq,
    {
        a == Self::zero() || a.norm() > R::zero()
    }
}
