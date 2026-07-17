use num_traits::{Zero, real::Real as _};
use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

use super::{Chart, Field, LieGroup, Real};
use crate::{impl_group_via_add, traits::Metric};

/// A finite-dimensional Euclidean space.
///
/// The space of all values of a type `E: Euclidean` is interpreted as
/// `R^N` (with `R := E::F` and `N := E::N`) — the canonical flat, *positive-
/// definite* space of dimension `N` over the field `R`. This is the space in
/// which local coordinate charts take their values, and in which tangent
/// vectors live.
///
/// `Euclidean` is the **definite real-valued refinement** of [`Quadratic`]:
/// it is a pseudo-Euclidean space (signature `(N, 0)`) that additionally
/// carries an [`InnerProduct`] — a positive-definite pairing inducing a genuine
/// `norm` and a [`Metric`]. Where the pseudo-Euclidean base has only a signed
/// [`Bilinear`] scalar product, a Euclidean space has all the metric-space
/// structure on top, because definiteness is exactly what makes
/// `sqrt(⟨v,v⟩)` real and the induced distance a metric.
///
/// Beyond the algebraic structure of a vector space (`Add`, `Sub`, `Mul`,
/// `Neg`, `Zero`), it carries that inner product and a canonical tangent
/// bundle ([`TangentBundle`]) whose charts are globally defined with infinite
/// injectivity radius — reflecting the flatness of the space.
///
/// # Flatness
/// Unlike a general Riemannian manifold, a Euclidean space is flat: geodesics
/// are straight lines, parallel transport is path-independent, and the
/// exponential map is a global isomorphism rather than merely a local one.
/// These properties are verified by the `check_*` methods inherited from
/// [`TangentBundle`] and [`Quadratic`] (`check_global_chart`,
/// `check_global_geodesic_scaling`, `check_translation_invariance`), together
/// with the definite-only `check_pythagorean` below.
///
/// # Implementing
/// Use the `test_euclidean!` macro to verify that your implementation
/// satisfies the Euclidean axioms. (For an indefinite space, implement only
/// [`Quadratic`] and use `test_pseudo_euclidean!` instead.)
///
/// [`Bilinear`]: crate::traits::Bilinear
/// [`InnerProduct`]: crate::traits::InnerProduct
/// [`Metric`]: crate::traits::Metric
/// [`TangentBundle`]: crate::traits::TangentBundle
pub trait Euclidean: Quadratic + InnerProduct {
    // Pythagorean theorem: d(a, b)² == |a - b|²
    #[cfg(feature = "testing")]
    fn check_pythagorean(a: &Self, b: &Self) -> bool
    where
        Self: Sub<Output = Self> + Clone,
    {
        let dist_sq = a.distance(b);
        let dist_sq = dist_sq * dist_sq;
        let diff = a.clone() - b.clone();
        let norm_sq = diff.norm_squared();
        dist_sq == norm_sq
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Dual<V: Vector>(V);

impl<V: Vector> Dual<V> {
    /// This is a naive constructor! Do not use this
    /// for geometric computation. It exists only to help
    /// with the implementation of `Form` on types.
    pub fn from_raw(v: V) -> Self {
        Self(v)
    }

    /// This is a naive projection! Do not use this
    /// for geometric computation. It exists only to help
    /// with the implementation of `Form` on types.
    pub fn to_raw(v: Self) -> V {
        v.0
    }
}

impl<V: Vector> Vector for Dual<V> {
    type F = V::F;

    const N: usize = V::N;

    type Iter<'a>
        = V::Iter<'a>
    where
        Self: 'a;

    fn iter(&self) -> Self::Iter<'_> {
        self.0.iter()
    }

    fn from_fn(f: impl Fn(usize) -> Self::F) -> Self {
        Self(V::from_fn(f))
    }
}

impl<V: Vector> Zero for Dual<V> {
    fn zero() -> Self {
        Self(V::zero())
    }

    fn is_zero(&self) -> bool {
        V::is_zero(&self.0)
    }
}

impl<V: Vector> Add<Self> for Dual<V> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl<V: Vector> Neg for Dual<V> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl<V: Vector> Sub<Self> for Dual<V> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl<V: Vector> Mul<V::F> for Dual<V> {
    type Output = Self;

    fn mul(self, rhs: V::F) -> Self::Output {
        Self(self.0 * rhs)
    }
}

impl<V: Vector> Index<usize> for Dual<V> {
    type Output = V::F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<V: Vector> IndexMut<usize> for Dual<V> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

pub trait Vector:
    LieGroup<Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Self::F, Output = Self>
    + Neg<Output = Self>
    + Zero
    + Index<usize, Output = Self::F>
    + IndexMut<usize>
    + Copy
    + std::fmt::Debug
{
    type F: Field;
    const N: usize;

    type Iter<'a>: Iterator<Item = &'a Self::F>
    where
        Self: 'a;
    fn iter(&self) -> Self::Iter<'_>;
    fn from_fn(f: impl Fn(usize) -> Self::F) -> Self;

    /// The canonical pairing (V, V*) -> F. Must remain the normal dot product.
    fn pairing(&self, rhs: &Dual<Self>) -> Self::F {
        self.iter()
            .zip(rhs.iter())
            .fold(Self::F::zero(), |acc, (&a, &b)| acc + a * b)
    }

    fn collapse(v: Dual<Dual<Self>>) -> Self {
        v.0.0
    }

    fn from_array<const N: usize>(arr: [Self::F; N]) -> Self {
        const { assert!(Self::N == N) }
        Self::from_fn(|i| arr[i])
    }

    fn to_array<const N: usize>(self) -> [Self::F; N] {
        const { assert!(Self::N == N) }
        std::array::from_fn(|i| self[i])
    }

    // Flat space has no singularities — to_local is always Some
    #[cfg(feature = "testing")]
    fn check_global_chart(p: &Self, q: &Self) -> bool {
        let chart = Self::chart_at(p);
        chart.to_local(q).is_some()
    }
}

// Form says: "this space has a lowering map."
// Nondegenerate says: "that lowering map is invertible."
// Sesquilinear says: "the lowering map interacts with scalar multiplication according to an involution."
// Bilinear says: "that involution is trivial."
// InnerProduct says: "the "
pub trait Form: Vector {
    fn flat(&self) -> Dual<Self>;

    fn dot(&self, b: &Self) -> Self::F {
        self.pairing(&b.flat())
    }

    fn self_dot(&self) -> Self::F {
        self.dot(self)
    }

    #[cfg(feature = "testing")]
    fn check_dot_agrees_with_pairing(a: &Self, b: &Self) -> bool {
        a.pairing(&b.flat()) == a.dot(b)
    }

    // Translation invariance: Q((a+c) - (b+c)) == Q(a - b),
    // where Q(v) = ⟨v,v⟩ is the form.
    //
    // Stated on norm_squared rather than a distance, since a pseudo-Euclidean
    // space has no metric: the difference is the same vector either way
    // ((a+c) - (b+c) = a - b), so the form agrees exactly.
    #[cfg(feature = "testing")]
    fn check_translation_invariance(a: &Self, b: &Self, c: &Self) -> bool
    where
        Self: Add<Output = Self> + Sub<Output = Self> + Clone,
    {
        let diff = a.clone() - b.clone();
        let diff_translated = (a.clone() + c.clone()) - (b.clone() + c.clone());
        diff.self_dot() == diff_translated.self_dot()
    }

    // Geodesic scaling holds globally (infinite injectivity radius):
    // to_global(v * t) is parallel to to_global(v) AND scaled by t exactly
    #[cfg(feature = "testing")]
    fn check_global_geodesic_scaling(p: &Self, v: Self, t: Self::F) -> bool
    where
        Self: PartialEq,
    {
        let chart = Self::chart_at(p);
        match (
            chart.to_local(&chart.to_global(v * t)),
            chart.to_local(&chart.to_global(v)),
        ) {
            (Some(tv_local), Some(v_local)) => tv_local == v_local * t,
            _ => false,
        }
    }
}

pub trait Nondegenerate: Form {
    fn sharp(v: Dual<Self>) -> Self;

    // check flat/sharp inverse functions
    #[cfg(feature = "testing")]
    fn check_isomorphism(a: &Self) -> bool
    where
        Self: PartialEq<Self>,
    {
        let flat = a.flat();

        Self::sharp(flat) == *a && Dual::<Self>::sharp(flat.flat()) == flat
    }
}

impl<V: Nondegenerate> Form for Dual<V> {
    fn flat(&self) -> Dual<Self> {
        Dual(Dual(V::sharp(*self)))
    }
}

impl<V: Nondegenerate> Nondegenerate for Dual<V> {
    fn sharp(v: Dual<Self>) -> Self {
        v.0.0.flat()
    }
}

/// A finite-dimensional pseudo-Euclidean space.
///
/// The space of all values of a type `V: Quadratic` is interpreted as
/// flat coordinate space `F^N` (`N := V::N`, `F := V::F`) equipped with a
/// symmetric [`Bilinear`] scalar product of *arbitrary signature*. The form
/// may be indefinite: a vector's quadratic form `⟨v,v⟩` (its `norm_squared`)
/// can be positive, negative, or zero. Minkowski spacetime is the archetype.
///
/// This is the **indefinite base**; [`Euclidean`] is its positive-definite
/// refinement. Because the form is only [`Bilinear`], a pseudo-Euclidean
/// space has **no norm and no [`Metric`]** — `sqrt(⟨v,v⟩)` need not be real,
/// null vectors give distinct points at zero separation, and the triangle
/// inequality reverses on timelike triples. Operations that need a genuine
/// norm or distance (e.g. `check_pythagorean`, `local_distance`,
/// `max_sectional_curvature`) are therefore available only on the definite
/// [`Euclidean`] refinement, gated by trait bounds rather than runtime checks.
///
/// # Flatness
/// Like a Euclidean space, a pseudo-Euclidean space is flat: geodesics are
/// straight lines, parallel transport is path-independent, and the
/// exponential map is a global isomorphism rather than merely a local one.
/// These properties are verified by the `check_*` methods inherited from
/// [`TangentBundle`] and by the signature-agnostic checks below —
/// `check_global_chart`, `check_global_geodesic_scaling`, and
/// `check_translation_invariance` — all stated on the *signed* quadratic form
/// so they hold in any signature. Compatibility of the exponential map with
/// the scalar product is certified separately by [`PseudoRiemannian`].
///
/// # Implementing
/// Use the `test_pseudo_euclidean!` macro to verify the pseudo-Euclidean
/// axioms. If the space is positive-definite, implement [`Euclidean`] as well
/// and use `test_euclidean!`, which additionally certifies the metric-space
/// and inner-product structure.
///
/// [`Bilinear`]: crate::traits::Bilinear
/// [`Euclidean`]: crate::traits::Euclidean
/// [`Metric`]: crate::traits::Metric
/// [`TangentBundle`]: crate::traits::TangentBundle
/// [`PseudoRiemannian`]: crate::traits::PseudoRiemannian
pub trait Quadratic: Bilinear {
    // todo: tests for Quadratic spaces
}

impl_group_via_add!(V, V: Vector);

impl<E: Vector> LieGroup<E> for E {
    fn identity_exp(v: E) -> Self {
        v
    }

    fn identity_log(p: &Self) -> Option<E> {
        Some(*p)
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
pub trait Bilinear: Sesquilinear {}
impl<F: Field<Fixed = F>, V: Sesquilinear<F = F>> Bilinear for V {}

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
pub trait Sesquilinear: Form {
    // Hermitian spaces are exactly the spaces where
    // self.dot(self) lands in the fixed field of F
    fn norm_squared(&self) -> <Self::F as Field>::Fixed {
        self.dot(self).to_fixed()
    }

    // ⟨v,w⟩ = conj(⟨w,v⟩) — Hermitian symmetry, the sesquilinear analogue
    // of Bilinear::check_symmetry. Additivity and conjugate-linearity in
    // the second argument both follow from this plus linearity in the
    // first, and aren't separately checked for the same reason Bilinear
    // doesn't separately check them.
    #[cfg(feature = "testing")]
    fn check_hermitian_symmetry(a: Self, b: Self) -> bool {
        a.dot(&b) == b.dot(&a).conj()
    }

    #[cfg(feature = "testing")]
    fn check_additivity(a: Self, b: Self, c: Self) -> bool
    where
        Self: Add<Output = Self> + Clone,
    {
        (a.clone() + b.clone()).dot(&c) == a.dot(&c) + b.dot(&c)
    }

    #[cfg(feature = "testing")]
    fn check_scalar_linearity(a: Self, c: Self, k: Self::F) -> bool
    where
        Self: Mul<Self::F, Output = Self> + Clone,
    {
        (a.clone() * k).dot(&c) == k * a.dot(&c)
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
pub trait InnerProduct: Sesquilinear<F: Real> + Metric<R = <Self::F as Field>::Fixed> {
    /// The norm `‖v‖ = sqrt(⟨v,v⟩)`. Well-defined and real because the form
    /// is positive-definite. On an indefinite [`Bilinear`] space this would
    /// not be real — which is why it lives here, not on the base.
    fn norm(&self) -> <Self::F as Field>::Fixed {
        self.norm_squared().sqrt()
    }

    #[cfg(feature = "testing")]
    fn check_positive_definite(a: Self) -> bool
    where
        Self: Zero + PartialEq,
    {
        a == Self::zero() || a.norm() > <Self::F as Field>::Fixed::zero()
    }

    #[cfg(feature = "testing")]
    fn check_metric_compatibility(a: Self, b: Self) -> bool {
        a.sub(b).norm_squared().sqrt() == a.distance(&b)
    }
}

impl<P: Sesquilinear<F: Real> + Metric<R = <Self::F as Field>::Fixed>> InnerProduct for P {}
