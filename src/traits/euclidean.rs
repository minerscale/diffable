use num_traits::Zero;
use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

use super::{InnerProduct, LieGroup, Scalar, TangentBundle};
use crate::{impl_group_via_add, traits::Bilinear};

/// A finite-dimensional Euclidean space.
///
/// The space of all values of a type `E: Euclidean` is interpreted as
/// `R^N` (with `R := E::F` and `N := E::N`) — the canonical flat, *positive-
/// definite* space of dimension `N` over the field `R`. This is the space in
/// which local coordinate charts take their values, and in which tangent
/// vectors live.
///
/// `Euclidean` is the **definite refinement** of [`PseudoEuclidean`]: it is a
/// pseudo-Euclidean space (signature `(N, 0)`) that additionally carries an
/// [`InnerProduct`] — a positive-definite pairing inducing a genuine `norm`
/// and a [`Metric`]. Where the pseudo-Euclidean base has only a signed
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
/// [`TangentBundle`] and [`PseudoEuclidean`] (`check_global_chart`,
/// `check_global_geodesic_scaling`, `check_translation_invariance`), together
/// with the definite-only `check_pythagorean` below.
///
/// # Implementing
/// Use the `test_euclidean!` macro to verify that your implementation
/// satisfies the Euclidean axioms. (For an indefinite space, implement only
/// [`PseudoEuclidean`] and use `test_pseudo_euclidean!` instead.)
///
/// [`Bilinear`]: crate::traits::Bilinear
/// [`InnerProduct`]: crate::traits::InnerProduct
/// [`Metric`]: crate::traits::Metric
/// [`TangentBundle`]: crate::traits::TangentBundle
pub trait Euclidean: PseudoEuclidean + InnerProduct<Self::F> {
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

/// A finite-dimensional pseudo-Euclidean space.
///
/// The space of all values of a type `V: PseudoEuclidean` is interpreted as
/// flat coordinate space `R^N` (`N := V::N`, `R := V::F`) equipped with a
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
pub trait PseudoEuclidean:
    Bilinear<Self::F>
    + TangentBundle<Self, Self>
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
    type F: Scalar;
    const N: usize;

    type Iter<'a>: Iterator<Item = &'a Self::F>
    where
        Self: 'a;
    fn iter(&self) -> Self::Iter<'_>;

    fn from_fn(f: impl Fn(usize) -> Self::F) -> Self;
    fn from_array<const N: usize>(arr: [Self::F; N]) -> Self;
    fn to_array<const N: usize>(self) -> [Self::F; N] {
        std::array::from_fn(|i| self[i])
    }

    // Flat space has no singularities — to_local is always Some
    #[cfg(feature = "testing")]
    fn check_global_chart(p: &Self, q: &Self) -> bool {
        let chart = Self::chart_at(p);
        chart.to_local(q).is_some()
    }

    // Translation invariance: Q((a+c) - (b+c)) == Q(a - b),
    // where Q(v) = ⟨v,v⟩ is the (signed) quadratic form.
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
        diff.norm_squared() == diff_translated.norm_squared()
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

impl_group_via_add!(V, V: PseudoEuclidean);

impl<E: PseudoEuclidean> LieGroup<E> for E {
    fn identity_exp(v: E) -> Self {
        v
    }

    fn identity_log(p: &Self) -> Option<E> {
        Some(*p)
    }
}
