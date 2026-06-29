use std::{
    marker::PhantomData,
    ops::{Add, Index, IndexMut, Mul, Neg, Sub},
};

use num_traits::{Zero, real::Real};

/// A point on a manifold.
///
/// The space of all values of a type `P: Point` is interpreted as a
/// smooth manifold `M`. The mathematical structure of `M` is expressed
/// through the associated traits [`Chart`], [`TangentBundle`], [`Metric`] etc.
///
/// The only meaningful requirement is equality — a point is defined by
/// its identity, the ability to ask whether two points are the same.
/// This should rightly be [`Eq`], but [`PartialEq`] is used in practice
/// because `f64`, the typical scalar type, implements only [`PartialEq`]
/// due to `NaN` — an IEEE 754 artifact with no geometric meaning.
pub trait Point: Clone + PartialEq {}

impl<T: PartialEq + Clone> Point for T {}

/// A chart in an atlas of a manifold.
///
/// The space of all values of a type `C: Chart<P, R, N>` is interpreted
/// as an atlas of the manifold `M` (the space of `P: Point`), covering
/// it with local coordinate neighbourhoods modelled on `R^N` (`Coords`).
/// The atlas axiom — that every point is covered — is expressed by
/// `chart_at`, which must produce a valid chart in a neighbourhood of
/// any given point.
///
/// `to_local` and `to_global` are the coordinate maps, with `to_local`
/// returning `None` at the singularities of the chart.
pub trait Chart<P: Point, const N: usize, Rn: Euclidean<N>>: Sized {
    fn to_local(&self, point: &P) -> Option<Rn>;
    fn to_global(&self, coord: Rn) -> P;
    fn chart_at(p: &P) -> Self;
}

/// A scalar field for use as the coordinate type of a Euclidean space.
///
/// Bundles the requirements that a scalar must satisfy to be usable
/// throughout diffable — real arithmetic and debuggability.
pub trait Scalar: Real + std::fmt::Debug {}

impl<R: Real + std::fmt::Debug> Scalar for R {}

/// A finite-dimensional Euclidean space.
///
/// The space of all values of a type `E: Euclidean<N>` is interpreted
/// as R^N (with R := E::Scalar) — the canonical flat Euclidean space of dimension `N` over the
/// field `R`. This is the space in which all local coordinate charts take
/// their values, and in which tangent vectors live.
///
/// Beyond the algebraic structure of a vector space (`Add`, `Sub`, `Mul`,
/// `Neg`, `Zero`), a Euclidean space carries an inner product (`InnerProduct`)
/// which induces a norm and metric, and a canonical tangent bundle
/// (`TangentBundle`) whose charts are globally defined with infinite
/// injectivity radius — reflecting the flatness of the space.
///
/// # Flatness
/// Unlike a general Riemannian manifold, a Euclidean space is flat: geodesics
/// are straight lines, parallel transport is path-independent, and the
/// exponential map is a global isomorphism rather than merely a local one.
/// These properties are verified by the `check_*` methods inherited from
/// `TangentBundle` and by the additional checks in `check_global_chart`,
/// `check_global_geodesic_scaling`, and `check_pythagorean`.
///
/// # Implementing
/// Use the `test_euclidean!` macro to verify that your implementation
/// satisfies the Euclidean axioms.
pub trait Euclidean<const N: usize>:
    InnerProduct<Self::Scalar>
    + TangentBundle<Self, N, Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Self::Scalar, Output = Self>
    + Neg<Output = Self>
    + Zero
    + From<[Self::Scalar; N]>
    + Into<[Self::Scalar; N]>
    + Index<usize, Output = Self::Scalar>
    + IndexMut<usize>
    + Copy
{
    type Scalar: Scalar;

    type Iter<'a>: Iterator<Item = &'a Self::Scalar>
    where
        Self: 'a,
        Self::Scalar: 'a;
    fn iter(&self) -> Self::Iter<'_>;

    // Flat space has no singularities — to_local is always Some
    fn check_global_chart(p: &Self, q: &Self) -> bool {
        let chart = Self::chart_at(p);
        chart.to_local(q).is_some()
    }

    // Translation invariance: d(a + c, b + c) == d(a, b)
    fn check_translation_invariance(a: &Self, b: &Self, c: &Self, epsilon: Self::Scalar) -> bool
    where
        Self: Add<Output = Self> + Clone,
    {
        let dist_ab = a.distance(b);
        let dist_translated = (a.clone() + c.clone()).distance(&(b.clone() + c.clone()));
        (dist_ab - dist_translated).abs() < epsilon
    }

    // Geodesic scaling holds globally (infinite injectivity radius):
    // to_global(v * t) is parallel to to_global(v) AND scaled by t exactly
    fn check_global_geodesic_scaling(
        p: &Self,
        v: Self,
        t: Self::Scalar,
        epsilon: Self::Scalar,
    ) -> bool {
        let chart = Self::chart_at(p);
        match (
            chart.to_local(&chart.to_global(v * t)),
            chart.to_local(&chart.to_global(v)),
        ) {
            (Some(tv_local), Some(v_local)) => {
                // parallel
                let dot = tv_local.dot(&v_local);
                let parallel =
                    (dot * dot - tv_local.norm_squared() * v_local.norm_squared()).abs() < epsilon;
                // scaled exactly
                let scaled = (tv_local - v_local * t).norm() < epsilon;
                parallel && scaled
            }
            _ => false, // None is never acceptable in flat space
        }
    }

    // Pythagorean theorem: d(a, b)² == |a - b|²
    fn check_pythagorean(a: &Self, b: &Self, epsilon: Self::Scalar) -> bool
    where
        Self: Sub<Output = Self> + Clone,
    {
        let dist_sq = a.distance(b);
        let dist_sq = dist_sq * dist_sq;
        let diff = a.clone() - b.clone();
        let norm_sq = diff.norm_squared();
        (dist_sq - norm_sq).abs() < epsilon
    }
}

impl<const N: usize, E: Euclidean<N>> LieGroup<N> for E {
    fn identity() -> Self {
        Self::zero()
    }

    fn compose(&self, other: &Self) -> Self {
        *self + *other
    }

    fn inverse(&self) -> Self {
        -*self
    }
}

pub trait LieGroup<const N: usize>: Point {
    fn identity() -> Self;
    fn compose(&self, other: &Self) -> Self;
    fn inverse(&self) -> Self;
}

pub trait Metric<R: Real>: Point {
    fn distance(&self, other: &Self) -> R;

    fn within(&self, other: &Self, epsilon: R) -> bool {
        self.distance(other) < epsilon
    }
}

pub trait InnerProduct<R: Real>: Metric<R> {
    fn dot(&self, other: &Self) -> R;

    fn norm(&self) -> R {
        self.norm_squared().sqrt()
    }

    fn norm_squared(&self) -> R {
        self.dot(self)
    }
}

/// By implementing ExpMap you certify that for C<P, R, N>: ExpMap<P, R, N> that
/// straight lines through the origin in R^N map to geodesics on M, and
/// that distances from the origin equal arc lengths along those geodesics.
///
/// Additionally, when `P: LieGroup<N>`, you certify that the base point
/// of this chart is the group identity — i.e. `self.to_global(zero) == P::identity()`.
/// This is what makes the left-translation construction in [`LeftTranslationChart`] valid.
pub trait ExpMap<P: Point, const N: usize, Rn: Euclidean<N>>: Chart<P, N, Rn> {
    fn base_point(&self) -> P {
        self.to_global(Rn::zero())
    }

    // Tests that base_point() is consistent with to_local.
    // Meaningful only when base_point() is overridden, since
    // the default impl makes this trivially true by construction.
    fn check_base_point_is_origin(&self, epsilon: Rn::Scalar) -> bool {
        self.to_local(&self.base_point())
            .map_or(false, |c| c.norm() < epsilon)
    }

    // Tests that log(exp(0)) == 0, i.e. that the
    // round trip at the origin is the identity.
    fn check_preservation_of_origin(&self, epsilon: Rn::Scalar) -> bool {
        let zero = Rn::zero();
        let exp_zero = self.to_global(zero);
        self.to_local(&exp_zero)
            .map_or(false, |c| c.norm() < epsilon)
    }

    // geodesics are reversible: log(exp(v)) == -log(exp(-v))
    fn check_geodesic_symmetry(&self, v: Rn, epsilon: Rn::Scalar) -> bool {
        match (
            self.to_local(&self.to_global(v)),
            self.to_local(&self.to_global(-v)),
        ) {
            (Some(fwd), Some(bwd)) => fwd.within(&(-bwd), epsilon),
            _ => true, // at singularity, vacuously true
        }
    }

    // geodesics are straight lines: exp(tv) lies on the same geodesic as exp(v),
    // i.e. log(exp(tv)) and log(exp(v)) are parallel in local coords.
    fn check_geodesic_scaling(&self, v: Rn, t: Rn::Scalar, epsilon: Rn::Scalar) -> bool {
        match (
            self.to_local(&self.to_global(v * t)),
            self.to_local(&self.to_global(v)),
        ) {
            (Some(tv_local), Some(v_local)) => {
                // tv_local and v_local should be parallel:
                // tv_local × v_local == 0, i.e. dot(tv_local, v_local)² == |tv_local|² * |v_local|²
                let dot = tv_local.dot(&v_local);
                let lhs = dot * dot;
                let rhs = tv_local.norm_squared() * v_local.norm_squared();
                (lhs - rhs).abs() < epsilon
            }
            _ => true,
        }
    }

    // isometry to first order: |exp(epsilon * v) - base| / epsilon → |v|
    fn check_first_order_isometry(&self, v: Rn, coef: Rn::Scalar, epsilon: Rn::Scalar) -> bool {
        let small_v = v * coef;
        self.to_local(&self.to_global(small_v))
            .map_or(true, |local| {
                let lhs = local.norm() / coef;
                let rhs = v.norm();
                (lhs - rhs).abs() < epsilon
            })
    }

    fn check_identity_base_point() -> bool
    where
        P: LieGroup<N>,
    {
        let chart = Self::chart_at(&P::identity());

        chart.base_point() == P::identity()
    }
}

/// A tangent bundle structure on a manifold.
///
/// The space of all values of a type `C: TangentBundle<P, N, Rn>` is
/// interpreted as the tangent bundle `TM` of the manifold `M` (the space
/// of `P: Point`). Each instance is a single tangent space `T_p M` at
/// the base point `p`, coordinatised by `Rn`.
///
/// By implementing `TangentBundle` you certify that for all `p: P`:
/// `C::chart_at(&p).to_global(Rn::zero()) == p`
///
/// That is, the chart produced at any point is centred at that point —
/// the origin of the local coordinate system corresponds to the base point
/// on the manifold. This is what distinguishes a tangent bundle from a
/// bare [`Chart`] or [`ExpMap`].
///
/// Use the `test_tangent_bundle!` macro to verify this invariant.
pub trait TangentBundle<P: Point, const N: usize, Rn: Euclidean<N>>: ExpMap<P, N, Rn> {
    // p is the point on the manifold which is the base point.
    fn check_invariant(p: &P, epsilon: Rn::Scalar) -> bool {
        let chart = Self::chart_at(p);

        chart.check_preservation_of_origin(epsilon) && chart.check_base_point_is_origin(epsilon)
    }
}

/// A tangent bundle on a Lie group constructed via left translation.
///
/// Given an [`ExpMap`] `E` centred at the group identity, this chart
/// extends it to every point on the group by left translation — for a
/// base point `p`, the chart maps a tangent vector `v` to `p * exp(v)`,
/// and its inverse maps a point `q` to `log(p⁻¹ * q)`.
///
/// This construction works because a Lie group is homogeneous — it looks
/// the same at every point, and left translation is a smooth isometry that
/// carries the geometry at the identity to any other point. The [`ExpMap`]
/// at the identity is therefore sufficient to generate a full [`TangentBundle`]
/// over the entire group.
///
/// # Invariant
/// The `identity_exp_map` field must be centred at the group identity —
/// i.e. `identity_exp_map.to_global(Rn::zero()) == P::identity()`. This
/// is certified by implementing [`ExpMap`] on a `P: LieGroup<N>`, which
/// requires that `chart_at(&P::identity())` returns a chart centred there.
pub struct LeftTranslationChart<
    const N: usize,
    P: LieGroup<N>,
    Rn: Euclidean<N>,
    E: ExpMap<P, N, Rn>,
> {
    base_point: P,
    identity_exp_map: E,
    _rn: PhantomData<Rn>,
}

impl<const N: usize, P: LieGroup<N>, Rn: Euclidean<N>, E: ExpMap<P, N, Rn>> Chart<P, N, Rn>
    for LeftTranslationChart<N, P, Rn, E>
{
    fn to_local(&self, point: &P) -> Option<Rn> {
        let translated = self.base_point.inverse().compose(point);
        self.identity_exp_map.to_local(&translated)
    }
    fn to_global(&self, coord: Rn) -> P {
        let translated = self.identity_exp_map.to_global(coord);
        self.base_point.compose(&translated)
    }
    fn chart_at(p: &P) -> Self {
        Self {
            base_point: p.clone(),
            // Guaranteed by definition of ExpMap that
            // this is the exp map centered at the identity
            identity_exp_map: E::chart_at(&P::identity()),
            _rn: PhantomData,
        }
    }
}

impl<P: LieGroup<N>, const N: usize, Rn: Euclidean<N>, E: ExpMap<P, N, Rn>> ExpMap<P, N, Rn>
    for LeftTranslationChart<N, P, Rn, E>
{
    // optimisation
    fn base_point(&self) -> P {
        self.base_point.clone()
    }
}

impl<P: LieGroup<N>, const N: usize, Rn: Euclidean<N>, E: ExpMap<P, N, Rn>> TangentBundle<P, N, Rn>
    for LeftTranslationChart<N, P, Rn, E>
{
}

#[cfg(feature = "testing")]
pub mod testing {
    // ---------------------------------------------------------------------------
    // Trait test macros
    // These generate the full suite of invariant tests for any implementation
    // of Chart, ExpMap, TangentBundle, LieGroup, and Metric. To test a new
    // manifold, just invoke the relevant macro with appropriate generators.
    // ---------------------------------------------------------------------------

    #[macro_export]
    macro_rules! test_euclidean {
        ($mod_name:ident, $space:ty, $arb_point:expr, $arb_vec:expr) => {
            mod $mod_name {
                use super::*;

                // inherit all TangentFibre tests
                test_tangent_bundle!(tangent_bundle, $space, $space, $arb_point, $arb_vec);

                proptest! {
                    #[test]
                    fn global_chart(p in $arb_point, q in $arb_point) {
                        prop_assert!(<$space>::check_global_chart(&p, &q));
                    }

                    #[test]
                    fn translation_invariance(
                        a in $arb_point,
                        b in $arb_point,
                        c in $arb_point,
                    ) {
                        prop_assert!(<$space>::check_translation_invariance(&a, &b, &c, EPSILON));
                    }

                    #[test]
                    fn global_geodesic_scaling(
                        p in $arb_point,
                        v in $arb_vec,
                        t in -10.0f64..10.0f64, // unbounded t, flat space has no injectivity radius
                    ) {
                        prop_assert!(<$space>::check_global_geodesic_scaling(&p, v, t, EPSILON));
                    }

                    #[test]
                    fn pythagorean(a in $arb_point, b in $arb_point) {
                        prop_assert!(<$space>::check_pythagorean(&a, &b, EPSILON));
                    }
                }
            }
        };
    }

    /// Tests the chart roundtrip invariant: to_global(to_local(p)) == p
    /// for any chart type and point generator.
    #[macro_export]
    macro_rules! test_chart {
        ($mod_name:ident, $chart:ty, $point:ty, $arb_point:expr) => {
            mod $mod_name {
                use super::*;
                proptest! {
                    #[test]
                    fn roundtrip(p in $arb_point) {
                        let chart = <$chart>::chart_at(&p);
                        if let Some(local) = chart.to_local(&p) {
                            let recovered = chart.to_global(local);
                            prop_assert!(
                                p.within(&recovered, EPSILON),
                                "roundtrip failed: {:?} -> {:?} | local: {:?}",
                                p, recovered, local
                            );
                        }
                    }
                }
            }
        };
    }

    /// Tests the ExpMap invariants: preservation of origin, geodesic symmetry,
    /// geodesic scaling, and first-order isometry. The chart is constructed
    /// via chart_at on a generated base point.
    #[macro_export]
    macro_rules! test_exp_map {
        ($mod_name:ident, $chart:ty, $point:ty, $arb_point:expr, $arb_vec:expr) => {
            mod $mod_name {
                use super::*;
                proptest! {
                    #[test]
                    fn preservation_of_origin(p in $arb_point) {
                        let chart = <$chart>::chart_at(&p);
                        prop_assert!(chart.check_preservation_of_origin(EPSILON));
                    }

                    #[test]
                    fn base_point_is_origin(p in $arb_point) {
                        let chart = <$chart>::chart_at(&p);
                        prop_assert!(chart.check_base_point_is_origin(EPSILON));
                    }

                    #[test]
                    fn geodesic_symmetry(p in $arb_point, v in $arb_vec) {
                        let chart = <$chart>::chart_at(&p);
                        prop_assert!(chart.check_geodesic_symmetry(v, EPSILON));
                    }

                    #[test]
                    fn geodesic_scaling(p in $arb_point, v in $arb_vec, t in 0.0f64..1.0f64) {
                        let chart = <$chart>::chart_at(&p);
                        prop_assert!(chart.check_geodesic_scaling(v, t, EPSILON));
                    }

                    #[test]
                    fn first_order_isometry(p in $arb_point, v in $arb_vec) {
                        let chart = <$chart>::chart_at(&p);
                        prop_assert!(chart.check_first_order_isometry(v, 1e-5, EPSILON));
                    }
                }
            }
        };
    }

    #[macro_export]
    macro_rules! test_exp_map_lie_group {
        ($mod_name:ident, $chart:ty, $point:ty, $n:literal, $arb_point:expr, $arb_vec:expr) => {
            mod $mod_name {
                use super::*;

                // inherit all ExpMap tests
                test_exp_map!(exp_map, $chart, $point, $arb_point, $arb_vec);

                #[test]
                fn identity_base_point() {
                    assert!(<$chart as ExpMap<$point, $n, _>>::check_identity_base_point());
                }
            }
        };
    }

    /// Tests the TangentBundle invariant on top of all ExpMap invariants.
    #[macro_export]
    macro_rules! test_tangent_bundle {
        ($mod_name:ident, $chart:ty, $point:ty, $arb_point:expr, $arb_vec:expr) => {
            mod $mod_name {
                use super::*;

                // inherit all ExpMap tests
                test_exp_map!(exp_map, $chart, $point, $arb_point, $arb_vec);

                proptest! {
                    // The TangentFibre invariant: chart_at(&p).to_global(zero) == p
                    #[test]
                    fn tangent_fibre_invariant(p in $arb_point) {
                        prop_assert!(<$chart>::check_invariant(&p, EPSILON));
                    }

                    // Full roundtrip: base and point are both arbitrary
                    #[test]
                    fn roundtrip(base in $arb_point, point in $arb_point) {
                        let chart = <$chart>::chart_at(&base);
                        prop_assert!(<$chart>::check_invariant(&base, EPSILON));
                        if let Some(local) = chart.to_local(&point) {
                            let recovered = chart.to_global(local);
                            prop_assert!(
                                point.within(&recovered, EPSILON),
                                "roundtrip failed: base={:?} point={:?} recovered={:?}",
                                base, point, recovered
                            );
                        }
                    }
                }
            }
        };
    }

    /// Tests the LieGroup axioms: identity, inverses, associativity.
    #[macro_export]
    macro_rules! test_lie_group {
        ($mod_name:ident, $point:ty, $arb_point:expr) => {
            mod $mod_name {
                use super::*;
                proptest! {
                    #[test]
                    fn left_identity(p in $arb_point) {
                        let id = <$point>::identity();
                        prop_assert!(id.compose(&p).within(&p, EPSILON));
                    }

                    #[test]
                    fn right_identity(p in $arb_point) {
                        let id = <$point>::identity();
                        prop_assert!(p.compose(&id).within(&p, EPSILON));
                    }

                    #[test]
                    fn left_inverse(p in $arb_point) {
                        let id = <$point>::identity();
                        prop_assert!(p.inverse().compose(&p).within(&id, EPSILON),
                            "p={:?} id={:?} composition={:?}", p, id, p.inverse().compose(&p));
                    }

                    #[test]
                    fn right_inverse(p in $arb_point) {
                        let id = <$point>::identity();
                        prop_assert!(p.compose(&p.inverse()).within(&id, EPSILON));
                    }

                    #[test]
                    fn associativity(a in $arb_point, b in $arb_point, c in $arb_point) {
                        prop_assert!(
                            a.compose(&b).compose(&c)
                                .within(&a.compose(&b.compose(&c)), EPSILON)
                        );
                    }
                }
            }
        };
    }

    /// Tests the Metric axioms: non-negativity, symmetry, self-distance is zero.
    #[macro_export]
    macro_rules! test_metric {
        ($mod_name:ident, $point:ty, $arb_point:expr) => {
            mod $mod_name {
                use super::*;
                proptest! {
                    #[test]
                    fn non_negative(a in $arb_point, b in $arb_point) {
                        prop_assert!(a.distance(&b) >= 0.0);
                    }

                    #[test]
                    fn symmetry(a in $arb_point, b in $arb_point) {
                        prop_assert!((a.distance(&b) - b.distance(&a)).abs() < EPSILON);
                    }

                    #[test]
                    fn self_distance_zero(p in $arb_point) {
                        prop_assert!(p.distance(&p) < EPSILON);
                    }
                }
            }
        };
    }
}
