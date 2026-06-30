use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

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
/// The space of all values of a type `C: Chart<P, Rn>` is interpreted
/// as an atlas of the manifold `M` (the space of `P: Point`), covering
/// it with local coordinate neighbourhoods modelled on `R^N` (`Coords`).
/// The atlas axiom — that every point is covered — is expressed by
/// `chart_at`, which must produce a valid chart in a neighbourhood of
/// any given point.
///
/// `to_local` and `to_global` are the coordinate maps, with `to_local`
/// returning `None` at the singularities of the chart.
pub trait Chart<P: Point, Rn: Euclidean>: Sized {
    fn to_local(&self, point: &P) -> Option<Rn>;
    fn to_global(&self, coord: Rn) -> P;
    fn chart_at(p: &P) -> Self;

    /// Checks whether `b` lies within `epsilon` of this chart's base point, as
    /// measured in local coordinates.
    ///
    /// Because every [`Chart`] is a homeomorphism onto an open subset of `Rn`,
    /// this check is always well-defined topologically: shrinking `epsilon`
    /// shrinks the corresponding neighbourhood on the manifold, and the chart's
    /// pulled-back metric generates the manifold's own topology. What it does
    /// *not* guarantee, for a bare `Chart`, is that `epsilon` corresponds to any
    /// particular distance on the manifold — different charts at the same point
    /// can disagree numerically about how "close" `b` is, since only an
    /// [`ExpMap`] additionally certifies that local coordinate distance equals
    /// geodesic distance to first order.
    ///
    /// The where `P: Chart<P, Rn>` bound forces `Self = P`, letting
    /// `self` serve as both the chart and the point being measured from.
    fn in_neighbourhood(&self, b: &P, epsilon: Rn::Scalar) -> bool
    where
        P: Chart<P, Rn>,
    {
        self.to_local(b).map_or(false, |x| x.norm() <= epsilon)
    }

    /// Best-effort check for whether `a` and `b` are close,
    /// without the trait bound P: Chart<P, Rn>.
    ///
    /// Tries the chart centred at `a`, then at `b`, succeeding if either
    /// places the other point within `epsilon` in local coordinates. A `true`
    /// result is trustworthy, by the same topological argument as
    /// [`Chart::in_neighbourhood`]. A `false` result is not proof of distance —
    /// only that neither chart could confirm closeness. There is no way to
    /// enumerate every chart covering a point (the space is typically infinite),
    /// so only the two `chart_at` already guarantees exist are tried.
    ///
    /// This is more reliable than the worst case suggests if `chart_at` places
    /// its singularity sensibly. [`crate::hypersphere::Stereographic`] always picks the pole
    /// opposite the input point, so `attempt(a)` can only miss a genuinely close
    /// `b` if `b` is near that far pole — but then `a`, being close to `b`,
    /// would have to be near it too, contradicting `chart_at(a)`'s own choice.
    /// So for charts shaped like this, `attempt(a)` alone rarely fails for
    /// points that are actually close. This is a property of well-behaved
    /// `chart_at` implementations, not something the trait enforces.
    ///
    /// If `P` implements `Chart<P, Rn>` directly, prefer
    /// [`Chart::in_neighbourhood`] instead — it has a `false` case that's also
    /// meaningful.
    fn in_neighbourhood_heuristic(a: &P, b: &P, epsilon: Rn::Scalar) -> bool {
        let attempt = |chart_base: &P| {
            let chart = Self::chart_at(chart_base);

            if let (Some(a), Some(b)) = (chart.to_local(a), chart.to_local(b)) {
                a.distance(&b) <= epsilon
            } else {
                false
            }
        };

        attempt(a) || attempt(b)
    }

    #[cfg(feature = "testing")]
    fn check_chart_roundtrip(p: P, epsilon: Rn::Scalar) -> bool {
        let chart = Self::chart_at(&p);
        if let Some(local) = chart.to_local(&p) {
            let recovered = chart.to_global(local);
            Self::in_neighbourhood_heuristic(&p, &recovered, epsilon)
        } else {
            false
        }
    }
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
/// as R^N (with R := E::Scalar) — the canonical flat Euclidean space of dimension `N`
/// over the field `R`. This is the space in which all local coordinate charts take
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
/// `check_global_geodesic_scaling`, `check_translation_invariance`, and `check_pythagorean`.
///
/// # Implementing
/// Use the `test_euclidean!` macro to verify that your implementation
/// satisfies the Euclidean axioms.
pub trait Euclidean:
    InnerProduct<Self::Scalar>
    + TangentBundle<Self, Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Self::Scalar, Output = Self>
    + Neg<Output = Self>
    + Zero
    + Index<usize, Output = Self::Scalar>
    + IndexMut<usize>
    + Copy
{
    type Scalar: Scalar;

    type Iter<'a>: Iterator<Item = &'a Self::Scalar>
    where
        Self: 'a;
    fn iter(&self) -> Self::Iter<'_>;

    fn from_array<const N: usize>(arr: [Self::Scalar; N]) -> Self;
    fn to_array<const N: usize>(self) -> [Self::Scalar; N] {
        std::array::from_fn(|i| self[i])
    }

    // Flat space has no singularities — to_local is always Some
    #[cfg(feature = "testing")]
    fn check_global_chart(p: &Self, q: &Self) -> bool {
        let chart = Self::chart_at(p);
        chart.to_local(q).is_some()
    }

    // Translation invariance: d(a + c, b + c) == d(a, b)
    #[cfg(feature = "testing")]
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
    #[cfg(feature = "testing")]
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
    #[cfg(feature = "testing")]
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

/// A Lie group structure on a manifold.
///
/// The space of all values of a type `G: LieGroup<Rn>` is interpreted as
/// a Lie group — a manifold that is also a group, where the group operations
/// are smooth maps. `Rn` is the Euclidean space coordinatising the group's
/// tangent space at the identity.
///
/// # Group axioms
/// - **Identity**: there exists an element `e` such that `e * g = g * e = g`
/// - **Inverses**: for every `g` there exists `g⁻¹` such that `g * g⁻¹ = g⁻¹ * g = e`
/// - **Associativity**: `(a * b) * c = a * (b * c)`
///
/// These are not enforced by the type system but are certified by implementing
/// this trait, and verified empirically by the `test_lie_group!` macro.
///
/// # Exponential map at the identity
/// `identity_exp` and `identity_log` are the exponential and logarithm maps
/// centred at the group identity — they witness that `Rn`, the tangent space
/// at the identity, genuinely linearises the group there. They are not
/// required to work, or have any particular meaning, at any other base point.
///
/// # Automatic tangent bundle
/// Implementing `LieGroup` automatically certifies [`Chart`], [`ExpMap`], and
/// [`TangentBundle`] for `Self` via a blanket implementation: a chart centred
/// at any base point `p` is constructed by left translation — `to_global(v) =
/// p * identity_exp(v)` and `to_local(q) = identity_log(p⁻¹ * q)`. This works
/// because a Lie group is homogeneous: left translation by `p` is a smooth
/// isometry carrying the geometry at the identity to every other point, so
/// the exponential map at the identity alone is sufficient to generate a
/// full tangent bundle over the entire group, with no separate wrapper type
/// needed.
pub trait LieGroup<Rn: Euclidean>: Point {
    fn identity() -> Self;
    fn compose(&self, other: &Self) -> Self;
    fn inverse(&self) -> Self;

    fn identity_exp(v: Rn) -> Self;
    fn identity_log(p: &Self) -> Option<Rn>;

    #[cfg(feature = "testing")]
    fn check_left_identity(&self, epsilon: Rn::Scalar) -> bool {
        let id = Self::identity();

        self.in_neighbourhood(&id.compose(self), epsilon)
    }

    #[cfg(feature = "testing")]
    fn check_right_identity(&self, epsilon: Rn::Scalar) -> bool {
        let id = Self::identity();

        self.in_neighbourhood(&self.compose(&id), epsilon)
    }

    #[cfg(feature = "testing")]
    fn check_left_inverse(&self, epsilon: Rn::Scalar) -> bool {
        let id = Self::identity();

        self.inverse().compose(&self).in_neighbourhood(&id, epsilon)
    }

    #[cfg(feature = "testing")]
    fn check_right_inverse(&self, epsilon: Rn::Scalar) -> bool {
        let id = Self::identity();

        self.compose(&self.inverse()).in_neighbourhood(&id, epsilon)
    }

    #[cfg(feature = "testing")]
    fn check_associativity(a: Self, b: Self, c: Self, epsilon: Rn::Scalar) -> bool {
        a.compose(&b)
            .compose(&c)
            .in_neighbourhood(&a.compose(&b.compose(&c)), epsilon)
    }
}

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
pub trait Metric<R: Real>: Point {
    fn distance(&self, other: &Self) -> R;

    fn within(&self, other: &Self, epsilon: R) -> bool {
        self.distance(other) < epsilon
    }

    #[cfg(feature = "testing")]
    fn check_self_distance_zero(a: Self, epsilon: R) -> bool {
        a.distance(&a) <= epsilon
    }

    #[cfg(feature = "testing")]
    fn check_triangle_inequality(a: Self, b: Self, c: Self, epsilon: R) -> bool {
        let sum = a.distance(&b) + b.distance(&c);
        a.distance(&c) <= sum * (R::one() + epsilon)
    }

    #[cfg(feature = "testing")]
    fn check_non_negative(a: Self, b: Self) -> bool {
        a.distance(&b) >= R::zero()
    }

    #[cfg(feature = "testing")]
    fn check_metric_symmetry(a: Self, b: Self, epsilon: R) -> bool {
        (a.distance(&b) - b.distance(&a)).abs() < epsilon
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
    fn check_inner_product_symmetry(a: Self, b: Self, epsilon: R) -> bool {
        (a.dot(&b) - b.dot(&a)).abs() < epsilon
    }

    #[cfg(feature = "testing")]
    fn check_additivity(a: Self, b: Self, c: Self, epsilon: R) -> bool
    where
        Self: Add<Output = Self> + Clone,
    {
        let lhs = (a.clone() + b.clone()).dot(&c);
        let rhs = a.dot(&c) + b.dot(&c);
        (lhs - rhs).abs() < epsilon
    }

    #[cfg(feature = "testing")]
    fn check_scalar_linearity(a: Self, c: Self, k: R, epsilon: R) -> bool
    where
        Self: Mul<R, Output = Self> + Clone,
    {
        let lhs = (a.clone() * k).dot(&c);
        let rhs = k * a.dot(&c);
        (lhs - rhs).abs() < epsilon
    }

    #[cfg(feature = "testing")]
    fn check_positive_definite(a: Self, epsilon: R) -> bool
    where
        Self: Zero,
    {
        let self_dot = a.dot(&a);
        if a.is_zero() {
            self_dot.abs() < epsilon
        } else {
            self_dot > epsilon
        }
    }
}

/// By implementing ExpMap you certify that for C<P, Rn>: ExpMap<P, Rn> that
/// straight lines through the origin in R^N map to geodesics on M, and
/// that distances from the origin equal arc lengths along those geodesics.
pub trait ExpMap<P: Point, Rn: Euclidean>: Chart<P, Rn> {
    fn base_point(&self) -> P {
        self.to_global(Rn::zero())
    }

    // Tests that base_point() is consistent with to_local.
    // Meaningful only when base_point() is overridden, since
    // the default impl makes this trivially true by construction.
    #[cfg(feature = "testing")]
    fn check_base_point_is_origin(&self, epsilon: Rn::Scalar) -> bool {
        self.to_local(&self.base_point())
            .map_or(false, |c| c.norm() < epsilon)
    }

    // Tests that log(exp(0)) == 0, i.e. that the
    // round trip at the origin is the identity.
    #[cfg(feature = "testing")]
    fn check_preservation_of_origin(&self, epsilon: Rn::Scalar) -> bool {
        let zero = Rn::zero();
        let exp_zero = self.to_global(zero);
        self.to_local(&exp_zero)
            .map_or(false, |c| c.norm() < epsilon)
    }

    // geodesics are reversible: log(exp(v)) == -log(exp(-v))
    #[cfg(feature = "testing")]
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
    #[cfg(feature = "testing")]
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
    #[cfg(feature = "testing")]
    fn check_first_order_isometry(&self, v: Rn, coef: Rn::Scalar, epsilon: Rn::Scalar) -> bool {
        let small_v = v * coef;
        self.to_local(&self.to_global(small_v))
            .map_or(true, |local| {
                let lhs = local.norm() / coef;
                let rhs = v.norm();
                (lhs - rhs).abs() < epsilon
            })
    }
}

/// A tangent bundle structure on a manifold.
///
/// The space of all values of a type `C: TangentBundle<P, Rn>` is
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
pub trait TangentBundle<P: Point, Rn: Euclidean>: ExpMap<P, Rn> {
    // p is the point on the manifold which is the base point.
    #[cfg(feature = "testing")]
    fn check_tangent_bundle_invariant(p: P, epsilon: Rn::Scalar) -> bool {
        let chart = Self::chart_at(&p);

        chart.check_preservation_of_origin(epsilon) && chart.check_base_point_is_origin(epsilon)
    }

    #[cfg(feature = "testing")]
    fn check_tangent_bundle_roundtrip(base: P, point: P, epsilon: Rn::Scalar) -> bool {
        let chart = Self::chart_at(&base);

        chart.to_local(&point).map_or(true, |local| {
            Self::in_neighbourhood_heuristic(&point, &chart.to_global(local), epsilon)
        })
    }
}

impl<E: Euclidean> LieGroup<E> for E {
    fn identity() -> Self {
        Self::zero()
    }

    fn compose(&self, other: &Self) -> Self {
        *self + *other
    }

    fn inverse(&self) -> Self {
        -*self
    }

    fn identity_exp(v: E) -> Self {
        v
    }

    fn identity_log(p: &Self) -> Option<E> {
        Some(*p)
    }
}

impl<Rn: Euclidean, L: LieGroup<Rn>> Chart<Self, Rn> for L {
    fn to_local(&self, point: &Self) -> Option<Rn> {
        let translated = self.inverse().compose(point);
        Self::identity_log(&translated)
    }

    fn to_global(&self, coord: Rn) -> Self {
        let translated = Self::identity_exp(coord);
        self.compose(&translated)
    }

    fn chart_at(p: &Self) -> Self {
        p.clone()
    }
}

impl<Rn: Euclidean, L: LieGroup<Rn>> ExpMap<Self, Rn> for L {
    // optimisation
    fn base_point(&self) -> Self {
        self.clone()
    }
}

impl<Rn: Euclidean, L: LieGroup<Rn>> TangentBundle<Self, Rn> for L {}

#[cfg(feature = "testing")]
pub mod testing {
    // ---------------------------------------------------------------------------
    // Trait test macros
    // These generate the full suite of invariant tests for any implementation
    // of Chart, ExpMap, TangentBundle, LieGroup, and Metric. To test a new
    // manifold, just invoke the relevant macro with appropriate generators.
    // ---------------------------------------------------------------------------

    /// Tests that a space claiming to be a euclidean space is a euclidean space
    #[macro_export]
    macro_rules! test_euclidean {
        ($mod_name:ident, $space:ty, $arb_point:expr, $arb_vec:expr, $arb_scalar:expr) => {
            mod $mod_name {
                use super::*;

                // inherit all TangentFibre tests
                test_tangent_bundle!(tangent_bundle, $space, $space, $arb_point, $arb_vec);
                test_metric!(metric, $space, $arb_vec);
                test_inner_product!(inner_product, $space, $arb_point, $arb_scalar);

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
        ($mod_name:ident, $chart:ty, $arb_point:expr) => {
            mod $mod_name {
                use super::*;
                proptest! {
                    #[test]
                    fn roundtrip(p in $arb_point) {
                        prop_assert!(<$chart>::check_chart_roundtrip(p, EPSILON))
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
        ($mod_name:ident, $chart:ty, $arb_point:expr, $arb_vec:expr) => {
            mod $mod_name {
                use super::*;

                // inherit all Chart tests
                test_chart!(chart, $chart, $arb_point);

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

    /// Tests the TangentBundle invariant on top of all ExpMap invariants.
    #[macro_export]
    macro_rules! test_tangent_bundle {
        ($mod_name:ident, $chart:ty, $point:ty, $arb_point:expr, $arb_vec:expr) => {
            mod $mod_name {
                use super::*;

                // inherit all ExpMap tests
                test_exp_map!(exp_map, $chart, $arb_point, $arb_vec);

                proptest! {
                    // The TangentFibre invariant: chart_at(&p).to_global(zero) == p
                    #[test]
                    fn tangent_fibre_invariant(p in $arb_point) {
                        prop_assert!(<$chart>::check_tangent_bundle_invariant(p, EPSILON));
                    }

                    // Full roundtrip: base and point are both arbitrary
                    #[test]
                    fn roundtrip(base in $arb_point, point in $arb_point) {
                        prop_assert!(<$chart>::check_tangent_bundle_roundtrip(base, point, EPSILON));
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
                        prop_assert!(p.check_left_identity(EPSILON));
                    }

                    #[test]
                    fn right_identity(p in $arb_point) {
                        prop_assert!(p.check_right_identity(EPSILON));
                    }

                    #[test]
                    fn left_inverse(p in $arb_point) {
                        prop_assert!(p.check_left_inverse(EPSILON));
                    }

                    #[test]
                    fn right_inverse(p in $arb_point) {
                        prop_assert!(p.check_right_inverse(EPSILON));
                    }

                    #[test]
                    fn associativity(a in $arb_point, b in $arb_point, c in $arb_point) {
                        prop_assert!(<$point>::check_associativity(a, b, c, EPSILON));
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
                        prop_assert!(<$point>::check_non_negative(a, b));
                    }

                    #[test]
                    fn symmetry(a in $arb_point, b in $arb_point) {
                        prop_assert!(<$point>::check_metric_symmetry(a, b, EPSILON));
                    }

                    #[test]
                    fn self_distance_zero(p in $arb_point) {
                        prop_assert!(<$point>::check_self_distance_zero(p, EPSILON))
                    }

                    #[test]
                    fn check_triangle_inequality(a in $arb_point, b in $arb_point, c in $arb_point) {
                        prop_assert!(<$point>::check_triangle_inequality(a, b, c, EPSILON))
                    }
                }
            }
        };
    }

    /// Tests the InnerProduct axioms: symmetry, bilinearity, positive-definiteness.
    #[macro_export]
    macro_rules! test_inner_product {
        ($mod_name:ident, $point:ty, $arb_point:expr, $arb_scalar:expr) => {
            mod $mod_name {
                use super::*;
                proptest! {
                    #[test]
                    fn symmetry(a in $arb_point, b in $arb_point) {
                        prop_assert!(<$point>::check_inner_product_symmetry(a, b, EPSILON));
                    }
    
                    #[test]
                    fn additivity(a in $arb_point, b in $arb_point, c in $arb_point) {
                        prop_assert!(<$point>::check_additivity(a, b, c, EPSILON));
                    }
    
                    #[test]
                    fn scalar_linearity(a in $arb_point, c in $arb_point, k in $arb_scalar) {
                        prop_assert!(<$point>::check_scalar_linearity(a, c, k, EPSILON));
                    }
    
                    #[test]
                    fn positive_definite(a in $arb_point) {
                        prop_assert!(<$point>::check_positive_definite(a, EPSILON));
                    }
                }
            }
        };
    }
}
