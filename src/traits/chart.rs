use super::{Euclidean, Point};
use itertools::Itertools;
use num_traits::{One, Zero};

/// A chart in an atlas of a manifold.
///
/// The space of all values of a type `C: Chart<P, V>` is interpreted
/// as an atlas of the manifold `M` (the space of `P: Point`), covering
/// it with local coordinate neighbourhoods modelled on `R^N` (`Coords`).
/// The atlas axiom — that every point is covered — is expressed by
/// `chart_at`, which must produce a valid chart in a neighbourhood of
/// any given point.
///
/// `to_local` and `to_global` are the coordinate maps, with `to_local`
/// returning `None` at the singularities of the chart.
pub trait Chart<P: Point, V: Euclidean>: Sized {
    fn to_local(&self, point: &P) -> Option<V>;
    fn to_global(&self, coord: V) -> P;
    fn chart_at(p: &P) -> Self;

    /// Calculates the distance between `self` and `other`
    /// in local coordinates, based at &self.
    fn local_distance(&self, other: &P) -> Option<V::F> {
        self.to_local(other).map(|v| v.norm())
    }

    #[cfg(feature = "testing")]
    fn check_local_inverse(p: &P) -> bool
    where
        P: PartialEq,
    {
        let chart = Self::chart_at(p);
        match chart.to_local(p) {
            Some(local) => p == &chart.to_global(local),
            None => false,
        }
    }
}

/// By implementing ExpMap you certify that for C<P, V>: ExpMap<P, V> that
/// straight lines through the origin in R^N map to geodesics on M, and
/// that distances from the origin equal arc lengths along those geodesics.
///
/// Additionally, you certify that Self::chart_at(&self.base_point()) == self
pub trait ExpMap<P: Point, V: Euclidean>: Chart<P, V> {
    fn base_point(&self) -> P {
        self.to_global(V::zero())
    }

    // Tests that base_point() is consistent with to_local.
    // Meaningful only when base_point() is overridden, since
    // the default impl makes this trivially true by construction.
    #[cfg(feature = "testing")]
    fn check_base_point_is_origin(&self) -> bool {
        self.to_local(&self.base_point())
            .map_or(false, |c| c.norm() == V::F::zero())
    }

    // Tests that log(exp(0)) == 0, i.e. that the
    // round trip at the origin is the identity.
    #[cfg(feature = "testing")]
    fn check_preservation_of_origin(&self) -> bool {
        let zero = V::zero();
        let exp_zero = self.to_global(zero);
        self.to_local(&exp_zero)
            .map_or(false, |c| c.norm() == V::F::zero())
    }

    /// If a chart centred at `p` exists, `chart_at(p)` returns it.
    /// Formally: `chart_at(p).base_point() == p` whenever `p` is
    /// the base point of some valid chart in this atlas.
    ///
    /// This is weaker than the [`TangentBundle`] centring invariant,
    /// which requires this to hold for *all* `p`. Here it is only
    /// required when `p` is already a base point of some chart —
    /// i.e. `chart_at` correctly identifies the chart when queried
    /// at a known base point.
    #[cfg(feature = "testing")]
    fn check_chart_at_base_point(&self) -> bool {
        Self::chart_at(&self.base_point()).check_preservation_of_origin()
    }

    // geodesics are reversible: log(exp(v)) == -log(exp(-v))
    #[cfg(feature = "testing")]
    fn check_geodesic_symmetry(&self, v: V) -> bool
    where
        V: PartialEq,
    {
        match (
            self.to_local(&self.to_global(v)),
            self.to_local(&self.to_global(-v)),
        ) {
            (Some(fwd), Some(bwd)) => fwd == -bwd,
            _ => true, // at singularity, vacuously true
        }
    }

    // geodesics are straight lines: exp(tv) lies on the same geodesic as exp(v),
    // i.e. log(exp(tv)) and log(exp(v)) are parallel in local coords.
    #[cfg(feature = "testing")]
    fn check_geodesic_scaling(&self, v: V, t: V::F) -> bool {
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
                lhs == rhs
            }
            _ => true,
        }
    }

    // isometry to first order: |exp(epsilon * v) - base| / epsilon → |v|
    #[cfg(feature = "testing")]
    fn check_first_order_isometry(&self, v: V, coef: V::F) -> bool {
        let small_v = v * coef;
        self.to_local(&self.to_global(small_v))
            .map_or(true, |local| local.norm() == v.norm() * coef)
    }
}

/// A tangent bundle structure on a manifold.
///
/// The space of all values of a type `C: TangentBundle<P, V>` is
/// interpreted as the tangent bundle `TM` of the manifold `M` (the space
/// of `P: Point`). Each instance is a single tangent space `T_p M` at
/// the base point `p`, coordinatised by `V`.
///
/// By implementing `TangentBundle` you certify that for all `p: P`:
/// `C::chart_at(&p).to_global(V::zero()) == p`
///
/// That is, the chart produced at any point is centred at that point —
/// the origin of the local coordinate system corresponds to the base point
/// on the manifold. This is what distinguishes a tangent bundle from a
/// bare [`Chart`] or [`ExpMap`].
///
/// Use the `test_tangent_bundle!` macro to verify this invariant.
pub trait TangentBundle<P: Point, V: Euclidean>: ExpMap<P, V> {
    fn sectional_curvature(&self, v: V, w: V, epsilon: V::F) -> V::F {
        let p2 = self.to_global(v + w * epsilon);
        let actual = (self.to_local(&p2).unwrap() - v).norm();
        let flat = (w * epsilon).norm();

        let two = V::F::one() + V::F::one();
        (two / (epsilon * epsilon)) * (V::F::one() - actual / flat)
    }

    fn max_sectional_curvature(&self, epsilon: V::F) -> V::F {
        (0..V::N)
            .array_combinations::<2>()
            .map(|[i, j]| {
                let v = V::from_fn(|k| if k == i { V::F::one() } else { V::F::zero() });
                let w = V::from_fn(|k| if k == j { V::F::one() } else { V::F::zero() });
                self.sectional_curvature(v, w, epsilon)
            })
            .fold(V::F::zero(), |max, k| if k > max { k } else { max })
    }

    // p is the point on the manifold which is the base point.
    #[cfg(feature = "testing")]
    fn check_universal_centring(p: P) -> bool {
        let chart = Self::chart_at(&p);
        chart.check_preservation_of_origin() && chart.check_base_point_is_origin()
    }
}

/// Intrinsic smooth structure on a manifold.
///
/// A type implementing `Smooth<V>` carries its own smooth structure:
/// every point determines a canonical chart centred at itself via `exp`
/// (the exponential map) and `log` (its inverse). This is the
/// self-charting case — no external atlas type is needed.
///
/// Implementing `Smooth<V>` automatically provides [`Chart<Self, V>`],
/// [`ExpMap<Self, V>`], and [`TangentBundle<Self, V>`] via blanket
/// implementations, so `exp` and `log` are the only methods an
/// implementor needs to write.
///
/// Implement `Smooth` for manifolds whose geodesic structure is
/// intrinsically determined but which are not Lie groups — spheres
/// of any dimension, hyperbolic spaces, and similar. For Lie groups,
/// implement [`LieGroup`] instead; a blanket implementation derives
/// `Smooth` from the group operation via left translation.
///
/// [`Chart<Self, V>`]: crate::traits::Chart
/// [`ExpMap<Self, V>`]: crate::traits::ExpMap
/// [`TangentBundle<Self, V>`]: crate::traits::TangentBundle
/// [`LieGroup`]: crate::traits::LieGroup
pub trait Smooth<V: Euclidean>: Point {
    /// The exponential map at `self`: sends a tangent vector `v` to the
    /// point reached by following the geodesic from `self` in direction
    /// `v` for unit time.
    fn exp(&self, v: V) -> Self;

    /// The logarithmic map at `self`: recovers the tangent vector whose
    /// geodesic reaches `other`, or `None` at the cut locus (e.g. the
    /// antipode on a sphere).
    fn log(&self, other: &Self) -> Option<V>;
}

impl<V: Euclidean, S: Smooth<V>> Chart<Self, V> for S {
    fn to_local(&self, point: &Self) -> Option<V> {
        self.log(point)
    }

    fn to_global(&self, coord: V) -> Self {
        self.exp(coord)
    }

    fn chart_at(p: &Self) -> Self {
        p.clone()
    }
}

impl<V: Euclidean, L: Smooth<V>> ExpMap<Self, V> for L {
    // optimisation
    fn base_point(&self) -> Self {
        self.clone()
    }
}

impl<V: Euclidean, L: Smooth<V>> TangentBundle<Self, V> for L {}
