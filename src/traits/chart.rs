use crate::traits::{Bilinear, DivRing, Euclidean, Field, Form, Interval, Real, Vector};

use super::Point;
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
pub trait Chart<P: Point, V: Vector>: Sized {
    fn to_local(&self, point: &P) -> Option<V>;
    fn to_global(&self, coord: V) -> P;
    fn chart_at(p: &P) -> Self;

    /// Calculates the distance between `self` and `other`
    /// in local coordinates, based at &self.
    fn local_distance(&self, other: &P) -> Option<<V::F as Field>::Fixed>
    where
        V: Euclidean,
    {
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
pub trait ExpMap<P: Point, V: Vector>: Chart<P, V> {
    fn base_point(&self) -> P {
        self.to_global(V::zero())
    }

    // Tests that base_point() is consistent with to_local.
    // Meaningful only when base_point() is overridden, since
    // the default impl makes this trivially true by construction.
    #[cfg(feature = "testing")]
    fn check_base_point_is_origin(&self) -> bool
    where
        V: Form,
    {
        self.to_local(&self.base_point())
            .map_or(false, |c| c.self_dot() == V::F::zero())
    }

    // Tests that log(exp(0)) == 0, i.e. that the
    // round trip at the origin is the identity.
    #[cfg(feature = "testing")]
    fn check_preservation_of_origin(&self) -> bool
    where
        V: Form,
    {
        let zero = V::zero();
        let exp_zero = self.to_global(zero);
        self.to_local(&exp_zero)
            .map_or(false, |c| c.self_dot() == V::F::zero())
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
    fn check_chart_at_base_point(&self) -> bool
    where
        V: Form,
    {
        Self::chart_at(&self.base_point()).check_preservation_of_origin()
    }

    // geodesics are reversible: log(exp(v)) == -log(exp(-v))
    #[cfg(feature = "testing")]
    fn check_geodesic_symmetry(&self, v: V) -> bool
    where
        V: Form + PartialEq,
    {
        let fwd = match self.to_local(&self.to_global(v)) {
            Some(x) => x,
            None => return true,
        };
        let bwd = match self.to_local(&self.to_global(-v)) {
            Some(x) => x,
            None => return true,
        };
        // gate: skip if either geodesic wrapped (folded norm ≠ input norm)
        if fwd.self_dot() != v.self_dot() || bwd.self_dot() != (-v).self_dot() {
            return true;
        }
        fwd == -bwd
    }

    // geodesics are straight lines: exp(tv) lies on the same geodesic as exp(v),
    // i.e. log(exp(tv)) and log(exp(v)) are parallel in local coords.
    #[cfg(feature = "testing")]
    fn check_geodesic_scaling(&self, v: V, t: <V::F as Field>::Fixed) -> bool
    where
        V: Form,
    {
        let t_as_f = V::F::from_fixed(t);
        let v_local = match self.to_local(&self.to_global(v)) {
            Some(x) => x,
            None => return true,
        };
        let tv_local = match self.to_local(&self.to_global(v * t_as_f)) {
            Some(x) => x,
            None => return true,
        };
        // Gate: did either geodesic wrap? exp parametrises by arc length, so
        // ‖log(exp(w))‖ ≤ ‖w‖ always, with equality iff no wrapping. If the
        // folded coord is shorter than the input, it wrapped — skip.
        if v_local.self_dot() != v.dot(&v) || tv_local.self_dot() != (v * t_as_f).self_dot() {
            return true;
        }
        let dot = tv_local.dot(&v_local);
        dot * dot == tv_local.self_dot() * v_local.self_dot()
    }
}

/// A manifold whose exponential map agrees with its scalar product — a
/// pseudo-Riemannian manifold.
///
/// [`ExpMap`] supplies geodesics and exponential coordinates; [`Bilinear`]
/// supplies the (possibly indefinite) scalar product on the tangent space.
/// This trait certifies that the two *coincide*: the signed interval of the
/// geodesic from `p` in direction `v` equals the value of the quadratic form
/// `Q(v) = ⟨v,v⟩`. Equivalently, `exp` is a radial isometry of the scalar
/// product — `Q(log_p(exp_p v)) = Q(v)` — within the injectivity domain.
///
/// This is stated on the **signed** form rather than a distance, because a
/// pseudo-Riemannian manifold need not be a metric space: for timelike `v`,
/// `Q(v) < 0` and the invariant is (minus) the squared proper time; for
/// spacelike `v`, the squared proper distance; for null `v`, zero. No `sqrt`
/// and no non-negativity is assumed, so the check is valid in any signature.
///
/// In the definite (`M = 0`) case this reduces to the usual Riemannian
/// statement `d(p, exp_p v) = ‖v‖`, recovered by taking `√Q`.
///
/// Verified by `test_pseudo_riemannian!`.
///
/// [`Bilinear`]: crate::traits::Bilinear
pub trait PseudoRiemannian<V: Bilinear<F: Real>>: ExpMap<Self, V> + Interval<R = V::F> {
    #[cfg(feature = "testing")]
    fn check_isometry(&self, v: V) -> bool {
        let global = self.to_global(v);
        // Re-log: the wrapped representative, guaranteed inside the injectivity
        // domain. On compact manifolds exp isn't injective, so |v| itself may
        // exceed the injectivity radius and NOT equal the interval — but
        // log(exp(v)) does.
        let local = match self.to_local(&global) {
            Some(u) => u,
            None => return true, // outside restricted log domain — skip
        };
        let s = self.base_point().interval(&global);

        s * s == local.norm_squared().into() // signed interval vs re-logged tangent form
    }
}

impl<V: Bilinear<F: Real>, E: ExpMap<Self, V> + Interval<R = V::F>> PseudoRiemannian<V> for E {}

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
pub trait TangentBundle<P: Point, V: Vector>: ExpMap<P, V> {
    fn sectional_curvature(&self, v: V, w: V, epsilon: V::F) -> Option<V::F>
    where
        V: Form,
    {
        // Denominator: signed Gram determinant of the 2-plane span(v, w).
        //   G = Q(v)·Q(w) − ⟨v,w⟩²
        // Zero ⟺ the plane is degenerate (contains a null direction, or v,w
        // dependent). Sectional curvature is undefined there — return None.
        let qv = v.self_dot();
        let qw = w.self_dot();
        let vw = v.dot(&w);
        let gram = qv * qw - vw * vw;
        if gram == V::F::zero() {
            return None;
        }

        // Deviation vector: how the exp-image of the perturbed direction differs
        // from the flat prediction, pulled back to the tangent space.
        //   δ = log_p( exp_p(v + ε w) ) − v
        let perturbed = self.to_global(v + w * epsilon);
        let delta = self.to_local(&perturbed)? - v; // None if outside injectivity domain

        // Second-order metric defect. In flat space Q(δ) = ε²·Q(w) exactly;
        // curvature is the O(ε⁴) correction:
        //   Q(δ) = ε²·Q(w) − (1/3)·ε⁴·⟨R(w,v)v,w⟩ + O(ε⁵)
        // Solve for the numerator ⟨R(w,v)v,w⟩.
        let q_delta = delta.self_dot();
        let eps2 = epsilon * epsilon;
        let three = V::F::one() + V::F::one() + V::F::one();
        let numerator = three * (eps2 * qw - q_delta).div(eps2 * eps2);

        Some(numerator.div(gram))
    }

    fn max_sectional_curvature(&self, epsilon: V::F) -> Option<V::F>
    where
        V: Euclidean,
    {
        (0..V::N)
            .array_combinations::<2>()
            .filter_map(|[i, j]| {
                let v = V::from_fn(|k| if k == i { V::F::one() } else { V::F::zero() });
                let w = V::from_fn(|k| if k == j { V::F::one() } else { V::F::zero() });
                self.sectional_curvature(v, w, epsilon)
            })
            .reduce(|max, k| if k > max { k } else { max })
    }

    // p is the point on the manifold which is the base point.
    #[cfg(feature = "testing")]
    fn check_universal_centring(p: P) -> bool
    where
        V: Form,
    {
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
pub trait Smooth<V: Vector>: Point {
    /// The exponential map at `self`: sends a tangent vector `v` to the
    /// point reached by following the geodesic from `self` in direction
    /// `v` for unit time.
    fn exp(&self, v: V) -> Self;

    /// The logarithmic map at `self`: recovers the tangent vector whose
    /// geodesic reaches `other`, or `None` at the cut locus (e.g. the
    /// antipode on a sphere).
    fn log(&self, other: &Self) -> Option<V>;
}

impl<V: Vector, S: Smooth<V>> Chart<Self, V> for S {
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

impl<V: Vector, L: Smooth<V>> ExpMap<Self, V> for L {
    // optimisation
    fn base_point(&self) -> Self {
        self.clone()
    }
}

impl<V: Vector, L: Smooth<V>> TangentBundle<Self, V> for L {}
