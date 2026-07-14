use std::{marker::PhantomData, ops::Mul};

use crate::{
    complex::Complex,
    coords::Coords,
    impl_group_via_mul, impl_lie_group_via_quotient, impl_tangent_bundle_via_bounded,
    traits::{
        Bounded, BuildNodes, Chart, Euclidean, ExpMap, InnerProduct, Interval, LieGroup, Metric,
        NerveComplexParameters, Quotient, Real, Smooth, TangentBundle,
    },
};

use num_traits::{Inv, NumCast, One, Zero, real::Real as _};

#[derive(Debug, PartialEq, Clone)]
pub struct Sphere<const N: usize, V: Euclidean> {
    real: V::F,
    imag: V,
}

#[derive(Clone, Debug)]
pub struct Stereographic<V: Euclidean>(StereographicPole, PhantomData<V>);

impl<V: Euclidean> Stereographic<V> {
    pub const fn south_pole() -> Self {
        Self(StereographicPole::SouthPole, PhantomData)
    }
    pub const fn north_pole() -> Self {
        Self(StereographicPole::NorthPole, PhantomData)
    }
}

#[derive(Clone, Debug)]
enum StereographicPole {
    SouthPole,
    NorthPole,
}

pub const EPSILON: f64 = 1e-3;

impl<const N: usize, V: Euclidean> Chart<Sphere<N, V>, V> for Stereographic<V> {
    fn to_local(&self, point: &Sphere<N, V>) -> Option<V> {
        let first = match self.0 {
            StereographicPole::NorthPole => point.real,
            StereographicPole::SouthPole => -point.real,
        };

        let epsilon = <V::F as NumCast>::from(EPSILON).unwrap();

        let denom = V::F::one() - first;
        if denom.abs() < epsilon {
            return None;
        } // at north pole

        let recip = denom.recip();
        Some(point.imag * recip)
    }

    fn to_global(&self, coord: V) -> Sphere<N, V> {
        let two = V::F::one() + V::F::one();
        let r_sq = coord.norm_squared();
        let denom = V::F::one() + r_sq;
        Sphere::new(
            match self.0 {
                StereographicPole::NorthPole => (r_sq - V::F::one()) / denom,
                StereographicPole::SouthPole => (V::F::one() - r_sq) / denom,
            },
            coord * (two / denom),
        )
    }

    fn chart_at(p: &Sphere<N, V>) -> Self {
        if p.real > V::F::zero() {
            Self::south_pole()
        } else {
            Self::north_pole()
        }
    }
}

impl<const N: usize, V: Euclidean> Sphere<N, V> {
    pub fn real(&self) -> V::F {
        self.real
    }
    pub fn imag(&self) -> V {
        self.imag
    }

    fn normalised(self) -> Self {
        let real = self.real;
        let imag = self.imag;
        let sum = real * real + imag.iter().fold(V::F::zero(), |acc, &v| acc + v * v);

        assert!(sum != V::F::zero());
        let q_rsqrt = V::F::sqrt(sum).recip(); // What the f***?

        Self {
            real: real * q_rsqrt,
            imag: imag * q_rsqrt,
        }
    }

    fn identity() -> Self {
        Sphere::new(V::F::one(), V::zero())
    }

    fn is_identity(&self) -> bool {
        self.real.is_zero() && self.imag.is_zero()
    }

    pub fn new(real: V::F, imag: V) -> Self {
        let sphere = Sphere { real, imag };

        sphere.normalised()
    }
}

impl<const N: usize, V: Euclidean> Smooth<V> for Sphere<N, V> {
    fn exp(&self, v: V) -> Self {
        let eps = <V::F as NumCast>::from(EPSILON).unwrap();

        // identity-frame exp, centred at +e0: (cos α, v · sinc α)
        let alpha = v.norm();

        let (sin_a, cos_a) = alpha.sin_cos();
        let sinc = sinc_from(alpha, sin_a, eps);

        // transport the identity-frame point to self's frame
        self.transport_from_identity(cos_a, v * sinc)
    }

    fn log(&self, other: &Self) -> Option<V> {
        let one = V::F::one();
        let eps = <V::F as NumCast>::from(EPSILON).unwrap();

        // transport `other` into the +e0 identity frame
        let p = self.transport_to_identity(other.real, other.imag);

        // identity-frame log: invert (cos α, v · sinc α)
        if (p.real + one).abs() < eps {
            return None; // antipodal to self: cut locus
        }
        let alpha = V::F::atan2(p.imag.norm(), p.real);

        let sinc_recip = sinc_recip(alpha, eps);
        Some(p.imag * sinc_recip)
    }
}

/// The cardinal sine `sin(α)/α`, with a Taylor fallback near zero to
/// avoid the `0/0` at the origin.
///
/// Series: `sin(α)/α = 1 − α²/6 + α⁴/120 − …`
/// The two-term approximation `1 − α²/6` is used for `α < eps`; its
/// error there is the dropped `α⁴/120` term (~8×10⁻¹⁵ at eps = 1e-3),
/// far below the R64 tolerance.
fn sinc_from<F: Real>(alpha: F, sin_a: F, eps: F) -> F {
    let one = F::one();
    if alpha < eps {
        let six = (one + one) * (one + one + one);
        one - alpha * alpha / six
    } else {
        sin_a / alpha
    }
}

/// The reciprocal cardinal sine `α/sin(α)`, with a Taylor fallback near
/// zero.
///
/// Series: `α/sin(α) = 1 + α²/6 + 7α⁴/360 + …`
/// Note this is **not** a sign-flipped copy of [`sinc`]'s series: only the
/// α² term flips sign; the α⁴ coefficient is `7/360`, not `±1/120`
/// (because `1/(1−x) ≠ 1 ∓ x` beyond first order). The two-term
/// approximation `1 + α²/6` is used for `α < eps`; its error there is the
/// dropped `7α⁴/360` term (~2×10⁻¹⁴ at eps = 1e-3), below the R64
/// tolerance.
fn sinc_recip<F: Real>(alpha: F, eps: F) -> F {
    let one = F::one();
    if alpha < eps {
        let six = (one + one) * (one + one + one);
        one + alpha * alpha / six
    } else {
        alpha / alpha.sin()
    }
}

impl<const N: usize, V: Euclidean> Sphere<N, V> {
    // s = -sign(self.real): reflect from the far pole (no self.real∓1 cancellation).
    fn far_pole_sign(&self) -> V::F {
        if self.real > V::F::zero() {
            -V::F::one()
        } else {
            V::F::one()
        }
    }

    // Householder swapping self ↔ s·e0, applied to (x_real, x_imag).
    fn reflect(&self, s: V::F, x_real: V::F, x_imag: V) -> (V::F, V) {
        let two = V::F::one() + V::F::one();
        let u_real = self.real - s; // = self.real ∓ 1, but s is the FAR pole so no cancellation
        let u_imag = self.imag;
        let u_dot_u = u_real * u_real + u_imag.norm_squared(); // ≥ 2
        let u_dot_x = u_real * x_real + u_imag.dot(&x_imag);
        let c = two * u_dot_x / u_dot_u;
        (x_real - c * u_real, x_imag - u_imag * c)
    }

    // self-frame → +e0 identity frame  (used by log)
    fn transport_to_identity(&self, x_real: V::F, x_imag: V) -> Self {
        let s = self.far_pole_sign();
        let (r, im) = self.reflect(s, x_real, x_imag); // self → s·e0
        if s < V::F::zero() {
            Sphere::new(-r, im)
        } else {
            Sphere::new(r, im)
        } // F if s=-1
    }

    // +e0 identity frame → self-frame  (used by exp): inverse of to_identity
    fn transport_from_identity(&self, x_real: V::F, x_imag: V) -> Self {
        let s = self.far_pole_sign();
        // inverse: apply F first (if s=-1), then H
        let (x_real, x_imag) = if s < V::F::zero() {
            (-x_real, x_imag)
        } else {
            (x_real, x_imag)
        };
        let (r, im) = self.reflect(s, x_real, x_imag);
        Sphere::new(r, im)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct S0<V: Euclidean>(pub Sphere<0, V>);
impl_group_via_mul!(S0<V>, V: Euclidean);

#[derive(Debug, Clone, PartialEq)]
pub struct UnitComplex<V: Euclidean>(pub Sphere<1, V>);
impl_group_via_mul!(UnitComplex<V>, V: Euclidean);

#[derive(Debug, Clone, PartialEq)]
pub struct S3<V: Euclidean>(pub Sphere<3, V>);
impl_group_via_mul!(S3<V>, V: Euclidean);

impl<V: Euclidean> Interval<V::F> for S0<V> {
    fn interval(&self, other: &Self) -> Complex<V::F> {
        self.0.interval(&other.0)
    }
}
impl<V: Euclidean> Metric<V::F> for S0<V> {}

impl<V: Euclidean> Interval<V::F> for UnitComplex<V> {
    fn interval(&self, other: &Self) -> Complex<V::F> {
        self.0.interval(&other.0)
    }
}
impl<V: Euclidean> Metric<V::F> for UnitComplex<V> {}

impl<V: Euclidean> Interval<V::F> for S3<V> {
    fn interval(&self, other: &Self) -> Complex<V::F> {
        self.0.interval(&other.0)
    }
}
impl<V: Euclidean> Metric<V::F> for S3<V> {}

impl<V: Euclidean> One for S0<V> {
    fn one() -> Self {
        Self(Sphere::identity())
    }

    fn is_one(&self) -> bool {
        self.0.is_identity()
    }
}

impl<V: Euclidean> Mul for S0<V> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(Sphere::new(self.0.real * rhs.0.real, V::zero()))
    }
}

impl<V: Euclidean> Inv for S0<V> {
    type Output = Self;

    fn inv(self) -> Self::Output {
        Self(Sphere::new(self.0.real, V::zero()))
    }
}

impl<V: Euclidean> LieGroup<V> for S0<V> {
    fn identity_exp(_: V) -> Self {
        Self::one()
    }

    fn identity_log(p: &Self) -> Option<V> {
        if p.0.real > V::F::zero() {
            Some(V::zero())
        } else {
            None
        }
    }
}

impl<V: Euclidean> One for UnitComplex<V> {
    fn one() -> Self {
        Self(Sphere::identity())
    }

    fn is_one(&self) -> bool {
        self.0.is_identity()
    }
}

impl<V: Euclidean> Mul for UnitComplex<V> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let (a1, [b1]) = (self.0.real, self.0.imag.to_array());
        let (a2, [b2]) = (rhs.0.real, rhs.0.imag.to_array());

        Self(Sphere::new(
            a1 * a2 - b1 * b2,
            V::from_array([a1 * b2 + a2 * b1]),
        ))
    }
}

impl<V: Euclidean> Inv for UnitComplex<V> {
    type Output = Self;

    fn inv(self) -> Self::Output {
        Self(Sphere::new(self.0.real, -self.0.imag))
    }
}

impl<V: Euclidean> LieGroup<V> for UnitComplex<V> {
    fn identity_exp(v: V) -> Self {
        let alpha = v[0];

        UnitComplex(Sphere::new(alpha.cos(), V::from_array([alpha.sin()])))
    }

    fn identity_log(p: &Self) -> Option<V> {
        Some(V::from_array([V::F::atan2(p.0.imag[0], p.0.real)]))
    }
}

impl<V: Euclidean> One for S3<V> {
    fn one() -> Self {
        Self(Sphere::identity())
    }

    fn is_one(&self) -> bool {
        self.0.is_identity()
    }
}

impl<V: Euclidean> Mul for S3<V> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let (a1, [b1, c1, d1]) = (self.0.real, self.0.imag.to_array());
        let (a2, [b2, c2, d2]) = (rhs.0.real, rhs.0.imag.to_array());
        Self(Sphere::new(
            a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
            V::from_array([
                a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
                a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
            ]),
        ))
    }
}

impl<V: Euclidean> Inv for S3<V> {
    type Output = Self;

    fn inv(self) -> Self::Output {
        let (a, [b, c, d]) = (self.0.real, self.0.imag.to_array());

        Self(Sphere::new(a, V::from_array([-b, -c, -d])))
    }
}

impl<V: Euclidean> LieGroup<V> for S3<V> {
    fn identity_exp(v: V) -> Self {
        let alpha = V::F::sqrt(v.iter().fold(V::F::zero(), |acc, &x| acc + x * x));
        let (sin, cos) = alpha.sin_cos();

        let sinc = sinc_from(alpha, sin, <V::F as NumCast>::from(EPSILON).unwrap());
        Self(Sphere::new(cos, v * sinc))
    }

    fn identity_log(p: &Self) -> Option<V> {
        let eps = <V::F as NumCast>::from(EPSILON).unwrap();
        if (p.0.real + V::F::one()).abs() < eps {
            return None; // antipode singularity
        }
        // use atan2 instead of acos for numerical stability
        let imag_norm = p.0.imag.norm();
        let alpha = V::F::atan2(imag_norm, p.0.real);

        let sinc_recip = sinc_recip(alpha, eps);
        Some(p.0.imag * sinc_recip)
    }
}

impl<const N: usize, V: Euclidean> Interval<V::F> for Sphere<N, V> {
    fn interval(&self, other: &Self) -> Complex<V::F> {
        // ambient inner product of two unit vectors = cos(geodesic distance)
        let cos_d = self.real * other.real + self.imag.dot(&other.imag);
        // the perpendicular component gives sin(geodesic distance):
        // ‖q − (p·q)p‖ = sin(θ)
        let w_real = other.real - cos_d * self.real;
        let w_imag = other.imag - self.imag * cos_d;
        let sin_d = (w_real * w_real + w_imag.norm_squared()).sqrt();
        // θ = atan2(sin, cos), stable everywhere including antipode (θ=π)
        V::F::atan2(sin_d, cos_d).into()
    }
}
impl<const N: usize, V: Euclidean> Metric<V::F> for Sphere<N, V> {}

#[derive(Clone, Debug, PartialEq)]
pub struct So3<V: Euclidean>(S3<V>);

impl<V: Euclidean> Quotient<S3<V>, S0<V>, V> for So3<V> {
    fn new(g: S3<V>) -> Self {
        // lexographic ordering on the fields
        match g
            .0
            .real()
            .partial_cmp(&V::F::zero())
            .unwrap()
            .then(g.0.imag().iter().partial_cmp(V::zero().iter()).unwrap())
        {
            std::cmp::Ordering::Less => So3(S3(Sphere::new(-g.0.real(), -g.0.imag()))),
            std::cmp::Ordering::Equal | std::cmp::Ordering::Greater => So3(g),
        }
    }

    fn lift(&self) -> S3<V> {
        self.0.clone()
    }

    fn embed(h: S0<V>) -> S3<V> {
        S3(Sphere::new(h.0.real(), V::zero()))
    }
}

impl_lie_group_via_quotient!(So3<V>, S3<V>, S0<V>);

use crate::epsilon_metric::R64;

#[derive(PartialEq, Debug, Clone)]
pub struct S1Cover(UnitComplex<Coords<R64, 1>>);

impl Bounded<UnitComplex<Coords<R64, 1>>, UnitComplex<Coords<R64, 1>>, Coords<R64, 1>> for S1Cover {
    // Each node's domain is the open arc of radius ρ = π/6 + 0.05 about its
    // base point. Six such arcs centred at the sixth roots of unity form an
    // open good cover of S¹:
    //   - covering:   arcs of half-length ρ > π/6 centred π/3 apart cover S¹
    //   - goodness:   arcs and their pairwise intersections are arcs (or
    //                 empty), hence contractible
    //   - nerve:      adjacent arcs (d = π/3 < 2ρ ≈ 1.147) overlap;
    //                 next-nearest (d = 2π/3 > 2ρ) do not — the nerve is a
    //                 hexagon, whose π₁ is free on one generator: π₁(S¹) = Z
    fn sdf(&self, v: &Coords<R64, 1>) -> R64 {
        v.norm() - R64(std::f64::consts::PI / 6.0 + 0.05)
    }
}

impl From<UnitComplex<Coords<R64, 1>>> for S1Cover {
    fn from(value: UnitComplex<Coords<R64, 1>>) -> Self {
        Self(value)
    }
}

impl AsRef<UnitComplex<Coords<R64, 1>>> for S1Cover {
    fn as_ref(&self) -> &UnitComplex<Coords<R64, 1>> {
        &self.0
    }
}

impl_tangent_bundle_via_bounded!(
    S1Cover, UnitComplex<Coords<R64, 1>>, UnitComplex<Coords<R64, 1>>, Coords<R64, 1>,
);

impl BuildNodes<S1Cover> for S1Cover {
    fn build_nodes() -> Vec<Self> {
        (0..6)
            .map(|i| {
                let angle: R64 = R64(i.into()) * R64(std::f64::consts::TAU) / R64(6.0);
                S1Cover(UnitComplex(Sphere::new(angle.cos(), [angle.sin()].into())))
            })
            .collect()
    }
}

impl
    NerveComplexParameters<
        UnitComplex<Coords<R64, 1>>,
        Coords<R64, 1>,
        UnitComplex<Coords<R64, 1>>,
        S1Cover,
    > for S1Cover
{
}

#[derive(PartialEq, Debug, Clone)]
pub struct So3Cover(So3<Coords<R64, 3>>);

impl Chart<So3<Coords<R64, 3>>, Coords<R64, 3>> for So3Cover {
    fn to_local(&self, point: &So3<Coords<R64, 3>>) -> Option<Coords<R64, 3>> {
        self.0.to_local(point)
    }
    fn to_global(&self, coord: Coords<R64, 3>) -> So3<Coords<R64, 3>> {
        self.0.to_global(coord)
    }
    fn chart_at(p: &So3<Coords<R64, 3>>) -> Self {
        Self(So3::chart_at(p))
    }
}

impl ExpMap<So3<Coords<R64, 3>>, Coords<R64, 3>> for So3Cover {}

impl TangentBundle<So3<Coords<R64, 3>>, Coords<R64, 3>> for So3Cover {}

/// Radius of the geodesic-ball domains of [`So3Cover`].
///
/// The 60 nodes are the icosahedral rotation group I ≅ A₅ ⊂ SO(3) — the
/// image of the 120 icosian unit quaternions (the vertices of the 600-cell)
/// under the double cover S³ → SO(3). In the bi-invariant metric
/// `d = |identity_log|` (half the rotation angle; diameter π/2), the
/// pairwise distances realised between nodes are exactly
///
/// ```text
///   π/5 ≈ 0.628,   π/3 ≈ 1.047,   2π/5 ≈ 1.257,   π/2 ≈ 1.571
/// ```
///
/// and the covering radius of the node set is ≈ 0.3857 (the circumradius
/// of a cell of the 600-cell). The radius ρ = 0.42 is chosen so that:
///
/// - **covering**: ρ > 0.3857, so the 60 open balls cover SO(3);
/// - **goodness**: ρ < π/4, the convexity radius of SO(3) ≅ RP³, so every
///   ball is geodesically convex and all intersections of balls are convex,
///   hence contractible or empty — an open *good* cover;
/// - **faithful 1-skeleton**: two equal balls overlap iff their centres are
///   closer than 2ρ = 0.84, which separates π/5 from π/3 with a wide margin
///   on both sides — the nerve's edges are exactly the 600-cell's edges
///   (mod ±1), and the computation is robust to floating-point error;
/// - **faithful 2-skeleton**: every triangle of the overlap graph is an
///   equilateral triangle of side π/5 with spherical circumradius ≈ 0.365
///   < ρ, so all three balls genuinely share a point — mutual pairwise
///   overlap coincides with triple intersection, and the triangles of the
///   nerve are exactly the 600-cell's 2-faces (mod ±1).
///
/// The nerve of this cover is therefore the *hemi-600-cell*: the classical
/// vertex-transitive 60-vertex triangulation of RP³ with f-vector
/// (60, 360, 600, 300), obtained from the boundary complex of the 600-cell
/// by identifying antipodes. By the nerve theorem the nerve is homotopy
/// equivalent to SO(3), and π₁ computed from its 2-skeleton is
/// ⟨x | x²⟩ ≅ Z/2Z.
impl Bounded<So3<Coords<R64, 3>>, So3<Coords<R64, 3>>, Coords<R64, 3>> for So3Cover {
    // Open geodesic ball of radius 0.42 about the base point.
    // In an exponential chart the geodesic distance from the base point is
    // exactly the coordinate norm, so the ball's true signed distance field
    // is radial.
    fn sdf(&self, v: &Coords<R64, 3>) -> R64 {
        v.norm() - R64(0.42)
    }
}

impl From<So3<Coords<R64, 3>>> for So3Cover {
    fn from(value: So3<Coords<R64, 3>>) -> Self {
        Self(value)
    }
}

impl AsRef<So3<Coords<R64, 3>>> for So3Cover {
    fn as_ref(&self) -> &So3<Coords<R64, 3>> {
        &self.0
    }
}

impl BuildNodes<Self> for So3Cover {
    fn build_nodes() -> Vec<Self> {
        // The 120 icosians: vertices of the 600-cell on S³.
        let phi = (1.0 + 5f64.sqrt()) / 2.0;
        let mut quats: Vec<[f64; 4]> = Vec::new();

        // 8 unit quaternions: ±1, ±i, ±j, ±k
        for i in 0..4 {
            for s in [-1.0, 1.0] {
                let mut q = [0.0; 4];
                q[i] = s;
                quats.push(q);
            }
        }
        // 16: (±1 ± i ± j ± k)/2
        for a in [-0.5, 0.5] {
            for b in [-0.5, 0.5] {
                for c in [-0.5, 0.5] {
                    for d in [-0.5, 0.5] {
                        quats.push([a, b, c, d]);
                    }
                }
            }
        }
        // 96: all even permutations of (±φ, ±1, ±1/φ, 0)/2
        let even_perms: [[usize; 4]; 12] = [
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2],
            [1, 0, 3, 2],
            [1, 2, 0, 3],
            [1, 3, 2, 0],
            [2, 0, 1, 3],
            [2, 1, 3, 0],
            [2, 3, 0, 1],
            [3, 0, 2, 1],
            [3, 1, 0, 2],
            [3, 2, 1, 0],
        ];
        let base = [phi / 2.0, 0.5, 1.0 / (2.0 * phi), 0.0];
        for p in even_perms {
            for s0 in [-1.0, 1.0] {
                for s1 in [-1.0, 1.0] {
                    for s2 in [-1.0, 1.0] {
                        let vals = [s0 * base[0], s1 * base[1], s2 * base[2], base[3]];
                        let mut q = [0.0; 4];
                        for i in 0..4 {
                            q[p[i]] = vals[i];
                        }
                        quats.push(q);
                    }
                }
            }
        }
        debug_assert_eq!(quats.len(), 120);

        // Quotient by ±1: canonicalise the sign (first non-zero
        // coordinate positive) and deduplicate, leaving one
        // representative per rotation — 60 in total.
        let mut seen = std::collections::HashSet::new();
        let mut nodes = Vec::new();
        for mut q in quats {
            if let Some(c) = q.iter().find(|c| c.abs() > 1e-9)
                && *c < 0.0
            {
                q = q.map(|x| -x);
            }
            if seen.insert(q.map(|c| (c * 1e6).round() as i64)) {
                let [w, x, y, z] = q.map(R64);
                nodes.push(So3Cover(So3::new(S3(Sphere::new(w, [x, y, z].into())))));
            }
        }
        debug_assert_eq!(nodes.len(), 60);
        nodes
    }
}

impl NerveComplexParameters<So3<Coords<R64, 3>>, Coords<R64, 3>, So3<Coords<R64, 3>>, So3Cover>
    for So3Cover
{
}
