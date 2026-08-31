//! Spheres, their Lie-group refinements, and finite covers.
//!
//! [`Sphere`] supplies intrinsic spherical geometry and [`Stereographic`] an
//! external atlas. [`S0`], [`UnitComplex`], and [`S3`] add the group structures
//! available in dimensions zero, one, and three; [`So3`] then forms the
//! antipodal quotient of `S3`.

use core::{marker::PhantomData, ops::Mul};

use crate::{
    complex::Complex,
    impl_group_via_mul, impl_lie_group_via_quotient, include_point,
    quaternion::Quaternion,
    traits::{
        Chart, Euclidean, Group, Interval, LieGroup, Metric, Quotient, Real, RootOfUnity,
        Sesquilinear, Smooth, Tensor,
        calculus::{CommutesJet, Jet, JetVector, JetVectorIn, Tangent},
        𝐅𝐥𝐝, 𝐑𝐞𝐚𝐥,
    },
};

use num_traits::{Inv, NumCast, One, Zero, real::Real as _};

/// The unit `N`-sphere `Sⁿ ⊂ V::F ⊕ V`.
///
/// A point is split as a scalar `real` part and a vector `imag` part,
/// constrained to `real² + ‖imag‖² = 1`. This splitting is what the
/// [`Stereographic`] chart projects from and what the geodesic distance
/// (`cos θ = ⟨p, q⟩`) is computed against. `V: Euclidean` supplies the
/// positive-definite inner product that makes "unit" and "distance" meaningful.
#[derive(Debug, PartialEq, Clone)]
pub struct Sphere<V: Euclidean> {
    real: V::F,
    imag: V,
}

include_point!(
    Sphere<V>,
    V: Euclidean
);

/// A [`Chart`] on the [`Sphere`] by stereographic projection from a chosen pole.
///
/// Projecting from one pole leaves the *opposite* pole as the chart's single
/// missing point, so two charts (north and south) cover the sphere. Construct
/// with [`south_pole`](Stereographic::south_pole) or
/// [`north_pole`](Stereographic::north_pole).
#[derive(Clone, Debug)]
pub struct Stereographic<V: Euclidean>(StereographicPole, PhantomData<V>);

impl<V: Euclidean> Stereographic<V> {
    /// Constructs the stereographic chart projecting from the south pole.
    pub const fn south_pole() -> Self {
        Self(StereographicPole::SouthPole, PhantomData)
    }
    /// Constructs the stereographic chart projecting from the north pole.
    pub const fn north_pole() -> Self {
        Self(StereographicPole::NorthPole, PhantomData)
    }
}

#[derive(Clone, Debug)]
enum StereographicPole {
    SouthPole,
    NorthPole,
}

/// Numerical exclusion radius around a [`Stereographic`] chart's missing pole.
pub const EPSILON: f64 = 1e-3;

impl<V: Euclidean> Chart<Sphere<V>, V> for Stereographic<V> {
    type Global = Sphere<V>;

    fn to_local(&self, point: &Sphere<V>) -> Option<V> {
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
        Some(point.imag.clone() * recip)
    }

    fn to_global(&self, coord: V) -> Sphere<V> {
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

    fn chart_at(p: &Sphere<V>) -> Self {
        if p.real > V::F::zero() {
            Self::south_pole()
        } else {
            Self::north_pole()
        }
    }
}

impl<V: Euclidean> Sphere<V> {
    /// Returns the scalar coordinate in the splitting `V::F ⊕ V`.
    pub fn real(&self) -> V::F {
        self.real
    }
    /// Returns the vector coordinate in the splitting `V::F ⊕ V`.
    pub fn imag(&self) -> V {
        self.imag.clone()
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
        self.real.is_one() && self.imag.is_zero()
    }

    /// Constructs and normalises a sphere point from scalar and vector parts.
    pub fn new(real: V::F, imag: V) -> Self {
        let sphere = Sphere { real, imag };

        sphere.normalised()
    }

    fn geodesic_distance(&self, other: &Self) -> V::F {
        let cos_d = self.real * other.real + self.imag.dot(&other.imag);
        let w_real = other.real - cos_d * self.real;
        let w_imag = other.imag.clone() - self.imag.clone() * cos_d;
        let sin_d = (w_real * w_real + w_imag.norm_squared()).sqrt();
        V::F::atan2(sin_d, cos_d) // θ, stable through the antipode
    }
}

/// The hopf map. See https://en.wikipedia.org/wiki/Hopf_fibration for more
pub fn hopf<U: Euclidean, V: Euclidean<F = U::F>>(q: Sphere<U>) -> Sphere<V> {
    const {
        assert!(U::N == 3);
        assert!(V::N == 2);
    }

    let a = q.real();
    let imag = q.imag();

    let [b, c, d] = [imag[0], imag[1], imag[2]];

    let two = U::F::one() + U::F::one();

    Sphere::new(
        a * a + b * b - c * c - d * d,
        V::from_iter([two * (b * c + a * d), two * (b * d - a * c)]),
    )
}

#[allow(type_alias_bounds)]
type SphereJet<V: Euclidean, const N: usize> = Sphere<JetVectorIn<𝐑𝐞𝐚𝐥::𝒞, V, N>>;

fn sphere_constant_jet<V: Euclidean, const N: usize>(value: Sphere<V>) -> SphereJet<V, N> {
    Sphere {
        real: Jet::<𝐑𝐞𝐚𝐥::𝒞, V::F, N>::from_parts(value.real, [Zero::zero(); N]),
        imag: JetVectorIn::<𝐑𝐞𝐚𝐥::𝒞, V, N>::constant(value.imag),
    }
}

// Reassemble the split tangent presentation as the sphere-valued curve
// t ↦ exp_value.0(value.1(t)).
fn sphere_assemble_jet<V: Euclidean, const N: usize>(
    value: Tangent<Sphere<V>, V, N>,
) -> SphereJet<V, N> {
    sphere_exp_coordinate_jet(&sphere_constant_jet(value.0), value.1.retag::<𝐑𝐞𝐚𝐥::𝒞>())
}

// Split a sphere-valued curve into its value at zero and exponential-chart
// coordinates about that value. The logarithm cannot encounter the cut locus:
// both curves have the same coefficient-zero point by construction.
fn sphere_split_jet<V: Euclidean, const N: usize>(
    value: SphereJet<V, N>,
) -> Tangent<Sphere<V>, V, N> {
    let point = sphere_jet_primal(&value);
    let base = sphere_constant_jet(point.clone());

    let coordinate = sphere_log_coordinate_jet(&base, &value).unwrap();

    Tangent::new(point, coordinate.retag::<𝐅𝐥𝐝::𝒞>())
}

fn sphere_exp_coordinate_jet<V: Euclidean, const N: usize>(
    base: &SphereJet<V, N>,
    coordinate: JetVectorIn<𝐑𝐞𝐚𝐥::𝒞, V, N>,
) -> SphereJet<V, N> {
    let (cos, sinc) = sphere_exp_factors(coordinate.norm_squared());

    base.transport_from_identity(cos, coordinate * sinc)
}

fn sphere_log_coordinate_jet<V: Euclidean, const N: usize>(
    base: &SphereJet<V, N>,
    point: &SphereJet<V, N>,
) -> Option<JetVectorIn<𝐑𝐞𝐚𝐥::𝒞, V, N>> {
    let point = base.transport_to_identity(point.real, point.imag.clone());

    let eps = <V::F as NumCast>::from(EPSILON).unwrap();

    // The antipode is the cut locus of the exponential chart.
    if (point.real[0] + V::F::one()).abs() < eps {
        return None;
    }

    let factor = sphere_log_factor(point.imag.norm_squared(), point.real);

    Some(point.imag * factor)
}

impl<V: Euclidean> Smooth<V> for Sphere<V> {
    type Global<const N: usize> = Tangent<Self, V, N>;

    fn exp<const N: usize>(
        base: Tangent<Self, V, N>,
        coordinate: JetVector<V, N>,
    ) -> Self::Global<N> {
        let base = sphere_assemble_jet(base);

        sphere_split_jet(sphere_exp_coordinate_jet(
            &base,
            coordinate.retag::<𝐑𝐞𝐚𝐥::𝒞>(),
        ))
    }

    fn log<const N: usize>(
        base: Tangent<Self, V, N>,
        point: Tangent<Self, V, N>,
    ) -> Option<JetVector<V, N>> {
        let base = sphere_assemble_jet(base);
        let point = sphere_assemble_jet(point);

        sphere_log_coordinate_jet(&base, &point).map(|coordinate| coordinate.retag::<𝐅𝐥𝐝::𝒞>())
    }
}

impl<V: Euclidean> Sphere<V> {
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
        let u_imag = self.imag.clone();
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

/// `S⁰ = {±1}` — the two-point sphere, the unit-norm reals, a group under multiplication.
#[derive(Debug, Clone, PartialEq)]
pub struct S0<V: Euclidean>(Sphere<V>);
impl_group_via_mul!(S0<V>, V: Euclidean);

/// `S¹ ⊂ ℂ` — the unit complex numbers `U(1)`, a group under multiplication.
#[derive(Debug, Clone, PartialEq)]
pub struct UnitComplex<V: Euclidean>(Sphere<V>);
impl_group_via_mul!(UnitComplex<V>, V: Euclidean);

/// `S³ ⊂ ℍ` — the unit quaternions `SU(2)`, a group under multiplication and
/// the double cover of [`So3`].
#[derive(Debug, Clone, PartialEq)]
pub struct S3<V: Euclidean>(Sphere<V>);
impl_group_via_mul!(S3<V>, V: Euclidean);

impl<V: Euclidean> Interval for S0<V> {
    type R = V::F;

    fn interval_squared(&self, other: &Self) -> V::F {
        self.0.interval_squared(&other.0)
    }
}

impl<V: Euclidean> Metric for S0<V> {}

impl<V: Euclidean> S0<V> {
    /// Wraps a sphere point with the Lie-group structure of `S⁰`.
    pub fn new(s: Sphere<V>) -> Self {
        // Dim(V) + 1 dimensions must embed
        // the unit circle.
        const { assert!(V::N == 0) }

        Self(s)
    }

    /// Removes the `S⁰` group wrapper.
    pub fn to_inner(self) -> Sphere<V> {
        self.0
    }

    /// Borrows the underlying sphere point.
    pub fn inner(&self) -> &Sphere<V> {
        &self.0
    }
}

impl<V: Euclidean> UnitComplex<V> {
    /// Wraps a sphere point with unit-complex multiplication.
    pub fn new(s: Sphere<V>) -> Self {
        // Dim(V) + 1 dimensions must embed
        // the unit circle.
        const { assert!(V::N == 1) }

        Self(s)
    }

    /// Removes the unit-complex group wrapper.
    pub fn to_inner(self) -> Sphere<V> {
        self.0
    }

    /// Borrows the underlying sphere point.
    pub fn inner(&self) -> &Sphere<V> {
        &self.0
    }
}

fn sphere_exp_factors<R: Real, const N: usize>(
    norm_squared: Jet<𝐑𝐞𝐚𝐥::𝒞, R, N>,
) -> (Jet<𝐑𝐞𝐚𝐥::𝒞, R, N>, Jet<𝐑𝐞𝐚𝐥::𝒞, R, N>) {
    let eps = <R as NumCast>::from(EPSILON).unwrap();

    if norm_squared[0] >= eps * eps {
        let alpha = norm_squared.sqrt();
        let (sin, cos) = alpha.sin_cos();

        return (cos, sin / alpha);
    }

    // cos(sqrt(q))
    //     = 1 - q/2! + q²/4! - q³/6! + ...
    //
    // sinc(sqrt(q))
    //     = 1 - q/3! + q²/5! - q³/7! + ...
    let mut cos = Jet::zero();
    let mut sinc = Jet::zero();
    let mut power = Jet::one();

    let mut cos_coefficient = R::one();
    let mut sinc_coefficient = R::one();

    for k in 0..=(N + 8) {
        cos = cos + power * Jet::from_parts(cos_coefficient, [R::zero(); N]);

        sinc = sinc + power * Jet::from_parts(sinc_coefficient, [R::zero(); N]);

        let cos_denominator = <R as NumCast>::from((2 * k + 1) * (2 * k + 2)).unwrap();

        let sinc_denominator = <R as NumCast>::from((2 * k + 2) * (2 * k + 3)).unwrap();

        cos_coefficient = -cos_coefficient / cos_denominator;

        sinc_coefficient = -sinc_coefficient / sinc_denominator;

        power = power * norm_squared;
    }

    (cos, sinc)
}

impl<V: Euclidean, const N: usize> CommutesJet<Sphere<V>, V, N>
    for Sphere<JetVectorIn<𝐑𝐞𝐚𝐥::𝒞, V, N>>
{
    fn commute_jet(value: Tangent<Sphere<V>, V, N>) -> Self {
        sphere_assemble_jet(value)
    }

    fn uncommute_jet(value: Self) -> Tangent<Sphere<V>, V, N> {
        sphere_split_jet(value)
    }
}

impl<V: Euclidean, const N: usize> CommutesJet<S0<V>, V, N> for S0<JetVectorIn<𝐑𝐞𝐚𝐥::𝒞, V, N>> {
    fn commute_jet(value: Tangent<S0<V>, V, N>) -> Self {
        S0(sphere_assemble_jet(Tangent::new(value.0.0, value.1)))
    }

    fn uncommute_jet(value: Self) -> Tangent<S0<V>, V, N> {
        let split = sphere_split_jet(value.0);
        Tangent::new(S0(split.0), split.1)
    }
}

impl<V: Euclidean, const N: usize> CommutesJet<UnitComplex<V>, V, N>
    for UnitComplex<JetVectorIn<𝐑𝐞𝐚𝐥::𝒞, V, N>>
{
    fn commute_jet(value: Tangent<UnitComplex<V>, V, N>) -> Self {
        UnitComplex(sphere_assemble_jet(Tangent::new(value.0.0, value.1)))
    }

    fn uncommute_jet(value: Self) -> Tangent<UnitComplex<V>, V, N> {
        let split = sphere_split_jet(value.0);
        Tangent::new(UnitComplex(split.0), split.1)
    }
}

impl<V: Euclidean, const N: usize> CommutesJet<S3<V>, V, N> for S3<JetVectorIn<𝐑𝐞𝐚𝐥::𝒞, V, N>> {
    fn commute_jet(value: Tangent<S3<V>, V, N>) -> Self {
        S3(sphere_constant_jet(value.0.0)) * S3(sphere_identity_exp_jet(value.1))
    }

    fn uncommute_jet(value: Self) -> Tangent<S3<V>, V, N> {
        let point = S3(sphere_jet_primal(&value.0));

        let local = S3(sphere_constant_jet(point.0.clone())).inverse() * value;

        Tangent::new(point, sphere_identity_log_jet(&local.0).unwrap())
    }
}

fn sphere_log_factor<R: Real, const N: usize>(
    norm_squared: Jet<𝐑𝐞𝐚𝐥::𝒞, R, N>,
    real: Jet<𝐑𝐞𝐚𝐥::𝒞, R, N>,
) -> Jet<𝐑𝐞𝐚𝐥::𝒞, R, N> {
    let eps = <R as NumCast>::from(EPSILON).unwrap();

    if norm_squared[0] >= eps * eps || real[0] <= R::zero() {
        let norm = norm_squared.sqrt();

        return norm.atan2(real) / norm;
    }

    // Near the identity, the sphere constraint gives
    //
    //     atan2(sqrt(q), real) = asin(sqrt(q)).
    //
    // Therefore the required factor is
    //
    //     asin(sqrt(q)) / sqrt(q)
    //         = 1 + q/6 + 3q²/40 + 5q³/112 + ...
    let mut result = Jet::zero();
    let mut power = Jet::one();
    let mut coefficient = R::one();

    for k in 0..=(N + 8) {
        result = result + power * Jet::from_parts(coefficient, [R::zero(); N]);

        let numerator = <R as NumCast>::from((2 * k + 1) * (2 * k + 1)).unwrap();

        let denominator = <R as NumCast>::from(2 * (k + 1) * (2 * k + 3)).unwrap();

        coefficient = coefficient * numerator / denominator;

        power = power * norm_squared;
    }

    result
}

fn sphere_identity_exp_jet<V: Euclidean, const N: usize>(
    coordinate: JetVector<V, N>,
) -> Sphere<JetVectorIn<𝐑𝐞𝐚𝐥::𝒞, V, N>> {
    let coordinate = coordinate.retag::<𝐑𝐞𝐚𝐥::𝒞>();

    // exp(v) = (cos ‖v‖, v sinc ‖v‖).
    //
    // Evaluate both radial functions through q = ‖v‖² so that the
    // derivatives remain well-defined at v = 0.
    let (cos, sinc) = sphere_exp_factors(coordinate.norm_squared());

    Sphere::new(cos, coordinate * sinc)
}

fn sphere_identity_log_jet<V: Euclidean, const N: usize>(
    point: &Sphere<JetVectorIn<𝐑𝐞𝐚𝐥::𝒞, V, N>>,
) -> Option<JetVector<V, N>> {
    let eps = <V::F as NumCast>::from(EPSILON).unwrap();

    // The antipode -1 is the cut locus of the identity.
    if (point.real[0] + V::F::one()).abs() < eps {
        return None;
    }

    // log(a, w) = w · atan2(‖w‖, a) / ‖w‖.
    //
    // Again, evaluate the scalar factor as an analytic function of
    // q = ‖w‖² to avoid differentiating sqrt at the identity.
    let factor = sphere_log_factor(point.imag.norm_squared(), point.real);

    Some((point.imag.clone() * factor).retag::<𝐅𝐥𝐝::𝒞>())
}

fn sphere_jet_primal<V: Euclidean, const N: usize>(value: &SphereJet<V, N>) -> Sphere<V> {
    Sphere {
        real: value.real[0],
        imag: V::from_iter(value.imag.iter().map(|coordinate| coordinate[0])),
    }
}

impl<V: Euclidean> S3<V> {
    /// Wraps a sphere point with unit-quaternion multiplication.
    pub fn new(s: Sphere<V>) -> Self {
        // Dim(V) + 1 dimensions must embed
        // the unit circle.
        const { assert!(V::N == 3) }

        Self(s)
    }

    /// Removes the `S³` group wrapper.
    pub fn to_inner(self) -> Sphere<V> {
        self.0
    }

    /// Borrows the underlying sphere point.
    pub fn inner(&self) -> &Sphere<V> {
        &self.0
    }

    /// Regard a point of `S³` as a unit quaternion.
    pub fn to_quaternion(&self) -> Quaternion<V::F> {
        Quaternion::new(self.0.real, self.0.imag[0], self.0.imag[1], self.0.imag[2])
    }

    /// Project a quaternion onto S3.
    pub fn from_quaternion(quaternion: Quaternion<V::F>) -> Self {
        let [real, i, j, k] = quaternion.into();
        Self::new(Sphere::new(real, V::from_iter([i, j, k])))
    }

    fn assemble_jet<const N: usize>(
        value: Tangent<Self, V, N>,
    ) -> S3<JetVectorIn<𝐑𝐞𝐚𝐥::𝒞, V, N>> {
        const { assert!(V::N == 3) }

        S3::<JetVectorIn<𝐑𝐞𝐚𝐥::𝒞, V, N>>::commute_jet(value)
    }

    fn split_jet<const N: usize>(
        value: S3<JetVectorIn<𝐑𝐞𝐚𝐥::𝒞, V, N>>
    ) -> Tangent<Self, V, N> {
        const { assert!(V::N == 3) }

        S3::<JetVectorIn<𝐑𝐞𝐚𝐥::𝒞, V, N>>::uncommute_jet(value)
    }
}

impl<V: Euclidean> Interval for UnitComplex<V> {
    type R = V::F;

    fn interval_squared(&self, other: &Self) -> V::F {
        self.0.interval_squared(&other.0)
    }
}
impl<V: Euclidean> Metric for UnitComplex<V> {}

impl<V: Euclidean> Interval for S3<V> {
    type R = V::F;

    fn interval_squared(&self, other: &Self) -> V::F {
        self.0.interval_squared(&other.0)
    }
}
impl<V: Euclidean> Metric for S3<V> {}

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
    fn compose_jet<const N: usize>(
        lhs: Tangent<Self, V, N>,
        rhs: Tangent<Self, V, N>,
    ) -> Tangent<Self, V, N> {
        const { assert!(V::N == 0) }

        Tangent::new(lhs.0.compose(&rhs.0), lhs.1.compose(&rhs.1))
    }

    fn inverse_jet<const N: usize>(value: Tangent<Self, V, N>) -> Tangent<Self, V, N> {
        const { assert!(V::N == 0) }

        Tangent::new(value.0.inverse(), value.1.inverse())
    }

    fn identity_exp<const N: usize>(coordinate: JetVector<V, N>) -> Tangent<Self, V, N> {
        const { assert!(V::N == 0) }

        Tangent::new(Self::identity(), coordinate)
    }

    fn identity_log<const N: usize>(point: Tangent<Self, V, N>) -> Option<JetVector<V, N>> {
        const { assert!(V::N == 0) }

        (point.0.0.real() > V::F::zero()).then_some(point.1)
    }
}

impl<V: Euclidean> One for UnitComplex<V> {
    fn one() -> Self {
        Self::new(Sphere::identity())
    }

    fn is_one(&self) -> bool {
        self.0.is_identity()
    }
}

impl<V: Euclidean> Mul for UnitComplex<V> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let (a1, b1) = (self.0.real, self.0.imag[0]);
        let (a2, b2) = (rhs.0.real, rhs.0.imag[0]);

        Self(Sphere::new(
            a1 * a2 - b1 * b2,
            V::from_iter([a1 * b2 + a2 * b1]),
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
    fn compose_jet<const N: usize>(
        lhs: Tangent<Self, V, N>,
        rhs: Tangent<Self, V, N>,
    ) -> Tangent<Self, V, N> {
        const { assert!(V::N == 1) }

        Tangent::new(lhs.0.compose(&rhs.0), lhs.1.compose(&rhs.1))
    }

    fn inverse_jet<const N: usize>(value: Tangent<Self, V, N>) -> Tangent<Self, V, N> {
        const { assert!(V::N == 1) }

        Tangent::new(value.0.inverse(), value.1.inverse())
    }

    fn identity_exp<const N: usize>(coordinate: JetVector<V, N>) -> Tangent<Self, V, N> {
        const { assert!(V::N == 1) }

        coordinate.into_tangent(|coordinate| {
            let alpha = coordinate[0];
            let (sin, cos) = alpha.sin_cos();

            Self::new(Sphere::new(cos, V::from_iter([sin])))
        })
    }

    fn identity_log<const N: usize>(point: Tangent<Self, V, N>) -> Option<JetVector<V, N>> {
        const { assert!(V::N == 1) }

        let eps = <V::F as NumCast>::from(EPSILON).unwrap();

        if (point.0.0.real + V::F::one()).abs() < eps {
            return None;
        }

        Some(point.into_jet(|point| V::from_iter([point.0.imag[0].atan2(point.0.real)])))
    }
}

impl<V: Euclidean> One for S3<V> {
    fn one() -> Self {
        Self::new(Sphere::identity())
    }

    fn is_one(&self) -> bool {
        self.0.is_identity()
    }
}

impl<V: Euclidean> Mul for S3<V> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let (a1, a2) = (self.0.real, rhs.0.real);

        let im1 = self.0.imag;
        let im2 = rhs.0.imag;
        let (b1, c1, d1, b2, c2, d2) = (im1[0], im1[1], im1[2], im2[0], im2[1], im2[2]);

        Self(Sphere::new(
            a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
            V::from_iter([
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
        let a = self.0.real();
        let im = self.0.imag;
        let (b, c, d) = (im[0], im[1], im[2]);

        Self(Sphere::new(a, V::from_iter([-b, -c, -d])))
    }
}

impl<V: Euclidean> LieGroup<V> for S3<V> {
    fn compose_jet<const N: usize>(
        lhs: Tangent<Self, V, N>,
        rhs: Tangent<Self, V, N>,
    ) -> Tangent<Self, V, N> {
        Self::split_jet(Self::assemble_jet(lhs) * Self::assemble_jet(rhs))
    }

    fn inverse_jet<const N: usize>(value: Tangent<Self, V, N>) -> Tangent<Self, V, N> {
        Self::split_jet(Self::assemble_jet(value).inverse())
    }

    fn identity_exp<const N: usize>(coordinate: JetVector<V, N>) -> Tangent<Self, V, N> {
        Self::split_jet(S3(sphere_identity_exp_jet(coordinate)))
    }

    fn identity_log<const N: usize>(point: Tangent<Self, V, N>) -> Option<JetVector<V, N>> {
        sphere_identity_log_jet(&Self::assemble_jet(point).0)
    }
}
impl<V: Euclidean> Interval for Sphere<V> {
    type R = V::F;

    fn interval(&self, other: &Self) -> Complex<V::F> {
        self.geodesic_distance(other).into()
    }
    fn interval_squared(&self, other: &Self) -> V::F {
        let d = self.geodesic_distance(other);
        d * d
    }
}

impl<V: Euclidean> Metric for Sphere<V> {}

/// The rotation group `SO(3)`, as `S³` quotiented by `{±1}` (`RP³`).
#[derive(Clone, Debug, PartialEq)]
pub struct So3<V: Euclidean>(S3<V>);

impl<V: Euclidean> Quotient<S3<V>, RootOfUnity<V::F, 2>, V> for So3<V> {
    fn new(g: S3<V>) -> Self {
        // lexographic ordering on the fields
        match g
            .0
            .real()
            .partial_cmp(&V::F::zero())
            .unwrap()
            .then(g.0.imag().iter().partial_cmp(V::zero().iter()).unwrap())
        {
            core::cmp::Ordering::Less => So3(S3(Sphere::new(-g.0.real(), -g.0.imag()))),
            core::cmp::Ordering::Equal | core::cmp::Ordering::Greater => So3(g),
        }
    }

    fn lift(&self) -> S3<V> {
        self.0.clone()
    }

    fn embed(h: RootOfUnity<V::F, 2>) -> S3<V> {
        S3(Sphere::new(h.inner(), V::zero()))
    }
}

impl_lie_group_via_quotient!(
    So3<V>, S3<V>, RootOfUnity<V::F, 2>, V,
    [V: Euclidean];

    commutes_jet<const N: usize> {
        quotient = So3<JetVectorIn<𝐑𝐞𝐚𝐥::𝒞, V, N>>,
        cover = S3<JetVectorIn<𝐑𝐞𝐚𝐥::𝒞, V, N>>,
        subgroup = RootOfUnity<Jet<𝐑𝐞𝐚𝐥::𝒞, V::F, N>, 2>,
        model = JetVectorIn<𝐑𝐞𝐚𝐥::𝒞, V, N>,
    }
);

#[cfg(feature = "simplicial")]
mod simplicial {
    use super::*;
    use crate::epsilon_metric::R64;
    use crate::{
        coords::Coords,
        impl_tangent_bundle_via_bounded,
        traits::{
            ExpMap, InnerProduct, TangentBundle,
            simplicial::{Bounded, BuildNodes, NerveComplexParameters},
        },
    };
    use std::vec::Vec;

    /// The six-chart good cover of [`UnitComplex`] used by [`NerveComplex`](crate::traits::simplicial::NerveComplex).
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

    /// A finite geodesic-ball cover of [`So3`] centred on icosahedral rotations.
    ///
    /// The cover supplies [`Bounded`] domains and nodes for the global simplicial
    /// and geodesic algorithms.
    #[derive(PartialEq, Debug, Clone)]
    pub struct So3Cover(So3<Coords<R64, 3>>);

    impl Chart<So3<Coords<R64, 3>>, Coords<R64, 3>> for So3Cover {
        type Global = So3<Coords<R64, 3>>;

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
}

#[cfg(feature = "simplicial")]
pub use simplicial::*;
