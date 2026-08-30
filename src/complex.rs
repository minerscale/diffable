//! Complex scalars and their real-valued square-root convention.
//!
//! [`Complex`] extends a [`Real`] field with its canonical
//! involution and Hermitian norm. It is the scalar field used by constructions
//! such as [`Sl2c`](crate::spacetime::Sl2c).

use core::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

use num_traits::{Inv, One, Zero, real::Real as _};

use crate::{
    coords::Coords,
    impl_group_via_add, include_as,
    traits::{
        CField, Chart, Field, FieldExp, Group, Interval, LieGroup, Metric, NatZero, NonZero, Real,
        Sesquilinear, Tensor,
        calculus::{CommutesJet, Jet, JetVector, Tangent},
        𝐑𝐞𝐚𝐥,
    },
};

/// Complex numbers a + bi, backed by R^2.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Complex<R: Real>(pub Coords<R, 2, 0>);

impl<R: Real> FieldExp for Complex<R> {
    fn exp(&self) -> Self {
        self.exp_by_series()
    }
}

impl<R: Real> Complex<R> {
    pub fn new(real: R, imag: R) -> Self {
        Self(Coords([real, imag]))
    }

    /// Returns the principal square root of a real value in the complex plane.
    ///
    /// Non-negative inputs lie on the real axis and negative inputs on the
    /// positive imaginary axis. This is the convention used by [`Interval::interval`].
    pub fn real_sqrt(r: R) -> Self {
        if r.is_sign_negative() {
            [R::zero(), (-r).sqrt()].into()
        } else {
            [r.sqrt(), R::zero()].into()
        }
    }
}

impl<R: Real> From<R> for Complex<R> {
    fn from(value: R) -> Self {
        Self([value, R::zero()].into())
    }
}

impl<R: Real> From<Coords<R, 2, 0>> for Complex<R> {
    fn from(value: Coords<R, 2, 0>) -> Self {
        Self(value)
    }
}

impl<R: Real> From<[R; 2]> for Complex<R> {
    fn from(value: [R; 2]) -> Self {
        Coords::from(value).into()
    }
}

impl<R: Real> From<Complex<R>> for [R; 2] {
    fn from(value: Complex<R>) -> Self {
        value.0.into()
    }
}

impl<R: Real> One for Complex<R> {
    fn one() -> Self {
        Self([R::one(), R::zero()].into())
    }
}

impl<R: Real> Zero for Complex<R> {
    fn zero() -> Self {
        Self([R::zero(), R::zero()].into())
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<R: Real> Add<Self> for Complex<R> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl<R: Real> Sub<Self> for Complex<R> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl<R: Real> Neg for Complex<R> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl<R: Real> Mul<Self> for Complex<R> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let [a, b] = self.0.into();
        let [c, d] = rhs.0.into();

        Self([a * c - b * d, b * c + a * d].into())
    }
}

impl<R: Real> Mul<R> for Complex<R> {
    type Output = Self;

    fn mul(self, rhs: R) -> Self::Output {
        Self(self.0 * rhs)
    }
}

impl<R: Real> Index<usize> for Complex<R> {
    type Output = R;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<R: Real> IndexMut<usize> for Complex<R> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl_group_via_add!(Complex<R>, R: Real);

impl<R: Real> LieGroup<Coords<R, 2>> for Complex<R> {
    fn compose_jet<const N: usize>(
        lhs: Tangent<Self, Coords<R, 2>, N>,
        rhs: Tangent<Self, Coords<R, 2>, N>,
    ) -> Tangent<Self, Coords<R, 2>, N> {
        Tangent::new(lhs.0.compose(&rhs.0), lhs.1.compose(&rhs.1))
    }

    fn inverse_jet<const N: usize>(
        value: Tangent<Self, Coords<R, 2>, N>,
    ) -> Tangent<Self, Coords<R, 2>, N> {
        Tangent::new(value.0.inverse(), value.1.inverse())
    }

    // The identity map Coords<R, 2> -> Complex<R>.
    fn identity_exp<const N: usize>(
        coordinate: JetVector<Coords<R, 2>, N>,
    ) -> Tangent<Self, Coords<R, 2>, N> {
        coordinate.into_tangent(Complex::from)
    }

    // The inverse identification Complex<R> -> Coords<R, 2>.
    fn identity_log<const N: usize>(
        point: Tangent<Self, Coords<R, 2>, N>,
    ) -> Option<JetVector<Coords<R, 2>, N>> {
        Some(point.into_jet(|x| x.0))
    }
}

impl<R: Real, const N: usize> CommutesJet<Complex<R>, Coords<R, 2>, N>
    for Complex<Jet<𝐑𝐞𝐚𝐥::𝒞, R, N>>
{
    fn commute_jet(value: Tangent<Complex<R>, Coords<R, 2>, N>) -> Self {
        let value = value.into_jet(|point| point.0).retag();

        Complex::from([value[0], value[1]])
    }

    fn uncommute_jet(value: Self) -> Tangent<Complex<R>, Coords<R, 2>, N> {
        JetVector::from_iter([value.0[0], value.0[1]].map(Jet::retag)).into_tangent(Complex::from)
    }
}

impl<R: Real> Metric for Complex<R> {}

impl<R: Real> Inv for NonZero<Complex<R>> {
    type Output = Self;

    fn inv(self) -> Self::Output {
        Self(self.0.conj() * self.0.norm_squared().recip())
    }
}

impl<R: Real> LieGroup<Coords<R, 2>> for NonZero<Complex<R>> {
    fn compose_jet<const N: usize>(
        lhs: Tangent<Self, Coords<R, 2>, N>,
        rhs: Tangent<Self, Coords<R, 2>, N>,
    ) -> Tangent<Self, Coords<R, 2>, N> {
        let lhs = lhs.into_jet(|p| p.0.0);
        let rhs = rhs.into_jet(|p| p.0.0);

        JetVector::from_fn(|i| {
            if i == 0 {
                lhs[0] * rhs[0] - lhs[1] * rhs[1]
            } else {
                lhs[0] * rhs[1] + lhs[1] * rhs[0]
            }
        })
        .into_tangent(|coordinate| NonZero(Complex::from(coordinate)))
    }

    fn inverse_jet<const N: usize>(
        value: Tangent<Self, Coords<R, 2>, N>,
    ) -> Tangent<Self, Coords<R, 2>, N> {
        let value = value.into_jet(|p| p.0.0).retag();

        let norm_squared = value[0] * value[0] + value[1] * value[1];

        JetVector::from_fn(|i| {
            if i == 0 {
                value[0] / norm_squared
            } else {
                -value[1] / norm_squared
            }
            .retag()
        })
        .into_tangent(|coordinate| NonZero(Complex::from(coordinate)))
    }

    // e^(a + bi) = e^a(cos(b) + i sin(b))
    fn identity_exp<const N: usize>(
        coordinate: JetVector<Coords<R, 2>, N>,
    ) -> Tangent<Self, Coords<R, 2>, N> {
        let coordinate = coordinate.retag();

        let [a, b] = [coordinate[0], coordinate[1]];
        let (sin, cos) = b.sin_cos();
        let radius = a.exp();

        JetVector::from_fn(|i| if i == 0 { radius * cos } else { radius * sin }.retag())
            .into_tangent(|coordinate| NonZero(Complex::from(coordinate)))
    }

    // Log(a + bi) = ln(sqrt(a² + b²)) + i atan2(b, a)
    fn identity_log<const N: usize>(
        point: Tangent<Self, Coords<R, 2>, N>,
    ) -> Option<JetVector<Coords<R, 2>, N>> {
        let point = point.into_jet(|p| p.0.0).retag();

        let [a, b] = [point[0], point[1]];
        let radius = (a * a + b * b).sqrt();
        let theta = b.atan2(a);

        Some(JetVector::from_fn(|i| {
            if i == 0 { radius.ln() } else { theta }.retag()
        }))
    }
}

#[allow(type_alias_bounds)]
type NonZeroComplexJet<R: Real, const N: usize> = NonZero<Complex<Jet<𝐑𝐞𝐚𝐥::𝒞, R, N>>>;

impl<R: Real, const N: usize> CommutesJet<NonZero<Complex<R>>, Coords<R, 2>, N>
    for NonZeroComplexJet<R, N>
{
    fn commute_jet(value: Tangent<NonZero<Complex<R>>, Coords<R, 2>, N>) -> Self {
        let value = value.into_jet(|point| point.0.0).retag();

        NonZero(Complex::from([value[0], value[1]]))
    }

    fn uncommute_jet(value: Self) -> Tangent<NonZero<Complex<R>>, Coords<R, 2>, N> {
        JetVector::<Coords<R, 2>, N>::from_fn(|i| value.0.0[i].retag())
            .into_tangent(|coordinate| NonZero(Complex::from(coordinate)))
    }
}

impl<R: Real> Interval for NonZero<Complex<R>> {
    type R = R;

    fn interval_squared(&self, other: &Self) -> R {
        self.to_local(other).unwrap().norm_squared()
    }
}

impl<R: Real> Field for Complex<R> {
    type Fixed = R;
    type Characteristic = NatZero;

    fn conj(&self) -> Self {
        let [a, b] = (*self).into();

        [a, -b].into()
    }

    fn to_fixed(self) -> R {
        let [a, _] = self.into();
        a
    }

    fn from_fixed(x: R) -> Self {
        [x, R::zero()].into()
    }
}

impl<R: Real> CField for Complex<R> {}

include_as!(Complex<R> => CField, R: Real);
