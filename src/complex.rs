use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

use num_traits::{Inv, One, Zero};

use crate::{
    coords::Coords,
    traits::{
        Bilinear, Euclidean, Field, InnerProduct, Interval, InvolutiveField, LieGroup, Metric,
        NonZero, Quadratic, Real, Smooth,
    },
};

/// Complex numbers a + bi, backed by R^2.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Complex<R: Real>(Coords<R, 2, 0>);

impl<R: Real> InvolutiveField for Complex<R> {
    type Fixed = R;

    fn conj(&self) -> Self {
        let [a, b] = (*self).into();

        [a, -b].into()
    }

    fn to_fixed(self) -> R {
        let [a, _] = self.into();

        a
    }

    fn from_fixed(x: R) -> Self {
        x.into()
    }
}

impl<R: Real> Complex<R> {
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

impl<R: Real> Metric<R> for Complex<R> {
    fn distance(&self, other: &Self) -> R {
        self.0.distance(&other.0)
    }
}

impl<R: Real> Interval<R> for Complex<R> {
    fn interval(&self, other: &Self) -> Complex<R> {
        self.0.distance(&other.0).into()
    }
}

impl<R: Real> Bilinear<R> for Complex<R> {
    fn dot(&self, other: &Self) -> R {
        self.0.dot(&other.0)
    }
}

impl<R: Real> Quadratic for Complex<R> {
    type F = R;

    const N: usize = 2;

    type Iter<'a>
        = std::slice::Iter<'a, R>
    where
        Self: 'a;

    fn iter(&self) -> Self::Iter<'_> {
        self.0.iter()
    }

    fn from_fn(f: impl Fn(usize) -> Self::F) -> Self {
        Self(Coords::<R, 2>::from_fn(f))
    }

    fn from_array<const N: usize>(arr: [Self::F; N]) -> Self {
        Self(Coords::<R, 2>::from_array(arr))
    }
}

impl<R: Real> Euclidean for Complex<R> {}

impl<R: Real> Interval<R> for NonZero<Complex<R>> {
    fn interval(&self, other: &Self) -> Complex<R> {
        self.log(other).unwrap().norm().into()
    }
}

impl<R: Real> Inv for NonZero<Complex<R>> {
    type Output = Self;

    fn inv(self) -> Self::Output {
        let [a, b] = self.0.into();
        let denom = self.0.dot(&self.0);
        Self([a / denom, -b / denom].into())
    }
}

impl<R: Real> LieGroup<Complex<R>> for NonZero<Complex<R>> {
    // e^z
    fn identity_exp(v: Complex<R>) -> Self {
        let [a, b] = v.into();

        let (sin, cos) = b.sin_cos();
        Self(Complex::from([cos, sin]) * a.exp())
    }

    // Log(p)
    fn identity_log(p: &Self) -> Option<Complex<R>> {
        let [a, b] = p.0.into();
        let r = p.0.norm();

        let theta = b.atan2(a);

        Some([r.ln(), theta].into())
    }
}

impl<R: Real> Field for Complex<R> {}
