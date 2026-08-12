//! The quaternion division algebra.
//!
//! [`Quaternion`] is the canonical noncommutative scalar example for
//! [`Field`], exercising multiplication order, handedness, duality, and
//! sesquilinear forms throughout the library.

use core::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

use num_traits::{Inv, One, Zero};

use crate::{
    complex::Complex,
    coords::Coords,
    impl_group_via_add,
    traits::{
        Field, FieldExp, Interpretation, LieGroup, Metric, NatZero, NonZero, Object, Real, 𝐅𝐥𝐝,
    },
};

/// The quaternion skew field `H`, with coordinates `a + bi + cj + dk`.
///
/// Multiplication is Hamilton multiplication:
/// `i² = j² = k² = ijk = -1`. In particular it is associative but not
/// commutative (`ij = k`, while `ji = -k`), making this the canonical test
/// case for the library's noncommutative [`Field`] support.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Quaternion<R: Real>(pub Coords<R, 4, 0>);

impl<R: Real> Quaternion<R> {
    /// Constructs a quaternion from its `(real, i, j, k)` coordinates.
    pub fn new(real: R, i: R, j: R, k: R) -> Self {
        [real, i, j, k].into()
    }

    /// Returns the quaternion basis element `i`.
    pub fn i() -> Self {
        Self::new(R::zero(), R::one(), R::zero(), R::zero())
    }

    /// Returns the quaternion basis element `j`.
    pub fn j() -> Self {
        Self::new(R::zero(), R::zero(), R::one(), R::zero())
    }

    /// Returns the quaternion basis element `k`.
    pub fn k() -> Self {
        Self::new(R::zero(), R::zero(), R::zero(), R::one())
    }
}

impl<R: Real> From<R> for Quaternion<R> {
    fn from(value: R) -> Self {
        Self::new(value, R::zero(), R::zero(), R::zero())
    }
}

impl<R: Real> From<Coords<R, 4, 0>> for Quaternion<R> {
    fn from(value: Coords<R, 4, 0>) -> Self {
        Self(value)
    }
}

impl<R: Real> From<Complex<R>> for Quaternion<R> {
    fn from(z: Complex<R>) -> Self {
        let [re, im] = z.into();

        Self::new(re, im, R::zero(), R::zero())
    }
}

impl<R: Real> From<[R; 4]> for Quaternion<R> {
    fn from(value: [R; 4]) -> Self {
        Coords::from(value).into()
    }
}

impl<R: Real> From<Quaternion<R>> for [R; 4] {
    fn from(value: Quaternion<R>) -> Self {
        value.0.into()
    }
}

impl<R: Real> One for Quaternion<R> {
    fn one() -> Self {
        R::one().into()
    }
}

impl<R: Real> Zero for Quaternion<R> {
    fn zero() -> Self {
        R::zero().into()
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<R: Real> Add for Quaternion<R> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl<R: Real> Sub for Quaternion<R> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl<R: Real> Neg for Quaternion<R> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl<R: Real> Mul for Quaternion<R> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let [a, b, c, d] = self.into();
        let [e, f, g, h] = rhs.into();

        Self::new(
            a * e - b * f - c * g - d * h,
            a * f + b * e + c * h - d * g,
            a * g - b * h + c * e + d * f,
            a * h + b * g - c * f + d * e,
        )
    }
}

impl<R: Real> Mul<R> for Quaternion<R> {
    type Output = Self;

    fn mul(self, rhs: R) -> Self::Output {
        Self(self.0 * rhs)
    }
}

impl<R: Real> Index<usize> for Quaternion<R> {
    type Output = R;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<R: Real> IndexMut<usize> for Quaternion<R> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl_group_via_add!(Quaternion<R>, R: Real);

impl<R: Real> LieGroup<Coords<R, 4>> for Quaternion<R> {
    fn identity_exp(v: Coords<R, 4>) -> Self {
        v.into()
    }

    fn identity_log(p: &Self) -> Option<Coords<R, 4>> {
        Some(p.0)
    }
}

impl<R: Real> Metric for Quaternion<R> {}
impl<R: Real> FieldExp for Quaternion<R> {
    fn exp(&self) -> Self {
        self.exp_by_series()
    }
}

impl<R: Real> Inv for NonZero<Quaternion<R>> {
    type Output = Self;

    fn inv(self) -> Self::Output {
        Self(self.0.conj() * self.0.norm_squared().recip())
    }
}

impl<R: Real> Field for Quaternion<R> {
    type Fixed = R;
    type Characteristic = NatZero;

    fn conj(&self) -> Self {
        let [a, b, c, d] = (*self).into();
        Self::new(a, -b, -c, -d)
    }

    fn to_fixed(self) -> R {
        self[0]
    }

    fn from_fixed(x: R) -> Self {
        x.into()
    }
}

impl<R: Real> Object for Quaternion<R> {
    type Context = Interpretation<𝐅𝐥𝐝, Self>;
}
