use std::ops::{Add, Deref, DerefMut, Index, IndexMut, Mul, Neg, Sub};

use num_traits::{ConstZero, Zero, real::Real};

use crate::traits::{Euclidean, InnerProduct, Metric, Scalar};

/// The canonical model of real coordinate space R^N.
///
/// This is the standard flat Euclidean space of dimension `N` over the
/// field `R` — the space in which all local coordinate charts take their
/// values, and in which tangent vectors live. It is intentionally minimal:
/// a fixed-size array with the algebraic structure of a vector space.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Coords<R: Real, const N: usize>([R; N]);

impl<R: Real, const N: usize> Zero for Coords<R, N> {
    fn zero() -> Self {
        [R::zero(); N].into()
    }

    fn is_zero(&self) -> bool {
        self == &Self::zero()
    }
}

impl<R: Real + ConstZero, const N: usize> ConstZero for Coords<R, N> {
    const ZERO: Self = Self([R::ZERO; N]);
}

impl<R: Real, const N: usize> Deref for Coords<R, N> {
    type Target = [R; N];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<R: Real, const N: usize> DerefMut for Coords<R, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<R: Real, const N: usize> From<[R; N]> for Coords<R, N> {
    fn from(arr: [R; N]) -> Self {
        Self(arr)
    }
}

impl<R: Real, const N: usize> From<Coords<R, N>> for [R; N] {
    fn from(c: Coords<R, N>) -> Self {
        c.0
    }
}

impl<R: Real, const N: usize> Index<usize> for Coords<R, N> {
    type Output = R;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<R: Real, const N: usize> IndexMut<usize> for Coords<R, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

fn array_zip_map<A, B, C, const N: usize>(a: [A; N], b: [B; N], f: fn(&A, &B) -> C) -> [C; N] {
    std::array::from_fn(|i| f(&a[i], &b[i]))
}

impl<R: Real, const N: usize> Add for Coords<R, N> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        array_zip_map(*self, *rhs, |&a, &b| a + b).into()
    }
}

impl<R: Real, const N: usize> Sub for Coords<R, N> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        array_zip_map(*self, *rhs, |&a, &b| a - b).into()
    }
}

impl<R: Real, const N: usize> Mul<R> for Coords<R, N> {
    type Output = Self;

    fn mul(self, rhs: R) -> Self::Output {
        self.map(|x| x * rhs).into()
    }
}

impl<R: Real, const N: usize> Neg for Coords<R, N> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.map(|x| -x).into()
    }
}

impl<R: Real, const N: usize> Metric<R> for Coords<R, N> {
    fn distance(&self, other: &Self) -> R {
        self.iter()
            .zip(other.iter())
            .fold(R::zero(), |acc, (&a, &b)| acc + (b - a) * (b - a))
            .sqrt()
    }
}

impl<R: Real, const N: usize> InnerProduct<R> for Coords<R, N> {
    fn dot(&self, other: &Self) -> R {
        self.iter()
            .zip(other.iter())
            .fold(R::zero(), |acc, (&a, &b)| acc + a * b)
    }
}

impl<R: Scalar, const N: usize> Euclidean for Coords<R, N> {
    type F = R;
    const N: usize = N;

    type Iter<'a>
        = std::slice::Iter<'a, R>
    where
        Self: 'a;

    fn iter(&self) -> Self::Iter<'_> {
        self.0.iter()
    }

    fn from_array<const M: usize>(s: [R; M]) -> Self {
        const { assert!(M == N) };
        std::array::from_fn(|i| s[i]).into()
    }

    fn from_fn(f: impl Fn(usize) -> Self::F) -> Self {
        std::array::from_fn(f).into()
    }
}
