use std::ops::{Add, Deref, DerefMut, Index, IndexMut, Mul, Neg, Sub};

use num_traits::{ConstZero, Zero};

use crate::{
    complex::Complex,
    traits::{Bilinear, Euclidean, Field, Interval, Metric, Quadratic, Real},
};

/// The canonical model of flat pseudo-Euclidean coordinate space `R^(N−M, M)`.
///
/// A fixed-size array of `N` coordinates over the field `R`, carrying the
/// algebraic structure of a vector space together with a symmetric bilinear
/// form of signature `(N − M, M)`: `N − M` positive (spacelike) directions
/// and `M` negative (timelike) ones. The first `M` coordinates are the
/// negative-signature directions; the remaining `N − M` are positive.
///
/// This is the space in which local coordinate charts take their values and
/// in which tangent vectors live. With the default `M = 0` it is ordinary
/// flat Euclidean space `R^N` — positive-definite, hence carrying a genuine
/// norm and [`Metric`]. With `M > 0` the form is indefinite: it is a
/// [`Bilinear`] scalar product only, with no norm and no metric (a timelike
/// vector has negative `norm_squared`, and null vectors give distinct points
/// at zero separation). Minkowski spacetime is `Coords<R, 4, 1>`.
///
/// `M` is expected in `0..=N`; values `M > N` are safe but redundant,
/// behaving identically to `M = N` (fully negative-definite), since the
/// scalar product only ranges over the `N` present coordinates.
///
/// # Trait scoping
/// The definite (`M = 0`) case implements [`InnerProduct`], [`Metric`], and
/// [`Euclidean`]; the general case implements [`Bilinear`] and
/// [`Quadratic`]. Operations requiring positive-definiteness — `norm`,
/// `distance`, sectional-curvature maxima — are therefore available only at
/// `M = 0`, enforced by the trait bounds rather than by runtime checks.
///
/// [`Bilinear`]: crate::traits::Bilinear
/// [`InnerProduct`]: crate::traits::InnerProduct
/// [`Metric`]: crate::traits::Metric
/// [`Euclidean`]: crate::traits::Euclidean
/// [`Quadratic`]: crate::traits::Quadratic
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Coords<F: Field, const N: usize, const M: usize = 0>([F; N]);

impl<F: Field, const N: usize, const M: usize> Zero for Coords<F, N, M> {
    fn zero() -> Self {
        [F::zero(); N].into()
    }

    fn is_zero(&self) -> bool {
        self == &Self::zero()
    }
}

impl<F: Field + ConstZero, const N: usize, const M: usize> ConstZero for Coords<F, N, M> {
    const ZERO: Self = Self([F::ZERO; N]);
}

impl<F: Field, const N: usize, const M: usize> Deref for Coords<F, N, M> {
    type Target = [F; N];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F: Field, const N: usize, const M: usize> DerefMut for Coords<F, N, M> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<F: Field, const N: usize, const M: usize> From<[F; N]> for Coords<F, N, M> {
    fn from(arr: [F; N]) -> Self {
        Self(arr)
    }
}

impl<F: Field, const N: usize, const M: usize> From<Coords<F, N, M>> for [F; N] {
    fn from(c: Coords<F, N, M>) -> Self {
        c.0
    }
}

impl<F: Field, const N: usize, const M: usize> Index<usize> for Coords<F, N, M> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<F: Field, const N: usize, const M: usize> IndexMut<usize> for Coords<F, N, M> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

pub(crate) fn array_zip_map<A, B, C, const N: usize>(
    a: [A; N],
    b: [B; N],
    f: fn(&A, &B) -> C,
) -> [C; N] {
    std::array::from_fn(|i| f(&a[i], &b[i]))
}

impl<F: Field, const N: usize, const M: usize> Add for Coords<F, N, M> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        array_zip_map(*self, *rhs, |&a, &b| a + b).into()
    }
}

impl<F: Field, const N: usize, const M: usize> Sub for Coords<F, N, M> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        array_zip_map(*self, *rhs, |&a, &b| a.sub(&b)).into()
    }
}

impl<F: Field, const N: usize, const M: usize> Mul<F> for Coords<F, N, M> {
    type Output = Self;

    fn mul(self, rhs: F) -> Self::Output {
        self.map(|x| x * rhs).into()
    }
}

impl<F: Field, const N: usize, const M: usize> Neg for Coords<F, N, M> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.map(|x| -x).into()
    }
}

impl<F: Field, const N: usize, const M: usize> Bilinear<F> for Coords<F, N, M> {
    fn dot(&self, other: &Self) -> F {
        self.iter()
            .zip(other.iter())
            .enumerate()
            .fold(F::zero(), |acc, (m, (&a, &b))| {
                if m < M {
                    acc.sub(&(a * b))
                } else {
                    acc + a * b
                }
            })
    }
}

impl<F: Field, const N: usize, const M: usize> Quadratic for Coords<F, N, M> {
    type F = F;
    const N: usize = N;

    type Iter<'a>
        = std::slice::Iter<'a, F>
    where
        Self: 'a;

    fn iter(&self) -> Self::Iter<'_> {
        self.0.iter()
    }

    fn from_array<const K: usize>(s: [F; K]) -> Self {
        const { assert!(K == N) };
        std::array::from_fn(|i| s[i]).into()
    }

    fn from_fn(f: impl Fn(usize) -> Self::F) -> Self {
        std::array::from_fn(f).into()
    }
}

impl<R: Field + Real, const N: usize, const M: usize> Interval<R> for Coords<R, N, M> {
    fn interval(&self, other: &Self) -> Complex<R> {
        let displacement = *self - *other;

        Complex::real_sqrt(displacement.dot(&displacement))
    }
}

impl<R: Field + Real, const N: usize> Metric<R> for Coords<R, N, 0> {
    fn distance(&self, other: &Self) -> R {
        let displacement = *self - *other;
        displacement.dot(&displacement).sqrt()
    }
}

impl<R: Field + Real, const N: usize> Euclidean for Coords<R, N, 0> {}
