use std::ops::{Add, Deref, DerefMut, Index, IndexMut, Mul, Neg, Sub};

use num_traits::{ConstZero, Zero, real::Real};

use crate::traits::{Bilinear, Euclidean, InnerProduct, Metric, PseudoEuclidean, Scalar};

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
/// [`PseudoEuclidean`]. Operations requiring positive-definiteness — `norm`,
/// `distance`, sectional-curvature maxima — are therefore available only at
/// `M = 0`, enforced by the trait bounds rather than by runtime checks.
///
/// [`Bilinear`]: crate::traits::Bilinear
/// [`InnerProduct`]: crate::traits::InnerProduct
/// [`Metric`]: crate::traits::Metric
/// [`Euclidean`]: crate::traits::Euclidean
/// [`PseudoEuclidean`]: crate::traits::PseudoEuclidean
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Coords<R: Real, const N: usize, const M: usize = 0>([R; N]);

impl<R: Real, const N: usize, const M: usize> Zero for Coords<R, N, M> {
    fn zero() -> Self {
        [R::zero(); N].into()
    }

    fn is_zero(&self) -> bool {
        self == &Self::zero()
    }
}

impl<R: Real + ConstZero, const N: usize, const M: usize> ConstZero for Coords<R, N, M> {
    const ZERO: Self = Self([R::ZERO; N]);
}

impl<R: Real, const N: usize, const M: usize> Deref for Coords<R, N, M> {
    type Target = [R; N];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<R: Real, const N: usize, const M: usize> DerefMut for Coords<R, N, M> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<R: Real, const N: usize, const M: usize> From<[R; N]> for Coords<R, N, M> {
    fn from(arr: [R; N]) -> Self {
        Self(arr)
    }
}

impl<R: Real, const N: usize, const M: usize> From<Coords<R, N, M>> for [R; N] {
    fn from(c: Coords<R, N, M>) -> Self {
        c.0
    }
}

impl<R: Real, const N: usize, const M: usize> Index<usize> for Coords<R, N, M> {
    type Output = R;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<R: Real, const N: usize, const M: usize> IndexMut<usize> for Coords<R, N, M> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

fn array_zip_map<A, B, C, const N: usize>(a: [A; N], b: [B; N], f: fn(&A, &B) -> C) -> [C; N] {
    std::array::from_fn(|i| f(&a[i], &b[i]))
}

impl<R: Real, const N: usize, const M: usize> Add for Coords<R, N, M> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        array_zip_map(*self, *rhs, |&a, &b| a + b).into()
    }
}

impl<R: Real, const N: usize, const M: usize> Sub for Coords<R, N, M> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        array_zip_map(*self, *rhs, |&a, &b| a - b).into()
    }
}

impl<R: Real, const N: usize, const M: usize> Mul<R> for Coords<R, N, M> {
    type Output = Self;

    fn mul(self, rhs: R) -> Self::Output {
        self.map(|x| x * rhs).into()
    }
}

impl<R: Real, const N: usize, const M: usize> Neg for Coords<R, N, M> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.map(|x| -x).into()
    }
}

impl<R: Real, const N: usize, const M: usize> Bilinear<R> for Coords<R, N, M> {
    fn dot(&self, other: &Self) -> R {
        self.iter()
            .zip(other.iter())
            .enumerate()
            .fold(
                R::zero(),
                |acc, (m, (&a, &b))| if m < M { acc - a * b } else { acc + a * b },
            )
    }
}

impl<R: Scalar, const N: usize, const M: usize> PseudoEuclidean for Coords<R, N, M> {
    type F = R;
    const N: usize = N;

    type Iter<'a>
        = std::slice::Iter<'a, R>
    where
        Self: 'a;

    fn iter(&self) -> Self::Iter<'_> {
        self.0.iter()
    }

    fn from_array<const K: usize>(s: [R; K]) -> Self {
        const { assert!(K == N) };
        std::array::from_fn(|i| s[i]).into()
    }

    fn from_fn(f: impl Fn(usize) -> Self::F) -> Self {
        std::array::from_fn(f).into()
    }
}

impl<R: Real, const N: usize> Metric<R> for Coords<R, N, 0> {
    fn distance(&self, other: &Self) -> R {
        self.iter()
            .zip(other.iter())
            .fold(R::zero(), |acc, (&a, &b)| acc + (b - a) * (b - a))
            .sqrt()
    }
}

impl<R: Real, const N: usize> InnerProduct<R> for Coords<R, N, 0> {}
impl<R: Scalar, const N: usize> Euclidean for Coords<R, N, 0> {}
