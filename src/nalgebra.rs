use crate::traits::{Euclidean, InnerProduct, Metric};
use nalgebra::{RealField, SVector};
use num_traits::real::Real;

impl<R: Real + RealField, const N: usize> Metric<R> for SVector<R, N> {
    fn distance(&self, other: &Self) -> R {
        (self - other).norm()
    }
}

impl<R: Real + RealField, const N: usize> InnerProduct<R> for SVector<R, N> {
    fn dot(&self, other: &Self) -> R {
        self.dot(other)
    }
}

impl<R: Real + RealField, const N: usize> Euclidean for SVector<R, N> {
    type Scalar = R;

    type Iter<'a>
        = nalgebra::base::iter::MatrixIter<
        'a,
        R,
        nalgebra::Const<N>,
        nalgebra::Const<1>,
        nalgebra::ArrayStorage<R, N, 1>,
    >
    where
        Self: 'a;

    fn iter(&self) -> Self::Iter<'_> {
        nalgebra::Matrix::iter(self)
    }

    fn from_array<const M: usize>(s: [R; M]) -> Self {
        const { assert!(M == N) };
        std::array::from_fn(|i| s[i]).into()
    }
}
