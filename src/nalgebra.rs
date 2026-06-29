use crate::traits::{Chart, Euclidean, ExpMap, InnerProduct, Metric, TangentBundle};
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

impl<R: Real + RealField, const N: usize> Chart<Self, N, Self> for SVector<R, N> {
    fn to_local(&self, point: &Self) -> Option<Self> {
        Some((point - self).into())
    }
    fn to_global(&self, coord: Self) -> Self {
        self + SVector::from(coord)
    }
    fn chart_at(p: &Self) -> Self {
        p.clone()
    }
}

impl<R: Real + RealField, const N: usize> ExpMap<Self, N, Self> for SVector<R, N> {}
impl<R: Real + RealField, const N: usize> TangentBundle<Self, N, Self> for SVector<R, N> {}

impl<R: Real + RealField, const N: usize> Euclidean<N> for SVector<R, N> {
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
        Self: 'a,
        R: 'a;

    fn iter(&self) -> Self::Iter<'_> {
        nalgebra::Matrix::iter(self)
    }
}
