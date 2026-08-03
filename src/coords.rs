use std::ops::{Deref, DerefMut, Index, IndexMut};

use num_traits::{ConstZero, Zero};

use crate::{
    impl_vector_ops,
    traits::{
        BothSided, DivRing, Dual, Euclidean, Field, Form, FormLift, Interval, JetMode, JetVector,
        Metric, Nondegenerate, NondegenerateLift, Point, Real, Right, Sesquilinear, Tensor,
    },
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
/// [`Euclidean`]; the general case implements [`Sesquilinear`] and, in cases
/// where the fixed field is itself, [`Bilinear`]
///
/// [`Bilinear`]: crate::traits::Bilinear
/// [`InnerProduct`]: crate::traits::InnerProduct
/// [`Metric`]: crate::traits::Metric
/// [`Euclidean`]: crate::traits::Euclidean
/// [`Sesquilinear`]: crate::traits::Sesquilinear
#[derive(Debug, Copy, Clone)]
pub struct Coords<F: Field, const N: usize, const M: usize = 0>(pub [F; N]);

impl<F: Field + ConstZero, const N: usize, const M: usize> ConstZero for Coords<F, N, M> {
    const ZERO: Self = Self([F::ZERO; N]);
}

impl<F: Field> From<F> for Coords<F, 1> {
    fn from(value: F) -> Self {
        Self([value])
    }
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

impl_vector_ops!(Coords<F, N, M>, F: Field, const N: usize, const M: usize);

impl<F: Field, const N: usize, const M: usize> Tensor for Coords<F, N, M> {
    type F = F;
    type Hand = Right;
    type Action = BothSided;

    type Array<T: Point> = [T; N];

    fn from_fn(f: impl FnMut(usize) -> Self::F) -> Self {
        Self(std::array::from_fn(f))
    }
}

impl<F: Field, const N: usize, const M: usize> AsRef<[F; N]> for Coords<F, N, M> {
    fn as_ref(&self) -> &[F; N] {
        &self.0
    }
}

impl<F: Field, const N: usize, const M: usize> AsMut<[F; N]> for Coords<F, N, M> {
    fn as_mut(&mut self) -> &mut [F; N] {
        &mut self.0
    }
}

impl<R: Real, F: Field<Fixed = R>, const N: usize, const M: usize> Interval for Coords<F, N, M> {
    type R = R;

    fn interval_squared(&self, other: &Self) -> R {
        (*self - *other).norm_squared()
    }
}

impl<R: Field + Real, const N: usize> Metric for Coords<R, N, 0> {
    fn distance(&self, other: &Self) -> R {
        let displacement = *self - *other;
        displacement.dot(&displacement).sqrt()
    }
}

impl<F: Field, const N: usize, const M: usize> PartialEq for Coords<F, N, M> {
    fn eq(&self, other: &Self) -> bool {
        // Coordinatewise closeness, scaled against the WHOLE vector rather
        // than each coordinate's own magnitude — otherwise a coordinate
        // that's much smaller than its neighbours gets an unfairly tight
        // absolute budget carved out of rounding error the larger
        // coordinates already spent (see the Complex<R64> incident).
        //
        // Deliberately uses InvolutiveField::norm_squared, not
        // Bilinear::dot: this is a question about the floating-point
        // REPRESENTATION, independent of whatever (possibly indefinite,
        // possibly non-Hermitian) geometric pairing M gives the vector —
        // Bilinear::dot can be negative or zero for a genuinely nonzero
        // vector on an indefinite or complex-non-Hermitian form, which
        // would make THAT a strictly worse comparison than the one being
        // fixed.
        let scale = self
            .iter()
            .fold(F::Fixed::zero(), |acc, x| acc + x.norm_squared());

        self.iter().zip(other.iter()).all(|(&a, &b)| {
            let diff_sq = (a + (-b)).norm_squared();
            if scale == F::Fixed::zero() {
                diff_sq == F::Fixed::zero()
            } else {
                F::Fixed::zero() == diff_sq.div(scale) // reuses Fixed's own tolerant `==`
            }
        })
    }
}

impl<R: Field, const N: usize, const M: usize> Form for Coords<R, N, M> {
    fn flat(&self) -> Dual<Self> {
        Dual::from_fn(|i| if i < M { -self[i] } else { self[i] }.conj())
    }
}

impl<R: Field, const N: usize, const M: usize> Nondegenerate for Coords<R, N, M> {
    fn sharp(v: Dual<Self>) -> Self {
        Self::from_fn(|i| if i < M { -v[i] } else { v[i] }.conj())
    }
}

impl<R: Field, const N: usize, const M: usize> FormLift for Coords<R, N, M> {
    fn jet_flat<S: Field, Mode: JetMode, const K: usize>(
        value: &JetVector<Self, Mode, K, S>,
    ) -> Dual<JetVector<Self, Mode, K, S>>
    where
        JetVector<Self, Mode, K, S>: Tensor,
    {
        Dual::from_fn(|i| if i < M { -value[i] } else { value[i] }.conj())
    }
}

impl<R: Field, const N: usize, const M: usize> NondegenerateLift for Coords<R, N, M> {
    fn jet_sharp<S: Field, Mode: JetMode, const K: usize>(
        value: Dual<JetVector<Self, Mode, K, S>>,
    ) -> JetVector<Self, Mode, K, S>
    where
        JetVector<Self, Mode, K, S>: Tensor,
    {
        JetVector::from_fn(|i| if i < M { -value[i] } else { value[i] }.conj())
    }
}

impl<F: Field, const N: usize, const M: usize> Sesquilinear for Coords<F, N, M> {}
impl<R: Real, const N: usize> Euclidean for Coords<R, N, 0> {}
