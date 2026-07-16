use std::ops::{Add, Deref, DerefMut, Index, IndexMut, Mul, Neg, Sub};

use num_traits::{ConstZero, Zero};

use crate::{
    complex::Complex,
    traits::{
        Bilinear, DivRing, Euclidean, Field, Interval, InvolutiveField, Metric, Quadratic, Real,
        RealStructure, Sesquilinear,
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
#[derive(Debug, Copy, Clone)]
pub struct Coords<F: Field, const N: usize, const M: usize = 0>([F; N]);

impl<F: Field, const N: usize, const M: usize> Zero for Coords<F, N, M> {
    fn zero() -> Self {
        [F::zero(); N].into()
    }

    fn is_zero(&self) -> bool {
        self.iter().all(|x| x == &F::zero())
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

/// A Hermitian form on an `F`-vector space, `F: InvolutiveField`.
///
/// The space of all values of a type `V: Sesquilinear<F>` carries a pairing
/// `⟨·,·⟩: V × V → F` that is linear in its first argument and
/// conjugate-linear in its second — the standard convention for a Hermitian
/// form. Unlike [`Bilinear`], which this trait deliberately does not
/// require, the pairing here is aware of `F`'s conjugation and its
/// self-pairing `⟨v,v⟩` is always self-adjoint (the `a == b` case of
/// Hermitian symmetry below), which is what licenses `norm_squared`
/// dropping it into `F::Fixed` via [`InvolutiveField::to_fixed`].
///
/// - **Hermitian symmetry**: `⟨v,w⟩ = conj(⟨w,v⟩)`.
/// - **Additivity** (first argument): `⟨v₁+v₂,w⟩ = ⟨v₁,w⟩ + ⟨v₂,w⟩`.
/// - **Scalar linearity** (first argument): `⟨v·k,w⟩ = k·⟨v,w⟩`.
///
/// Additivity and conjugate-linearity in the *second* argument are
/// corollaries of these three and are not independently certified, for the
/// same reason [`Bilinear`] doesn't separately test its own second
/// argument.
///
/// There is deliberately no blanket implementation of this trait over
/// [`Quadratic`]: a Hermitian form has to agree with whatever coordinate
/// structure a type actually carries, and that agreement is not mechanical
/// in general. It is free for `Coords`, whose form is a signed coordinatewise
/// sum in a basis where that sum is already meaningful; it is not free for a
/// type like a Lie algebra whose natural coordinates (root vectors, Cartan
/// generators) are not an eigenbasis of its own [`Bilinear`] pairing at all —
/// there, a Hermitian form is a genuine additional choice (which real form)
/// and must be supplied by hand. See [`RealStructure`] for what it means for
/// a `Sesquilinear` form to be *compatible* with a [`Bilinear`] one on the
/// same type, which is likewise never assumed automatically.
///
/// Certified by implementing this trait; verified by `test_sesquilinear!`.
///
/// [`Bilinear`]: crate::traits::Bilinear
/// [`Quadratic`]: crate::traits::Quadratic
/// [`RealStructure`]: crate::traits::RealStructure
impl<F: InvolutiveField, const N: usize, const M: usize> Sesquilinear<F> for Coords<F, N, M> {
    fn hermitian(&self, rhs: &Self) -> F {
        self.iter()
            .zip(rhs.iter())
            .enumerate()
            .fold(F::zero(), |acc, (m, (&a, &b))| {
                let x = a * b.conj();
                if m < M { acc.sub(&x) } else { acc + x }
            })
    }
}

/// A real structure on an `F`-vector space that already carries both a
/// [`Bilinear`] pairing and a [`Sesquilinear`] form: an antilinear involution
/// relating the two.
///
/// The name is not decorative. An antilinear involution `σ` on an
/// `F`-vector space `V` is exactly the data of a real form: `V₀ :=
/// {v : σ(v) = v}` is a genuine `Fixed`-vector space with `V ≅ V₀ ⊗_Fixed
/// F`, the same fixed-point construction [`InvolutiveField`] performs on the
/// scalar field itself, one level up. `su(n)` is precisely the fixed
/// subspace of `X ↦ -X†` on `sl(n,ℂ)` in this exact sense — the standard
/// meaning of calling `su(n)` *the* (compact) real form of `sl(n,ℂ)` in Lie
/// theory. Different valid choices of `conj` on the same `V` correspond to
/// different, inequivalent real forms (`su(n)` vs. `su(p,q)` vs. `sl(n,ℝ)`,
/// all real forms of the same complexified Lie algebra) — so `conj` is
/// genuine mathematical content the implementor supplies, not something
/// derivable from `V`'s dimension or field alone.
///
/// - **Involution**: `v.conj().conj() == v`.
/// - **Antilinearity**: `(v·k).conj() == v.conj() · k.conj()` — the scalar
///   picks up [`InvolutiveField::conj`] crossing this trait's `conj`,
///   mirroring the conjugate-linearity of [`Sesquilinear`]'s second
///   argument.
/// - **Compatibility of forms**: `Sesquilinear::hermitian(a,b) ==
///   Bilinear::dot(a, b.conj())`. This is the sole reason `RealStructure`
///   requires both supertraits at once — it certifies that the `Bilinear`
///   and `Sesquilinear` structures on `Self` were built to agree, rather
///   than merely coexisting.
///
/// None of these three are corollaries of the others. Involution is not
/// derivable from compatibility of forms at all. Antilinearity *would* be
/// derivable from compatibility together with non-degeneracy of `Bilinear`'s
/// pairing — but this library permits degenerate `Bilinear` forms on purpose
/// (a null hypersurface's induced pairing, or the Killing form of a
/// solvable Lie algebra, are both legitimately degenerate and not excluded
/// by [`Bilinear`]'s definition), so that premise is never available and all
/// three are independently certified.
///
/// `Coords<F,N,M>` gets this automatically (`conj` is entrywise
/// [`InvolutiveField::conj`]), precisely because its [`Bilinear`] and
/// [`Sesquilinear`] pairings are both literally the same signed coordinate
/// sum, one with a `.conj()` inserted — which is a fact about how `Coords`
/// happens to be built, not a property every [`Quadratic`] type shares, and
/// is why this trait still has no blanket implementation over `Quadratic`
/// in general.
///
/// Certified by implementing this trait; verified by `test_real_structure!`.
///
/// [`Bilinear`]: crate::traits::Bilinear
/// [`Sesquilinear`]: crate::traits::Sesquilinear
/// [`InvolutiveField`]: crate::traits::InvolutiveField
/// [`Quadratic`]: crate::traits::Quadratic
impl<F: InvolutiveField, const N: usize, const M: usize> RealStructure<F> for Coords<F, N, M> {
    fn conj(&self) -> Self {
        Self::from_fn(|i| self[i].conj())
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

impl<F: InvolutiveField, const N: usize, const M: usize> PartialEq for Coords<F, N, M> {
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
