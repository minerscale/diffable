use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

use num_traits::{Inv, One, Zero};

use crate::{
    complex::Complex,
    coords::Coords,
    impl_group_via_mul, impl_lie_group_via_quotient,
    matrix::{Matrix, MatrixExponential},
    traits::{
        Bilinear, Field, InvolutiveField, LieGroup, Quadratic, Quotient, Real, RootOfUnity,
    },
};

pub type Minkowski<R> = Coords<R, 4, 1>;

#[derive(Debug, Copy, Clone)]
pub struct Sl<const N: usize, F: Field>(Matrix<N, F>);

impl<const N: usize, F: InvolutiveField> PartialEq for Sl<N, F> {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

pub type Sl2c<R> = Sl<2, Complex<R>>;

/// The restricted lorentz group
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Lorentz<R: Real>(Sl2c<R>);

impl<R: Real> Quotient<Sl2c<R>, RootOfUnity<Complex<R>, 2>, SlAlgebra<Complex<R>, 2, 3>>
    for Lorentz<R>
{
    fn new(g: Sl2c<R>) -> Self {
        let neg_g = Sl(g.0 * (-Complex::<R>::one()));
        let [re, im] = g.0.trace().into();
    
        // Tolerant comparison, deliberately — a Re(tr) that's zero up to
        // R's own tolerance should be treated as if it were exactly zero,
        // falling through to the next tiebreak, rather than having the
        // outcome depend on which way floating-point noise happened to
        // push a value that's morally on the boundary. Unlike the
        // Matrix::exp/log convergence loop, there's no risk of stranding
        // anything here — "fall through" always lands on a fresh, separate
        // comparison, never on the same one repeating forever.
        if R::zero() < re { return Self(g); }
        if re < R::zero() { return Self(neg_g); }
        if R::zero() < im { return Self(g); }
        if im < R::zero() { return Self(neg_g); }
    
        // Last resort: tr(g) is zero up to tolerance. g_ij and (-g)_ij are
        // exact negatives (negation by -1 is bit-exact), so the same
        // antisymmetry argument applies entrywise. Termination is
        // guaranteed by a different fact than before, though: it's not
        // that g != -g as literal values, it's that det(g) = 1 exactly
        // forbids every entry from being simultaneously tolerantly-zero
        // (a matrix that's tolerantly all-zero would have a
        // tolerantly-zero determinant, not 1).
        let g_wins = g.0.flat_iter().zip(neg_g.0.flat_iter())
            .find_map(|(&a, &b)| {
                let [are, aim] = a.into();
                let [bre, bim] = b.into();
                if are < bre { Some(true) }
                else if bre < are { Some(false) }
                else if aim < bim { Some(true) }
                else if bim < aim { Some(false) }
                else { None }
            })
            .expect("g tolerantly equals -g despite det(g) = 1 — shouldn't be possible");
    
        if g_wins { Self(g) } else { Self(neg_g) }
    }

    fn lift(&self) -> Sl2c<R> {
        self.0
    }

    fn embed(h: RootOfUnity<Complex<R>, 2>) -> Sl2c<R> {
        if h.is_one() {
            Sl2c::one()
        } else {
            // -I is in Sl<N, F> when N is even.
            Sl(-Matrix::one())
        }
    }
}

impl_lie_group_via_quotient!(Lorentz<R>, Sl2c<R>, RootOfUnity<Complex<R>,2>, SlAlgebra<Complex<R>, 2, 3>, R: Real);

impl<const N: usize, F: Field> Sl<N, F> {
    pub fn trace(&self) -> F {
        self.0.trace()
    }
}

impl<const N: usize, F: Field> One for Sl<N, F> {
    fn one() -> Self {
        Self(Matrix::one())
    }
}

impl<const N: usize, F: Field> Mul for Sl<N, F> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl<const N: usize, F: Field> Inv for Sl<N, F> {
    type Output = Self;

    fn inv(self) -> Self::Output {
        match N {
            0 => self,
            1 => self,
            2 => {
                let [[a, b], [c, d]] = self.0.destructure();

                let mut output = Matrix::zero();

                output[(0, 0)] = d;
                output[(0, 1)] = -b;
                output[(1, 0)] = -c;
                output[(1, 1)] = a;

                Sl(output)
            }
            _ => Sl(self.0.gauss_jordan()),
        }
    }
}

impl_group_via_mul!(Sl<N, F>, const N: usize, F: Field);

impl<F: Field> LieGroup<SlAlgebra<F, 2, 3>> for Sl<2, F>
where
    Matrix<2, F>: MatrixExponential,
{
    fn identity_exp(v: SlAlgebra<F, 2, 3>) -> Self {
        Self(Matrix::exp(&v.to_matrix()))
    }

    fn identity_log(p: &Self) -> Option<SlAlgebra<F, 2, 3>> {
        Matrix::log(&p.0).map(|x| SlAlgebra::from_matrix(x))
    }
}

#[derive(Debug, Copy, Clone)]
pub struct SlAlgebra<F: Field, const N: usize, const D: usize>(Coords<F, D>);

impl<F: InvolutiveField, const N: usize, const D: usize> PartialEq for SlAlgebra<F, N, D> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<F: Field, const N: usize, const D: usize> From<Coords<F, D>> for SlAlgebra<F, N, D> {
    fn from(value: Coords<F, D>) -> Self {
        Self(value)
    }
}

impl<F: Field, const N: usize, const D: usize> From<[F; D]> for SlAlgebra<F, N, D> {
    fn from(value: [F; D]) -> Self {
        Coords::from(value).into()
    }
}

impl<F: Field, const N: usize, const D: usize> From<SlAlgebra<F, N, D>> for [F; D] {
    fn from(value: SlAlgebra<F, N, D>) -> Self {
        value.0.into()
    }
}

impl<F: Field, const N: usize, const D: usize> Zero for SlAlgebra<F, N, D> {
    fn zero() -> Self {
        Self(Coords::zero())
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<F: Field, const N: usize, const D: usize> Add<Self> for SlAlgebra<F, N, D> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl<F: Field, const N: usize, const D: usize> Mul<F> for SlAlgebra<F, N, D> {
    type Output = Self;

    fn mul(self, rhs: F) -> Self::Output {
        Self(self.0 * rhs)
    }
}

impl<F: Field, const N: usize, const D: usize> Sub<Self> for SlAlgebra<F, N, D> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl<F: Field, const N: usize, const D: usize> Neg for SlAlgebra<F, N, D> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl<F: Field, const N: usize, const D: usize> Index<usize> for SlAlgebra<F, N, D> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<F: Field, const N: usize, const D: usize> IndexMut<usize> for SlAlgebra<F, N, D> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<F: Field, const N: usize, const D: usize> SlAlgebra<F, N, D> {
    fn to_matrix(&self) -> Matrix<N, F> {
        let mut out = [[F::zero(); N]; N];

        let mut index = 0;

        // Off diagonal E_ij
        for i in 0..N {
            for j in 0..N {
                if i != j {
                    out[i][j] = self[index];
                    index += 1;
                }
            }
        }

        // Diagonal H_k = E_kk - E_(k+1)(k+1)
        for k in 0..N - 1 {
            let c = self[index];

            out[k][k] = out[k][k] + c;
            out[k + 1][k + 1] = out[k + 1][k + 1].sub(&c);

            index += 1;
        }

        Matrix::new(out)
    }

    fn from_matrix(m: Matrix<N, F>) -> Self {
        let mut out = [F::zero(); D];

        let mut index = 0;

        // Off diagonal E_ij coefficients
        for i in 0..N {
            for j in 0..N {
                if i != j {
                    out[index] = m[(i, j)];
                    index += 1;
                }
            }
        }

        // Diagonal H_k coefficients
        let mut accum = F::zero();

        for k in 0..N - 1 {
            accum = accum + m[(k, k)];
            out[index] = accum;
            index += 1;
        }

        Self(out.into())
    }
}

// Here, D is N*N - 1, we do this this way because of const-generics restrictions
// we statically assert in the constructor that D = N*N - 1.
impl<F: Field, const N: usize, const D: usize> Bilinear<F> for SlAlgebra<F, N, D> {
    // Killing form / 2*N
    fn dot(&self, other: &Self) -> F {
        let x = self.to_matrix();
        let y = other.to_matrix();

        (x * y).trace()
    }
}

impl<F: Field, const N: usize, const D: usize> Quadratic for SlAlgebra<F, N, D> {
    type F = F;

    const N: usize = D;

    type Iter<'a>
        = std::slice::Iter<'a, F>
    where
        Self: 'a;

    fn iter(&self) -> Self::Iter<'_> {
        self.0.iter()
    }

    fn from_fn(f: impl Fn(usize) -> Self::F) -> Self {
        Self(Coords::<F, D>::from_fn(f))
    }

    fn from_array<const K: usize>(arr: [Self::F; K]) -> Self {
        Self(Coords::<F, D>::from_array(arr))
    }
}
