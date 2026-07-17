use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

use num_traits::{Inv, One, Zero};

use crate::{
    complex::Complex,
    coords::Coords,
    impl_group_via_mul, impl_lie_group_via_quotient,
    matrix::{Matrix, MatrixExponential},
    traits::{
        Bilinear, Dual, Field, Form, LieGroup, Nondegenerate, Quadratic, Quotient, Real,
        RootOfUnity, Sesquilinear, Vector,
    },
};

pub type Minkowski<R> = Coords<R, 4, 1>;

/// The special linear group `SL(V)` — automorphisms of `V` with determinant one.
///
/// The determinant-one invariant is maintained *by construction*: there is no
/// raw constructor. Values arise only from the group operations (the identity,
/// products, inverses) and from [`exp`](crate::matrix::MatrixExponential::exp)
/// of the traceless [`SlAlgebra`] — all of which preserve `det = 1`
/// (`det(AB) = det(A)det(B)`, `det(exp X) = e^{tr X} = e^0`). Since the Lie
/// algebra has no invalid representations either, every reachable `Sl` value is
/// genuinely in the group; membership is a theorem about reachability, not a
/// runtime check.
#[derive(Debug, Copy, Clone)]
pub struct Sl<V: Vector, const N: usize>(Matrix<V, N>);

/// `SL(2, ℂ)` — the double cover of the restricted Lorentz group. See [`Lorentz`].
pub type Sl2c<R> = Sl<Coords<Complex<R>, 2>, 2>;

impl<V: Vector, const N: usize> PartialEq for Sl<V, N> {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

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
        if R::zero() < re {
            return Self(g);
        }
        if re < R::zero() {
            return Self(neg_g);
        }
        if R::zero() < im {
            return Self(g);
        }
        if im < R::zero() {
            return Self(neg_g);
        }

        // Last resort: tr(g) is zero up to tolerance. g_ij and (-g)_ij are
        // exact negatives (negation by -1 is bit-exact), so the same
        // antisymmetry argument applies entrywise. Termination is
        // guaranteed by a different fact than before, though: it's not
        // that g != -g as literal values, it's that det(g) = 1 exactly
        // forbids every entry from being simultaneously tolerantly-zero
        // (a matrix that's tolerantly all-zero would have a
        // tolerantly-zero determinant, not 1).
        let g_wins =
            g.0.flat_iter()
                .zip(neg_g.0.flat_iter())
                .find_map(|(&a, &b)| {
                    let [are, aim] = a.into();
                    let [bre, bim] = b.into();
                    if are < bre {
                        Some(true)
                    } else if bre < are {
                        Some(false)
                    } else if aim < bim {
                        Some(true)
                    } else if bim < aim {
                        Some(false)
                    } else {
                        None
                    }
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

impl<V: Vector, const N: usize> Sl<V, N> {
    pub fn trace(&self) -> V::F {
        self.0.trace()
    }
}

impl<V: Vector, const N: usize> One for Sl<V, N> {
    fn one() -> Self {
        Self(Matrix::one())
    }
}

impl<V: Vector, const N: usize> Mul for Sl<V, N> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl<V: Vector, const N: usize> Inv for Sl<V, N> {
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

impl_group_via_mul!(Sl<V, N>, V: Vector, const N: usize);

impl<F: Field> LieGroup<SlAlgebra<F, 2, 3>> for Sl<Coords<F, 2>, 2>
where
    Matrix<Coords<F, 2>, 2>: MatrixExponential,
{
    fn identity_exp(v: SlAlgebra<F, 2, 3>) -> Self {
        Self(Matrix::exp(&v.to_matrix()))
    }

    fn identity_log(p: &Self) -> Option<SlAlgebra<F, 2, 3>> {
        Matrix::log(&p.0).map(|x| SlAlgebra::from_matrix(x))
    }
}

/// The Lie algebra `𝔰𝔩(N)` — the traceless `N×N` matrices, tangent space to
/// [`Sl`].
///
/// Stored in coordinates as `Coords<F, D>` with `D = N² − 1` (the const
/// assertion in the constructor enforces the relation, which stable const
/// generics can't state directly). Every representation is valid — tracelessness
/// is built into the basis, so there are no invalid elements to exclude. Its
/// [`flat`](`crate::traits::Form`)/[`sharp`](`crate::traits::Nondegenerate`)
/// implement the (normalised) Killing form `⟨X, Y⟩ = tr(XY)`,
/// with the Cartan block carrying the `A_{N−1}` Cartan matrix and its inverse.
#[derive(Debug, Copy, Clone)]
pub struct SlAlgebra<F: Field, const N: usize, const D: usize>(Coords<F, D>);

impl<F: Field, const N: usize, const D: usize> PartialEq for SlAlgebra<F, N, D> {
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
    fn to_matrix(&self) -> Matrix<Coords<F, N>, N> {
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
            out[k + 1][k + 1] = out[k + 1][k + 1] - c;

            index += 1;
        }

        Matrix::new(out)
    }

    fn from_matrix(m: Matrix<Coords<F, N>, N>) -> Self {
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

fn offdiag_index<const N: usize>(i: usize, j: usize) -> usize {
    debug_assert!(i != j);

    let before = i * (N - 1);

    before + if j < i { j } else { j - 1 }
}

impl<F: Field, const N: usize, const D: usize> Form for SlAlgebra<F, N, D> {
    fn flat(&self) -> Dual<Self> {
        let mut out = *self;

        // ----- Root spaces -----
        // B(E_ij,E_kl)=δ_jk δ_il
        // so E_ij maps to the dual of E_ji.
        for i in 0..N {
            for j in (i + 1)..N {
                let a = offdiag_index::<N>(i, j);
                let b = offdiag_index::<N>(j, i);

                out.0.swap(a, b);
            }
        }

        // ----- Cartan -----
        // Multiply by the A_{N-1} Cartan matrix.
        let base = N * (N - 1);

        for i in 0..N - 1 {
            let mut x = self[base + i] + self[base + i];

            if i > 0 {
                x = x - self[base + i - 1];
            }

            if i + 1 < N - 1 {
                x = x - self[base + i + 1];
            }

            out[base + i] = x;
        }

        Dual::from_raw(out)
    }
}

impl<F: Field, const N: usize, const D: usize> Nondegenerate for SlAlgebra<F, N, D> {
    fn sharp(v: Dual<Self>) -> Self {
        let mut out = Dual::to_raw(v);

        // Root spaces:
        for i in 0..N {
            for j in (i + 1)..N {
                let a = offdiag_index::<N>(i, j);
                let b = offdiag_index::<N>(j, i);

                out.0.swap(a, b);
            }
        }

        let base = N * (N - 1);

        // Need the original RHS while overwriting.
        // So do one coordinate at a time into a temporary scalar.
        for i in 0..N - 1 {
            let mut sum = F::zero();

            for j in 0..N - 1 {
                let coeff_num = (usize::min(i, j) + 1) * (N - usize::max(i, j) - 1);

                let coeff = F::from_nat(coeff_num).div(F::from_nat(N));

                sum = sum + coeff * v[j + base];
            }

            out[base + i] = sum;
        }

        out
    }
}

impl<F: Field<Fixed = F>, const N: usize, const D: usize> Sesquilinear for SlAlgebra<F, N, D> {}
impl<F: Field, const N: usize, const D: usize> Quadratic for SlAlgebra<F, N, D> where
    SlAlgebra<F, N, D>: Bilinear
{
}

impl<F: Field, const N: usize, const D: usize> Vector for SlAlgebra<F, N, D> {
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
}
