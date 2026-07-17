use std::{
    array::from_fn,
    ops::{Add, Index, IndexMut, Mul, Neg, Sub},
};

use num_traits::{Inv, NumCast, One, Zero, real::Real as _};

use crate::{
    coords::array_zip_map,
    traits::{DivRing, Dual, ExactCmp, Field, Metric, NatZero, NonZero, Vector},
};

/// A matrix, interpreted as the (1, 1) tensor V ⊗ V*.
/// N must be equal to V::N. This is enforced by all constructors
/// at compile time. This is due to limitations in Rust's const generics.
#[derive(Debug, Copy, Clone)]
pub struct Matrix<V: Vector, const N: usize>([[V::F; N]; N]);

impl<V: Vector, const N: usize> Index<(usize, usize)> for Matrix<V, N> {
    type Output = V::F;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.0[index.0][index.1]
    }
}

impl<V: Vector, const N: usize> IndexMut<(usize, usize)> for Matrix<V, N> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.0[index.0][index.1]
    }
}

impl<V: Vector, const N: usize> PartialEq for Matrix<V, N> {
    fn eq(&self, other: &Self) -> bool {
        // Scale is computed from `self` alone, not chained with
        // `other` — this looks like it should break symmetry (self.eq(other) vs
        // other.eq(self) using different denominators) but it doesn't, because of
        // some cool math:
        //
        // If self.eq(other) holds, every coordinate satisfies
        // (self_i - other_i)² < scale(self)·ε, so each |diff_i| is bounded by
        // √(scale(self)·ε) — tiny relative to √scale(self). Expanding
        // scale(other) = Σ(self_i - diff_i)² and bounding the cross term by
        // Cauchy–Schwarz gives scale(other) = scale(self)·(1 ± O(√ε)): the two
        // scales can only differ by a relative amount on the order of √ε
        // (~1e-6), nowhere near enough to move the ratio across the tolerance
        // boundary. So whenever the comparison would say "equal," the two
        // scales already agree closely enough that it doesn't matter whose you
        // used.
        //
        // And whenever they *don't* agree closely, they're obviously unequal,
        // so they'll not be equal.
        let zero = <V::F as Field>::Fixed::zero();

        let scale = self
            .0
            .as_flattened()
            .iter()
            .fold(zero, |acc, x| acc + x.norm_squared());

        self.0
            .as_flattened()
            .iter()
            .zip(other.0.as_flattened().iter())
            .all(|(&a, &b)| {
                let diff_sq = (a + (-b)).norm_squared();
                if scale == zero {
                    diff_sq == zero
                } else {
                    zero == diff_sq.div(scale)
                }
            })
    }
}

impl<F: Field, V: Vector<F = F>, const N: usize> Matrix<V, N> {
    /// The contraction (V ⊗ V*) ⊗ (V ⊗ V*) -> (V ⊗ V*)
    pub fn mul(&self, rhs: &Self) -> Self {
        Self(matrix_mul(self.0, rhs.0))
    }

    /// The contraction (V ⊗ V*) ⊗ V -> V
    pub fn mul_v(&self, v: &V) -> V {
        V::from_fn(|i| (0..N).fold(V::F::zero(), |acc, j| acc + self[(i, j)] * v[j]))
    }

    /// The contraction V* ⊗ (V ⊗ V*) -> V*
    pub fn mul_dual_v(&self, v: &Dual<V>) -> Dual<V> {
        Dual::from_fn(|j| (0..N).fold(V::F::zero(), |acc, i| acc + v[i] * self[(i, j)]))
    }

    /// The transpose V ⊗ V* -> V* ⊗ V** ≅ V* ⊗ V
    pub fn transpose(self) -> Matrix<Dual<V>, N> {
        Matrix::new(std::array::from_fn(|i| {
            std::array::from_fn(|j| self[(j, i)])
        }))
    }

    pub fn new(m: [[F; N]; N]) -> Self {
        const {
            assert!(V::N == N);
        }

        Matrix(m)
    }

    pub fn flat_iter<'a>(&'a self) -> impl Iterator<Item = &'a F>
    where
        F: 'a,
    {
        self.0.as_flattened().iter()
    }

    pub fn trace(&self) -> F {
        matrix_trace(self.0)
    }

    pub fn destructure<const M: usize>(&self) -> [[F; M]; M] {
        const { assert!(N == M) }
        from_fn(|i| from_fn(|j| self.0[i][j].clone()))
    }

    pub fn gauss_jordan(&self) -> Self {
        // Mutate a copy of our inner arrays (A) and an identity matrix (I)
        let mut mat = self.0;
        let mut inv: [[F; N]; N] = Matrix::<V, N>::one().0;

        for i in 0..N {
            // 1. Find the pivot row (first non-zero element in column `i` downwards)
            let mut pivot = i;

            // Assuming your Field can check for zero.
            // If using `num_traits::Zero`, use `.is_zero()`.
            // If using `PartialEq`, use `mat[pivot][i] == F::zero()`.
            while pivot < N && mat[pivot][i].is_zero() {
                pivot += 1;
            }

            // A matrix in the Special Linear group (det = 1) is strictly invertible.
            // If we hit N, the matrix is singular (det = 0), which violates the type's invariants.
            assert!(
                pivot < N,
                "Matrix is singular; violates Sl group properties."
            );

            // 2. Swap the current row with the pivot row in both matrices
            // Rust's slice::swap handles this beautifully and safely in place.
            mat.swap(i, pivot);
            inv.swap(i, pivot);

            // 3. Scale the pivot row so the pivot element becomes exactly 1
            let pivot_inv =
                <F as DivRing>::Mul::inv(NonZero::new(mat[i][i].clone()).unwrap().into())
                    .into()
                    .0;
            for j in 0..N {
                mat[i][j] = mat[i][j].clone() * pivot_inv.clone();
                inv[i][j] = inv[i][j].clone() * pivot_inv.clone();
            }

            // 4. Eliminate all other entries in the current column `i`
            for k in 0..N {
                if k != i {
                    let factor = mat[k][i].clone();

                    for j in 0..N {
                        // Equivalent to: row[k] = row[k] - (factor * row[i])
                        let mat_sub = factor.clone() * mat[i][j].clone();
                        mat[k][j] = mat[k][j] - mat_sub;

                        let inv_sub = factor.clone() * inv[i][j].clone();
                        inv[k][j] = inv[k][j] - inv_sub;
                    }
                }
            }
        }

        // Return the transformed identity matrix
        Matrix(inv)
    }
}

impl<F: Field + Metric, V: Vector<F = F>, const N: usize> Matrix<V, N> {
    pub fn frobenius_norm(&self) -> F::R {
        self.0
            .as_flattened()
            .iter()
            .fold(F::R::zero(), |acc, x| acc + x.interval_squared(&F::zero()))
            .sqrt()
    }
}

impl<F: Field, V: Vector<F = F>, const N: usize> Zero for Matrix<V, N> {
    fn zero() -> Self {
        const { assert!(V::N == N) }

        Self(from_fn(|_| from_fn(|_| F::zero())))
    }

    fn is_zero(&self) -> bool {
        self.0.as_flattened().iter().all(|x| x.is_zero())
    }
}

impl<F: Field, V: Vector<F = F>, const N: usize> One for Matrix<V, N> {
    fn one() -> Self {
        const { assert!(V::N == N) }

        Self(from_fn(|i| {
            from_fn(|j| if i == j { F::one() } else { F::zero() })
        }))
    }
}

impl<F: Field, V: Vector<F = F>, const N: usize> Mul<Self> for Matrix<V, N> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(matrix_mul(self.0, rhs.0))
    }
}

impl<F: Field, V: Vector<F = F>, const N: usize> Add<Self> for Matrix<V, N> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Matrix(array_zip_map(self.0, rhs.0, |&v, &u| {
            array_zip_map(v, u, |&a, &b| a + b)
        }))
    }
}

impl<F: Field, V: Vector<F = F>, const N: usize> Sub<Self> for Matrix<V, N> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Matrix(array_zip_map(self.0, rhs.0, |&v, &u| {
            array_zip_map(v, u, |&a, &b| a - b)
        }))
    }
}

impl<F: Field, V: Vector<F = F>, const N: usize> Neg for Matrix<V, N> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(self.0.map(|v| v.map(|x| -x)))
    }
}

impl<F: Field, V: Vector<F = F>, const N: usize> Mul<F> for Matrix<V, N> {
    type Output = Self;

    fn mul(self, rhs: F) -> Self::Output {
        Self(self.0.map(|v| v.map(|x| x * rhs)))
    }
}

pub trait MatrixExponential: Sized {
    fn exp(&self) -> Self;
    fn log(&self) -> Option<Self>;
}

macro_rules! todo_warn {
    ($msg:expr) => {{
        #[must_use = $msg]
        struct Warning;
        // We use an expression block to force the compiler to notice
        // that a `must_use` structure was dropped without being used.
        Warning
    }};
}

impl<const N: usize, F: Field<Characteristic = NatZero> + Metric, V: Vector<F = F>>
    MatrixExponential for Matrix<V, N>
{
    fn exp(&self) -> Self {
        todo_warn!(
            "\n\n⚠️ SHITTY IMPLEMENTATION ALERT:\nUses unstable Taylor series. Come back and refactor to Scaling and Squaring!\n"
        );

        let mut result = Matrix::one();
        let mut term = Matrix::one();

        let epsilon = F::R::epsilon();

        let mut k_as_f = F::one();
        for k in 1.. {
            term = term * *self;
            term = term * F::one().div(k_as_f);
            k_as_f = F::one() + k_as_f;

            result = result + term;

            if term
                .frobenius_norm()
                .exact_le(epsilon * result.frobenius_norm() + epsilon)
            {
                break;
            }

            if k > 256 {
                panic!("exp failed to converge");
            }
        }

        result
    }

    fn log(&self) -> Option<Self> {
        let log_radius: F::R = <F::R as NumCast>::from(0.1).unwrap();
        let x = *self - Matrix::one();

        let norm = x.frobenius_norm();

        if norm >= log_radius {
            return None;
        }

        let epsilon = F::R::epsilon();

        let mut result = x;
        let mut term = x;

        let mut k_as_f = F::one() + F::one();
        for k in 2.. {
            term = term * x;

            let next = term * (if k % 2 == 0 { -F::one() } else { F::one() }).div(k_as_f);

            k_as_f = k_as_f + F::one();

            result = result + next;

            if next
                .frobenius_norm()
                .exact_le(epsilon * result.frobenius_norm())
            {
                return Some(result);
            }

            if k > 256 {
                panic!("log failed to converge");
            }
        }

        None
    }
}

fn matrix_trace<const N: usize, F: Field>(a: [[F; N]; N]) -> F {
    a.iter()
        .enumerate()
        .fold(F::zero(), |acc, (i, v)| acc + v[i])
}

fn matrix_mul<const N: usize, F: Field>(a: [[F; N]; N], b: [[F; N]; N]) -> [[F; N]; N] {
    let mut output = from_fn(|_| from_fn(|_| F::zero()));

    for i in 0..N {
        for j in 0..N {
            output[i][j] = (0..N).fold(F::zero(), |acc, k| acc + a[i][k] * b[k][j])
        }
    }

    output
}
