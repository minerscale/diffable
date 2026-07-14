use std::{
    array::from_fn,
    ops::{Add, Index, IndexMut, Mul, Neg, Sub},
};

use num_traits::{Inv, One, Zero};

use crate::{
    complex::Complex,
    coords::{Coords, array_zip_map},
    impl_group_via_mul,
    traits::{Bilinear, Field, InvolutiveField, LieGroup, NonZero, Quadratic, Real},
};

pub type Minkowski<R> = Coords<R, 4, 1>;

#[derive(Debug, Copy, Clone)]
pub struct Sl<const N: usize, F: Field>(Matrix<N, F>);

#[derive(Debug, Copy, Clone)]
pub struct Matrix<const N: usize, F: Field>([[F; N]; N]);

impl<const N: usize, F: Field> Matrix<N, F> {
    pub fn trace(&self) -> F {
        matrix_trace(self.0)
    }
}

impl<const N: usize, R: Real, F: InvolutiveField<Fixed = R>> Matrix<N, F> {
    pub fn frobenius_norm(&self) -> R {
        self.0
            .as_flattened()
            .iter()
            .fold(R::zero(), |acc, x| acc + x.norm_squared())
            .sqrt()
    }
}

impl<const N: usize, F: Field> Matrix<N, F> {
    fn zero() -> Self {
        Self(from_fn(|_| from_fn(|_| F::zero())))
    }
}

impl<const N: usize, F: Field> One for Matrix<N, F> {
    fn one() -> Self {
        Self(from_fn(|i| {
            from_fn(|j| if i == j { F::one() } else { F::zero() })
        }))
    }
}

impl<const N: usize, F: Field> Mul<Self> for Matrix<N, F> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(matrix_mul(self.0, rhs.0))
    }
}

impl<const N: usize, F: Field> Add<Self> for Matrix<N, F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Matrix(array_zip_map(self.0, rhs.0, |&v, &u| {
            array_zip_map(v, u, |&a, &b| a + b)
        }))
    }
}

impl<const N: usize, F: Field> Sub<Self> for Matrix<N, F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Matrix(array_zip_map(self.0, rhs.0, |&v, &u| {
            array_zip_map(v, u, |&a, &b| a.sub(&b))
        }))
    }
}

impl<const N: usize, F: Field> Mul<F> for Matrix<N, F> {
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

impl<const N: usize, R: Real, F: InvolutiveField<Fixed = R>> MatrixExponential for Matrix<N, F> { 
    fn exp(&self) -> Self {
        todo_warn!("\n\n⚠️ SHITTY IMPLEMENTATION ALERT:\nUses unstable Taylor series. Come back and refactor to Scaling and Squaring!\n");

        let mut result = Matrix::one();
        let mut term = Matrix::one();

        let epsilon = R::epsilon();

        for k in 1.. {
            term = term * *self;
            term = term * F::from_fixed(R::one() / R::from(k).unwrap());

            result = result + term;

            if term.frobenius_norm() < epsilon * result.frobenius_norm() {
                break;
            }

            if k > 256 {
                panic!("exp failed to converge");
            }
        }

        result
    }

    fn log(&self) -> Option<Self> {
        let log_radius: R = R::from(0.1).unwrap();
        let x = *self - Matrix::one();
    
        let norm = x.frobenius_norm();
    
        if norm >= log_radius {
            return None;
        }
    
        let epsilon =
            R::epsilon() * R::from(10.0).unwrap();
    
        let mut result = x;
        let mut term = x;
    
        for k in 2.. {
            term = term * x;
    
            let next =
                term * F::from_fixed((
                    if k % 2 == 0 {
                        -R::one()
                    } else {
                        R::one()
                    }
                )
                / R::from(k).unwrap());
    
            result = result + next;
    
            if next.frobenius_norm() < epsilon * result.frobenius_norm() {
                return Some(result);
            }
    
            if k > 256 {
                return None;
            }
        }
    
        None
    }
}

pub type SL2C<R> = Sl<2, Complex<R>>;

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

fn destructure<const N: usize, const M: usize, F: Field>(m: Matrix<N, F>) -> [[F; M]; M] {
    const { assert!(N == M) }
    from_fn(|i| from_fn(|j| m.0[i][j].clone()))
}

fn gauss_jordan<const N: usize, F: Field>(m: Matrix<N, F>) -> Matrix<N, F> {
    // Mutate a copy of our inner arrays (A) and an identity matrix (I)
    let mut mat = m.0;
    let mut inv: [[F; N]; N] = Matrix::one().0;

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
        let pivot_inv = F::Mul::inv(NonZero::new(mat[i][i].clone()).unwrap().into())
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
                    mat[k][j] = mat[k][j].sub(&mat_sub);

                    let inv_sub = factor.clone() * inv[i][j].clone();
                    inv[k][j] = inv[k][j].sub(&inv_sub);
                }
            }
        }
    }

    // Return the transformed identity matrix
    Matrix(inv)
}

impl<const N: usize, F: Field> Inv for Sl<N, F> {
    type Output = Self;

    fn inv(self) -> Self::Output {
        match N {
            0 => self,
            1 => self,
            2 => {
                let [[a, b], [c, d]] = destructure(self.0);

                let mut output = Matrix::zero();

                output.0[0][0] = d;
                output.0[0][1] = -b;
                output.0[1][0] = -c;
                output.0[1][1] = a;

                Sl(output)
            }
            _ => Sl(gauss_jordan(self.0)),
        }
    }
}

impl_group_via_mul!(Sl<N, F>, const N: usize, F: Field);

impl<F: Field, const N: usize, const D: usize> LieGroup<SlAlgebra<F, N, D>> for Sl<N, F>
where
    Matrix<N, F>: MatrixExponential,
{
    fn identity_exp(v: SlAlgebra<F, N, D>) -> Self {
        Self(Matrix::exp(&v.to_matrix()))
    }

    fn identity_log(p: &Self) -> Option<SlAlgebra<F, N, D>> {
        Matrix::log(&p.0).map(|x| SlAlgebra::from_matrix(x))
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct SlAlgebra<F: Field, const N: usize, const D: usize>(Coords<F, D>);

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

        Matrix(out)
    }

    fn from_matrix(m: Matrix<N, F>) -> Self {
        let mut out = [F::zero(); D];

        let mut index = 0;

        // Off diagonal E_ij coefficients
        for i in 0..N {
            for j in 0..N {
                if i != j {
                    out[index] = m.0[i][j];
                    index += 1;
                }
            }
        }

        // Diagonal H_k coefficients
        let mut accum = F::zero();

        for k in 0..N - 1 {
            accum = accum + m.0[k][k];
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
