use std::{
    array::from_fn,
    ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub},
};

use num_traits::{Inv, NumCast, One, Zero, real::Real as _};

use crate::{
    coords::array_zip_map,
    traits::{
        DivRing, Dual, ExactCmp, Field, FieldExp, FromReal, Metric, NatZero, NonZero, Vector,
    },
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
    /// Wraps a raw `N×N` array as a matrix, checking `V::N == N` at compile
    /// time. The const assertion is the crate's stand-in for `Matrix<V, {V::N}>`,
    /// which stable const generics can't express — it guarantees the matrix's
    /// dimension matches the space it acts on.
    pub fn new(m: [[F; N]; N]) -> Self {
        const {
            assert!(V::N == N);
        }

        Matrix(m)
    }

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

    /// Iterates all `N²` entries in row-major order.
    pub fn flat_iter<'a>(&'a self) -> impl Iterator<Item = &'a F>
    where
        F: 'a,
    {
        self.0.as_flattened().iter()
    }

    /// The trace `Σᵢ Mᵢᵢ` — the contraction `V ⊗ V* -> F`.
    pub fn trace(&self) -> F {
        matrix_trace(self.0)
    }

    /// Extracts the raw entry array, checking `N == M` at compile time. Escape
    /// hatch back to a plain `[[F; M]; M]` for callers that need the components
    /// directly.
    pub fn destructure<const M: usize>(&self) -> [[F; M]; M] {
        const { assert!(N == M) }
        from_fn(|i| from_fn(|j| self.0[i][j]))
    }

    /// Solves A * X = B by Gauss–Jordan elimination.
    ///
    /// Assumes A is invertible.
    pub fn solve(&self, rhs: Self) -> Self {
        let mut mat = self.0;
        let mut out = rhs.0;

        for i in 0..N {
            // Pivot must be non-zero according to the scalar's equality semantics.
            assert_ne!(
                mat[i][i],
                F::zero(),
                "Matrix is singular during Gauss-Jordan elimination."
            );

            // Scale pivot row so the diagonal entry becomes one.
            let pivot_inv =
                <F as DivRing>::Mul::inv(NonZero::new(mat[i][i]).unwrap().into())
                    .into()
                    .0;

            for j in 0..N {
                mat[i][j] = mat[i][j] * pivot_inv;
                out[i][j] = out[i][j] * pivot_inv;
            }

            // Eliminate the pivot column from every other row.
            for k in 0..N {
                if k == i {
                    continue;
                }

                let factor = mat[k][i];

                for j in 0..N {
                    mat[k][j] = mat[k][j] - factor * mat[i][j];

                    out[k][j] = out[k][j] - factor * out[i][j];
                }
            }
        }

        Matrix(out)
    }

    fn swap_rows(&mut self, a: usize, b: usize) {
        if a != b {
            self.0.swap(a, b);
        }
    }

    /// Inverts the matrix by Gauss–Jordan elimination.
    ///
    /// Assumes invertibility: it `panic!`s on a zero pivot (a singular matrix).
    /// For an [`Sl`](crate::spacetime::Sl) element that panic is unreachable —
    /// determinant one is never singular — so this is a total operation on the
    /// special linear group, which is where it's used.
    pub fn inverse(&self) -> Self {
        self.solve(Matrix::one())
    }
}

impl<F: Field + Metric, V: Vector<F = F>, const N: usize> Matrix<V, N> {
    /// The Frobenius norm `√(Σᵢⱼ |Mᵢⱼ|²)`, valued in the real field `F::R`.
    ///
    /// Requires `F: Metric` so each entry has a *definite* squared magnitude
    /// (`interval_squared` against zero), keeping the sum a non-negative real.
    /// This is the norm the [`MatrixExponential`] Taylor series measures
    /// convergence against; it is submultiplicative, which is what makes that
    /// series converge.
    pub fn frobenius_norm(&self) -> F::R {
        self.0
            .as_flattened()
            .iter()
            .fold(F::R::zero(), |acc, x| acc + x.interval_squared(&F::zero()))
            .sqrt()
    }

    pub fn one_norm(&self) -> F::R {
        let mut max = F::R::zero();

        for col in 0..N {
            let mut sum = F::R::zero();

            for row in 0..N {
                sum = sum + self[(row, col)].distance(&F::zero());
            }

            if sum > max {
                max = sum;
            }
        }

        max
    }

    /// Solves A * X = B using Gauss–Jordan elimination with partial pivoting.
    ///
    /// Pivot rows are chosen by maximizing the scalar metric magnitude.
    /// This improves numerical stability for approximate fields.
    ///
    /// Assumes A is invertible.
    pub fn solve_pivoted(&self, rhs: Self) -> Self {
        let mut mat = self.0;
        let mut out = rhs.0;

        for i in 0..N {
            // Find the row with the largest pivot magnitude.
            let mut pivot = i;
            let mut pivot_norm = mat[i][i].interval_squared(&F::zero());

            for r in (i + 1)..N {
                let norm = mat[r][i].interval_squared(&F::zero());

                if norm > pivot_norm {
                    pivot = r;
                    pivot_norm = norm;
                }
            }

            assert!(
                !mat[pivot][i].is_zero(),
                "Matrix is singular during Gauss-Jordan elimination."
            );

            // Move the pivot into place.
            mat.swap(i, pivot);
            out.swap(i, pivot);

            // Normalize pivot row.
            let pivot_inv =
                <F as DivRing>::Mul::inv(NonZero::new(mat[i][i]).unwrap().into())
                    .into()
                    .0;

            for j in 0..N {
                mat[i][j] = mat[i][j] * pivot_inv;
                out[i][j] = out[i][j] * pivot_inv;
            }

            // Eliminate this column everywhere else.
            for r in 0..N {
                if r == i {
                    continue;
                }

                let factor = mat[r][i];

                for j in 0..N {
                    mat[r][j] = mat[r][j] - factor * mat[i][j];
                    out[r][j] = out[r][j] - factor * out[i][j];
                }
            }
        }

        Matrix(out)
    }

    pub fn inverse_pivoted(&self) -> Self {
        self.solve_pivoted(Self::one())
    }

    pub fn det(&self) -> F {
        let mut lu = *self;

        let mut perm: [usize; N] = core::array::from_fn(|i| i);
        let mut odd = false;

        for k in 0..N {
            //
            // Pivot search
            //
            let mut pivot = k;
            let mut best = lu[(k, k)].interval_squared(&F::zero());

            for r in (k + 1)..N {
                let norm = lu[(r, k)].interval_squared(&F::zero());

                if best.exact_le(norm) {
                    best = norm;
                    pivot = r;
                }
            }

            assert!(!lu[(pivot, k)].is_zero(), "Matrix is singular.");

            //
            // Swap rows.
            //
            if pivot != k {
                lu.swap_rows(k, pivot);
                perm.swap(k, pivot);
                odd = !odd;
            }

            //
            // Eliminate.
            //
            let pivot_inv = <F as DivRing>::Mul::inv(NonZero::new(lu[(k, k)]).unwrap().into())
                .into()
                .0;

            for i in (k + 1)..N {
                let multiplier = lu[(i, k)] * pivot_inv;

                //
                // Store L in-place.
                //
                lu[(i, k)] = multiplier;

                //
                // Update remaining row.
                //
                for j in (k + 1)..N {
                    lu[(i, j)] = lu[(i, j)] - multiplier * lu[(k, j)];
                }
            }
        }

        let mut det = if odd { -F::one() } else { F::one() };

        for i in 0..N {
            det = det * lu[(i, i)];
        }

        det
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

impl<F: Field, V: Vector<F = F>, const N: usize> Div<F> for Matrix<V, N> {
    type Output = Self;

    fn div(self, rhs: F) -> Self::Output {
        Self(self.0.map(|v| v.map(|x| x.div(rhs))))
    }
}

/// Matrices that can be exponentiated and (locally) logged.
///
/// [`exp`](MatrixExponential::exp) is the Lie-theoretic exponential
/// `Σ Aⁿ/n!`; [`log`](MatrixExponential::log) is its local inverse, defined
/// only within a small radius of the identity (returns `None` outside it).
/// Because the series needs `1/k!`, this is only implemented for scalar fields
/// of characteristic zero with a real metric — see the impl's bounds.
pub trait MatrixExponential: Sized {
    fn exp(&self) -> Self;
    fn log(&self) -> Option<Self>;
}

pub fn nth_root_near_one<F: Field + Metric>(a: &F, n: usize) -> F {
    assert!(n > 0);

    if n == 1 {
        return *a;
    }

    let n_f = F::from_nat(n);
    let mut y = F::one();

    let epsilon = F::R::epsilon();

    for _ in 0..32 {
        let y_pow = y.pow(n - 1);

        let next = ((F::from_nat(n - 1) * y) + a.div(y_pow)).div(n_f);

        let diff = next - y;

        y = next;

        if diff.distance(&F::zero()).exact_le(epsilon) {
            return y;
        }
    }

    panic!("didn't converge!");
}

impl<
    const N: usize,
    F: Field<Characteristic = NatZero> + Metric + FromReal + FieldExp,
    V: Vector<F = F>,
> MatrixExponential for Matrix<V, N>
{
    fn exp(&self) -> Self {
        let theta = <F::R as NumCast>::from(5.371920351148152).unwrap();

        // 13/13 pade approximant
        const B: [usize; 14] = const {
            [
                64764752532480000,
                32382376266240000,
                7771770303897600,
                1187353796428800,
                129060195264000,
                10559470521600,
                670442572800,
                33522128640,
                1323241920,
                40840800,
                960960,
                16380,
                182,
                1,
            ]
        };

        let b = B.map(|x| F::from_real(<F::R as NumCast>::from(x).unwrap()));

        let norm = self.one_norm();

        let s = if norm.exact_le(theta) {
            0
        } else {
            <usize as NumCast>::from((norm / theta).log2().ceil()).unwrap()
        };

        let two = F::one() + F::one();
        let mut a = *self;
        for _ in 0..s {
            a = a / two;
        }

        let a2 = a * a;
        let a4 = a2 * a2;
        let a6 = a4 * a2;

        let i = Matrix::one();

        let u = a
            * (a6 * (a6 * b[13] + a4 * b[11] + a2 * b[9])
                + a6 * b[7]
                + a4 * b[5]
                + a2 * b[3]
                + i * b[1]);

        let v = a6 * (a6 * b[12] + a4 * b[10] + a2 * b[8])
            + a6 * b[6]
            + a4 * b[4]
            + a2 * b[2]
            + i * b[0];

        let mut r = (v - u).solve_pivoted(v + u);

        for _ in 0..s {
            r = r * r;
        }

        r
    }

    fn log(&self) -> Option<Self> {
        let log_radius: F::R = <F::R as NumCast>::from(1.0).unwrap();
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
