use std::{
    array::from_fn,
    ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub},
};

use num_traits::{Inv, NumCast, One, Zero, real::Real as _};

use crate::{
    coords::array_zip_map,
    traits::{
        CField, DivRing, Dual, ExactCmp, Field, FieldExp, FromReal, Hand, Handedness, Interval,
        Metric, NatZero, NonZero, Vector,
    },
};

/// An endomorphism of `V`, represented as a `(1, 1)` tensor.
///
/// Its tensor interpretation follows the handedness of `V`:
///
/// - right-handed: `V ⊗ V*`;
/// - left-handed: `V* ⊗ V`.
///
/// The same raw array therefore has different index semantics on opposite
/// hands. Matrix multiplication and application preserve abstract composition
/// order in both cases.
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

    /// Scales every component by an element of the field's fixed subfield.
    ///
    /// The fixed subfield embeds into the centre of `V::F`, so this operation
    /// preserves the represented endomorphism even when `V::F` is
    /// noncommutative.
    pub fn scale_fixed(self, fixed: <V::F as Field>::Fixed) -> Self {
        Self(self.0.map(|v| v.map(|x| x * V::F::from_fixed(fixed))))
    }

    /// Applies this endomorphism to `v`.
    ///
    /// In tensor form, the contraction is:
    /// - `(V ⊗ V*) ⊗ V → V` for right modules;
    /// - `V ⊗ (V* ⊗ V) → V` for left modules.
    pub fn mul_v(&self, v: &V) -> V {
        V::from_fn(|i| {
            (0..N).fold(V::F::zero(), |acc, j| {
                acc + match V::Hand::H {
                    Hand::Right => self[(i, j)] * v[j],
                    Hand::Left => v[j] * self[(j, i)],
                }
            })
        })
    }

    /// Applies the induced dual endomorphism to `v`.
    ///
    /// This contracts `V* ⊗ (V ⊗ V*) → V*` when `V` is right-handed, or
    /// `(V* ⊗ V) ⊗ V* → V*` when `V` is left-handed.
    pub fn mul_dual_v(&self, v: &Dual<V>) -> Dual<V> {
        Dual::from_fn(|j| {
            (0..N).fold(V::F::zero(), |acc, i| {
                acc + match V::Hand::H {
                    Hand::Right => v[i] * self[(i, j)],
                    Hand::Left => self[(j, i)] * v[i],
                }
            })
        })
    }

    /// Reinterprets this endomorphism as one on the opposite-handed dual space.
    ///
    /// Abstractly this is
    /// `V ⊗ V* → V* ⊗ V** ≅ V* ⊗ V` (or the handedness-reversed analogue).
    /// It does **not** physically transpose the stored array: changing from `V`
    /// to `Dual<V>` changes which raw index represents the input and output.
    pub fn transpose(self) -> Matrix<Dual<V>, N> {
        Matrix::new(self.0)
    }

    /// Iterates all `N²` entries in row-major order.
    pub fn flat_iter<'a>(&'a self) -> impl Iterator<Item = &'a F>
    where
        F: 'a,
    {
        self.0.as_flattened().iter()
    }

    /// The trace `Σᵢ Mᵢᵢ` — contraction of the input and output indices.
    ///
    /// This is `V ⊗ V* → F` for right-handed `V` and
    /// `V* ⊗ V → F` for left-handed `V`.
    pub fn trace(&self) -> F
    where
        F: CField,
    {
        matrix_trace(self.0)
    }

    /// Extracts the raw entry array, checking `N == M` at compile time. Escape
    /// hatch back to a plain `[[F; M]; M]` for callers that need the components
    /// directly.
    pub fn destructure<const M: usize>(&self) -> [[F; M]; M] {
        const { assert!(N == M) }
        from_fn(|i| from_fn(|j| self.0[i][j]))
    }

    /// Solves the abstract composition equation `A ∘ X = B` by pivoted
    /// Gauss–Jordan elimination.
    ///
    /// In the raw representation this is `AX = B` for right-handed matrices
    /// and `XA = B` for left-handed matrices. Virtual rows are therefore
    /// physical rows on the right and physical columns on the left.
    ///
    /// At each step, the first remaining nonzero pivot is selected. This is
    /// purely algebraic and requires no metric or ordering.
    ///
    /// Assumes `A` is invertible.
    pub fn solve(&self, rhs: Self) -> Self {
        let get = |matrix: &[[V::F; N]; N], i: usize, j: usize| match V::Hand::H {
            Hand::Right => matrix[i][j],
            Hand::Left => matrix[j][i],
        };

        let set = |matrix: &mut [[V::F; N]; N], i: usize, j: usize, value: V::F| match V::Hand::H {
            Hand::Right => matrix[i][j] = value,
            Hand::Left => matrix[j][i] = value,
        };

        let mul = |lhs: V::F, rhs: V::F| match V::Hand::H {
            Hand::Right => lhs * rhs,
            Hand::Left => rhs * lhs,
        };

        let mut mat = self.0;
        let mut out = rhs.0;

        for i in 0..N {
            // Choose any nonzero pivot from the remaining virtual rows.
            let pivot_row = (i..N)
                .find(|&k| get(&mat, k, i) != V::F::zero())
                .expect("Matrix is singular during Gauss-Jordan elimination.");

            // Swap virtual rows in both sides of the equation. For a left-handed
            // matrix these are physically columns.
            if pivot_row != i {
                for j in 0..N {
                    let a = get(&mat, i, j);
                    let b = get(&mat, pivot_row, j);
                    set(&mut mat, i, j, b);
                    set(&mut mat, pivot_row, j, a);

                    let a = get(&out, i, j);
                    let b = get(&out, pivot_row, j);
                    set(&mut out, i, j, b);
                    set(&mut out, pivot_row, j, a);
                }
            }

            let pivot = get(&mat, i, i);
            let pivot_inv = <V::F as DivRing>::Mul::inv(NonZero::new(pivot).unwrap().into())
                .into()
                .0;

            // Normalize the virtual pivot row.
            for j in 0..N {
                let mat_value = mul(pivot_inv, get(&mat, i, j));
                let out_value = mul(pivot_inv, get(&out, i, j));

                set(&mut mat, i, j, mat_value);
                set(&mut out, i, j, out_value);
            }

            // Eliminate this coordinate from every other virtual row.
            for k in 0..N {
                if k == i {
                    continue;
                }

                let factor = get(&mat, k, i);

                for j in 0..N {
                    let mat_value = get(&mat, k, j) - mul(factor, get(&mat, i, j));

                    let out_value = get(&out, k, j) - mul(factor, get(&out, i, j));

                    set(&mut mat, k, j, mat_value);
                    set(&mut out, k, j, out_value);
                }
            }
        }

        Self::new(out)
    }

    /// Inverts the matrix by pivoted Gauss–Jordan elimination.
    ///
    /// Assumes invertibility and panics if the matrix is singular. For an
    /// [`Sl`](crate::spacetime::Sl) element that panic is unreachable because
    /// determinant one implies invertibility.
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
                sum = sum
                    + match V::Hand::H {
                        Hand::Right => self[(row, col)],
                        Hand::Left => self[(col, row)],
                    }
                    .distance(&F::zero());
            }

            if sum > max {
                max = sum;
            }
        }

        max
    }

    /// Solves the abstract composition equation `A ∘ X = B` using
    /// Gauss–Jordan elimination with partial pivoting.
    ///
    /// In the raw representation this is `AX = B` for right-handed matrices
    /// and `XA = B` for left-handed matrices. Pivot rows are physical rows on
    /// the right and physical columns on the left. They are chosen by maximizing
    /// the scalar metric magnitude, improving numerical stability for
    /// approximate fields.
    ///
    /// Assumes A is invertible.
    pub fn solve_pivoted(&self, rhs: Self) -> Self {
        let get = |matrix: &[[V::F; N]; N], i: usize, j: usize| match V::Hand::H {
            Hand::Right => matrix[i][j],
            Hand::Left => matrix[j][i],
        };

        let set = |matrix: &mut [[V::F; N]; N], i: usize, j: usize, value: V::F| match V::Hand::H {
            Hand::Right => matrix[i][j] = value,
            Hand::Left => matrix[j][i] = value,
        };

        let mul = |lhs: V::F, rhs: V::F| match V::Hand::H {
            Hand::Right => lhs * rhs,
            Hand::Left => rhs * lhs,
        };

        let swap_rows = |matrix: &mut [[V::F; N]; N], lhs: usize, rhs: usize| {
            match V::Hand::H {
                Hand::Right => matrix.swap(lhs, rhs),

                // Virtual rows are physical columns.
                Hand::Left => {
                    for row in matrix {
                        row.swap(lhs, rhs);
                    }
                }
            }
        };

        let mut mat = self.0;
        let mut out = rhs.0;

        for i in 0..N {
            let mut pivot_index = i;
            let mut pivot_norm = get(&mat, i, i).interval_squared(&V::F::zero());

            for k in (i + 1)..N {
                let norm = get(&mat, k, i).interval_squared(&V::F::zero());

                if norm > pivot_norm {
                    pivot_index = k;
                    pivot_norm = norm;
                }
            }

            assert!(
                !get(&mat, pivot_index, i).is_zero(),
                "Matrix is singular during Gauss-Jordan elimination."
            );

            swap_rows(&mut mat, i, pivot_index);
            swap_rows(&mut out, i, pivot_index);

            let pivot = get(&mat, i, i);

            let pivot_inv = <V::F as DivRing>::Mul::inv(NonZero::new(pivot).unwrap().into())
                .into()
                .0;

            for j in 0..N {
                let mat_value = mul(pivot_inv, get(&mat, i, j));
                let out_value = mul(pivot_inv, get(&out, i, j));

                set(&mut mat, i, j, mat_value);
                set(&mut out, i, j, out_value);
            }

            for k in 0..N {
                if k == i {
                    continue;
                }

                let factor = get(&mat, k, i);

                for j in 0..N {
                    let mat_value = get(&mat, k, j) - mul(factor, get(&mat, i, j));

                    let out_value = get(&out, k, j) - mul(factor, get(&out, i, j));

                    set(&mut mat, k, j, mat_value);
                    set(&mut out, k, j, out_value);
                }
            }
        }

        Self::new(out)
    }

    pub fn inverse_pivoted(&self) -> Self {
        self.solve_pivoted(Self::one())
    }
}

impl<F: CField + Metric, V: Vector<F = F>, const N: usize> Matrix<V, N> {
    fn swap_rows(&mut self, a: usize, b: usize) {
        if a != b {
            self.0.swap(a, b);
        }
    }

    pub fn det(&self) -> F {
        let mut lu = self.clone();

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

    fn mul(self, rhs: Self) -> Self {
        Self::new(from_fn(|i| {
            from_fn(|j| {
                (0..N).fold(V::F::zero(), |acc, k| {
                    acc + match V::Hand::H {
                        Hand::Right => self[(i, k)] * rhs[(k, j)],
                        Hand::Left => rhs[(i, k)] * self[(k, j)],
                    }
                })
            })
        }))
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

impl<F: CField, V: Vector<F = F>, const N: usize> Mul<F> for Matrix<V, N> {
    type Output = Self;

    fn mul(self, rhs: F) -> Self::Output {
        Self(self.0.map(|v| v.map(|x| x * rhs)))
    }
}

impl<F: CField, V: Vector<F = F>, const N: usize> Div<F> for Matrix<V, N> {
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
    /// Computes the matrix exponential using scaling and squaring with a
    /// degree-13 Padé approximant.
    ///
    /// This operation is total, but—as with any finite-precision matrix
    /// exponential—large or ill-conditioned inputs may suffer substantial
    /// numerical error through rounding, cancellation, and repeated squaring.
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
    F: Field<Characteristic = NatZero, Fixed: FromReal + Metric> + Metric + FromReal + FieldExp,
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

        let b = B
            .map(|x| F::Fixed::from_real(<<F::Fixed as Interval>::R as NumCast>::from(x).unwrap()));

        let norm = self.one_norm();

        let s = if norm.exact_le(theta) {
            0
        } else {
            <usize as NumCast>::from((norm / theta).log2().ceil()).unwrap()
        };

        let fone = F::Fixed::one();
        let half = fone.div(fone + fone);
        let mut a = self.clone();
        for _ in 0..s {
            a = a.scale_fixed(half);
        }

        let a2 = a.clone() * a.clone();
        let a4 = a2.clone() * a2.clone();
        let a6 = a4.clone() * a2.clone();

        let i = Matrix::one();

        let u = a
            * (a6.clone()
                * (a6.clone().scale_fixed(b[13])
                    + a4.clone().scale_fixed(b[11])
                    + a2.clone().scale_fixed(b[9]))
                + a6.clone().scale_fixed(b[7])
                + a4.clone().scale_fixed(b[5])
                + a2.clone().scale_fixed(b[3])
                + i.clone().scale_fixed(b[1]));

        let v = a6.clone()
            * (a6.clone().scale_fixed(b[12])
                + a4.clone().scale_fixed(b[10])
                + a2.clone().scale_fixed(b[8]))
            + a6.scale_fixed(b[6])
            + a4.scale_fixed(b[4])
            + a2.scale_fixed(b[2])
            + i.scale_fixed(b[0]);

        let mut r = (v.clone() - u.clone()).solve_pivoted(v + u);

        for _ in 0..s {
            r = r.clone() * r;
        }

        r
    }

    fn log(&self) -> Option<Self> {
        let log_radius: F::R = <F::R as NumCast>::from(1.0).unwrap();
        let x = self.clone() - Matrix::one();

        let norm = x.frobenius_norm();

        if norm >= log_radius {
            return None;
        }

        let epsilon = F::R::epsilon();

        let mut result = x.clone();
        let mut term = x.clone();

        let fone = F::Fixed::one();
        let mut k_as_f = fone + fone;
        for k in 2.. {
            term = term * x.clone();

            let next = term
                .clone()
                .scale_fixed((if k % 2 == 0 { -fone } else { fone }).div(k_as_f));

            k_as_f = k_as_f + fone;

            result = result + next.clone();

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
