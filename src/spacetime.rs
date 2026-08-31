//! Linear and Lie-theoretic models used in spacetime geometry.
//!
//! [`Minkowski`] is the canonical `(3, 1)` coordinate space. [`Sl`] and
//! [`SlAlgebra`] represent the special linear group and its traceless Lie
//! algebra, while [`Lorentz`] realises the restricted Lorentz group as the
//! quotient of [`Sl2c`] by `{+1, -1}`.

use core::ops::Mul;

use num_traits::{Inv, One, Zero};

use crate::{
    complex::Complex,
    coords::Coords,
    impl_group_via_mul, impl_lie_group_via_quotient, impl_vector_ops,
    matrix::{Matrix, MatrixExponential},
    traits::{
        Atomic, BothSided, CField, Dual, FieldExp, Form, FromReal, Group, LieGroup, Metric,
        NatZero, Nondegenerate, Point, Quotient, Real, Right, RootOfUnity, Sesquilinear, Tensor,
        calculus::{CommutesJet, Jet, JetVector, Tangent},
        𝐅𝐥𝐝, 𝐑𝐞𝐚𝐥,
    },
};

/// Four-dimensional Minkowski space with signature `(3, 1)`.
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
pub struct Sl<V: Tensor<F: CField>, const N: usize>(Matrix<V, N>);

/// `SL(2, ℂ)` — the double cover of the restricted Lorentz group. See [`Lorentz`].
pub type Sl2c<R> = Sl<Coords<Complex<R>, 2>, 2>;

impl<V: Tensor<F: CField>, const N: usize> PartialEq for Sl<V, N> {
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

#[allow(type_alias_bounds)]
type RealJet<R: Real, const N: usize> = Jet<𝐑𝐞𝐚𝐥::𝒞, R, N>;

#[allow(type_alias_bounds)]
type NominalSl2cJet<R: Real, const N: usize> = Sl2c<RealJet<R, N>>;

fn complex_jet_to_jetted_complex<R: Real, const N: usize>(
    value: Jet<𝐅𝐥𝐝::𝒞, Complex<R>, N>,
) -> Complex<RealJet<R, N>> {
    let real = Jet::from_parts(value[0][0], core::array::from_fn(|i| value[i + 1][0]));

    let imag = Jet::from_parts(value[0][1], core::array::from_fn(|i| value[i + 1][1]));

    Complex::from([real, imag])
}

fn jetted_complex_to_complex_jet<R: Real, const N: usize>(
    value: Complex<RealJet<R, N>>,
) -> Jet<𝐅𝐥𝐝::𝒞, Complex<R>, N> {
    Jet::from_parts(
        Complex::from([value[0][0], value[1][0]]),
        core::array::from_fn(|i| Complex::from([value[0][i + 1], value[1][i + 1]])),
    )
}

fn sl2_matrix_jet_to_nominal<R: Real, const N: usize>(
    value: Sl2Jetted<Complex<R>, N>,
) -> NominalSl2cJet<R, N> {
    Sl(Matrix::new(core::array::from_fn(|i| {
        core::array::from_fn(|j| complex_jet_to_jetted_complex(value[(i, j)]))
    })))
}

fn nominal_sl2_to_matrix_jet<R: Real, const N: usize>(
    value: NominalSl2cJet<R, N>,
) -> Sl2Jetted<Complex<R>, N> {
    Matrix::new(core::array::from_fn(|i| {
        core::array::from_fn(|j| jetted_complex_to_complex_jet(value.0[(i, j)]))
    }))
}

impl<R: Real, const N: usize> CommutesJet<Sl2c<R>, SlAlgebra<Complex<R>, 2, 3>, N>
    for NominalSl2cJet<R, N>
{
    fn commute_jet(value: Tangent<Sl2c<R>, SlAlgebra<Complex<R>, 2, 3>, N>) -> Self {
        sl2_matrix_jet_to_nominal(sl2_assemble_group_jet(value))
    }

    fn uncommute_jet(value: Self) -> Tangent<Sl2c<R>, SlAlgebra<Complex<R>, 2, 3>, N> {
        sl2_split_group_jet(nominal_sl2_to_matrix_jet(value))
    }
}

impl_lie_group_via_quotient!(
    Lorentz<R>, Sl2c<R>, RootOfUnity<Complex<R>, 2>, SlAlgebra<Complex<R>, 2, 3>,
    [R: Real];

    commutes_jet<const N: usize> {
        quotient = Lorentz<RealJet<R, N>>,
        cover = Sl2c<RealJet<R, N>>,
        subgroup = RootOfUnity<Complex<RealJet<R, N>>, 2>,
        model = SlAlgebra<Complex<RealJet<R, N>>, 2, 3>,
    }
);

impl<V: Tensor<F: CField>, const N: usize> Sl<V, N> {
    /// Returns the trace of the represented special-linear transformation.
    pub fn trace(&self) -> V::F {
        self.0.trace()
    }
}

impl<V: Tensor<F: CField>, const N: usize> One for Sl<V, N> {
    fn one() -> Self {
        Self(Matrix::one())
    }
}

impl<V: Tensor<F: CField>, const N: usize> Mul for Sl<V, N> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl<V: Tensor<F: CField>, const N: usize> Inv for Sl<V, N> {
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
            _ => Sl(self.0.inverse()),
        }
    }
}

impl_group_via_mul!(Sl<V, N>, V: Tensor<F: CField>, const N: usize);

type Sl2Matrix<F> = Matrix<Coords<F, 2>, 2>;
type Sl2Jetted<F, const M: usize> = Matrix<JetVector<Coords<F, 2>, M>, 2>;

fn sl2_assemble_group_jet<F, const M: usize>(
    value: Tangent<Sl<Coords<F, 2>, 2>, SlAlgebra<F, 2, 3>, M>,
) -> Sl2Jetted<F, M>
where
    F: CField<Characteristic = NatZero, Fixed: FromReal + Metric> + Metric + FromReal + FieldExp,
{
    Sl2Matrix::<F>::constant_jet::<M>(value.0.0)
        * Sl2Matrix::<F>::exp_jet::<M>(SlAlgebra::matrix_jet(value.1))
}

fn sl2_split_group_jet<F, const M: usize>(
    value: Sl2Jetted<F, M>,
) -> Tangent<Sl<Coords<F, 2>, 2>, SlAlgebra<F, 2, 3>, M>
where
    F: CField<Characteristic = NatZero, Fixed: FromReal + Metric> + Metric + FromReal + FieldExp,
{
    let point = Sl(Sl2Matrix::<F>::primal::<M>(&value));

    let local_group = Sl2Matrix::<F>::constant_jet::<M>(point.clone().inverse().0) * value;

    let local = Sl2Matrix::<F>::log_jet::<M>(local_group).unwrap();

    let mut tangent = SlAlgebra::from_matrix_jet(local);

    tangent[0][0] = F::zero();
    tangent[1][0] = F::zero();
    tangent[2][0] = F::zero();

    Tangent::new(point, tangent)
}

impl<F: CField<Characteristic = NatZero>> SlAlgebra<F, 2, 3> {
    fn matrix_jet<const M: usize>(value: JetVector<Self, M>) -> Sl2Jetted<F, M> {
        let matrix = SlAlgebra::<Jet<𝐅𝐥𝐝::𝒞, F, M>, 2, 3>::from_fn(|i| value[i]).matrix();

        Matrix::new(core::array::from_fn(|i| {
            core::array::from_fn(|j| matrix[(i, j)])
        }))
    }

    fn from_matrix_jet<const M: usize>(value: Sl2Jetted<F, M>) -> JetVector<Self, M> {
        let matrix =
            Matrix::<Coords<Jet<𝐅𝐥𝐝::𝒞, F, M>, 2>, 2>::new(core::array::from_fn(|i| {
                core::array::from_fn(|j| value[(i, j)])
            }));

        let algebra = SlAlgebra::<Jet<𝐅𝐥𝐝::𝒞, F, M>, 2, 3>::from_matrix(matrix);

        JetVector::from_fn(|i| algebra[i])
    }
}

impl<F> LieGroup<SlAlgebra<F, 2, 3>> for Sl<Coords<F, 2>, 2>
where
    F: CField<Characteristic = NatZero, Fixed: FromReal + Metric> + Metric + FromReal + FieldExp,
{
    fn compose_jet<const N: usize>(
        lhs: Tangent<Self, SlAlgebra<F, 2, 3>, N>,
        rhs: Tangent<Self, SlAlgebra<F, 2, 3>, N>,
    ) -> Tangent<Self, SlAlgebra<F, 2, 3>, N> {
        if N == 0 {
            return Tangent::new(lhs.0.compose(&rhs.0), lhs.1.compose(&rhs.1));
        }

        sl2_split_group_jet(sl2_assemble_group_jet(lhs) * sl2_assemble_group_jet(rhs))
    }

    fn inverse_jet<const N: usize>(
        value: Tangent<Self, SlAlgebra<F, 2, 3>, N>,
    ) -> Tangent<Self, SlAlgebra<F, 2, 3>, N> {
        if N == 0 {
            return Tangent::new(value.0.inverse(), value.1.inverse());
        }

        let value = Sl(sl2_assemble_group_jet(value));

        sl2_split_group_jet(value.inverse().0)
    }

    fn identity_exp<const N: usize>(
        coordinate: JetVector<SlAlgebra<F, 2, 3>, N>,
    ) -> Tangent<Self, SlAlgebra<F, 2, 3>, N> {
        if N == 0 {
            return coordinate.into_tangent(|coordinate| Self(Matrix::exp(&coordinate.matrix())));
        }

        sl2_split_group_jet(Sl2Matrix::<F>::exp_jet::<N>(SlAlgebra::matrix_jet(
            coordinate,
        )))
    }

    fn identity_log<const N: usize>(
        point: Tangent<Self, SlAlgebra<F, 2, 3>, N>,
    ) -> Option<JetVector<SlAlgebra<F, 2, 3>, N>> {
        if N == 0 {
            let coordinate: SlAlgebra<_, _, 3> =
                Matrix::log(&point.0.0).map(SlAlgebra::from_matrix)?;

            return Some(JetVector::from_iter(
                coordinate
                    .iter()
                    .map(|&x| Jet::from_parts(x, [Zero::zero(); N])),
            ));
        }

        Sl2Matrix::<F>::log_jet::<N>(sl2_assemble_group_jet(point)).map(SlAlgebra::from_matrix_jet)
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
pub struct SlAlgebra<F: CField<Characteristic = NatZero>, const N: usize, const D: usize>(
    Coords<F, D>,
);

impl<F: CField<Characteristic = NatZero>, const N: usize, const D: usize> PartialEq
    for SlAlgebra<F, N, D>
{
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<F: CField<Characteristic = NatZero>, const N: usize, const D: usize> From<Coords<F, D>>
    for SlAlgebra<F, N, D>
{
    fn from(value: Coords<F, D>) -> Self {
        const {
            assert!(D == N * N - 1);
        }
        Self(value)
    }
}

impl<F: CField<Characteristic = NatZero>, const N: usize, const D: usize> From<[F; D]>
    for SlAlgebra<F, N, D>
{
    fn from(value: [F; D]) -> Self {
        const {
            assert!(D == N * N - 1);
        }
        Coords::from(value).into()
    }
}

impl<F: CField<Characteristic = NatZero>, const N: usize, const D: usize> From<SlAlgebra<F, N, D>>
    for [F; D]
{
    fn from(value: SlAlgebra<F, N, D>) -> Self {
        value.0.into()
    }
}

impl_vector_ops!(SlAlgebra<F, N, D>, F: CField<Characteristic = NatZero>, const N: usize, const D: usize);

impl<F: CField<Characteristic = NatZero>, const N: usize, const D: usize> SlAlgebra<F, N, D> {
    fn matrix(&self) -> Matrix<Coords<F, N>, N> {
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
        const {
            assert!(D == N * N - 1);
        }
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

impl<F: CField<Characteristic = NatZero>, const N: usize, const D: usize> Form
    for SlAlgebra<F, N, D>
{
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

impl<F: CField<Characteristic = NatZero>, const N: usize, const D: usize> Nondegenerate
    for SlAlgebra<F, N, D>
{
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

impl<F: CField<Fixed = F, Characteristic = NatZero>, const N: usize, const D: usize> Sesquilinear
    for SlAlgebra<F, N, D>
{
}

impl<F: CField<Characteristic = NatZero>, const N: usize, const D: usize> Tensor
    for SlAlgebra<F, N, D>
{
    type F = F;
    type Hand = Right;
    type Action = BothSided;
    type Normalization = Atomic;

    type Array<T: Point> = [T; D];

    fn from_fn(f: impl FnMut(usize) -> Self::F) -> Self {
        const {
            assert!(D == N * N - 1);
        }
        Self(Coords::<F, D>::from_fn(f))
    }
}
impl<F: CField<Characteristic = NatZero>, const N: usize, const D: usize> AsRef<[F; D]>
    for SlAlgebra<F, N, D>
{
    fn as_ref(&self) -> &[F; D] {
        &self.0
    }
}

impl<F: CField<Characteristic = NatZero>, const N: usize, const D: usize> AsMut<[F; D]>
    for SlAlgebra<F, N, D>
{
    fn as_mut(&mut self) -> &mut [F; D] {
        &mut self.0
    }
}
