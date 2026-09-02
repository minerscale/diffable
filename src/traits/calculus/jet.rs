use core::{
    marker::PhantomData,
    ops::{Add, Div, Index, IndexMut, Mul, Neg, Rem, Sub},
};

use num_traits::{Euclid, Inv, Num, NumCast, One, ToPrimitive, Zero};

use crate::{
    coords::Coords,
    impl_vector_ops, include_as,
    traits::{
        Absent, Array, AssocName, Atomic, BindsReflected, BothSided, CField, Cat, Category,
        DivRing, Dual, Euclidean, ExactCmp, Field, Form, Interval, Jetted, Metric, NonZero,
        Nondegenerate, Point, Real, Reflect, ReflectedContext, Sesquilinear, Sinister, Tensor,
        TensorOf, Vector,
        calculus::{CommutesJet, DirectSum, DirectSumArray, Tangent},
        jet, tensor_of, Ø, ː, ι, π, Ⱶ, 𝐅𝐥𝐝, 𝐑𝐞𝐚𝐥, 𝐓𝐞𝐧𝐬, 𝒯,
    },
};

#[allow(type_alias_bounds)]
type ReflectedFunctorImage<𝒞: Cat, Name: AssocName, Payload: Reflect<𝒞>, C: Category> = 𝒯<
    ː<BindsReflected<Name, 𝒞, Payload>, <C as Category>::Structure>,
    <C as Category>::Properties,
    <C as Category>::Equations,
>;

/// Concrete representation used when a tensor is re-presented over a new scalar.
///
/// Scalar re-presentation is a construction on tensors, not a new category in
/// the reflected ontology. The logical tensor shape, array family, handedness,
/// and available scalar actions come from `V`; only the coordinate scalar is
/// replaced. No scalar map `V::F -> S` is asserted here; a construction using
/// this representation supplies the map which makes that base change functorial.
#[doc(hidden)]
#[derive(Copy, Clone, Debug)]
pub struct TensorOver<V: Tensor, S: Point>(
    pub(crate) V::Array<S>,
    pub(crate) PhantomData<fn() -> V>,
);

impl<V: Tensor, S: Point + PartialEq> PartialEq for TensorOver<V, S> {
    fn eq(&self, other: &Self) -> bool {
        self.0.iter().eq(other.0.iter())
    }
}

impl<V: Tensor, S: Field> Tensor for TensorOver<V, S> {
    type Normalization = Atomic;
    type F = S;

    type Array<T: Point> = V::Array<T>;

    type Hand = V::Hand;
    type Action = V::Action;

    fn from_fn(f: impl FnMut(usize) -> Self::F) -> Self {
        Self(V::Array::from_fn(f), PhantomData)
    }
}

impl<V: Tensor, S: Point> AsRef<V::Array<S>> for TensorOver<V, S> {
    fn as_ref(&self) -> &V::Array<S> {
        &self.0
    }
}

impl<V: Tensor, S: Point> AsMut<V::Array<S>> for TensorOver<V, S> {
    fn as_mut(&mut self) -> &mut V::Array<S> {
        &mut self.0
    }
}

impl<𝒞: Cat, U: Tensor + From<[F; K]>, F: Field, const N: usize, const K: usize>
    From<[Jet<𝒞, F, N>; K]> for JetVectorIn<𝒞, U, N>
where
    Jet<𝒞, U::F, N>: Field,
{
    fn from(value: [Jet<𝒞, F, N>; K]) -> Self {
        Self::from_fn(|coordinate| {
            let primal = U::from(core::array::from_fn(|i| value[i][0]));

            let coefficients = core::array::from_fn(|order| {
                U::from(core::array::from_fn(|i| value[i][order + 1]))[coordinate]
            });

            Jet::from_parts(primal[coordinate], coefficients)
        })
    }
}

impl_vector_ops!(TensorOver<V, S>, V: Tensor, S: Field);

impl<𝒞: Cat, V, S> Reflect<TensorOf<𝒞>> for TensorOver<V, S>
where
    V: Tensor + Reflect<𝒞>,
    S: Field,
    Self: Reflect<𝒞>,
    ReflectedFunctorImage<𝒞, tensor_of::Payload, V, ReflectedContext<𝒞, Self>>: Ⱶ<TensorOf<𝒞>>,
{
    type Body = ReflectedFunctorImage<𝒞, tensor_of::Payload, V, ReflectedContext<𝒞, Self>>;
}

impl<V: Tensor, S: Field> TensorOver<V, S> {
    /// Re-present `value` over a new scalar while preserving `𝒞`.
    ///
    /// This method is the object map of scalar re-presentation on `𝒞`. The
    /// concrete image type is retained, so Rust keeps every native interface
    /// implemented by that image while [`Reflect`] records the functorial
    /// metadata. `map_scalar` supplies the scalar morphism `V::F -> S`.
    ///
    /// The method exists exactly when both the source and image genuinely
    /// reflect as `𝒞`; no separate functor-domain registry is required.
    pub fn new<𝒞: Cat>(value: V, mut map_scalar: impl FnMut(V::F) -> S) -> Self
    where
        V: Reflect<𝒞>,
        Self: Reflect<𝒞>,
    {
        Self(
            V::Array::<S>::from_fn(|i| map_scalar(value[i])),
            PhantomData,
        )
    }
}

impl<𝒞: Cat, V: Tensor, const N: usize> JetVectorIn<𝒞, V, N> {
    pub fn retag<𝒟: Cat>(self) -> JetVectorIn<𝒟, V, N>
    where
        Jet<𝒟, V::F, N>: Field,
    {
        JetVectorIn::from_fn(|i| self.0[i].retag::<𝒟>())
    }

    /// Truncates every scalar jet to order `M`.
    ///
    /// Requires `M <= N`.
    pub fn truncate<const M: usize>(self) -> JetVectorIn<𝒞, V, M>
    where
        Jet<𝒞, V::F, M>: Field,
    {
        const { assert!(M <= N) };

        JetVectorIn::<𝒞, V, M>::from_fn(|i| self.0[i].truncate::<M>())
    }
}

/// A tensor whose scalar coordinates are jets.
///
/// This is intentionally only notation for the concrete witness used internally.
/// Mathematically the construction composes [`Jet::constant`] with
/// [`TensorOver::new`]; the named presentation lets the differentiation
/// interpreter express nested images without erasing their native Rust structure.
#[allow(type_alias_bounds)]
pub type JetVectorIn<𝒞: Cat, V: Tensor, const N: usize = 1, S: Field = <V as Tensor>::F> =
    TensorOver<V, Jet<𝒞, S, N>>;

#[allow(type_alias_bounds)]
pub type JetVector<V: Tensor, const N: usize = 1, S: Field = <V as Tensor>::F> =
    TensorOver<V, Jet<𝐅𝐥𝐝::𝒞, S, N>>;

impl<𝒞: Cat, V: Tensor<F: ι<C: JetRegion<𝒞>>>, const N: usize> TensorOver<V, Jet<𝒞, V::F, N>> {
    /// Embeds every coordinate of `v` through the jet functor.
    pub fn constant(v: V) -> Self {
        Self(
            V::Array::<Jet<𝒞, V::F, N>>::from_fn(|i| Jet::constant(v[i])),
            PhantomData,
        )
    }
}

type JetCoords<F, const N: usize> = DirectSum<Coords<F, 1>, Coords<F, N>>;

/// Concrete truncated-power-series representation used by jettification.
///
/// Jettification is a construction on an existing category rather than a
/// reflected category of its own. The category parameter selects which existing
/// structure the representation is required to preserve; the specialised trait
/// implementations provide the actual proof that the structure survives.
///
/// A value represents
///
/// ```text
/// a₀ + a₁ε + a₂ε² + ⋯ + aₙεⁿ,    εⁿ⁺¹ = 0.
/// ```
///
/// Index `0` is the primal value and index `k` is the Taylor coefficient
/// `f⁽ᵏ⁾/k!`, not the unscaled derivative. Multiplication is truncated
/// convolution. The [`d`] interpreter normally uses first-order jets and nests
/// them when independent derivative slots are required.
#[doc(hidden)]
#[derive(Debug, Copy, Clone)]
pub struct Jet<𝒞: Cat, F: Field, const N: usize = 1>(JetCoords<F, N>, PhantomData<𝒞>);

impl<𝒞: Cat, F: Field, const N: usize> Jet<𝒞, F, N> {
    #[inline]
    pub fn retag<𝒟: Cat>(self) -> Jet<𝒟, F, N> {
        Jet::from_fn(|i| self[i])
    }

    pub fn from_parts(value: F, coefficients: [F; N]) -> Self {
        Self(
            DirectSum(DirectSumArray([value], coefficients, PhantomData)),
            PhantomData,
        )
    }

    /// Truncates this jet to order `M`.
    ///
    /// Requires `M <= N`.
    pub fn truncate<const M: usize>(self) -> Jet<𝒞, F, M> {
        const { assert!(M <= N) };

        Jet::<𝒞, F, M>::from_fn(|i| self[i])
    }

    /// Constructs all `N + 1` coefficients by index, beginning with the primal
    /// coefficient at index zero.
    fn from_fn(f: impl FnMut(usize) -> F) -> Self {
        Self(JetCoords::from_fn(f), PhantomData)
    }

    pub fn derivative(self) -> Self {
        Self::from_fn(|i| {
            if i < N {
                F::from_nat(i + 1) * self[i + 1]
            } else {
                F::zero()
            }
        })
    }

    pub fn integrate_from(primal: F, derivative: Self) -> Self {
        Self::from_fn(|i| {
            if i == 0 {
                primal
            } else {
                derivative[i - 1].div(F::from_nat(i))
            }
        })
    }
}

impl<V: Tensor, const N: usize> JetVector<V, N> {
    pub fn into_tangent<P: Point>(
        self,
        coordinate_to_point: impl FnOnce(V) -> P,
    ) -> Tangent<P, V, N> {
        let coordinate = V::from_iter(self.iter().map(|jet| jet[0]));
        let point = coordinate_to_point(coordinate);

        let tangent = JetVector::from_fn(|i| {
            let mut jet = self[i];
            jet[0] = V::F::zero();
            jet
        });

        Tangent::new(point, tangent)
    }
}

/// Proof that a scalar's richest categorical context selects jettification in `𝒞`.
///
/// This is not a domain registry: the two implementations below are derived from
/// the canonical context itself. Real-valued contexts select `𝐑𝐞𝐚𝐥`; field
/// contexts which constructively do not satisfy the Real theory select the
/// `𝐅𝐥𝐝` fallback. The trait exists only to present that disjoint proof to rustc
/// through one inherent constructor namespace.
pub trait JetRegion<𝒞: Cat>: Category {}

impl<C> JetRegion<𝐅𝐥𝐝::𝒞> for C where
    C: Ⱶ<𝐅𝐥𝐝::𝒞> + Ⱶ<𝐑𝐞𝐚𝐥::𝒞, Absent>
{
}
impl<C: Ⱶ<𝐑𝐞𝐚𝐥::𝒞>> JetRegion<𝐑𝐞𝐚𝐥::𝒞> for C {}

impl<𝒞: Cat, F: Field + ι, const N: usize> Jet<𝒞, F, N>
where
    F::C: JetRegion<𝒞>,
{
    /// Constructs a jet in the unique categorical region selected by `F`.
    pub fn new(value: F, coefficients: [F; N]) -> Self {
        Self::from_parts(value, coefficients)
    }

    /// Embeds a scalar as a constant jet in the unique selected region.
    pub fn constant(value: F) -> Self {
        Self::new(value, [F::zero(); N])
    }
}

impl<𝒞: Cat, F, const N: usize> Reflect<Jetted<𝒞>> for Jet<𝒞, F, N>
where
    F: Field + Reflect<𝒞>,
    Self: Reflect<𝒞>,
    ReflectedFunctorImage<𝒞, jet::Payload, F, ReflectedContext<𝒞, Self>>: Ⱶ<Jetted<𝒞>>,
{
    type Body = ReflectedFunctorImage<𝒞, jet::Payload, F, ReflectedContext<𝒞, Self>>;
}

#[allow(dead_code)]
fn reflected_functor_smoke<R: Real, V: Tensor<F = R>>(scalar: R, tensor: V) {
    let j = Jet::<𝐑𝐞𝐚𝐥::𝒞, R, 1>::constant(scalar);

    // The functor returns its concrete image, so native behaviour is retained.
    let _ = num_traits::real::Real::sin(j);

    fn sees_jetted_real<R: Real, X>(_: &X)
    where
        X: Reflect<Jetted<𝐑𝐞𝐚𝐥::𝒞>>,
        ReflectedContext<Jetted<𝐑𝐞𝐚𝐥::𝒞>, X>:
            π<jet::Payload, 𝒞 = 𝐑𝐞𝐚𝐥::𝒞, X = R> + Ⱶ<𝐑𝐞𝐚𝐥::𝒞>,
    {
    }

    sees_jetted_real::<R, _>(&j);

    let t = TensorOver::<V, Jet<𝐑𝐞𝐚𝐥::𝒞, R, 1>>::new::<𝐓𝐞𝐧𝐬::𝒞>(
        tensor,
        |x| Jet::<𝐑𝐞𝐚𝐥::𝒞, R, 1>::constant(x),
    );

    fn sees_tensor_image<V: Tensor, X>(_: &X)
    where
        X: Reflect<TensorOf<𝐓𝐞𝐧𝐬::𝒞>>,
        ReflectedContext<TensorOf<𝐓𝐞𝐧𝐬::𝒞>, X>: π<tensor_of::Payload, 𝒞 = 𝐓𝐞𝐧𝐬::𝒞, X = V>
            + Ⱶ<𝐓𝐞𝐧𝐬::𝒞>,
    {
    }

    sees_tensor_image::<V, _>(&t);
}

impl<R: Real, const N: usize> Jet<𝐑𝐞𝐚𝐥::𝒞, R, N> {
    fn sinh_cosh(self) -> (Self, Self) {
        let sinh_primal = self[0].sinh();
        let cosh_primal = self[0].cosh();

        let mut sinh_coefficients = [R::zero(); N];
        let mut cosh_coefficients = [R::zero(); N];

        for n in 1..=N {
            let mut sinh_sum = R::zero();
            let mut cosh_sum = R::zero();

            for k in 1..=n {
                let sinh_nk = if k == n {
                    sinh_primal
                } else {
                    sinh_coefficients[n - k - 1]
                };

                let cosh_nk = if k == n {
                    cosh_primal
                } else {
                    cosh_coefficients[n - k - 1]
                };

                let weighted_x = R::from_nat(k) * self[k];

                sinh_sum = sinh_sum + weighted_x * cosh_nk;
                cosh_sum = cosh_sum + weighted_x * sinh_nk;
            }

            sinh_coefficients[n - 1] = sinh_sum / R::from_nat(n);
            cosh_coefficients[n - 1] = cosh_sum / R::from_nat(n);
        }

        (
            Self::new(sinh_primal, sinh_coefficients),
            Self::new(cosh_primal, cosh_coefficients),
        )
    }
}

impl<F: CField, const N: usize> CField for Jet<𝐅𝐥𝐝::𝒞, F, N> {}

impl<𝒞: Cat, F: Field, const N: usize> PartialEq for Jet<𝒞, F, N> {
    fn eq(&self, other: &Self) -> bool {
        self[0] == other[0]
    }
}

impl<𝒞: Cat, F: Field, const N: usize> Index<usize> for Jet<𝒞, F, N> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<𝒞: Cat, F: Field, const N: usize> IndexMut<usize> for Jet<𝒞, F, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<F: Field, const N: usize> Field for Jet<𝐅𝐥𝐝::𝒞, F, N> {
    type Fixed = Jet<𝐅𝐥𝐝::𝒞, F::Fixed, N>;

    fn conj(&self) -> Self {
        Self::from_fn(|i| self[i].conj())
    }

    type Characteristic = F::Characteristic;

    fn to_fixed(self) -> Self::Fixed {
        Self::Fixed::from_fn(|i| self[i].to_fixed())
    }

    fn from_fixed(x: Self::Fixed) -> Self {
        Jet::from_fn(|i| F::from_fixed(x[i]))
    }
}

impl<𝒞: Cat, F: Field, const N: usize> Add for Jet<𝒞, F, N> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0, PhantomData)
    }
}

impl<𝒞: Cat, F: Field, const N: usize> Sub for Jet<𝒞, F, N> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0, PhantomData)
    }
}

impl<𝒞: Cat, F: Field, const N: usize> Mul for Jet<𝒞, F, N> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::from_fn(|n| {
            let mut coefficient = F::zero();

            for k in 0..=n {
                coefficient = coefficient + self[k] * rhs[n - k];
            }

            coefficient
        })
    }
}

impl<𝒞: Cat, F: Field, const N: usize> Neg for Jet<𝒞, F, N> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0, PhantomData)
    }
}

impl<𝒞: Cat, F: Field, const N: usize> One for Jet<𝒞, F, N> {
    fn one() -> Self {
        Self::from_fn(|x| if x == 0 { F::one() } else { F::zero() })
    }
}

impl<𝒞: Cat, F: Field, const N: usize> Zero for Jet<𝒞, F, N> {
    fn zero() -> Self {
        Self(DirectSum::zero(), PhantomData)
    }

    fn is_zero(&self) -> bool {
        self[0].is_zero()
    }
}

impl<𝒞: Cat, F: Field, const N: usize> Inv for NonZero<Jet<𝒞, F, N>> {
    type Output = NonZero<Jet<𝒞, F, N>>;

    fn inv(self) -> Self::Output {
        let input = self.0;

        // Spell this using your DivRing::Mul machinery.
        let constant_inverse: F = <F as DivRing>::Mul::from(NonZero::new_unchecked(input[0]))
            .inv()
            .into()
            .0;

        let mut output = Jet::<𝒞, F, N>::zero();
        output[0] = constant_inverse;

        for n in 1..=N {
            let mut sum = F::zero();

            for k in 1..=n {
                sum = sum + input[k] * output[n - k];
            }

            output[n] = -(constant_inverse * sum);
        }

        // Its constant coefficient is constant_inverse, hence nonzero.
        NonZero::new_unchecked(output)
    }
}

/// One step through a jet-valued scalar presentation in a [`ConstantRoute`].
///
/// `JetLayer<𝒞, N>` records that the current scalar was obtained by wrapping
/// the preceding scalar in [`Jet<𝒞, _, N>`].
#[derive(Debug, Copy, Clone)]
pub struct JetLayer<𝒞: Cat, const N: usize>(PhantomData<𝒞>);

/// Constructs the current scalar presentation from a scalar in the base field.
///
/// A [`JetMap`] carries this type-level route while differential operators add
/// jet layers. This lets operators which capture base-field constants inject
/// them into an arbitrarily deeply nested jet computation.
pub trait ConstantRoute<F: Field> {
    /// The scalar type reached after following this route from `F`.
    type Output: Field;

    /// Embeds a base-field value as a constant in the current presentation.
    fn constant(value: F) -> Self::Output;
}

impl<F: Field> ConstantRoute<F> for Ø {
    type Output = F;

    fn constant(value: F) -> Self::Output {
        value
    }
}

impl<𝒞: Cat, F, const N: usize, Tail> ConstantRoute<F> for ː<JetLayer<𝒞, N>, Tail>
where
    F: Field,
    Tail: ConstantRoute<F>,
    Jet<𝒞, Tail::Output, N>: Field,
{
    type Output = Jet<𝒞, Tail::Output, N>;

    fn constant(value: F) -> Self::Output {
        Jet::from_parts(Tail::constant(value), [Tail::Output::zero(); N])
    }
}

impl<𝒞: Cat, V: FormLift, const N: usize, S: Field> Form for JetVectorIn<𝒞, V, N, S>
where
    Jet<𝒞, S, N>: Field,
    Self: Tensor<F = Jet<𝒞, S, N>>,
{
    fn flat(&self) -> Dual<Self> {
        V::jet_flat::<𝒞, S, N>(self)
    }
}

impl<𝒞: Cat, V: NondegenerateLift, const N: usize, S: Field> Nondegenerate
    for JetVectorIn<𝒞, V, N, S>
where
    Jet<𝒞, S, N>: Field,
    Self: Form<F = Jet<𝒞, S, N>>,
{
    fn sharp(value: Dual<Self>) -> Self {
        V::jet_sharp::<𝒞, S, N>(value)
    }
}

impl<𝒞: Cat, V: Sesquilinear + Interval, const N: usize, S: Field> Interval
    for JetVectorIn<𝒞, V, N, S>
where
    Self: Sesquilinear<F: Field<Fixed: Real>>,
{
    type R = <<Self as Tensor>::F as Field>::Fixed;

    fn interval_squared(&self, other: &Self) -> Self::R {
        (self.clone() - other.clone()).norm_squared()
    }
}

impl<𝒞: Cat, V: Sesquilinear, const N: usize, S: Field> Sesquilinear for JetVectorIn<𝒞, V, N, S> where
    Self: Nondegenerate + Vector
{
}

impl<𝒞: Cat, V: Tensor + Metric, const N: usize, S: Field> Metric for JetVectorIn<𝒞, V, N, S> where
    Self: Interval
{
}

impl<V: Euclidean, const N: usize, S: Real> Euclidean for JetVectorIn<𝐑𝐞𝐚𝐥::𝒞, V, N, S> where
    Self: Vector<F = Jet<𝐑𝐞𝐚𝐥::𝒞, S, N>, Action = BothSided>
{
}

impl<R: Real, const N: usize> PartialOrd for Jet<𝐑𝐞𝐚𝐥::𝒞, R, N> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self[0].partial_cmp(&other[0])
    }
}

impl<R: Real, const N: usize> ToPrimitive for Jet<𝐑𝐞𝐚𝐥::𝒞, R, N> {
    fn to_i64(&self) -> Option<i64> {
        self[0].to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self[0].to_u64()
    }

    fn to_isize(&self) -> Option<isize> {
        self[0].to_isize()
    }

    fn to_i8(&self) -> Option<i8> {
        self[0].to_i8()
    }

    fn to_i16(&self) -> Option<i16> {
        self[0].to_i16()
    }

    fn to_i32(&self) -> Option<i32> {
        self[0].to_i32()
    }

    fn to_i128(&self) -> Option<i128> {
        self[0].to_i128()
    }

    fn to_usize(&self) -> Option<usize> {
        self[0].to_usize()
    }

    fn to_u8(&self) -> Option<u8> {
        self[0].to_u8()
    }

    fn to_u16(&self) -> Option<u16> {
        self[0].to_u16()
    }

    fn to_u32(&self) -> Option<u32> {
        self[0].to_u32()
    }

    fn to_u128(&self) -> Option<u128> {
        self[0].to_u128()
    }

    fn to_f32(&self) -> Option<f32> {
        self[0].to_f32()
    }

    fn to_f64(&self) -> Option<f64> {
        self[0].to_f64()
    }
}

impl<R: Real, const N: usize> NumCast for Jet<𝐑𝐞𝐚𝐥::𝒞, R, N> {
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        R::from(n).map(|x| Self::constant(x))
    }
}

impl<R: Real, const N: usize> Div<Self> for Jet<𝐑𝐞𝐚𝐥::𝒞, R, N> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self.mul(NonZero::new(rhs).unwrap().inv().0)
    }
}

impl<R: Real, const N: usize> Rem<Self> for Jet<𝐑𝐞𝐚𝐥::𝒞, R, N> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        let quotient = (self[0] / rhs[0]).trunc();
        let remainder = self[0] % rhs[0];

        Self::from_fn(|n| {
            if n == 0 {
                remainder
            } else {
                self[n] - quotient * rhs[n]
            }
        })
    }
}

impl<R: Real, const N: usize> Euclid for Jet<𝐑𝐞𝐚𝐥::𝒞, R, N> {
    fn div_euclid(&self, rhs: &Self) -> Self {
        let quotient = <R as Euclid>::div_euclid(&self[0], &rhs[0]);

        Self::constant(quotient)
    }

    fn rem_euclid(&self, rhs: &Self) -> Self {
        let quotient = <R as Euclid>::div_euclid(&self[0], &rhs[0]);

        let remainder = <R as Euclid>::rem_euclid(&self[0], &rhs[0]);

        Self::from_fn(|n| {
            if n == 0 {
                remainder
            } else {
                self[n] - quotient * rhs[n]
            }
        })
    }
}

impl<R: Real, const N: usize> Num for Jet<𝐑𝐞𝐚𝐥::𝒞, R, N> {
    type FromStrRadixErr = R::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        R::from_str_radix(str, radix).map(|x| Self::constant(x))
    }
}

impl<R: Real, const N: usize> num_traits::real::Real for Jet<𝐑𝐞𝐚𝐥::𝒞, R, N> {
    fn min_value() -> Self {
        Self::constant(R::min_value())
    }

    fn min_positive_value() -> Self {
        Self::constant(R::min_positive_value())
    }

    fn epsilon() -> Self {
        Self::constant(R::epsilon())
    }

    fn max_value() -> Self {
        Self::constant(R::max_value())
    }

    fn floor(self) -> Self {
        Self::constant(self[0].floor())
    }

    fn ceil(self) -> Self {
        Self::constant(self[0].ceil())
    }

    fn round(self) -> Self {
        Self::constant(self[0].round())
    }

    fn trunc(self) -> Self {
        Self::constant(self[0].trunc())
    }

    fn fract(self) -> Self {
        let whole = Self::constant(self[0].trunc());
        self - whole
    }

    fn abs(self) -> Self {
        if self[0].is_sign_negative() {
            -self
        } else {
            self
        }
    }

    fn signum(self) -> Self {
        Self::constant(self[0].signum())
    }

    fn is_sign_positive(self) -> bool {
        self[0].is_sign_positive()
    }

    fn is_sign_negative(self) -> bool {
        self[0].is_sign_negative()
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        Self::from_fn(|n| {
            let mut coefficient = b[n];

            for k in 0..=n {
                coefficient = self[k].mul_add(a[n - k], coefficient);
            }

            coefficient
        })
    }

    fn recip(self) -> Self {
        NonZero::new(self).unwrap().inv().0
    }

    fn powi(self, n: i32) -> Self {
        fn unsigned_pow<R: Real, const N: usize>(
            mut base: Jet<𝐑𝐞𝐚𝐥::𝒞, R, N>,
            mut exponent: u32,
        ) -> Jet<𝐑𝐞𝐚𝐥::𝒞, R, N> {
            let mut result = Jet::one();

            while exponent != 0 {
                if exponent & 1 != 0 {
                    result = result * base;
                }

                exponent >>= 1;

                if exponent != 0 {
                    base = base * base;
                }
            }

            result
        }

        let result = unsigned_pow(self, n.unsigned_abs());

        if n < 0 { result.recip() } else { result }
    }

    fn powf(self, n: Self) -> Self {
        if self[0].exact_le(R::zero()) {
            panic!("powf: non-positive base; use powi for integer powers")
        }

        (n * self.ln()).exp()
    }

    fn sqrt(self) -> Self {
        let primal = self[0].sqrt();

        if self[0].is_zero() {
            if (1..=N).all(|i| self[i].is_zero()) {
                return Self::constant(primal);
            }

            panic!("sqrt: not differentiable at a zero primal");
        }

        let mut coefficients = [R::zero(); N];
        let two_primal = R::from_nat(2) * primal;

        for n in 1..=N {
            let mut cross_terms = R::zero();

            for k in 1..n {
                cross_terms = cross_terms + coefficients[k - 1] * coefficients[n - k - 1];
            }

            coefficients[n - 1] = (self[n] - cross_terms) / two_primal;
        }

        Self::new(primal, coefficients)
    }

    fn exp(self) -> Self {
        let primal = self[0].exp();
        let mut coefficients = [R::zero(); N];

        for n in 1..=N {
            let mut sum = R::zero();

            for k in 1..=n {
                let y_nk = if n == k {
                    primal
                } else {
                    coefficients[n - k - 1]
                };

                sum = sum + R::from_nat(k) * self[k] * y_nk;
            }

            coefficients[n - 1] = sum / R::from_nat(n);
        }

        Self::new(primal, coefficients)
    }

    fn exp2(self) -> Self {
        let primal = self[0].exp2();
        let ln_2 = R::from_nat(2).ln();
        let mut coefficients = [R::zero(); N];

        for n in 1..=N {
            let mut sum = R::zero();

            for k in 1..=n {
                let y_nk = if n == k {
                    primal
                } else {
                    coefficients[n - k - 1]
                };

                sum = sum + R::from_nat(k) * self[k] * y_nk;
            }

            coefficients[n - 1] = ln_2 * sum / R::from_nat(n);
        }

        Self::new(primal, coefficients)
    }

    fn ln(self) -> Self {
        let primal = self[0].ln();
        let mut coefficients = [R::zero(); N];

        for n in 1..=N {
            let mut correction = R::zero();

            for k in 1..n {
                correction = correction + self[k] * R::from_nat(n - k) * coefficients[n - k - 1];
            }

            coefficients[n - 1] =
                (R::from_nat(n) * self[n] - correction) / (R::from_nat(n) * self[0]);
        }

        Self::new(primal, coefficients)
    }

    fn log(self, base: Self) -> Self {
        self.ln() / base.ln()
    }

    fn log2(self) -> Self {
        let primal = self[0].log2();
        let ln_2 = R::from_nat(2).ln();
        let logarithm = self.ln();

        Self::from_fn(|i| if i == 0 { primal } else { logarithm[i] / ln_2 })
    }

    fn log10(self) -> Self {
        let primal = self[0].log10();
        let ln_10 = R::from_nat(10).ln();
        let logarithm = self.ln();

        Self::from_fn(|i| if i == 0 { primal } else { logarithm[i] / ln_10 })
    }

    fn to_degrees(self) -> Self {
        Self::from_fn(|i| self[i].to_degrees())
    }

    fn to_radians(self) -> Self {
        Self::from_fn(|i| self[i].to_radians())
    }

    fn max(self, other: Self) -> Self {
        if self[0].exact_lt(other[0]) {
            other
        } else {
            self
        }
    }

    fn min(self, other: Self) -> Self {
        if other[0].exact_lt(self[0]) {
            other
        } else {
            self
        }
    }

    fn abs_sub(self, other: Self) -> Self {
        if other[0].exact_lt(self[0]) {
            self - other
        } else {
            Self::zero()
        }
    }

    fn cbrt(self) -> Self {
        let primal = self[0].cbrt();

        if self[0].is_zero() {
            if (1..=N).all(|i| self[i].is_zero()) {
                return Self::constant(primal);
            }

            panic!("cbrt: not differentiable at a zero primal");
        }

        let mut coefficients = [R::zero(); N];

        for n in 1..=N {
            let mut numerator = R::zero();

            for k in 0..n {
                let y = if n - 1 == k {
                    primal
                } else {
                    coefficients[n - k - 2]
                };

                numerator = numerator + R::from_nat(k + 1) * self[k + 1] * y;
            }

            for k in 1..n {
                numerator = numerator
                    - R::from_nat(3) * R::from_nat(n - k) * self[k] * coefficients[n - k - 1];
            }

            coefficients[n - 1] = numerator / (R::from_nat(3) * R::from_nat(n) * self[0]);
        }

        Self::new(primal, coefficients)
    }

    fn hypot(self, other: Self) -> Self {
        (self * self + other * other).sqrt()
    }

    fn sin(self) -> Self {
        self.sin_cos().0
    }

    fn cos(self) -> Self {
        self.sin_cos().1
    }

    fn tan(self) -> Self {
        let (sin, cos) = self.sin_cos();
        sin / cos
    }

    fn asin(self) -> Self {
        let primal = self[0].asin();
        let dx = self.derivative();

        let derivative = dx / (Self::one() - self * self).sqrt();

        Self::integrate_from(primal, derivative)
    }

    fn acos(self) -> Self {
        let primal = self[0].acos();
        let dx = self.derivative();

        let derivative = -(dx / (Self::one() - self * self).sqrt());

        Self::integrate_from(primal, derivative)
    }

    fn atan(self) -> Self {
        let primal = self[0].atan();
        let dx = self.derivative();

        let derivative = dx / (Self::one() + self * self);

        Self::integrate_from(primal, derivative)
    }

    fn atan2(self, other: Self) -> Self {
        let primal = self[0].atan2(other[0]);

        let dy = self.derivative();
        let dx = other.derivative();

        let derivative = (other * dy - self * dx) / (other * other + self * self);

        Self::integrate_from(primal, derivative)
    }

    fn sin_cos(self) -> (Self, Self) {
        let (sin_primal, cos_primal) = self[0].sin_cos();
        let mut sin_coefficients = [R::zero(); N];
        let mut cos_coefficients = [R::zero(); N];

        for n in 1..=N {
            let mut sin_sum = R::zero();
            let mut cos_sum = R::zero();

            for k in 1..=n {
                let sin_nk = if k == n {
                    sin_primal
                } else {
                    sin_coefficients[n - k - 1]
                };

                let cos_nk = if k == n {
                    cos_primal
                } else {
                    cos_coefficients[n - k - 1]
                };

                let weighted_x = R::from_nat(k) * self[k];

                sin_sum = sin_sum + weighted_x * cos_nk;
                cos_sum = cos_sum - weighted_x * sin_nk;
            }

            sin_coefficients[n - 1] = sin_sum / R::from_nat(n);
            cos_coefficients[n - 1] = cos_sum / R::from_nat(n);
        }

        (
            Self::new(sin_primal, sin_coefficients),
            Self::new(cos_primal, cos_coefficients),
        )
    }

    fn exp_m1(self) -> Self {
        let primal = self[0].exp_m1();
        let mut result = self.exp() - Self::one();

        result[0] = primal;
        result
    }

    fn ln_1p(self) -> Self {
        let primal = self[0].ln_1p();
        let mut result = (Self::one() + self).ln();

        result[0] = primal;
        result
    }

    fn sinh(self) -> Self {
        self.sinh_cosh().0
    }

    fn cosh(self) -> Self {
        self.sinh_cosh().1
    }

    fn tanh(self) -> Self {
        let primal = self[0].tanh();
        let mut coefficients = [R::zero(); N];
        let mut slope = [R::zero(); N];

        for n in 1..=N {
            // Coefficient n - 1 of 1 - y².
            let j = n - 1;
            let mut y_squared = R::zero();

            for i in 0..=j {
                let y_i = if i == 0 { primal } else { coefficients[i - 1] };

                let y_ji = if j == i {
                    primal
                } else {
                    coefficients[j - i - 1]
                };

                y_squared = y_squared + y_i * y_ji;
            }

            slope[j] = if j == 0 {
                R::one() - y_squared
            } else {
                -y_squared
            };

            let mut sum = R::zero();

            for k in 1..=n {
                sum = sum + R::from_nat(k) * self[k] * slope[n - k];
            }

            coefficients[n - 1] = sum / R::from_nat(n);
        }

        Self::new(primal, coefficients)
    }

    fn asinh(self) -> Self {
        let primal = self[0].asinh();
        let dx = self.derivative();

        let derivative = dx / (Self::one() + self * self).sqrt();

        Self::integrate_from(primal, derivative)
    }

    fn acosh(self) -> Self {
        let primal = self[0].acosh();
        let dx = self.derivative();

        let derivative = dx / ((self - Self::one()).sqrt() * (self + Self::one()).sqrt());

        Self::integrate_from(primal, derivative)
    }

    fn atanh(self) -> Self {
        let primal = self[0].atanh();
        let dx = self.derivative();

        let derivative = dx / (Self::one() - self * self);

        Self::integrate_from(primal, derivative)
    }
}

/// Extends a lowering map through arbitrary jet scalar presentations.
///
/// Implementors provide the array-level operation, which avoids assuming that
/// distinct tensor wrappers share a nominal array type. [`FormLift::jet_flat`]
/// supplies the public tensor-valued wrapper. For a semilinear form the
/// implementation must preserve the form's conjugation convention coefficient
/// by coefficient.
pub trait FormLift: Form {
    /// Applies the lifted lowering map to raw coordinate arrays.
    fn jet_flat_array<𝒞: Cat, S: Field, const N: usize>(
        value: &<Self as Tensor>::Array<Jet<𝒞, S, N>>,
    ) -> <Dual<Self> as Tensor>::Array<Jet<𝒞, S, N>>
    where
        Jet<𝒞, S, N>: Field;

    /// Applies the lifted lowering map to a [`JetVector`].
    fn jet_flat<𝒞: Cat, S: Field, const N: usize>(
        value: &JetVectorIn<𝒞, Self, N, S>,
    ) -> Dual<JetVectorIn<𝒞, Self, N, S>>
    where
        Jet<𝒞, S, N>: Field,
        JetVectorIn<𝒞, Self, N, S>: Tensor<F = Jet<𝒞, S, N>>,
    {
        let value = <Self as Tensor>::Array::from_fn(|coordinate| value[coordinate]);
        let flat = Self::jet_flat_array(&value);

        Dual::from_fn(|coordinate| flat[coordinate])
    }
}

/// Extends an invertible lowering map and its raising map through jets.
///
/// This is the recursive counterpart of [`Nondegenerate`]. Requiring it on a
/// Euclidean space ensures generic Euclidean functions remain valid when their
/// scalar and vector arguments acquire further derivative layers.
pub trait NondegenerateLift: Nondegenerate + FormLift {
    /// Applies the lifted raising map to raw coordinate arrays.
    fn jet_sharp_array<𝒞: Cat, S: Field, const N: usize>(
        value: &<Dual<Self> as Tensor>::Array<Jet<𝒞, S, N>>,
    ) -> <Self as Tensor>::Array<Jet<𝒞, S, N>>
    where
        Jet<𝒞, S, N>: Field;

    /// Applies the lifted raising map to a jet-valued covector.
    fn jet_sharp<𝒞: Cat, S: Field, const N: usize>(
        value: Dual<JetVectorIn<𝒞, Self, N, S>>,
    ) -> JetVectorIn<𝒞, Self, N, S>
    where
        Jet<𝒞, S, N>: Field,
        JetVectorIn<𝒞, Self, N, S>: Tensor<F = Jet<𝒞, S, N>>,
    {
        let value = Dual::to_raw(value);

        let value = <Dual<Self> as Tensor>::Array::from_fn(|coordinate| value[coordinate]);

        let sharp = Self::jet_sharp_array(&value);

        JetVectorIn::from_fn(|coordinate| sharp[coordinate])
    }
}

impl<𝒞: Cat, V, const N: usize, S> FormLift for JetVectorIn<𝒞, V, N, S>
where
    V: FormLift,
    S: Field,
    Jet<𝒞, S, N>: Field,
    Self: Form<F = Jet<𝒞, S, N>>,
{
    fn jet_flat_array<𝒟: Cat, T: Field, const K: usize>(
        value: &<Self as Tensor>::Array<Jet<𝒟, T, K>>,
    ) -> <Dual<Self> as Tensor>::Array<Jet<𝒟, T, K>>
    where
        Jet<𝒟, T, K>: Field,
    {
        let value = V::Array::from_fn(|coordinate| value[coordinate]);
        let flat = V::jet_flat_array::<𝒟, T, K>(&value);

        <Dual<Self> as Tensor>::Array::from_fn(|coordinate| flat[coordinate])
    }
}

impl<𝒞: Cat, V, const N: usize, S> NondegenerateLift for JetVectorIn<𝒞, V, N, S>
where
    V: NondegenerateLift,
    S: Field,
    Jet<𝒞, S, N>: Field,
    Self: Nondegenerate<F = Jet<𝒞, S, N>>,
{
    fn jet_sharp_array<𝒟: Cat, T: Field, const K: usize>(
        value: &<Dual<Self> as Tensor>::Array<Jet<𝒟, T, K>>,
    ) -> <Self as Tensor>::Array<Jet<𝒟, T, K>>
    where
        Jet<𝒟, T, K>: Field,
    {
        let value = <Dual<V> as Tensor>::Array::from_fn(|coordinate| value[coordinate]);
        let sharp = V::jet_sharp_array::<𝒟, T, K>(&value);

        <Self as Tensor>::Array::from_fn(|coordinate| sharp[coordinate])
    }
}

impl<V> FormLift for Dual<V>
where
    V: NondegenerateLift,
{
    fn jet_flat_array<𝒞: Cat, S: Field, const N: usize>(
        value: &<Self as Tensor>::Array<Jet<𝒞, S, N>>,
    ) -> <Dual<Self> as Tensor>::Array<Jet<𝒞, S, N>>
    where
        Jet<𝒞, S, N>: Field,
    {
        let value = <Dual<V> as Tensor>::Array::from_fn(|coordinate| value[coordinate]);
        let sharp = V::jet_sharp_array::<𝒞, S, N>(&value);

        <Dual<Self> as Tensor>::Array::from_fn(|coordinate| sharp[coordinate])
    }
}

impl<V> NondegenerateLift for Dual<V>
where
    V: NondegenerateLift,
{
    fn jet_sharp_array<𝒞: Cat, S: Field, const N: usize>(
        value: &<Dual<Self> as Tensor>::Array<Jet<𝒞, S, N>>,
    ) -> <Self as Tensor>::Array<Jet<𝒞, S, N>>
    where
        Jet<𝒞, S, N>: Field,
    {
        let value = <V as Tensor>::Array::from_fn(|coordinate| value[coordinate]);
        let flat = V::jet_flat_array::<𝒞, S, N>(&value);

        <Self as Tensor>::Array::from_fn(|coordinate| flat[coordinate])
    }
}

impl<V> FormLift for Sinister<V>
where
    V: FormLift<Action = BothSided>,
{
    fn jet_flat_array<𝒞: Cat, S: Field, const N: usize>(
        value: &<Self as Tensor>::Array<Jet<𝒞, S, N>>,
    ) -> <Dual<Self> as Tensor>::Array<Jet<𝒞, S, N>>
    where
        Jet<𝒞, S, N>: Field,
    {
        let value = <V as Tensor>::Array::from_fn(|coordinate| value[coordinate]);
        let flat = V::jet_flat_array::<𝒞, S, N>(&value);

        <Dual<Self> as Tensor>::Array::from_fn(|coordinate| flat[coordinate])
    }
}

impl<V> NondegenerateLift for Sinister<V>
where
    V: NondegenerateLift<Action = BothSided>,
{
    fn jet_sharp_array<𝒞: Cat, S: Field, const N: usize>(
        value: &<Dual<Self> as Tensor>::Array<Jet<𝒞, S, N>>,
    ) -> <Self as Tensor>::Array<Jet<𝒞, S, N>>
    where
        Jet<𝒞, S, N>: Field,
    {
        let value = <Dual<V> as Tensor>::Array::from_fn(|coordinate| value[coordinate]);
        let sharp = V::jet_sharp_array::<𝒞, S, N>(&value);

        <Self as Tensor>::Array::from_fn(|coordinate| sharp[coordinate])
    }
}

impl<𝒞, V, const K: usize> CommutesJet<V, V, K> for JetVectorIn<𝒞, V, K>
where
    𝒞: Cat,
    V: Vector,
    Jet<𝒞, V::F, K>: Field,
{
    fn commute_jet(value: Tangent<V, V, K>) -> Self {
        value.into_jet(|point| point).retag::<𝒞>()
    }

    fn uncommute_jet(value: Self) -> Tangent<V, V, K> {
        value.retag::<𝐅𝐥𝐝::𝒞>().into_tangent(|point| point)
    }
}

// Algebraic jets are canonically included at `Field`. Real jets are a
// distinct presentation, `Jet<𝐑𝐞𝐚𝐥::𝒞, _, _>`, and obtain their richer
// canonical inclusion through the blanket `Real -> ι` implementation.
// Keeping this admission restricted to `𝐅𝐥𝐝::𝒞` preserves the disjoint
// region selection performed by `JetRegion`.
include_as!(
    Jet<𝐅𝐥𝐝::𝒞, F, N> => Field,
    F: Field,
    const N: usize
);
