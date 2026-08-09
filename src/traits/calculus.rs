//! Tensor constructions, tangent lifts, and forward automatic differentiation.
//!
//! The differentiation API is a small typed language. [`d`] introduces a
//! derivative, [`d::along`] contracts one derivative slot with a direction,
//! and [`d::at`] or [`Along::at`] evaluates the completed program. Programs may
//! themselves be differentiated, so `d(d(f))` is the second derivative (the
//! Hessian for a scalar-valued map) and arbitrarily deep nestings use the same
//! machinery.
//!
//! Evaluation is implemented with truncated Taylor jets. [`Jet`] stores scalar
//! coefficients, [`JetVector`] gives a tensor a jet-valued scalar presentation,
//! and [`JetMap`] is the internal interpretation rule for ordinary functions
//! and differential programs. [`ConstantRoute`] records how captured constants
//! must be embedded through nested presentations. [`EvaluableAt`] is the final
//! interpreter boundary and provides the user-facing diagnostic when a program
//! cannot be evaluated.
//!
//! [`TangentLift`] extends the construction from vector spaces to tangent
//! bundles. [`FormLift`] and [`NondegenerateLift`] state that lowering and
//! raising maps extend coherently when coordinates are replaced by jets.

use core::{
    marker::PhantomData,
    ops::{Add, Deref, DerefMut, Div, Index, IndexMut, Mul, Neg, Rem, Sub},
};

use num_traits::{Euclid, Inv, Num, NumCast, One, ToPrimitive, Zero};

use crate::{
    coords::Coords,
    impl_vector_ops,
    traits::{
        ActionExists, ApplyTensorDecoration, Array, Atomic, BothSided, CField, Chart, Cons,
        DivRing, Dual, Euclidean, ExactCmp, ExpMap, Field, Form, Handedness, Interval, Left,
        Metric, NonZero, Nondegenerate, NormalizeWith, OneSided, Point, Real, Right, Sesquilinear,
        Sidedness, Sinister, TangentBundle, Tensor, TensorDecoration, TensorNormalization,
        TensorProductAction, Undecorated, Vector, Ø,
    },
};

#[doc(hidden)]
pub struct NormalizeDirectSum;

#[doc(hidden)]
pub struct NormalizeTensorProduct;

/// The external direct sum `U ⊕ V` of tensors with a common field and hand.
///
/// Coordinates of `U` precede coordinates of `V`. The result exposes only the
/// scalar actions available on both summands: its [`Tensor::Action`] is the
/// meet of `U::Action` and `V::Action`.
#[derive(Debug, Copy, Clone)]
pub struct DirectSum<U: Tensor<F = V::F>, V: Tensor>(
    DirectSumArray<V::F, U::Array<V::F>, V::Array<V::F>>,
);

impl<F: Field, H: Handedness, U: Tensor<F = F, Hand = H>, V: Tensor<F = F, Hand = H>>
    DirectSum<U, V>
{
    /// Applies the canonical isomorphism `(U ⊕ V)* ≅ U* ⊕ V*`.
    pub fn dual_isomorphism(dual: Dual<Self>) -> DirectSum<Dual<U>, Dual<V>> {
        DirectSum::<Dual<U>, Dual<V>>::from_fn(|i| dual[i])
    }

    /// Applies the inverse canonical isomorphism `U* ⊕ V* ≅ (U ⊕ V)*`.
    pub fn dual_isomorphism_inverse(dual: DirectSum<Dual<U>, Dual<V>>) -> Dual<Self> {
        Dual::<Self>::from_fn(|i| dual[i])
    }
}

/// The concatenated array representation used by [`DirectSum`].
///
/// The representation implements [`Array`] without requiring either summand
/// to use contiguous storage.
#[derive(Debug, Copy, Clone)]
pub struct DirectSumArray<T: Point, U: Array<T>, V: Array<T>>(U, V, PhantomData<T>);

impl<T: Point, U: Array<T>, V: Array<T>> Array<T> for DirectSumArray<T, U, V> {
    const N: usize = U::N + V::N;

    type Iter<'a>
        = core::iter::Chain<U::Iter<'a>, V::Iter<'a>>
    where
        Self: 'a,
        T: 'a;

    type IterMut<'a>
        = core::iter::Chain<U::IterMut<'a>, V::IterMut<'a>>
    where
        Self: 'a,
        T: 'a;

    fn iter(&self) -> Self::Iter<'_> {
        self.0.iter().chain(self.1.iter())
    }

    fn iter_mut(&mut self) -> Self::IterMut<'_> {
        self.0.iter_mut().chain(self.1.iter_mut())
    }

    fn from_fn(mut f: impl FnMut(usize) -> T) -> Self {
        Self(U::from_fn(&mut f), V::from_fn(|i| f(U::N + i)), PhantomData)
    }
}

impl<T: Point, U: Array<T>, V: Array<T>> Index<usize> for DirectSumArray<T, U, V> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        if index < U::N {
            &self.0[index]
        } else {
            &self.1[index - U::N]
        }
    }
}
impl<T: Point, U: Array<T>, V: Array<T>> IndexMut<usize> for DirectSumArray<T, U, V> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index < U::N {
            &mut self.0[index]
        } else {
            &mut self.1[index - U::N]
        }
    }
}
impl<T: Point, U: Array<T>, V: Array<T>> IntoIterator for DirectSumArray<T, U, V> {
    type Item = T;

    type IntoIter = core::iter::Chain<U::IntoIter, V::IntoIter>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter().chain(self.1)
    }
}

impl<F: Field, H: Handedness, U: Tensor<F = F, Hand = H>, V: Tensor<F = F, Hand = H>> Tensor
    for DirectSum<U, V>
{
    type Normalization = NormalizeDirectSum;
    type F = U::F;
    type Hand = H;
    type Action = <U::Action as Sidedness>::Meet<V::Action>;

    type Array<T: Point> = DirectSumArray<T, U::Array<T>, V::Array<T>>;

    fn from_fn(f: impl FnMut(usize) -> Self::F) -> Self {
        Self(Self::Array::<V::F>::from_fn(f))
    }
}
impl<U, V, D> TensorNormalization<DirectSum<U, V>, D> for NormalizeDirectSum
where
    U: Tensor + NormalizeWith<Undecorated>,
    V: Tensor<F = U::F, Hand = U::Hand> + NormalizeWith<Undecorated>,
    <U as NormalizeWith<Undecorated>>::Normalized:
        Tensor<F = U::F, Hand = U::Hand, Action = U::Action>,
    <V as NormalizeWith<Undecorated>>::Normalized:
        Tensor<F = U::F, Hand = U::Hand, Action = V::Action>,
    D: TensorDecoration
        + ApplyTensorDecoration<
            DirectSum<
                <U as NormalizeWith<Undecorated>>::Normalized,
                <V as NormalizeWith<Undecorated>>::Normalized,
            >,
        >,
{
    type Normalized = <D as ApplyTensorDecoration<
        DirectSum<
            <U as NormalizeWith<Undecorated>>::Normalized,
            <V as NormalizeWith<Undecorated>>::Normalized,
        >,
    >>::Output;

    fn normalize(tensor: DirectSum<U, V>) -> Self::Normalized {
        D::apply(DirectSum::from_fn(|i| tensor[i]))
    }
}

impl<F: Field, H: Handedness, U: Tensor<F = F, Hand = H>, V: Tensor<F = F, Hand = H>>
    AsRef<DirectSumArray<F, U::Array<F>, V::Array<F>>> for DirectSum<U, V>
{
    fn as_ref(&self) -> &DirectSumArray<F, U::Array<F>, V::Array<F>> {
        &self.0
    }
}

impl<F: Field, H: Handedness, U: Tensor<F = F, Hand = H>, V: Tensor<F = F, Hand = H>>
    AsMut<DirectSumArray<F, U::Array<F>, V::Array<F>>> for DirectSum<U, V>
{
    fn as_mut(&mut self) -> &mut DirectSumArray<F, U::Array<F>, V::Array<F>> {
        &mut self.0
    }
}

impl_vector_ops!(
    DirectSum<U, V>,
    F: Field,
    H: Handedness,
    U: Tensor<F = F, Hand = H>,
    V: Tensor<F = F, Hand = H>
);

/// The nested array representation used by [`TensorProduct`].
///
/// Its flat iteration and indexing order is outer coordinate first, then inner
/// coordinate. No claim is made that the nested arrays are contiguous.
#[derive(Debug, Copy, Clone)]
pub struct TensorProductArray<T: Point, U: Array<V>, V: Array<T>>(U, PhantomData<(T, V)>);

impl<T: Point, U: Array<V>, V: Array<T>> TensorProductArray<T, U, V> {
    pub fn from_fn_ij(mut f: impl FnMut(usize, usize) -> T) -> Self {
        Self::from_fn(|n| {
            let i = n / V::N;
            let j = n % V::N;

            f(i, j)
        })
    }
}

fn iter_inner<'a, T: Point, V: Array<T>>(v: &'a V) -> V::Iter<'a> {
    v.iter()
}

fn iter_inner_mut<'a, T: Point, V: Array<T>>(v: &'a mut V) -> V::IterMut<'a> {
    v.iter_mut()
}

impl<T: Point, U: Array<V>, V: Array<T>> Array<T> for TensorProductArray<T, U, V> {
    const N: usize = U::N * V::N;

    type Iter<'a>
        = core::iter::FlatMap<U::Iter<'a>, V::Iter<'a>, fn(&'a V) -> V::Iter<'a>>
    where
        Self: 'a,
        T: 'a;

    type IterMut<'a>
        = core::iter::FlatMap<U::IterMut<'a>, V::IterMut<'a>, fn(&'a mut V) -> V::IterMut<'a>>
    where
        Self: 'a,
        T: 'a;

    fn iter(&self) -> Self::Iter<'_> {
        self.0.iter().flat_map(iter_inner::<T, V>)
    }

    fn iter_mut(&mut self) -> Self::IterMut<'_> {
        self.0.iter_mut().flat_map(iter_inner_mut::<T, V>)
    }

    fn from_fn(mut f: impl FnMut(usize) -> T) -> Self {
        Self(U::from_fn(|i| V::from_fn(|j| f(i * V::N + j))), PhantomData)
    }
}

impl<T: Point, U: Array<V>, V: Array<T>> Index<usize> for TensorProductArray<T, U, V> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        &self.0[index / V::N][index % V::N]
    }
}

impl<T: Point, U: Array<V>, V: Array<T>> IndexMut<usize> for TensorProductArray<T, U, V> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.0[index / V::N][index % V::N]
    }
}

impl<T: Point, U: Array<V>, V: Array<T>> Index<(usize, usize)> for TensorProductArray<T, U, V> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &T {
        &self.0[index.0][index.1]
    }
}

impl<T: Point, U: Array<V>, V: Array<T>> IndexMut<(usize, usize)> for TensorProductArray<T, U, V> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut T {
        &mut self.0[index.0][index.1]
    }
}

impl<T: Point, U: Array<V>, V: Array<T>> IntoIterator for TensorProductArray<T, U, V> {
    type Item = T;
    type IntoIter = core::iter::Flatten<U::IntoIter>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter().flatten()
    }
}

/// The balanced tensor product `U ⊗ V`.
///
/// `U` must expose the right action used for balancing and `V` the left action.
/// The result's preferred hand and remaining scalar actions are computed by
/// [`TensorProductAction`]. These restrictions remain explicit even over a
/// commutative field, where switching hands is available through
/// [`Sinister`].
#[derive(Debug, Clone, Copy)]
pub struct TensorProduct<
    U: Tensor<Hand = Right, Action: TensorProductAction<V::Action>>,
    V: Tensor<F = U::F, Hand = Left, Action: ActionExists>,
>(TensorProductArray<V::F, U::Array<V::Array<V::F>>, V::Array<V::F>>);

impl<
    U: Tensor<Hand = Right, Action: TensorProductAction<V::Action>>,
    V: Tensor<F = U::F, Hand = Left, Action: ActionExists>,
> TensorProduct<U, V>
{
    pub fn pure(a: U, b: V) -> Self {
        Self::from_fn_ij(|i, j| a[i] * b[j])
    }

    pub fn from_fn_ij(f: impl FnMut(usize, usize) -> V::F) -> Self {
        Self(TensorProductArray::from_fn_ij(f))
    }
}

impl<
    U: Tensor<Hand = Right, Action: TensorProductAction<V::Action>>,
    V: Tensor<F = U::F, Hand = Left, Action: ActionExists>,
> Tensor for TensorProduct<U, V>
{
    type Normalization = NormalizeTensorProduct;
    type F = V::F;
    type Action = <U::Action as TensorProductAction<V::Action>>::Action;
    type Hand = <U::Action as TensorProductAction<V::Action>>::Hand;

    type Array<T: Point> = TensorProductArray<T, U::Array<V::Array<T>>, V::Array<T>>;

    fn from_fn(f: impl FnMut(usize) -> Self::F) -> Self {
        Self(Self::Array::from_fn(f))
    }
}
impl<U, V, D> TensorNormalization<TensorProduct<U, V>, D> for NormalizeTensorProduct
where
    U: Tensor<Hand = Right, Action: TensorProductAction<V::Action>> + NormalizeWith<Undecorated>,
    V: Tensor<F = U::F, Hand = Left, Action: ActionExists> + NormalizeWith<Undecorated>,
    <U as NormalizeWith<Undecorated>>::Normalized:
        Tensor<F = U::F, Hand = Right, Action = U::Action>,
    <V as NormalizeWith<Undecorated>>::Normalized:
        Tensor<F = U::F, Hand = Left, Action = V::Action>,
    D: TensorDecoration
        + ApplyTensorDecoration<
            TensorProduct<
                <U as NormalizeWith<Undecorated>>::Normalized,
                <V as NormalizeWith<Undecorated>>::Normalized,
            >,
        >,
{
    type Normalized = <D as ApplyTensorDecoration<
        TensorProduct<
            <U as NormalizeWith<Undecorated>>::Normalized,
            <V as NormalizeWith<Undecorated>>::Normalized,
        >,
    >>::Output;
    fn normalize(tensor: TensorProduct<U, V>) -> Self::Normalized {
        D::apply(TensorProduct::from_fn(|i| tensor[i]))
    }
}

impl<
    U: Tensor<Hand = Right, Action: TensorProductAction<V::Action>>,
    V: Tensor<F = U::F, Hand = Left, Action: ActionExists>,
> AsRef<TensorProductArray<V::F, U::Array<V::Array<V::F>>, V::Array<V::F>>>
    for TensorProduct<U, V>
{
    fn as_ref(&self) -> &TensorProductArray<V::F, U::Array<V::Array<V::F>>, V::Array<V::F>> {
        &self.0
    }
}

impl<
    U: Tensor<Hand = Right, Action: TensorProductAction<V::Action>>,
    V: Tensor<F = U::F, Hand = Left, Action: ActionExists>,
> AsMut<TensorProductArray<V::F, U::Array<V::Array<V::F>>, V::Array<V::F>>>
    for TensorProduct<U, V>
{
    fn as_mut(
        &mut self,
    ) -> &mut TensorProductArray<V::F, U::Array<V::Array<V::F>>, V::Array<V::F>> {
        &mut self.0
    }
}

impl<
    U: Tensor<Hand = Right, Action: TensorProductAction<V::Action>>,
    V: Tensor<F = U::F, Hand = Left, Action: ActionExists>,
> Index<(usize, usize)> for TensorProduct<U, V>
{
    type Output = V::F;

    fn index(&self, index: (usize, usize)) -> &V::F {
        &self.0[index]
    }
}

impl<
    U: Tensor<Hand = Right, Action: TensorProductAction<V::Action>>,
    V: Tensor<F = U::F, Hand = Left, Action: ActionExists>,
> IndexMut<(usize, usize)> for TensorProduct<U, V>
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut V::F {
        &mut self.0[index]
    }
}

impl_vector_ops!(
    TensorProduct<U, V>,
    U: Tensor<Hand = Right, Action: TensorProductAction<V::Action>>,
    V: Tensor<F = U::F, Hand = Left, Action: ActionExists>,
);

/// Descends into the left operand of a tensor product.
#[derive(Debug, Copy, Clone)]
pub struct OnLeft<P>(PhantomData<P>);

/// Descends into the right operand of a tensor product.
#[derive(Debug, Copy, Clone)]
pub struct OnRight<P>(PhantomData<P>);

/// Passes through an explicitly selected [`Sinister`] presentation.
#[derive(Debug, Copy, Clone)]
pub struct ThroughSinister<P>(PhantomData<P>);

/// Passes through an explicitly selected [`Dual`] presentation.
#[derive(Debug, Copy, Clone)]
pub struct ThroughDual<P>(PhantomData<P>);

/// Applies one reassociation at a node selected by a type-level tree path.
///
/// Tensor subtrees are atomic unless the path explicitly enters them. Thus a
/// `TensorProduct<A, B>` may be treated as one vector space by an operation at
/// its parent, or opened by [`OnLeft`] and [`OnRight`] when its construction is
/// relevant.
pub trait ReassociateKernel<P>: Tensor {
    /// The same tensor factors and coordinate order in the new presentation.
    type Reassociated: Tensor<F = Self::F, Hand = Self::Hand, Action = Self::Action>;

    /// Performs the selected coordinate-preserving associativity isomorphism.
    fn reassociate_kernel(self) -> Self::Reassociated;
}

pub trait Reassociate {
    fn reassociate<P>(
        self,
    ) -> <<Self as ReassociateKernel<P>>::Reassociated as NormalizeWith<Undecorated>>::Normalized
    where
        Self: ReassociateKernel<P>,
        <Self as ReassociateKernel<P>>::Reassociated: NormalizeWith<Undecorated>,
    {
        NormalizeWith::<Undecorated>::normalize_with(
            <Self as ReassociateKernel<P>>::reassociate_kernel(self),
        )
    }
}

impl<T> Reassociate for T {}

impl<A, B, C> ReassociateKernel<Right> for TensorProduct<TensorProduct<A, B>, C>
where
    A: Tensor<Hand = Right, Action = BothSided>,
    B: Tensor<F = A::F, Hand = Left, Action = BothSided>,
    C: Tensor<F = A::F, Hand = Left, Action = BothSided>,
{
    type Reassociated = TensorProduct<A, Sinister<TensorProduct<Sinister<B>, C>>>;

    fn reassociate_kernel(self) -> Self::Reassociated {
        Self::Reassociated::from_fn(|i| self[i])
    }
}

impl<A, B, C> ReassociateKernel<Left> for TensorProduct<A, Sinister<TensorProduct<B, C>>>
where
    A: Tensor<Hand = Right, Action = BothSided>,
    B: Tensor<F = A::F, Hand = Right, Action = BothSided>,
    C: Tensor<F = A::F, Hand = Left, Action = BothSided>,
{
    type Reassociated = TensorProduct<TensorProduct<A, Sinister<B>>, C>;

    fn reassociate_kernel(self) -> Self::Reassociated {
        Self::Reassociated::from_fn(|i| self[i])
    }
}

impl<A, B, P> ReassociateKernel<OnLeft<P>> for TensorProduct<A, B>
where
    A: Tensor<Hand = Right, Action: TensorProductAction<B::Action>> + ReassociateKernel<P>,
    B: Tensor<F = A::F, Hand = Left, Action: ActionExists>,
    <A as ReassociateKernel<P>>::Reassociated: Tensor<F = A::F, Hand = Right, Action = A::Action>,
{
    type Reassociated = TensorProduct<<A as ReassociateKernel<P>>::Reassociated, B>;

    fn reassociate_kernel(self) -> Self::Reassociated {
        Self::Reassociated::from_fn(|i| self[i])
    }
}

impl<A, B, P> ReassociateKernel<OnRight<P>> for TensorProduct<A, B>
where
    A: Tensor<Hand = Right, Action: TensorProductAction<B::Action>>,
    B: Tensor<F = A::F, Hand = Left, Action: ActionExists> + ReassociateKernel<P>,
    <B as ReassociateKernel<P>>::Reassociated: Tensor<F = A::F, Hand = Left, Action = B::Action>,
{
    type Reassociated = TensorProduct<A, <B as ReassociateKernel<P>>::Reassociated>;

    fn reassociate_kernel(self) -> Self::Reassociated {
        Self::Reassociated::from_fn(|i| self[i])
    }
}

impl<T, P> ReassociateKernel<ThroughSinister<P>> for Sinister<T>
where
    T: Tensor<Action = BothSided> + ReassociateKernel<P>,
    <T as ReassociateKernel<P>>::Reassociated: Tensor<F = T::F, Hand = T::Hand, Action = BothSided>,
{
    type Reassociated = Sinister<<T as ReassociateKernel<P>>::Reassociated>;

    fn reassociate_kernel(self) -> Self::Reassociated {
        Self::Reassociated::from_fn(|i| self[i])
    }
}

#[derive(Debug, Copy, Clone)]
pub enum Here {}

#[doc(hidden)]
pub trait SwapKernel<P>: Tensor {
    type Swapped: Tensor<F = Self::F, Hand = Self::Hand, Action = Self::Action>;
    fn source_index(output: usize) -> usize;
}

pub trait Swap: Tensor {
    fn swap<P>(self) -> <<Self as SwapKernel<P>>::Swapped as NormalizeWith<Undecorated>>::Normalized
    where
        Self: SwapKernel<P>,
        <Self as SwapKernel<P>>::Swapped: NormalizeWith<Undecorated>,
    {
        let raw = <Self as SwapKernel<P>>::Swapped::from_fn(|i| {
            self[<Self as SwapKernel<P>>::source_index(i)]
        });
        NormalizeWith::<Undecorated>::normalize_with(raw)
    }
}
impl<T: Tensor> Swap for T {}

impl<A, B> SwapKernel<Here> for TensorProduct<A, B>
where
    A: Tensor<Hand = Right, Action = BothSided>,
    B: Tensor<F = A::F, Hand = Left, Action = BothSided>,
{
    type Swapped = TensorProduct<Sinister<B>, Sinister<A>>;
    fn source_index(output: usize) -> usize {
        let right = output / A::N;
        let left = output % A::N;
        left * B::N + right
    }
}
impl<A, B, P> SwapKernel<OnLeft<P>> for TensorProduct<A, B>
where
    A: Tensor<Hand = Right, Action: TensorProductAction<B::Action>> + SwapKernel<P>,
    B: Tensor<F = A::F, Hand = Left, Action: ActionExists>,
    <A as SwapKernel<P>>::Swapped: Tensor<F = A::F, Hand = Right, Action = A::Action>,
{
    type Swapped = TensorProduct<<A as SwapKernel<P>>::Swapped, B>;
    fn source_index(output: usize) -> usize {
        let (left, right) = (output / B::N, output % B::N);
        <A as SwapKernel<P>>::source_index(left) * B::N + right
    }
}
impl<A, B, P> SwapKernel<OnRight<P>> for TensorProduct<A, B>
where
    A: Tensor<Hand = Right, Action: TensorProductAction<B::Action>>,
    B: Tensor<F = A::F, Hand = Left, Action: ActionExists> + SwapKernel<P>,
    <B as SwapKernel<P>>::Swapped: Tensor<F = A::F, Hand = Left, Action = B::Action>,
{
    type Swapped = TensorProduct<A, <B as SwapKernel<P>>::Swapped>;
    fn source_index(output: usize) -> usize {
        let (left, right) = (output / B::N, output % B::N);
        left * B::N + <B as SwapKernel<P>>::source_index(right)
    }
}

pub trait Contract: Tensor {
    fn contract<P>(
        self,
    ) -> <<Self as ContractKernel<P>>::Shape as NormalizedContractionShape<Self::F>>::Output
    where
        Self: ContractKernel<P>,
        <Self as ContractKernel<P>>::Shape: NormalizedContractionShape<Self::F>,
    {
        <<Self as ContractKernel<P>>::Shape as NormalizedContractionShape<Self::F>>::from_fn(
            |output| {
                (0..<Self as ContractKernel<P>>::CONTRACTED_N).fold(
                    Self::F::zero(),
                    |sum, contracted| {
                        sum + self[<Self as ContractKernel<P>>::source_index(output, contracted)]
                    },
                )
            },
        )
    }
}

impl<T: Tensor> Contract for T {}

#[doc(hidden)]
pub trait ContractionShape<F: Field> {
    type Output;

    fn from_fn(f: impl FnMut(usize) -> F) -> Self::Output;
}

#[doc(hidden)]
pub trait NormalizedContractionShape<F: Field>: ContractionShape<F> {
    type Output;
    fn from_fn(f: impl FnMut(usize) -> F) -> <Self as NormalizedContractionShape<F>>::Output;
}

#[doc(hidden)]
pub enum ScalarContraction {}

#[doc(hidden)]
pub struct TensorContraction<T: Tensor>(PhantomData<T>);

#[doc(hidden)]
pub trait SinisterContraction<F: Field>: ContractionShape<F> {
    type Shape: ContractionShape<F>;
}

impl<F: Field> SinisterContraction<F> for ScalarContraction {
    type Shape = ScalarContraction;
}

impl<F: Field, T: Tensor<F = F, Action = BothSided>> SinisterContraction<F>
    for TensorContraction<T>
{
    type Shape = TensorContraction<Sinister<T>>;
}

impl<F: Field> ContractionShape<F> for ScalarContraction {
    type Output = F;

    fn from_fn(mut f: impl FnMut(usize) -> F) -> Self::Output {
        f(0)
    }
}

impl<F: Field> NormalizedContractionShape<F> for ScalarContraction {
    type Output = F;
    fn from_fn(mut f: impl FnMut(usize) -> F) -> F {
        f(0)
    }
}

impl<F: Field, T: Tensor<F = F>> ContractionShape<F> for TensorContraction<T> {
    type Output = T;

    fn from_fn(f: impl FnMut(usize) -> F) -> Self::Output {
        T::from_fn(f)
    }
}

impl<F: Field, T: Tensor<F = F> + NormalizeWith<Undecorated>> NormalizedContractionShape<F>
    for TensorContraction<T>
{
    type Output = <T as NormalizeWith<Undecorated>>::Normalized;
    fn from_fn(f: impl FnMut(usize) -> F) -> <Self as NormalizedContractionShape<F>>::Output {
        NormalizeWith::<Undecorated>::normalize_with(T::from_fn(f))
    }
}

#[doc(hidden)]
pub trait AppendContractionRight<F: Field, B: Tensor<F = F>>: ContractionShape<F> {
    type Shape: ContractionShape<F>;

    fn split_output(index: usize) -> (usize, usize);
}

#[doc(hidden)]
pub trait AppendScalarContractionRight<F: Field, B: Tensor<F = F>>: Sidedness {
    type Shape: ContractionShape<F>;
}

impl<F: Field, B: Tensor<F = F, Hand = Left, Action = OneSided>> AppendScalarContractionRight<F, B>
    for OneSided
{
    type Shape = TensorContraction<B>;
}

impl<F: Field, B: Tensor<F = F, Hand = Left, Action = BothSided>> AppendScalarContractionRight<F, B>
    for BothSided
{
    type Shape = TensorContraction<Sinister<B>>;
}

impl<F, B> AppendContractionRight<F, B> for ScalarContraction
where
    F: Field,
    B: Tensor<F = F, Hand = Left>,
    B::Action: AppendScalarContractionRight<F, B>,
{
    type Shape = <B::Action as AppendScalarContractionRight<F, B>>::Shape;

    fn split_output(index: usize) -> (usize, usize) {
        (0, index)
    }
}

impl<F, A, B> AppendContractionRight<F, B> for TensorContraction<A>
where
    F: Field,
    A: Tensor<F = F, Hand = Right, Action: TensorProductAction<B::Action>>,
    B: Tensor<F = F, Hand = Left, Action: ActionExists>,
{
    type Shape = TensorContraction<TensorProduct<A, B>>;

    fn split_output(index: usize) -> (usize, usize) {
        (index / B::N, index % B::N)
    }
}

#[doc(hidden)]
pub trait AppendContractionLeft<F: Field, A: Tensor<F = F>>: ContractionShape<F> {
    type Shape: ContractionShape<F>;

    fn split_output(index: usize) -> (usize, usize);
}

impl<F: Field, A: Tensor<F = F>> AppendContractionLeft<F, A> for ScalarContraction {
    type Shape = TensorContraction<A>;

    fn split_output(index: usize) -> (usize, usize) {
        (index, 0)
    }
}

impl<F, A, B> AppendContractionLeft<F, A> for TensorContraction<B>
where
    F: Field,
    A: Tensor<Hand = Right, F = F, Action: TensorProductAction<B::Action>>,
    B: Tensor<Hand = Left, F = F, Action: ActionExists>,
{
    type Shape = TensorContraction<TensorProduct<A, B>>;

    fn split_output(index: usize) -> (usize, usize) {
        (index / B::N, index % B::N)
    }
}

#[doc(hidden)]
pub trait ContractKernel<P>: Tensor {
    type Shape: ContractionShape<Self::F>;
    const CONTRACTED_N: usize;

    fn source_index(output: usize, contracted: usize) -> usize;
}

#[doc(hidden)]
pub trait ContractibleWith<Rhs: Tensor<F = Self::F>>: Tensor {}

impl<V> ContractibleWith<Dual<V>> for V
where
    V: Tensor<Hand = Right, Action: TensorProductAction<V::Action>>,
    V::Action: ActionExists,
{
}

impl<V> ContractibleWith<V> for Dual<V>
where
    V: Tensor<Hand = Left, Action: ActionExists>,
    V::Action: TensorProductAction<V::Action>,
{
}

impl<V> ContractibleWith<Sinister<V>> for Sinister<Dual<V>> where
    V: Tensor<Hand = Right, Action = BothSided>
{
}

impl<V> ContractibleWith<Sinister<Dual<V>>> for Sinister<V> where
    V: Tensor<Hand = Left, Action = BothSided>
{
}

impl<A, B> ContractKernel<Here> for TensorProduct<A, B>
where
    A: Tensor<Hand = Right, Action: TensorProductAction<B::Action>> + ContractibleWith<B>,
    B: Tensor<F = A::F, Hand = Left, Action: ActionExists>,
{
    type Shape = ScalarContraction;
    const CONTRACTED_N: usize = A::N;

    fn source_index(_output: usize, contracted: usize) -> usize {
        contracted * B::N + contracted
    }
}

impl<A, B, P> ContractKernel<OnLeft<P>> for TensorProduct<A, B>
where
    A: Tensor<Hand = Right, Action: TensorProductAction<B::Action>> + ContractKernel<P>,
    B: Tensor<F = A::F, Hand = Left, Action: ActionExists>,
    <A as ContractKernel<P>>::Shape: AppendContractionRight<A::F, B>,
{
    type Shape = <<A as ContractKernel<P>>::Shape as AppendContractionRight<A::F, B>>::Shape;
    const CONTRACTED_N: usize = <A as ContractKernel<P>>::CONTRACTED_N;

    fn source_index(output: usize, contracted: usize) -> usize {
        let (left, right) =
            <<A as ContractKernel<P>>::Shape as AppendContractionRight<A::F, B>>::split_output(
                output,
            );
        <A as ContractKernel<P>>::source_index(left, contracted) * B::N + right
    }
}

impl<A, B, P> ContractKernel<OnRight<P>> for TensorProduct<A, B>
where
    A: Tensor<Hand = Right, Action: TensorProductAction<B::Action>>,
    B: Tensor<F = A::F, Hand = Left, Action: ActionExists> + ContractKernel<P>,
    <B as ContractKernel<P>>::Shape: AppendContractionLeft<A::F, A>,
{
    type Shape = <<B as ContractKernel<P>>::Shape as AppendContractionLeft<A::F, A>>::Shape;
    const CONTRACTED_N: usize = <B as ContractKernel<P>>::CONTRACTED_N;

    fn source_index(output: usize, contracted: usize) -> usize {
        let (left, right) =
            <<B as ContractKernel<P>>::Shape as AppendContractionLeft<A::F, A>>::split_output(
                output,
            );
        left * B::N + <B as ContractKernel<P>>::source_index(right, contracted)
    }
}

impl<T, P> ContractKernel<ThroughSinister<P>> for Sinister<T>
where
    T: Tensor<Action = BothSided> + ContractKernel<P>,
    <T as ContractKernel<P>>::Shape: SinisterContraction<T::F>,
{
    type Shape = <<T as ContractKernel<P>>::Shape as SinisterContraction<T::F>>::Shape;
    const CONTRACTED_N: usize = <T as ContractKernel<P>>::CONTRACTED_N;

    fn source_index(output: usize, contracted: usize) -> usize {
        <T as ContractKernel<P>>::source_index(output, contracted)
    }
}

type HomOf<BT, FT> = TensorProduct<FT, Dual<BT>>;

/// A derivative represented as a linear map between tangent spaces.
///
/// The carrier is `FT ⊗ BT*`, so coordinates are stored in
/// **output-by-input** order. `FP` and `Fiber` retain the geometric target of
/// the map even though the coordinate carrier depends only on `BT` and `FT`.
#[derive(Debug)]
pub struct TangentMap<
    BT: Tensor<Hand = Right, Action: ActionExists>,
    FP: Point,
    FT: Tensor<F = BT::F, Hand = Right, Action: TensorProductAction<BT::Action>>,
    Fiber: TangentBundle<FP, FT>,
>(HomOf<BT, FT>, PhantomData<fn() -> (FP, Fiber)>);

impl<
    BT: Tensor<Hand = Right, Action: ActionExists>,
    FP: Point,
    FT: Tensor<F = BT::F, Hand = Right, Action: TensorProductAction<BT::Action>>,
    Fiber: TangentBundle<FP, FT>,
> TangentMap<BT, FP, FT, Fiber>
{
    /// Wraps the tensor representing a tangent map.
    pub fn new(v: HomOf<BT, FT>) -> Self {
        Self(v, PhantomData)
    }
}

impl<
    BT: Tensor<Hand = Right, Action: ActionExists>,
    FP: Point,
    FT: Tensor<F = BT::F, Hand = Right, Action: TensorProductAction<BT::Action>>,
    Fiber: TangentBundle<FP, FT>,
> Clone for TangentMap<BT, FP, FT, Fiber>
{
    fn clone(&self) -> Self {
        Self(self.0.clone(), self.1)
    }
}

impl<
    BT: Tensor<Hand = Right, Action: ActionExists>,
    FP: Point,
    FT: Tensor<F = BT::F, Hand = Right, Action: TensorProductAction<BT::Action>>,
    Fiber: TangentBundle<FP, FT>,
>
    AsRef<
        TensorProductArray<
            BT::F,
            FT::Array<<Dual<BT> as Tensor>::Array<BT::F>>,
            <Dual<BT> as Tensor>::Array<BT::F>,
        >,
    > for TangentMap<BT, FP, FT, Fiber>
{
    fn as_ref(
        &self,
    ) -> &TensorProductArray<
        BT::F,
        FT::Array<<Dual<BT> as Tensor>::Array<BT::F>>,
        <Dual<BT> as Tensor>::Array<BT::F>,
    > {
        self.0.as_ref()
    }
}

impl<
    BT: Tensor<Hand = Right, Action: ActionExists>,
    FP: Point,
    FT: Tensor<F = BT::F, Hand = Right, Action: TensorProductAction<BT::Action>>,
    Fiber: TangentBundle<FP, FT>,
>
    AsMut<
        TensorProductArray<
            BT::F,
            FT::Array<<Dual<BT> as Tensor>::Array<BT::F>>,
            <Dual<BT> as Tensor>::Array<BT::F>,
        >,
    > for TangentMap<BT, FP, FT, Fiber>
{
    fn as_mut(
        &mut self,
    ) -> &mut TensorProductArray<
        BT::F,
        FT::Array<<Dual<BT> as Tensor>::Array<BT::F>>,
        <Dual<BT> as Tensor>::Array<BT::F>,
    > {
        self.0.as_mut()
    }
}

impl<
    BT: Tensor<Hand = Right, Action: ActionExists>,
    FP: Point,
    FT: Tensor<F = BT::F, Hand = Right, Action: TensorProductAction<BT::Action>>,
    Fiber: TangentBundle<FP, FT>,
> Tensor for TangentMap<BT, FP, FT, Fiber>
{
    type Normalization = Atomic;
    type F = <HomOf<BT, FT> as Tensor>::F;
    type Array<T: Point> = <HomOf<BT, FT> as Tensor>::Array<T>;
    type Hand = <HomOf<BT, FT> as Tensor>::Hand;
    type Action = <HomOf<BT, FT> as Tensor>::Action;

    fn from_fn(f: impl FnMut(usize) -> Self::F) -> Self {
        Self(HomOf::<BT, FT>::from_fn(f), PhantomData)
    }
}
impl_vector_ops!(TangentMap<BT, FP, FT, Fiber>,
    BT: Tensor<Hand = Right, Action: ActionExists>,
    FP: Point,
    FT: Tensor<F = BT::F, Hand = Right, Action: TensorProductAction<BT::Action>>,
    Fiber: TangentBundle<FP, FT>,
);

impl<
    BT: Tensor<Hand = Right, Action: ActionExists>,
    FP: Point,
    FT: Tensor<F = BT::F, Hand = Right, Action: TensorProductAction<BT::Action>>,
    Fiber: TangentBundle<FP, FT>,
> Deref for TangentMap<BT, FP, FT, Fiber>
{
    type Target = HomOf<BT, FT>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<
    BT: Tensor<Hand = Right, Action: ActionExists>,
    FP: Point,
    FT: Tensor<F = BT::F, Hand = Right, Action: TensorProductAction<BT::Action>>,
    Fiber: TangentBundle<FP, FT>,
> DerefMut for TangentMap<BT, FP, FT, Fiber>
{
    fn deref_mut(&mut self) -> &mut HomOf<BT, FT> {
        &mut self.0
    }
}

/// Selects the algebra implemented by a [`Jet`] presentation.
///
/// [`Algebraic`] works over arbitrary fields. [`JetReal`] additionally exposes
/// the projection semantics required by [`Real`], allowing generic real
/// functions to be evaluated by forward automatic differentiation.
pub trait JetMode: Copy + Clone + core::fmt::Debug {}

/// Algebraic jets over an arbitrary [`Field`].
#[derive(Debug, Copy, Clone)]
pub enum Algebraic {}
/// Real jets whose comparisons and nonanalytic operations project to the
/// constant coefficient.
#[derive(Debug, Copy, Clone)]
pub enum JetReal {}

impl JetMode for Algebraic {}
impl JetMode for JetReal {}

/// A tensor `V` whose coordinates are jets over a presented scalar `S`.
///
/// `V` remains the *logical* tensor: its dimension, array family, handedness,
/// and available scalar actions are preserved. `S` records the scalar
/// presentation currently carried by each coordinate. Keeping those roles
/// separate is what permits nested differentiation without constructing a
/// nominal tower such as `JetVector<JetVector<V>>`.
///
/// `N` is the highest retained Taylor coefficient. The default is a first-order
/// presentation, which is sufficient for one forward derivative layer.
#[derive(Copy, Clone, Debug)]
pub struct JetVector<
    V: Tensor,
    Mode: JetMode = Algebraic,
    const N: usize = 1,
    S: Field = <V as Tensor>::F,
>(V::Array<Jet<S, Mode, N>>);

impl<V: Tensor, M: JetMode, const N: usize> JetVector<V, M, N> {
    /// Embeds every coordinate of `v` as a constant jet.
    pub fn constant(v: V) -> Self {
        Self(V::Array::<Jet<V::F, M, N>>::from_fn(|i| {
            Jet::from_field(v[i])
        }))
    }
}

impl<V: Tensor, const N: usize, S: Field> Tensor for JetVector<V, Algebraic, N, S>
where
    Jet<S, Algebraic, N>: Field,
{
    type Normalization = Atomic;
    type F = Jet<S, Algebraic, N>;

    type Array<T: Point> = V::Array<T>;

    type Hand = V::Hand;
    type Action = V::Action;

    fn from_fn(f: impl FnMut(usize) -> Self::F) -> Self {
        Self(V::Array::from_fn(f))
    }
}
impl<V: Tensor<F: Real>, const N: usize, S: Real> Tensor for JetVector<V, JetReal, N, S> {
    type Normalization = Atomic;
    type F = Jet<S, JetReal, N>;

    type Array<T: Point> = V::Array<T>;

    type Hand = V::Hand;
    type Action = V::Action;

    fn from_fn(f: impl FnMut(usize) -> Self::F) -> Self {
        Self(V::Array::from_fn(f))
    }
}
impl<V: Tensor, M: JetMode, const N: usize, S: Field> AsRef<V::Array<Jet<S, M, N>>>
    for JetVector<V, M, N, S>
{
    fn as_ref(&self) -> &V::Array<Jet<S, M, N>> {
        &self.0
    }
}

impl<V: Tensor, M: JetMode, const N: usize, S: Field> AsMut<V::Array<Jet<S, M, N>>>
    for JetVector<V, M, N, S>
{
    fn as_mut(&mut self) -> &mut V::Array<Jet<S, M, N>> {
        &mut self.0
    }
}

impl_vector_ops!(JetVector<V, Algebraic, N, S>, V: Tensor, const N: usize, S: Field);
impl_vector_ops!(JetVector<V, JetReal, N, S>, V: Tensor<F: Real>, const N: usize, S: Real);

type JetCoords<F, const N: usize> = DirectSum<Coords<F, 1>, Coords<F, N>>;

/// A truncated formal power series used for forward automatic differentiation.
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
#[derive(Debug, Copy, Clone)]
pub struct Jet<F: Field, Mode: JetMode, const N: usize = 1>(JetCoords<F, N>, PhantomData<Mode>);

impl<F: Field, M: JetMode, const N: usize> Jet<F, M, N> {
    /// Embeds a field element as a constant jet.
    pub fn from_field(value: F) -> Self {
        Self(
            DirectSum(DirectSumArray([value], [F::zero(); N], PhantomData)),
            PhantomData,
        )
    }

    /// Constructs a jet from its primal value and coefficients of
    /// `ε¹, …, εᴺ`.
    pub fn new(value: F, coefficients: [F; N]) -> Self {
        Self(
            DirectSum(DirectSumArray([value], coefficients, PhantomData)),
            PhantomData,
        )
    }

    /// Constructs all `N + 1` coefficients by index, beginning with the primal
    /// coefficient at index zero.
    pub fn from_fn(f: impl FnMut(usize) -> F) -> Self {
        Self(JetCoords::from_fn(f), PhantomData)
    }

    fn derivative(self) -> Self {
        Self::from_fn(|i| {
            if i < N {
                F::from_nat(i + 1) * self[i + 1]
            } else {
                F::zero()
            }
        })
    }

    fn integrate_from(primal: F, derivative: Self) -> Self {
        Self::from_fn(|i| {
            if i == 0 {
                primal
            } else {
                derivative[i - 1].div(F::from_nat(i))
            }
        })
    }
}

impl<R: Real, const N: usize> Jet<R, JetReal, N> {
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

impl<F: CField, const N: usize> CField for Jet<F, Algebraic, N> {}

impl<F: Field, M: JetMode, const N: usize> PartialEq for Jet<F, M, N> {
    fn eq(&self, other: &Self) -> bool {
        self[0] == other[0]
    }
}

impl<F: Field, M: JetMode, const N: usize> Index<usize> for Jet<F, M, N> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<F: Field, M: JetMode, const N: usize> IndexMut<usize> for Jet<F, M, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<F: Field, const N: usize> Field for Jet<F, Algebraic, N> {
    type Fixed = Jet<F::Fixed, Algebraic, N>;

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

impl<F: Field, M: JetMode, const N: usize> Add for Jet<F, M, N> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0, PhantomData)
    }
}

impl<F: Field, M: JetMode, const N: usize> Sub for Jet<F, M, N> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0, PhantomData)
    }
}

impl<F: Field, M: JetMode, const N: usize> Mul for Jet<F, M, N> {
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

impl<F: Field, M: JetMode, const N: usize> Neg for Jet<F, M, N> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0, PhantomData)
    }
}

impl<F: Field, M: JetMode, const N: usize> One for Jet<F, M, N> {
    fn one() -> Self {
        Self::from_fn(|x| if x == 0 { F::one() } else { F::zero() })
    }
}

impl<F: Field, M: JetMode, const N: usize> Zero for Jet<F, M, N> {
    fn zero() -> Self {
        Self(DirectSum::zero(), PhantomData)
    }

    fn is_zero(&self) -> bool {
        self[0].is_zero()
    }
}

impl<F: Field, M: JetMode, const N: usize> Inv for NonZero<Jet<F, M, N>> {
    type Output = NonZero<Jet<F, M, N>>;

    fn inv(self) -> Self::Output {
        let input = self.0;

        // Spell this using your DivRing::Mul machinery.
        let constant_inverse: F = <F as DivRing>::Mul::from(NonZero::new_unchecked(input[0]))
            .inv()
            .into()
            .0;

        let mut output = Jet::<F, M, N>::zero();
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
/// `JetLayer<M, N>` records that the current scalar was obtained by wrapping
/// the preceding scalar in [`Jet<_, M, N>`].
#[derive(Debug, Copy, Clone)]
pub struct JetLayer<M: JetMode, const N: usize>(PhantomData<M>);

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

impl<F, M, const N: usize, Tail> ConstantRoute<F> for Cons<JetLayer<M, N>, Tail>
where
    F: Field,
    M: JetMode,
    Tail: ConstantRoute<F>,
    Jet<Tail::Output, M, N>: Field,
{
    type Output = Jet<Tail::Output, M, N>;

    fn constant(value: F) -> Self::Output {
        Jet::from_field(Tail::constant(value))
    }
}

/// A point together with a jet-valued tangent coordinate and its tower tag.
///
/// `Tower` distinguishes iterated tangent constructions that have identical
/// runtime representations. Use [`TangentElement::new`] rather than spelling
/// the marker explicitly.
#[derive(Debug, Clone)]
pub struct TangentElement<P: Point, V: Tensor, Tower>(pub P, pub JetVector<V>, PhantomData<Tower>);

impl<P: Point, V: Tensor, Tower> TangentElement<P, V, Tower> {
    /// Constructs a tangent element from its base point and local jet.
    pub fn new(p: P, v: JetVector<V>) -> Self {
        Self(p, v, PhantomData)
    }

    /// Returns a clone of the base point.
    pub fn base_point(&self) -> P {
        self.0.clone()
    }

    /// Borrows the jet-valued tangent coordinate.
    pub fn jet(&self) -> &JetVector<V> {
        &self.1
    }
}

type Prolongation<P, V, T> = TangentElement<P, V, Cons<T, Ø>>;

/// A first [`TangentElement`] at a point of `P`, expressed in `V` coordinates.
pub type Tangent<P, V> = TangentElement<P, V, Ø>;
/// An iterated [`TangentElement`] with explicit [`TangentBundle`] witnesses.
pub type TM<P, V, T, U> = TangentElement<P, V, Cons<T, Cons<U, Ø>>>;
/// The tangent bundle of `T`, represented by the canonical jet prolongation.
///
/// This is the concrete iterated-tangent representation constructed by
/// [`TangentLift`].
pub type LiftedTM<P, V, T> = TM<P, V, T, Prolongation<P, V, T>>;

impl<P: Point, V: Tensor, T: TangentBundle<P, V>, U: TangentBundle<Self, JetVector<V>>> Chart<P, V>
    for TM<P, V, T, U>
{
    type Global = T::Global;

    fn to_local(&self, point: &P) -> Option<V> {
        T::chart_at(&self.0).to_local(point)
    }

    fn to_global(&self, coord: V) -> Self::Global {
        T::chart_at(&self.0).to_global(coord)
    }

    fn chart_at(p: &P) -> Self {
        Self(p.clone(), JetVector::zero(), PhantomData)
    }
}
impl<P: Point, V: Tensor, T: TangentBundle<P, V>, U: TangentBundle<Self, JetVector<V>>> ExpMap<P, V>
    for TM<P, V, T, U>
{
}
impl<P: Point, V: Tensor, T: TangentBundle<P, V>, U: TangentBundle<Self, JetVector<V>>>
    TangentBundle<P, V> for TM<P, V, T, U>
{
}

impl<P: Point, V: Tensor, T: TangentBundle<P, V>, U: TangentBundle<Self, JetVector<V>>>
    Chart<Self, JetVector<V>> for TM<P, V, T, U>
{
    type Global = U::Global;

    fn to_local(&self, point: &Self) -> Option<JetVector<V>> {
        U::chart_at(self).to_local(point)
    }

    fn to_global(&self, coord: JetVector<V>) -> Self::Global {
        U::chart_at(self).to_global(coord)
    }

    fn chart_at(p: &Self) -> Self {
        p.clone()
    }
}
impl<P: Point, V: Tensor, T: TangentBundle<P, V>, U: TangentBundle<Self, JetVector<V>>>
    ExpMap<Self, JetVector<V>> for TM<P, V, T, U>
{
}

impl<P: Point, V: Tensor, T: TangentBundle<P, V>, U: TangentBundle<Self, JetVector<V>>>
    TangentBundle<Self, JetVector<V>> for TM<P, V, T, U>
{
}

impl<P: Point, V: Tensor, T: TangentLift<P, V>, U: TangentBundle<Self, JetVector<V>>>
    TangentLift<P, V> for TM<P, V, T, U>
{
    fn tangent_to_local(base: Tangent<P, V>, local: Tangent<P, V>) -> Option<JetVector<V>> {
        T::tangent_to_local(base, local)
    }

    fn tangent_to_global(base: Tangent<P, V>, coordinate: JetVector<V>) -> (P, JetVector<V>) {
        T::tangent_to_global(base, coordinate)
    }
}

/// Extends a tangent-bundle chart to jet-valued tangent coordinates.
///
/// Implementing this trait is the admission point for differentiating through
/// a manifold. It states how a tangent element is expressed in the local chart
/// at another tangent element, and how that local jet is returned to the global
/// bundle. [`Tangent`] is the first lifted element; [`TM`] and [`LiftedTM`]
/// describe its iterated tangent bundles. Vector spaces receive the canonical
/// translation-based implementation.
pub trait TangentLift<P: Point, V: Tensor>: TangentBundle<P, V> {
    /// Expresses `local` in the lifted chart centred at `base`.
    fn tangent_to_local(base: Tangent<P, V>, local: Tangent<P, V>) -> Option<JetVector<V>>;
    /// Reconstructs a global point and tangent jet from a lifted coordinate.
    fn tangent_to_global(base: Tangent<P, V>, coordinate: JetVector<V>) -> (P, JetVector<V>);
}

impl<P: Point, V: Tensor, T: TangentLift<P, V>> Chart<LiftedTM<P, V, T>, JetVector<V>>
    for Prolongation<P, V, T>
{
    type Global = LiftedTM<P, V, T>;

    fn to_local(&self, point: &LiftedTM<P, V, T>) -> Option<JetVector<V>> {
        T::tangent_to_local(
            TangentElement::new(self.0.clone(), self.1.clone()),
            TangentElement::new(point.0.clone(), point.1.clone()),
        )
    }

    fn to_global(&self, coordinate: JetVector<V>) -> Self::Global {
        let (base, jet) = T::tangent_to_global(
            TangentElement::new(self.0.clone(), self.1.clone()),
            coordinate,
        );

        TangentElement::new(base, jet)
    }

    fn chart_at(point: &LiftedTM<P, V, T>) -> Self {
        TangentElement::new(point.0.clone(), point.1.clone())
    }
}

impl<P: Point, V: Tensor, T: TangentLift<P, V>> ExpMap<LiftedTM<P, V, T>, JetVector<V>>
    for Prolongation<P, V, T>
{
}
impl<P: Point, V: Tensor, T: TangentLift<P, V>> TangentBundle<LiftedTM<P, V, T>, JetVector<V>>
    for Prolongation<P, V, T>
{
}

impl<V: Tensor> TangentLift<V, V> for V {
    fn tangent_to_local(base: Tangent<V, V>, local: Tangent<V, V>) -> Option<JetVector<V>> {
        Some(JetVector::from_fn(|i| {
            local.1[i] - base.1[i] + Jet::from_field(local.0[i] - base.0[i])
        }))
    }

    fn tangent_to_global(base: Tangent<V, V>, coordinate: JetVector<V>) -> (V, JetVector<V>) {
        let combined =
            JetVector::<V>::from_fn(|i| Jet::from_field(base.0[i]) + base.1[i] + coordinate[i]);

        let base = V::from_fn(|i| combined[i][0]);

        let tangent = JetVector::from_fn(|i| {
            let mut value = combined[i];
            value[0] = V::F::zero();
            value
        });

        (base, tangent)
    }
}

/// A composable differential program for `F`.
///
/// Construct it as `d(f)`. Calling [`d::at`] evaluates the full derivative,
/// while [`d::along`] contracts the next derivative slot with a direction.
/// Since `d<F>` itself implements [`JetMap`], differential programs can be
/// nested: `d(d(f))` and `d(d(d(f)))` use the same machinery as `d(f)`.
#[allow(non_camel_case_types)]
pub struct d<F>(pub F);

/// A differential program with its next input slot contracted with `direction`.
///
/// This represents the function `p ↦ Dfₚ(direction)`. It remains a
/// differentiable program until [`Along::at`] evaluates it.
pub struct Along<F, V> {
    f: F,
    direction: V,
}

impl<F> d<F> {
    /// Evaluates the full derivative at `point`.
    ///
    /// For `f: BT → FT`, the result is represented as `FT ⊗ BT*`, with
    /// output coordinates outermost and input coordinates innermost.
    pub fn at<
        BT: Tensor<Hand = Right, Action: ActionExists>,
        FT: Tensor<F = BT::F, Hand = Right, Action: TensorProductAction<BT::Action>>,
        M: JetMode,
    >(
        &self,
        point: BT,
    ) -> TangentMap<BT, FT, FT, FT>
    where
        Self: EvaluableAt<BT, TangentMap<BT, FT, FT, FT>, M>,
    {
        <Self as EvaluableAt<BT, TangentMap<BT, FT, FT, FT>, M>>::evaluate_at(self, point)
    }

    /// Contracts the next derivative slot with `direction`.
    pub fn along<V>(self, direction: V) -> Along<F, V> {
        Along {
            f: self.0,
            direction,
        }
    }
}

impl<F, BT> Along<F, BT> {
    /// Evaluates the directional derivative at `point`.
    pub fn at<FT, M>(&self, point: BT) -> FT
    where
        Self: EvaluableAt<BT, FT, M>,
    {
        <Self as EvaluableAt<BT, FT, M>>::evaluate_at(self, point)
    }
}

#[diagnostic::on_unimplemented(
    message = "this differential program cannot be evaluated at `{Point}`",
    label = "the composed differential operations are not defined for this point type",
    note = "the function may not accept the required jet presentation",
    note = "the input and output tensors may have incompatible fields, handedness, or actions",
    note = "a required form or musical isomorphism may not lift through nested jets"
)]
#[doc(hidden)]
/// The diagnostic evaluation boundary used by [`d::at`] and [`Along::at`].
///
/// Keeping their large proof obligations behind this trait replaces a wall of
/// nested associated-type failures with one explanation of why a differential
/// program is not evaluable at a particular point type.
pub trait EvaluableAt<Point, Output, Mode> {
    fn evaluate_at(&self, point: Point) -> Output;
}

impl<F, BT, FT, M> EvaluableAt<BT, TangentMap<BT, FT, FT, FT>, M> for d<F>
where
    BT: Tensor<Hand = Right, Action: ActionExists>,
    FT: Tensor<F = BT::F, Hand = Right, Action: TensorProductAction<BT::Action>>,
    M: JetMode,
    F: JetMap<BT, FT, M, 1, BT::F>,
    JetVector<BT, M>: Tensor<F = Jet<BT::F, M>>,
    JetVector<FT, M, 1, BT::F>: Tensor<F = Jet<BT::F, M>>,
{
    fn evaluate_at(&self, point: BT) -> TangentMap<BT, FT, FT, FT> {
        let columns: BT::Array<FT> = BT::Array::from_fn(|input_coordinate| {
            let input = JetVector::<BT, M>::from_fn(|coordinate| {
                Jet::new(
                    point[coordinate],
                    [if input_coordinate == coordinate {
                        BT::F::one()
                    } else {
                        BT::F::zero()
                    }],
                )
            });

            let output = <F as JetMap<BT, FT, M, 1, BT::F, Ø>>::jet_at(&self.0, input);

            FT::from_fn(|output_coordinate| output[output_coordinate][1])
        });

        let rows: FT::Array<<Dual<BT> as Tensor>::Array<BT::F>> =
            FT::Array::from_fn(|output_coordinate| {
                <Dual<BT> as Tensor>::Array::from_fn(|input_coordinate| {
                    columns[input_coordinate][output_coordinate]
                })
            });

        TangentMap::new(TensorProduct(TensorProductArray(rows, PhantomData)))
    }
}

impl<F, BT, FT, M> EvaluableAt<BT, FT, M> for Along<F, BT>
where
    BT: Vector,
    FT: Tensor<F = BT::F, Hand = Right, Action: TensorProductAction<BT::Action>>,
    M: JetMode,
    F: JetMap<BT, FT, M, 1, BT::F>,
    JetVector<BT, M, 1, BT::F>: Tensor<F = Jet<BT::F, M, 1>>,
    JetVector<FT, M, 1, BT::F>: Tensor<F = Jet<BT::F, M>>,
{
    fn evaluate_at(&self, point: BT) -> FT {
        let input = JetVector::<BT, M, 1, BT::F>::from_fn(|coordinate| {
            Jet::new(point[coordinate], [self.direction[coordinate]])
        });

        let output: JetVector<FT, M, 1, BT::F> =
            <F as JetMap<BT, FT, M, 1, BT::F, Ø>>::jet_at(&self.f, input);

        FT::from_fn(|coordinate| output[coordinate][1])
    }
}

/// A map that can be evaluated over a selected jet presentation.
///
/// Ordinary generic Rust functions implement this trait through the blanket
/// `Fn(JetVector<BT, ..>)` implementation. Differential programs implement it
/// recursively, adding jet layers while `Route` remembers how to inject
/// captured base-field constants into the current scalar type.
pub trait JetMap<BT: Tensor, FT: Tensor<F = BT::F>, M: JetMode, const N: usize, S: Field, Route = Ø>
{
    /// Evaluates the map without discarding any jet coefficients.
    fn jet_at(&self, input: JetVector<BT, M, N, S>) -> JetVector<FT, M, N, S>;
}

impl<
    F: Fn(JetVector<BT, M, N, S>) -> JetVector<FT, M, N, S>,
    BT: Tensor,
    FT: Tensor<F = BT::F>,
    M: JetMode,
    const N: usize,
    S: Field,
    Route,
> JetMap<BT, FT, M, N, S, Route> for F
{
    fn jet_at(&self, input: JetVector<BT, M, N, S>) -> JetVector<FT, M, N, S> {
        self(input)
    }
}

impl<
    F: JetMap<BT, FT, M, 1, Jet<S, M, N>, Cons<JetLayer<M, N>, Route>>,
    BT: Vector<F = FT::F, Hand = Right>,
    FT: Vector<Hand = Right, Action: TensorProductAction<BT::Action>>,
    M: JetMode,
    const N: usize,
    S: Field,
    Route: ConstantRoute<BT::F, Output = S>,
> JetMap<BT, HomOf<BT, FT>, M, N, S, Route> for d<F>
where
    // The outer presentation.
    JetVector<FT, M, N, S>: Vector<F = Jet<S, M, N>>,
    JetVector<BT, M, N, S>: Tensor<F = Jet<S, M, N>>,
    // One additional derivative layer over the existing outer scalar.
    JetVector<BT, M, 1, Jet<S, M, N>>: Tensor<F = Jet<Jet<S, M, N>, M>>,
    JetVector<FT, M, 1, Jet<S, M, N>>: Tensor<F = Jet<Jet<S, M, N>, M>>,
    Jet<S, M, N>: Field,
{
    fn jet_at(&self, input: JetVector<BT, M, N, S>) -> JetVector<HomOf<BT, FT>, M, N, S> {
        type OuterScalar<S, M, const N: usize> = Jet<S, M, N>;

        let columns: BT::Array<JetVector<FT, M, N, S>> = BT::Array::from_fn(|input_coordinate| {
            let nested_input = JetVector::<BT, M, 1, OuterScalar<S, M, N>>::from_fn(|coordinate| {
                Jet::new(
                    input[coordinate],
                    [if input_coordinate == coordinate {
                        OuterScalar::<S, M, N>::one()
                    } else {
                        OuterScalar::<S, M, N>::zero()
                    }],
                )
            });

            let nested_output: JetVector<FT, M, 1, OuterScalar<S, M, N>> = <F as JetMap<
                BT,
                FT,
                M,
                1,
                OuterScalar<S, M, N>,
                Cons<JetLayer<M, N>, Route>,
            >>::jet_at(
                &self.0, nested_input
            );

            JetVector::<FT, M, N, S>::from_fn(|output_coordinate| {
                nested_output[output_coordinate][1]
            })
        });

        let rows: FT::Array<<Dual<BT> as Tensor>::Array<OuterScalar<S, M, N>>> =
            FT::Array::from_fn(|output_coordinate| {
                <Dual<BT> as Tensor>::Array::from_fn(|input_coordinate| {
                    columns[input_coordinate][output_coordinate]
                })
            });

        JetVector::<HomOf<BT, FT>, M, N, S>(TensorProductArray(rows, PhantomData))
    }
}

impl<F, BT, FT, M, const N: usize, S, Route> JetMap<BT, FT, M, N, S, Route> for Along<F, BT>
where
    BT: Vector<F = FT::F>,
    FT: Vector,
    M: JetMode,
    S: Field,
    Route: ConstantRoute<BT::F, Output = S>,
    Jet<S, M, N>: Field,
    JetVector<FT, M, N, S>: Tensor<F = Jet<S, M, N>>,
    JetVector<BT, M, N, S>: Tensor<F = Jet<S, M, N>>,
    JetVector<BT, M, 1, Jet<S, M, N>>: Tensor<F = Jet<Jet<S, M, N>, M>>,
    JetVector<FT, M, 1, Jet<S, M, N>>: Tensor<F = Jet<Jet<S, M, N>, M>>,
    F: JetMap<BT, FT, M, 1, Jet<S, M, N>, Cons<JetLayer<M, N>, Route>>,
{
    fn jet_at(&self, input: JetVector<BT, M, N, S>) -> JetVector<FT, M, N, S> {
        type OuterScalar<S, M, const N: usize> = Jet<S, M, N>;

        let nested_input = JetVector::<BT, M, 1, OuterScalar<S, M, N>>::from_fn(|coordinate| {
            Jet::new(
                input[coordinate],
                [Jet::from_field(Route::constant(self.direction[coordinate]))],
            )
        });

        let nested_output: JetVector<FT, M, 1, OuterScalar<S, M, N>> =
            <F as JetMap<BT, FT, M, 1, OuterScalar<S, M, N>, Cons<JetLayer<M, N>, Route>>>::jet_at(
                &self.f,
                nested_input,
            );

        JetVector::<FT, M, N, S>::from_fn(|coordinate| nested_output[coordinate][1])
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
    fn jet_flat_array<S: Field, M: JetMode, const N: usize>(
        value: &<Self as Tensor>::Array<Jet<S, M, N>>,
    ) -> <Dual<Self> as Tensor>::Array<Jet<S, M, N>>
    where
        Jet<S, M, N>: Field;

    /// Applies the lifted lowering map to a [`JetVector`].
    fn jet_flat<S: Field, M: JetMode, const N: usize>(
        value: &JetVector<Self, M, N, S>,
    ) -> Dual<JetVector<Self, M, N, S>>
    where
        Jet<S, M, N>: Field,
        JetVector<Self, M, N, S>: Tensor<F = Jet<S, M, N>>,
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
    fn jet_sharp_array<S: Field, M: JetMode, const N: usize>(
        value: &<Dual<Self> as Tensor>::Array<Jet<S, M, N>>,
    ) -> <Self as Tensor>::Array<Jet<S, M, N>>
    where
        Jet<S, M, N>: Field;

    /// Applies the lifted raising map to a jet-valued covector.
    fn jet_sharp<S: Field, M: JetMode, const N: usize>(
        value: Dual<JetVector<Self, M, N, S>>,
    ) -> JetVector<Self, M, N, S>
    where
        Jet<S, M, N>: Field,
        JetVector<Self, M, N, S>: Tensor<F = Jet<S, M, N>>,
    {
        let value = Dual::to_raw(value);

        let value = <Dual<Self> as Tensor>::Array::from_fn(|coordinate| value[coordinate]);

        let sharp = Self::jet_sharp_array(&value);

        JetVector::from_fn(|coordinate| sharp[coordinate])
    }
}

impl<V, M, const N: usize, S> FormLift for JetVector<V, M, N, S>
where
    V: FormLift,
    M: JetMode,
    S: Field,
    Jet<S, M, N>: Field,
    Self: Form<F = Jet<S, M, N>>,
{
    fn jet_flat_array<T: Field, InnerMode: JetMode, const K: usize>(
        value: &<Self as Tensor>::Array<Jet<T, InnerMode, K>>,
    ) -> <Dual<Self> as Tensor>::Array<Jet<T, InnerMode, K>>
    where
        Jet<T, InnerMode, K>: Field,
    {
        let value = V::Array::from_fn(|coordinate| value[coordinate]);
        let flat = V::jet_flat_array::<T, InnerMode, K>(&value);

        <Dual<Self> as Tensor>::Array::from_fn(|coordinate| flat[coordinate])
    }
}

impl<V, M, const N: usize, S> NondegenerateLift for JetVector<V, M, N, S>
where
    V: NondegenerateLift,
    M: JetMode,
    S: Field,
    Jet<S, M, N>: Field,
    Self: Nondegenerate<F = Jet<S, M, N>>,
{
    fn jet_sharp_array<T: Field, InnerMode: JetMode, const K: usize>(
        value: &<Dual<Self> as Tensor>::Array<Jet<T, InnerMode, K>>,
    ) -> <Self as Tensor>::Array<Jet<T, InnerMode, K>>
    where
        Jet<T, InnerMode, K>: Field,
    {
        let value = <Dual<V> as Tensor>::Array::from_fn(|coordinate| value[coordinate]);
        let sharp = V::jet_sharp_array::<T, InnerMode, K>(&value);

        <Self as Tensor>::Array::from_fn(|coordinate| sharp[coordinate])
    }
}

impl<V> FormLift for Dual<V>
where
    V: NondegenerateLift,
{
    fn jet_flat_array<S: Field, M: JetMode, const N: usize>(
        value: &<Self as Tensor>::Array<Jet<S, M, N>>,
    ) -> <Dual<Self> as Tensor>::Array<Jet<S, M, N>>
    where
        Jet<S, M, N>: Field,
    {
        let value = <Dual<V> as Tensor>::Array::from_fn(|coordinate| value[coordinate]);
        let sharp = V::jet_sharp_array::<S, M, N>(&value);

        <Dual<Self> as Tensor>::Array::from_fn(|coordinate| sharp[coordinate])
    }
}

impl<V> NondegenerateLift for Dual<V>
where
    V: NondegenerateLift,
{
    fn jet_sharp_array<S: Field, M: JetMode, const N: usize>(
        value: &<Dual<Self> as Tensor>::Array<Jet<S, M, N>>,
    ) -> <Self as Tensor>::Array<Jet<S, M, N>>
    where
        Jet<S, M, N>: Field,
    {
        let value = <V as Tensor>::Array::from_fn(|coordinate| value[coordinate]);
        let flat = V::jet_flat_array::<S, M, N>(&value);

        <Self as Tensor>::Array::from_fn(|coordinate| flat[coordinate])
    }
}

impl<V> FormLift for Sinister<V>
where
    V: FormLift<Action = BothSided>,
{
    fn jet_flat_array<S: Field, M: JetMode, const N: usize>(
        value: &<Self as Tensor>::Array<Jet<S, M, N>>,
    ) -> <Dual<Self> as Tensor>::Array<Jet<S, M, N>>
    where
        Jet<S, M, N>: Field,
    {
        let value = <V as Tensor>::Array::from_fn(|coordinate| value[coordinate]);
        let flat = V::jet_flat_array::<S, M, N>(&value);

        <Dual<Self> as Tensor>::Array::from_fn(|coordinate| flat[coordinate])
    }
}

impl<V> NondegenerateLift for Sinister<V>
where
    V: NondegenerateLift<Action = BothSided>,
{
    fn jet_sharp_array<S: Field, M: JetMode, const N: usize>(
        value: &<Dual<Self> as Tensor>::Array<Jet<S, M, N>>,
    ) -> <Self as Tensor>::Array<Jet<S, M, N>>
    where
        Jet<S, M, N>: Field,
    {
        let value = <Dual<V> as Tensor>::Array::from_fn(|coordinate| value[coordinate]);
        let sharp = V::jet_sharp_array::<S, M, N>(&value);

        <Self as Tensor>::Array::from_fn(|coordinate| sharp[coordinate])
    }
}

impl<V: FormLift, M: JetMode, const N: usize, S: Field> Form for JetVector<V, M, N, S>
where
    Jet<S, M, N>: Field,
    Self: Tensor<F = Jet<S, M, N>>,
{
    fn flat(&self) -> Dual<Self> {
        V::jet_flat::<S, M, N>(self)
    }
}

impl<V: NondegenerateLift, M: JetMode, const N: usize, S: Field> Nondegenerate
    for JetVector<V, M, N, S>
where
    Jet<S, M, N>: Field,
    Self: Form<F = Jet<S, M, N>>,
{
    fn sharp(value: Dual<Self>) -> Self {
        V::jet_sharp::<S, M, N>(value)
    }
}

impl<V: Sesquilinear + Interval, const N: usize, M: JetMode, S: Field> Interval
    for JetVector<V, M, N, S>
where
    Self: Sesquilinear<F: Field<Fixed: Real>>,
{
    type R = <<Self as Tensor>::F as Field>::Fixed;

    fn interval_squared(&self, other: &Self) -> Self::R {
        (self.clone() - other.clone()).norm_squared()
    }
}

impl<V: Sesquilinear, const N: usize, M: JetMode, S: Field> Sesquilinear for JetVector<V, M, N, S> where
    Self: Nondegenerate + Vector
{
}

impl<V: Tensor + Metric, const N: usize, M: JetMode, S: Field> Metric for JetVector<V, M, N, S> where
    Self: Interval
{
}

impl<V: Euclidean, const N: usize, S: Real> Euclidean for JetVector<V, JetReal, N, S> where
    Self: Vector<F = Jet<S, JetReal, N>, Action = BothSided>
{
}

impl<R: Real, const N: usize> PartialOrd for Jet<R, JetReal, N> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self[0].partial_cmp(&other[0])
    }
}

impl<R: Real, const N: usize> ToPrimitive for Jet<R, JetReal, N> {
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

impl<R: Real, const N: usize> NumCast for Jet<R, JetReal, N> {
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        R::from(n).map(|x| Self::from_field(x))
    }
}

impl<R: Real, const N: usize> Div<Self> for Jet<R, JetReal, N> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self.mul(NonZero::new(rhs).unwrap().inv().0)
    }
}

impl<R: Real, const N: usize> Rem<Self> for Jet<R, JetReal, N> {
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

impl<R: Real, const N: usize> Euclid for Jet<R, JetReal, N> {
    fn div_euclid(&self, rhs: &Self) -> Self {
        let quotient = <R as Euclid>::div_euclid(&self[0], &rhs[0]);

        Self::from_field(quotient)
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

impl<R: Real, const N: usize> Num for Jet<R, JetReal, N> {
    type FromStrRadixErr = R::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        R::from_str_radix(str, radix).map(|x| Self::from_field(x))
    }
}

impl<R: Real, const N: usize> num_traits::real::Real for Jet<R, JetReal, N> {
    fn min_value() -> Self {
        Self::from_field(R::min_value())
    }

    fn min_positive_value() -> Self {
        Self::from_field(R::min_positive_value())
    }

    fn epsilon() -> Self {
        Self::from_field(R::epsilon())
    }

    fn max_value() -> Self {
        Self::from_field(R::max_value())
    }

    fn floor(self) -> Self {
        Self::from_field(self[0].floor())
    }

    fn ceil(self) -> Self {
        Self::from_field(self[0].ceil())
    }

    fn round(self) -> Self {
        Self::from_field(self[0].round())
    }

    fn trunc(self) -> Self {
        Self::from_field(self[0].trunc())
    }

    fn fract(self) -> Self {
        let whole = Self::from_field(self[0].trunc());
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
        Self::from_field(self[0].signum())
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
            mut base: Jet<R, JetReal, N>,
            mut exponent: u32,
        ) -> Jet<R, JetReal, N> {
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
                return Self::from_field(primal);
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
                return Self::from_field(primal);
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
