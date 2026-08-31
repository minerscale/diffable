//! Tensor constructions, tangent lifts, and forward automatic differentiation.
//!
//! The differentiation API is a small typed language. [`d`] introduces a
//! derivative, [`d::along`] contracts one derivative slot with a direction,
//! and [`d::at`] or [`Along::at`] evaluates the completed program. Programs may
//! themselves be differentiated, so `d(d(f))` is the second derivative (the
//! Hessian for a scalar-valued map) and arbitrarily deep nestings use the same
//! machinery.
//!
//! Evaluation is implemented with truncated Taylor jets. [`Jet::new`] and
//! [`Jet::constant`] are the category-restricted constructors of the jet image,
//! while [`TensorOver::new`] performs scalar re-presentation. Their concrete
//! return types remain visible to Rust so all native trait structure is preserved.
//! [`JetVector`] names the tensor presentation needed by extension traits, while
//! [`JetMap`] is the internal interpretation rule for ordinary functions and
//! differential programs. [`ConstantRoute`] records how captured constants
//! must be embedded through nested presentations. [`EvaluableAt`] is the final
//! interpreter boundary and provides the user-facing diagnostic when a program
//! cannot be evaluated.
//!
//! [`Connection`] extends the construction from vector spaces to tangent
//! bundles. [`FormLift`] and [`NondegenerateLift`] state that lowering and
//! raising maps extend coherently when coordinates are replaced by jets.

use core::{
    marker::PhantomData,
    ops::{Add, Deref, DerefMut, Div, Index, IndexMut, Mul, Neg, Rem, Sub},
};

use num_traits::{Euclid, Inv, Num, NumCast, One, ToPrimitive, Zero, real::Real as _};

use crate::{
    coords::Coords,
    impl_vector_ops,
    matrix::endomorphism_exp,
    traits::{
        Absent, ActionExists, Array, AssocName, Atomic, BindsReflected, BothSided, CField, Cat,
        Category, Chart, DivRing, Dual, Euclidean, ExactCmp, ExpMap, Field, Form, FromReal,
        Handedness, Interval, Jetted, Left, Metric, NonZero, Nondegenerate, Normalize,
        NormalizeWith, OneSided, OptionallyOption, Point, Real, Reflect, ReflectedContext,
        Rehandable, Right, Sesquilinear, Sidedness, Sinister, TangentBundle, Tensor,
        TensorNormalizer, TensorOf, TensorProductAction, Undecorated, Vector, jet, tensor_of, Ø, ː,
        ι, π, Ⱶ, 𝐃𝐢𝐟𝐟, 𝐅𝐥𝐝, 𝐅𝐨𝐫𝐦, 𝐌𝐞𝐭, 𝐑𝐞𝐚𝐥, 𝐓𝐞𝐧𝐬, 𝒯,
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

type NormalizedDirectSum<U, V> = DirectSum<
    <U as NormalizeWith<Undecorated>>::Normalized,
    <V as NormalizeWith<Undecorated>>::Normalized,
>;

type NormalizedTensorProduct<U, V> = TensorProduct<
    <U as NormalizeWith<Undecorated>>::Normalized,
    <V as NormalizeWith<Undecorated>>::Normalized,
>;

impl<U, V> TensorNormalizer<DirectSum<U, V>> for NormalizeDirectSum
where
    U: Tensor,
    V: Tensor<F = U::F, Hand = U::Hand>,
{
    type Undecorated = NormalizedDirectSum<U, V>;

    type Dualized = Dual<NormalizedDirectSum<U, V>>;

    type Sinistered
        = Sinister<Self::Undecorated>
    where
        <DirectSum<U, V> as Tensor>::Action: Rehandable;

    type DualSinistered
        = Sinister<Dual<Self::Undecorated>>
    where
        <DirectSum<U, V> as Tensor>::Action: Rehandable;

    fn undecorated(tensor: DirectSum<U, V>) -> Self::Undecorated {
        Self::Undecorated::from_fn(|i| tensor[i])
    }

    fn dualized(tensor: DirectSum<U, V>) -> Self::Dualized {
        Self::Dualized::from_fn(|i| tensor[i])
    }

    fn sinistered(tensor: DirectSum<U, V>) -> Self::Sinistered
    where
        <DirectSum<U, V> as Tensor>::Action: Rehandable,
    {
        Self::Sinistered::from_fn(|i| tensor[i])
    }

    fn dual_sinistered(tensor: DirectSum<U, V>) -> Self::DualSinistered
    where
        <DirectSum<U, V> as Tensor>::Action: Rehandable,
    {
        Self::DualSinistered::from_fn(|i| tensor[i])
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

    pub fn inverse(
        mut self,
    ) -> <TensorProduct<Dual<V>, Dual<U>> as NormalizeWith<Undecorated>>::Normalized
    where
        V::Action: TensorProductAction<U::Action>,
    {
        const { assert!(U::N == V::N, "cannot invert a non-square tensor product") }

        let n = U::N;

        let mut inverse = TensorProduct::<Dual<V>, Dual<U>>::from_fn_ij(|i, j| {
            if i == j { V::F::one() } else { V::F::zero() }
        });

        for column in 0..n {
            // Any nonzero pivot suffices algebraically; no metric/order required.
            let pivot_row = (column..n)
                .find(|&row| !self[row * n + column].is_zero())
                .expect("tensor product is singular during Gauss-Jordan elimination");

            if pivot_row != column {
                for j in 0..n {
                    let a = column * n + j;
                    let b = pivot_row * n + j;

                    let tmp = self[a];
                    self[a] = self[b];
                    self[b] = tmp;

                    let tmp = inverse[a];
                    inverse[a] = inverse[b];
                    inverse[b] = tmp;
                }
            }

            let pivot = self[column * n + column];

            let pivot_inv = <V::F as DivRing>::Mul::inv(
                NonZero::new(pivot)
                    .expect("pivot selected as nonzero")
                    .into(),
            )
            .into()
            .0;

            // Left-multiply the pivot row by pivot^{-1}.
            for j in 0..n {
                let index = column * n + j;

                self[index] = pivot_inv * self[index];
                inverse[index] = pivot_inv * inverse[index];
            }

            // Eliminate this column from every other row.
            for row in 0..n {
                if row == column {
                    continue;
                }

                let factor = self[row * n + column];

                if factor.is_zero() {
                    continue;
                }

                for j in 0..n {
                    let index = row * n + j;
                    let pivot_index = column * n + j;

                    self[index] = self[index] - factor * self[pivot_index];

                    inverse[index] = inverse[index] - factor * inverse[pivot_index];
                }
            }
        }

        inverse.normalize()
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

impl<U, V> TensorNormalizer<TensorProduct<U, V>> for NormalizeTensorProduct
where
    U: Tensor<Hand = Right, Action: TensorProductAction<V::Action>>,
    V: Tensor<F = U::F, Hand = Left, Action: ActionExists>,
{
    type Undecorated = NormalizedTensorProduct<U, V>;

    type Dualized = Dual<NormalizedTensorProduct<U, V>>;

    type Sinistered
        = Sinister<Self::Undecorated>
    where
        <TensorProduct<U, V> as Tensor>::Action: Rehandable;

    type DualSinistered
        = Sinister<Dual<Self::Undecorated>>
    where
        <TensorProduct<U, V> as Tensor>::Action: Rehandable;

    fn undecorated(tensor: TensorProduct<U, V>) -> Self::Undecorated {
        Self::Undecorated::from_fn(|i| tensor[i])
    }

    fn dualized(tensor: TensorProduct<U, V>) -> Self::Dualized {
        Self::Dualized::from_fn(|i| tensor[i])
    }

    fn sinistered(tensor: TensorProduct<U, V>) -> Self::Sinistered
    where
        <TensorProduct<U, V> as Tensor>::Action: Rehandable,
    {
        Self::Sinistered::from_fn(|i| tensor[i])
    }

    fn dual_sinistered(tensor: TensorProduct<U, V>) -> Self::DualSinistered
    where
        <TensorProduct<U, V> as Tensor>::Action: Rehandable,
    {
        Self::DualSinistered::from_fn(|i| tensor[i])
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

impl<T, P> SwapKernel<ThroughSinister<P>> for Sinister<T>
where
    T: Tensor<Action: Rehandable> + SwapKernel<P>,
    <T as SwapKernel<P>>::Swapped: Tensor<F = T::F, Hand = T::Hand, Action = T::Action>,
{
    type Swapped = Sinister<<T as SwapKernel<P>>::Swapped>;

    fn source_index(output: usize) -> usize {
        <T as SwapKernel<P>>::source_index(output)
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
    Fiber: TangentBundle<FP, FT> = FP,
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
pub struct TensorOver<V: Tensor, S: Point>(V::Array<S>, PhantomData<fn() -> V>);

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
    fn truncate<const M: usize>(self) -> JetVectorIn<𝒞, V, M>
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
    fn truncate<const M: usize>(self) -> Jet<𝒞, F, M> {
        const { assert!(M <= N) };

        Jet::<𝒞, F, M>::from_fn(|i| self[i])
    }

    /// Constructs all `N + 1` coefficients by index, beginning with the primal
    /// coefficient at index zero.
    fn from_fn(f: impl FnMut(usize) -> F) -> Self {
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

/// A point together with a jet-valued tangent coordinate and its tower tag.
///
/// `Tower` distinguishes iterated tangent constructions that have identical
/// runtime representations. Use [`TangentElement::new`] rather than spelling
/// the marker explicitly.
#[derive(Debug, Clone)]
pub struct TangentElement<P: Point, V: Tensor, Tower, const N: usize = 1>(
    pub P,
    pub JetVector<V, N>,
    PhantomData<Tower>,
);

impl<V: Tensor, S: Point + PartialEq> PartialEq for TensorOver<V, S> {
    fn eq(&self, other: &Self) -> bool {
        self.0.iter().eq(other.0.iter())
    }
}

impl<P: Point + PartialEq, V: Tensor, Tower, const N: usize> PartialEq
    for TangentElement<P, V, Tower, N>
{
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}

impl<P: Point, V: Tensor, Tower, const N: usize> TangentElement<P, V, Tower, N> {
    /// Constructs a tangent element from its base point and local jet.
    pub fn new(p: P, v: JetVector<V, N>) -> Self {
        Self(p, v, PhantomData)
    }

    /// Returns a clone of the base point.
    pub fn base_point(&self) -> P {
        self.0.clone()
    }

    /// Borrows the jet-valued tangent coordinate.
    pub fn jet(&self) -> &JetVector<V, N> {
        &self.1
    }
}

impl<P: Point, V: Tensor, const N: usize> Tangent<P, V, N> {
    pub fn into_jet(self, point_to_coordinate: impl FnOnce(P) -> V) -> JetVector<V, N> {
        let coordinate = point_to_coordinate(self.0);

        JetVector::from_fn(|i| {
            let mut jet = self.1[i];
            jet[0] = coordinate[i];
            jet
        })
    }
}

type Prolongation<P, V, T, const N: usize = 1> = TangentElement<P, V, ː<T, Ø>, N>;

/// A first [`TangentElement`] at a point of `P`, expressed in `V` coordinates.
pub type Tangent<P, V, const N: usize = 1> = TangentElement<P, V, Ø, N>;
/// An iterated [`TangentElement`] with explicit [`TangentBundle`] witnesses.
pub type TM<P, V, T, U, const N: usize = 1> = TangentElement<P, V, ː<T, ː<U, Ø>>, N>;
/// The tangent bundle of `T`, represented by the canonical jet prolongation.
///
/// This is the concrete iterated-tangent representation constructed by
/// [`Connection`].
pub type LiftedTM<P, V, T, const N: usize = 1> = TM<P, V, T, Prolongation<P, V, T, N>, N>;

impl<
    P: Point,
    V: Tensor,
    T: TangentBundle<P, V>,
    U: TangentBundle<Self, JetVector<V, N>>,
    const N: usize,
> Chart<P, V> for TM<P, V, T, U, N>
{
    type Global = T::Global;

    fn to_local(&self, point: &P) -> Option<V> {
        T::chart_at(&self.0).to_local(point)
    }

    fn to_global(&self, coord: V) -> Self::Global {
        T::chart_at(&self.0).to_global(coord)
    }

    fn chart_at(p: &P) -> Self {
        Self(p.clone(), JetVectorIn::zero(), PhantomData)
    }
}

impl<
    P: Point,
    V: Tensor,
    T: TangentBundle<P, V>,
    U: TangentBundle<Self, JetVector<V, N>>,
    const N: usize,
> ExpMap<P, V> for TM<P, V, T, U, N>
{
}
impl<
    P: Point,
    V: Tensor,
    T: TangentBundle<P, V>,
    U: TangentBundle<Self, JetVector<V, N>>,
    const N: usize,
> TangentBundle<P, V> for TM<P, V, T, U, N>
{
}

impl<
    P: Point,
    V: Tensor,
    T: TangentBundle<P, V>,
    U: TangentBundle<Self, JetVector<V, N>>,
    const N: usize,
> Chart<Self, JetVector<V, N>> for TM<P, V, T, U, N>
{
    type Global = U::Global;

    fn to_local(&self, point: &Self) -> Option<JetVector<V, N>> {
        U::chart_at(self).to_local(point)
    }

    fn to_global(&self, coord: JetVector<V, N>) -> Self::Global {
        U::chart_at(self).to_global(coord)
    }

    fn chart_at(p: &Self) -> Self {
        p.clone()
    }
}
impl<
    P: Point,
    V: Tensor,
    T: TangentBundle<P, V>,
    U: TangentBundle<Self, JetVector<V, N>>,
    const N: usize,
> ExpMap<Self, JetVector<V, N>> for TM<P, V, T, U, N>
{
}

impl<
    P: Point,
    V: Tensor,
    T: TangentBundle<P, V>,
    U: TangentBundle<Self, JetVector<V, N>>,
    const N: usize,
> TangentBundle<Self, JetVector<V, N>> for TM<P, V, T, U, N>
{
}

impl<P: Point, V: Tensor, T: Connection<P, V>, U: TangentBundle<Self, JetVector<V>>>
    Connection<P, V> for TM<P, V, T, U>
{
    fn tangent_to_local<const M: usize>(
        base: Tangent<P, V, M>,
        local: Tangent<P, V, M>,
    ) -> Option<JetVector<V, M>> {
        T::tangent_to_local(base, local)
    }

    fn tangent_to_global<const N: usize>(
        base: Tangent<P, V, N>,
        coordinate: JetVector<V, N>,
    ) -> <Self::Global as OptionallyOption<P>>::Mapped<(P, JetVector<V, N>)> {
        T::tangent_to_global(base, coordinate)
    }
}

impl<P: Point, V: Tensor, T: Connection<P, V>, const N: usize>
    Connection<LiftedTM<P, V, T, N>, JetVector<V, N>> for Prolongation<P, V, T, N>
{
    fn tangent_to_local<const M: usize>(
        base: Tangent<LiftedTM<P, V, T, N>, JetVector<V, N>, M>,
        local: Tangent<LiftedTM<P, V, T, N>, JetVector<V, N>, M>,
    ) -> Option<JetVector<JetVector<V, N>, M>> {
        let point = T::tangent_to_local::<N>(
            TangentElement::new(base.0.0.clone(), base.0.1.clone()),
            TangentElement::new(local.0.0.clone(), local.0.1.clone()),
        )?;

        Some(JetVectorIn::from_fn(|i| {
            local.1[i] - base.1[i] + Jet::from_parts(point[i], [Jet::<𝐅𝐥𝐝::𝒞, V::F, N>::zero(); M])
        }))
    }

    fn tangent_to_global<const M: usize>(
        base: Tangent<LiftedTM<P, V, T, N>, JetVector<V, N>, M>,
        coordinate: JetVector<JetVector<V, N>, M>,
    ) -> <Self::Global as OptionallyOption<LiftedTM<P, V, T, N>>>::Mapped<(
        LiftedTM<P, V, T, N>,
        JetVector<JetVector<V, N>, M>,
    )> {
        let combined =
            JetVectorIn::<𝐅𝐥𝐝::𝒞, JetVector<V, N>, M>::from_fn(
                |i| base.1[i] + coordinate[i],
            );

        // The outer constant coefficient is an ordinary coordinate in the
        // Prolongation chart on LiftedTM.
        let point_coordinate = JetVectorIn::<𝐅𝐥𝐝::𝒞, V, N>::from_fn(|i| combined[i][0]);

        T::tangent_to_global::<N>(
            TangentElement::new(base.0.0.clone(), base.0.1.clone()),
            point_coordinate,
        )
        .cast_option(|(point, jet)| {
            let point = TangentElement::new(point, jet);

            // Everything above outer order zero is the tangent part.
            let tangent = JetVectorIn::from_fn(|i| {
                let mut value = combined[i];
                value[0] = Jet::<𝐅𝐥𝐝::𝒞, V::F, N>::zero();
                value
            });

            (point, tangent)
        })
    }
}

pub type Christoffel<V> = TensorProduct<TensorProduct<V, Dual<V>>, Dual<V>>;

/// Extends a tangent-bundle chart to jet-valued tangent coordinates.
///
/// Implementing this trait is the admission point for differentiating through
/// a manifold. It states how a tangent element is expressed in the local chart
/// at another tangent element, and how that local jet is returned to the global
/// bundle. [`Tangent`] is the first lifted element; [`TM`] and [`LiftedTM`]
/// describe its iterated tangent bundles. Vector spaces receive the canonical
/// translation-based implementation.
pub trait Connection<P: Point, V: Tensor>: TangentBundle<P, V> {
    /// Expresses `local` in the lifted chart centred at `base`.
    fn tangent_to_local<const N: usize>(
        base: Tangent<P, V, N>,
        local: Tangent<P, V, N>,
    ) -> Option<JetVector<V, N>>;
    /// Reconstructs a global point and tangent jet from a lifted coordinate.
    fn tangent_to_global<const N: usize>(
        base: Tangent<P, V, N>,
        coordinate: JetVector<V, N>,
    ) -> <Self::Global as OptionallyOption<P>>::Mapped<(P, JetVector<V, N>)>;

    fn christoffel_symbols(&self, p: P) -> Option<Christoffel<V>>
    where
        V: Vector<Hand = Right, Action = BothSided>,
    {
        let origin = TangentElement::new(
            LiftedTM::<P, V, Self, 1>::new(p, Zero::zero()),
            Zero::zero(),
        );
        let observer = TangentElement::new(
            LiftedTM::<P, V, Self, 1>::new(self.base_point(), Zero::zero()),
            Zero::zero(),
        );

        let success = core::cell::Cell::new(true);

        let transition = |v: JetVector<V, 1, Jet<𝐅𝐥𝐝::𝒞, <V as Tensor>::F, 1>>| -> _ {
            match Prolongation::<P, V, Self, 1>::tangent_to_global::<1>(
                origin.clone(),
                TensorOver(v.0, PhantomData),
            )
            .into_option()
            {
                Some((point, tangent)) => {
                    match Prolongation::<P, V, Self, 1>::tangent_to_local::<1>(
                        observer.clone(),
                        TangentElement::new(point, tangent),
                    ) {
                        Some(local) => TensorOver(local.0, PhantomData),

                        None => {
                            success.set(false);
                            Zero::zero()
                        }
                    }
                }
                None => {
                    success.set(false);
                    Zero::zero()
                }
            }
        };

        let christoffel = -evaluate_derivative_at(&d(d(transition)), V::zero()).0;

        success.get().then_some(christoffel)
    }

    /// Returns the coordinate acceleration at `p` of the geodesic with
    /// initial tangent `v`, expressed in the fixed chart `self`.
    ///
    /// If
    /// ```text
    /// γ_v(t) = exp_p(t v)
    /// ```
    ///
    /// and `x_v(t)` is that geodesic expressed in the coordinates of
    /// `self`, this returns
    ///
    /// ```text
    ///     x_v''(0).
    /// ```
    ///
    /// The observing chart must contain `p`.
    #[cfg(feature = "testing")]
    fn geodesic_acceleration(&self, p: P, v: V) -> Option<V>
    where
        V: Vector,
    {
        // Observe everything in the fixed chart `self`.
        //
        // By the ExpMap law,
        //
        // Self::chart_at(&self.base_point()) == self,
        //
        // so the zero tangent based at `self.base_point()` selects exactly
        // this lifted chart.
        let observer =
            Tangent::<P, V, 2>::new(self.base_point(), JetVectorIn::<𝐅𝐥𝐝::𝒞, V, 2>::zero());

        // The exponential chart centred at p.
        let origin = Tangent::<P, V, 2>::new(p, JetVectorIn::<𝐅𝐥𝐝::𝒞, V, 2>::zero());

        // The 2-jet of t ↦ t v:
        //
        //     0 + v t + 0 t².
        //
        // Pushing this through the lifted exponential chart therefore
        // constructs the 2-jet of γ_v(t) = exp_p(t v).
        let radial = JetVectorIn::<𝐅𝐥𝐝::𝒞, V, 2>::from_fn(|i| {
            Jet::from_parts(V::F::zero(), [v[i], V::F::zero()])
        });

        let (point, tangent) = Self::tangent_to_global::<2>(origin, radial).into_option()?;

        let geodesic = Tangent::<P, V, 2>::new(point, tangent);

        // Re-express the geodesic in ONE FIXED chart. Using the chart
        // centred at p here would make its coordinate acceleration vanish
        // tautologically.
        let local = Self::tangent_to_local::<2>(observer, geodesic)?;

        // Jets store Taylor coefficients:
        //
        //     jet[2] = x''(0) / 2!
        //
        // so recover the actual acceleration.
        let two = V::F::from_nat(2);

        Some(V::from_fn(|i| local[i][2] * two))
    }

    /// Certifies that the geodesic spray is quadratic in tangent velocity.
    ///
    /// For a fixed chart and point p, define
    ///
    /// ```text
    /// A_p(v) = d²/dt² |₀ chart(exp_p(t v)).
    /// ```
    ///
    /// `Connection` requires:
    ///
    /// ```text
    /// A_p(u + v) + A_p(u - v) = 2 A_p(u) + 2 A_p(v)
    /// ```
    ///
    /// and
    ///
    /// ```text
    ///     A_p(a v) = a² A_p(v).
    /// ```
    ///
    /// Thus A_p is quadratic. Polarization therefore determines a unique
    /// symmetric bilinear Christoffel operation
    ///
    /// ```text
    ///     Γ_p(u, v)
    ///       = -½ (A_p(u + v) - A_p(u) - A_p(v)),
    /// ```
    ///
    /// so the lifted geodesic structure determines a torsion-free affine
    /// connection.
    #[cfg(feature = "testing")]
    fn check_quadratic_geodesic_acceleration(&self, p: P, u: V, v: V, a: V::F) -> bool
    where
        Self: Sized,
        V: Vector + PartialEq,
    {
        // The assertion is local to the observing chart. If p is not in
        // this chart, there is no coordinate acceleration here to test.
        if self.to_local(&p).is_none() {
            return true;
        }

        let Some(a_u) = self.geodesic_acceleration(p.clone(), u.clone()) else {
            return false;
        };

        let Some(a_v) = self.geodesic_acceleration(p.clone(), v.clone()) else {
            return false;
        };

        let Some(a_u_plus_v) = self.geodesic_acceleration(p.clone(), u.clone() + v.clone()) else {
            return false;
        };

        let Some(a_u_minus_v) = self.geodesic_acceleration(p.clone(), u - v.clone()) else {
            return false;
        };

        let two = V::F::from_nat(2);

        // Quadratic parallelogram identity.
        if a_u_plus_v + a_u_minus_v != (a_u + a_v.clone()) * two {
            return false;
        }

        let Some(a_av) = self.geodesic_acceleration(p, v * a) else {
            return false;
        };

        // Degree-two homogeneity.
        a_av == a_v * (a * a)
    }

    #[cfg(feature = "testing")]
    fn check_tangent_to_local_agrees_with_chart(base: P, point: P) -> bool
    where
        V: PartialEq,
        JetVector<V>: PartialEq,
    {
        let chart = Self::chart_at(&base);

        let Some(local) = chart.to_local(&point) else {
            return true;
        };

        let lifted_base = Tangent::new(base, JetVectorIn::<𝐅𝐥𝐝::𝒞, V>::zero());

        let lifted_point = Tangent::new(point, JetVectorIn::<𝐅𝐥𝐝::𝒞, V>::zero());

        let expected = constant_jet_vector(local);

        Self::tangent_to_local(lifted_base, lifted_point).is_some_and(|actual| actual == expected)
    }

    #[cfg(feature = "testing")]
    fn check_tangent_to_global_agrees_with_chart(base: P, local: V) -> bool
    where
        V: PartialEq,
        JetVector<V>: PartialEq,
    {
        use crate::traits::OptionallyOption;

        let chart = Self::chart_at(&base);

        let expected = match chart.to_global(local.clone()).into_option() {
            Some(point) => point,
            None => return true,
        };

        let lifted_base = Tangent::new(base, JetVectorIn::<𝐅𝐥𝐝::𝒞, V>::zero());

        let coordinate = constant_jet_vector(local);

        let (actual, tangent) = match Self::tangent_to_global(lifted_base, coordinate).into_option()
        {
            Some(x) => x,
            None => return true,
        };

        tangent == JetVectorIn::zero() && chart.to_local(&actual) == chart.to_local(&expected)
    }

    /// Certifies that the lifted tangent charts form a coherent tower under
    /// truncation.
    ///
    /// Given `M <= N`, truncating an order-`N` tangent coordinate before
    /// applying `tangent_to_global` must agree exactly with applying the
    /// order-`N` map first and then truncating its jet component.
    ///
    /// Coherence of `tangent_to_local` follows from `check_tangent_isomorphism`.
    #[cfg(feature = "testing")]
    fn check_truncation_coherence<const M: usize, const N: usize>(
        base: Tangent<P, V, N>,
        coordinate: JetVector<V, N>,
    ) -> bool
    where
        P: PartialEq,
        JetVector<V, M>: PartialEq,
    {
        const { assert!(M <= N) };

        let (point_n, tangent_n) =
            match Self::tangent_to_global::<N>(base.clone(), coordinate.clone()).into_option() {
                Some(x) => x,
                None => return true,
            };

        let base_m = Tangent::new(base.0, base.1.truncate::<M>());

        let coordinate_m = coordinate.truncate::<M>();

        let (point_m, tangent_m) =
            match Self::tangent_to_global::<M>(base_m, coordinate_m).into_option() {
                Some(x) => x,
                None => return true,
            };

        point_n == point_m && tangent_n.truncate::<M>() == tangent_m
    }

    /// Certifies that the lifted local and global tangent charts are mutual
    /// inverses at jet order `N`.
    ///
    /// On the domain of `tangent_to_local`,
    ///
    /// ```text
    /// tangent_to_global(base, tangent_to_local(base, point))
    ///     == point,
    /// ```
    ///
    /// while every lifted local coordinate must round-trip as
    ///
    /// ```text
    /// tangent_to_local(base, tangent_to_global(base, coordinate))
    ///     == Some(coordinate).
    /// ```
    ///
    /// Thus `tangent_to_local::<N>` and `tangent_to_global::<N>` describe a
    /// genuine lifted chart isomorphism rather than independent choices of
    /// higher-order tangent data.
    #[cfg(feature = "testing")]
    fn check_tangent_isomorphism<const N: usize>(
        base: Tangent<P, V, N>,
        local: Tangent<P, V, N>,
        coordinate: JetVector<V, N>,
    ) -> bool
    where
        V: PartialEq,
        JetVector<V, N>: PartialEq,
    {
        // First check:
        //
        //     local -> coordinate -> global == local
        //
        // `tangent_to_local` is partial, so points outside this lifted chart
        // impose no inversehood obligation.
        if let Some(local_coordinate) = Self::tangent_to_local::<N>(base.clone(), local.clone()) {
            let (point, tangent) =
                match Self::tangent_to_global::<N>(base.clone(), local_coordinate).into_option() {
                    Some(x) => x,
                    None => return true,
                };

            // Since `P` need not implement PartialEq, compare the reconstructed
            // point through the ordinary chart in which `local` was expressible.
            let chart = Self::chart_at(&base.0);

            if tangent != local.1 || chart.to_local(&point) != chart.to_local(&local.0) {
                return false;
            }
        }

        // Then check:
        //
        //     coordinate -> global -> local == coordinate
        //
        // This obligation applies when coefficient zero belongs to the branch
        // selected by the ordinary chart. `tangent_to_global` may be total even
        // when the exponential chart is not globally injective, so arbitrary
        // vectors need not be canonical local coordinates of their image.
        let coordinate_primal = V::from_fn(|index| coordinate[index][0]);

        let (point, tangent) =
            match Self::tangent_to_global::<N>(base.clone(), coordinate.clone()).into_option() {
                Some(value) => value,
                None => return true,
            };

        let chart = Self::chart_at(&base.0);

        // The generated coordinate lies outside the branch represented by this
        // chart. There is no inversehood obligation for this presentation.
        if chart.to_local(&point) != Some(coordinate_primal) {
            return true;
        }

        let reconstructed = Tangent::new(point, tangent);

        Self::tangent_to_local::<N>(base, reconstructed).is_some_and(|actual| actual == coordinate)
    }
}

/// A connection with an explicitly supplied metric tensor field.
///
/// `MetricTensor` refines [`Connection`]: it is an implementation strategy for a
/// differential structure in which the metric itself is available pointwise,
/// rather than reconstructed from parallel transport of the model-space form.
///
/// For a right-handed tangent space `V`, the covariant rank-two metric is
/// represented as `Sinister<V*> ⊗ V*`. The musical maps are consequences of
/// this tensor: lowering contracts `g_p` with a vector, while raising contracts
/// the inverse tensor with a covector.
pub trait MetricTensor<P: Point, V: Tensor<Hand = Right, Action = BothSided>>:
    Connection<P, V>
{
    /// Evaluate the supplied metric tensor in the tangent space selected by
    /// `target`, expressed in this connection's local tangent coordinates.
    fn g(&self, target: V) -> TensorProduct<Sinister<Dual<V>>, Dual<V>>;
}

fn tangent_lerp<
    P: Point,
    V: Vector<Hand = Right, Action = BothSided, F: FromReal>,
    T: Connection<P, V>,
    const N: usize,
>(
    connection: &T,
    target: V,
    t: Jet<𝐑𝐞𝐚𝐥::𝒞, <V::F as Interval>::R, N>,
) -> Tangent<P, V, N> {
    let t = Jet::<𝐅𝐥𝐝::𝒞, V::F, N>::from_parts(
        V::F::from_real(t[0]),
        core::array::from_fn(|i| V::F::from_real(t[i + 1])),
    );

    let radial = JetVectorIn::<𝐅𝐥𝐝::𝒞, V, N>::from_fn(|i| {
        Jet::<𝐅𝐥𝐝::𝒞, V::F, N>::from_parts(target[i], [Zero::zero(); N]) * t
    });

    let base = Tangent::<P, V, N>::new(connection.base_point(), JetVectorIn::zero());

    let (point, tangent) = T::tangent_to_global::<N>(base, radial)
        .into_option()
        .unwrap();

    TangentElement::new(point, tangent)
}

pub const TRANSPORT_ORDER: usize = 6;

pub struct Ordered<
    'a,
    𝒞: Cat,
    𝒟: Cat,
    P: Point,
    V: Vector<Hand = Right, Action = BothSided, F: FromReal> + ι<C: TransportRegion<𝒞>>,
    T: ParallelTransport<𝒞, 𝒟, P, V>,
    const N: usize,
> {
    connection: &'a T,
    _phantom: PhantomData<fn() -> (𝒞, 𝒟, P, V)>,
}

impl<
    'a,
    𝒞: Cat,
    𝒟: Cat,
    P: Point,
    V: Vector<Hand = Right, Action = BothSided, F: FromReal> + ι<C: TransportRegion<𝒞>>,
    T: ParallelTransport<𝒞, 𝒟, P, V>,
    const N: usize,
> Ordered<'a, 𝒞, 𝒟, P, V, T, N>
{
    pub fn transport(
        &self,
        curve: impl Fn(Jet<𝐑𝐞𝐚𝐥::𝒞, <V::F as Interval>::R, N>) -> Tangent<P, V, N>,
        from: <V::F as Interval>::R,
        to: <V::F as Interval>::R,
    ) -> TensorProduct<V, Dual<V>> {
        self.connection.transport_with::<N>(curve, from, to)
    }

    pub fn lower(&self, target: V, v: V) -> Dual<V>
    where
        V: Form,
        <T as ι>::C: MusicalRegion<𝒟, 𝒞, P, V, T>,
    {
        self.connection.lower_with::<N>(target, v)
    }

    pub fn raise(&self, target: V, v: Dual<V>) -> V
    where
        V: Nondegenerate,
        <T as ι>::C: MusicalRegion<𝒟, 𝒞, P, V, T>,
    {
        self.connection.raise_with::<N>(target, v)
    }
}

pub trait ParallelTransport<
    𝒞: Cat,
    𝒟: Cat,
    P: Point,
    V: Vector<Hand = Right, Action = BothSided, F: FromReal> + ι<C: TransportRegion<𝒞>>,
>: Connection<P, V> + ι
{
    fn transport_with<const N: usize>(
        &self,
        curve: impl Fn(Jet<𝐑𝐞𝐚𝐥::𝒞, <V::F as Interval>::R, N>) -> Tangent<P, V, N>,
        from: <V::F as Interval>::R,
        to: <V::F as Interval>::R,
    ) -> TensorProduct<V, Dual<V>>;

    fn order<'a, const N: usize>(&'a self) -> Ordered<'a, 𝒞, 𝒟, P, V, Self, N> {
        Ordered {
            connection: self,
            _phantom: PhantomData,
        }
    }

    fn transport(
        &self,
        curve: impl Fn(
            Jet<𝐑𝐞𝐚𝐥::𝒞, <V::F as Interval>::R, TRANSPORT_ORDER>,
        ) -> Tangent<P, V, TRANSPORT_ORDER>,
        from: <V::F as Interval>::R,
        to: <V::F as Interval>::R,
    ) -> TensorProduct<V, Dual<V>> {
        self.transport_with::<TRANSPORT_ORDER>(curve, from, to)
    }

    fn lower_with<const N: usize>(&self, target: V, v: V) -> Dual<V>
    where
        V: Form,
        <Self as ι>::C: MusicalRegion<𝒟, 𝒞, P, V, Self>,
    {
        <<Self as ι>::C as MusicalRegion<𝒟, 𝒞, P, V, Self>>::lower::<N>(self, target, v)
    }

    fn lower(&self, target: V, v: V) -> Dual<V>
    where
        V: Form,
        <Self as ι>::C: MusicalRegion<𝒟, 𝒞, P, V, Self>,
    {
        self.lower_with::<TRANSPORT_ORDER>(target, v)
    }

    fn raise_with<const N: usize>(&self, target: V, v: Dual<V>) -> V
    where
        V: Nondegenerate,
        <Self as ι>::C: MusicalRegion<𝒟, 𝒞, P, V, Self>,
    {
        <<Self as ι>::C as MusicalRegion<𝒟, 𝒞, P, V, Self>>::raise::<N>(self, target, v)
    }

    fn raise(&self, target: V, v: Dual<V>) -> V
    where
        V: Nondegenerate,
        <Self as ι>::C: MusicalRegion<𝒟, 𝒞, P, V, Self>,
    {
        self.raise_with::<TRANSPORT_ORDER>(target, v)
    }

    /// Certifies that parallel transport around every sufficiently small closed
    /// curve based at `target` preserves the form at that fibre.
    ///
    /// `closed_curve` is assumed by the test harness to satisfy
    ///
    /// ```text
    /// closed_curve(0) = closed_curve(1) = target
    /// ```
    ///
    /// in the corresponding tangent-bundle sense.
    ///
    /// For arbitrary `u, v ∈ T_target M`, this checks
    ///
    /// ```text
    /// g(Pγ u, Pγ v) = g(u, v),
    /// ```
    ///
    /// where `Pγ` is parallel transport around the closed curve `γ`.
    ///
    /// Since the manifold form is derived from the model-space form by parallel
    /// transport, this is precisely the path-independence condition required for
    /// that transported form to be well-defined.
    #[cfg(feature = "testing")]
    fn check_holonomy_preserves_form<const N: usize>(
        &self,
        target: V,
        closed_curve: impl Fn(Jet<𝐑𝐞𝐚𝐥::𝒞, <V::F as Interval>::R, N>) -> Tangent<P, V, N>,
        u: V,
        v: V,
    ) -> bool
    where
        V: Form + PartialEq,
        <Self as ι>::C: MusicalRegion<𝒟, 𝒞, P, V, Self>,
    {
        let base = self.order::<N>();
        let before = u.pairing(&base.lower(target.clone(), v.clone()));

        let transport = base.transport(&closed_curve, Zero::zero(), One::one());
        let u = transport.mul_v(&u);
        let v = transport.mul_v(&v);

        let after = u.pairing(&base.lower(target, v));

        before == after
    }
}

/// Closed structural region selecting the musical implementation for a connection.
///
/// The dispatch theory is inferred exactly like [`TransportRegion`]: a context
/// constructively outside `Met` selects the connection-derived implementation,
/// while a context refining `Met` selects the supplied metric tensor.
#[doc(hidden)]
pub trait MusicalRegion<
    𝒟: Cat,
    𝒞: Cat,
    P: Point,
    V: Vector<Hand = Right, Action = BothSided, F: FromReal> + ι<C: TransportRegion<𝒞>>,
    T: Connection<P, V>,
>: Category
{
    fn lower<const N: usize>(connection: &T, target: V, v: V) -> Dual<V>
    where
        V: Form;

    fn raise<const N: usize>(connection: &T, target: V, v: Dual<V>) -> V
    where
        V: Nondegenerate;
}

// Generic connection region: there is constructively no supplied metric tensor,
// so reconstruct the musical maps from parallel transport of the model-space form.
impl<
    C: Category,
    𝒞: Cat,
    P: Point,
    V: Vector<Hand = Right, Action = BothSided, F: FromReal> + ι<C: TransportRegion<𝒞>>,
    T: Connection<P, V>,
> MusicalRegion<𝐃𝐢𝐟𝐟::𝒞, 𝒞, P, V, T> for C
where
    C: Ⱶ<𝐌𝐞𝐭::𝒞, Absent>,
{
    fn lower<const N: usize>(connection: &T, target: V, v: V) -> Dual<V>
    where
        V: Form,
    {
        const {
            assert!(N > 0, "lowering requires a positive Taylor order");
        }

        let zero = <V::F as Interval>::R::zero();
        let one = <V::F as Interval>::R::one();
        let curve =
            |t: Jet<𝐑𝐞𝐚𝐥::𝒞, <V::F as Interval>::R, N>| tangent_lerp(connection, target.clone(), t);
        let transport = <V as ι>::C::parallel_transport(connection, &curve, one, zero);
        let v = transport.mul_v(&v);

        Dual::<V>::from_fn(|i| {
            let basis = V::from_fn(|j| if i == j { V::F::one() } else { V::F::zero() });
            transport.mul_v(&basis).dot(&v)
        })
    }

    fn raise<const N: usize>(connection: &T, target: V, v: Dual<V>) -> V
    where
        V: Nondegenerate,
    {
        const {
            assert!(N > 0, "raising requires a positive Taylor order");
        }

        let zero = <V::F as Interval>::R::zero();
        let one = <V::F as Interval>::R::one();
        let curve =
            |t: Jet<𝐑𝐞𝐚𝐥::𝒞, <V::F as Interval>::R, N>| tangent_lerp(connection, target.clone(), t);
        let transport = <V as ι>::C::parallel_transport(connection, &curve, zero, one);

        let v = Dual::<V>::from_fn(|i| {
            let basis = V::from_fn(|j| if i == j { V::F::one() } else { V::F::zero() });

            transport.mul_v(&basis).pairing(&v)
        });

        transport.mul_v(&V::sharp(v))
    }
}

// Metric region: use the supplied metric tensor directly in the tangent
// coordinate space selected by `target`.
impl<
    C: Category,
    𝒞: Cat,
    P: Point,
    V: Vector<Hand = Right, Action = BothSided, F: FromReal, Normalization = Atomic>
        + ι<C: TransportRegion<𝒞>>,
    T: Connection<P, V> + MetricTensor<P, V>,
> MusicalRegion<𝐌𝐞𝐭::𝒞, 𝒞, P, V, T> for C
where
    C: Ⱶ<𝐌𝐞𝐭::𝒞>,
{
    fn lower<const N: usize>(connection: &T, target: V, v: V) -> Dual<V>
    where
        V: Form,
    {
        let product = TensorProduct::pure(connection.g(target), Sinister(v));
        let reassociated = ReassociateKernel::<Right>::reassociate_kernel(product);
        let lowered: Sinister<Dual<V>> = reassociated.contract::<OnRight<ThroughSinister<Here>>>();

        Sinister(lowered).collapse()
    }

    fn raise<const N: usize>(connection: &T, target: V, v: Dual<V>) -> V
    where
        V: Nondegenerate,
    {
        let inverse: TensorProduct<V, Sinister<V>> = connection.g(target).inverse();
        let product = TensorProduct::pure(Sinister(v), Sinister(inverse));
        let reassociated = ReassociateKernel::<Left>::reassociate_kernel(product);

        reassociated.contract::<OnLeft<Here>>()
    }
}

impl<𝒞, 𝒟, P, V, T> ParallelTransport<𝒞, 𝒟, P, V> for T
where
    𝒞: Cat,
    𝒟: Cat,
    P: Point,
    V: Vector<Hand = Right, Action = BothSided, F: FromReal> + ι<C: TransportRegion<𝒞>>,
    T: Connection<P, V> + ι,
    <T as ι>::C: MusicalRegion<𝒟, 𝒞, P, V, T>,
{
    fn transport_with<const N: usize>(
        &self,
        curve: impl Fn(Jet<𝐑𝐞𝐚𝐥::𝒞, <V::F as Interval>::R, N>) -> Tangent<P, V, N>,
        from: <V::F as Interval>::R,
        to: <V::F as Interval>::R,
    ) -> TensorProduct<V, Dual<V>> {
        <V as ι>::C::parallel_transport(self, curve, from, to)
    }
}

#[doc(hidden)]
pub trait TransportRegion<𝒞: Cat>: Category {
    fn parallel_transport<
        P: Point,
        V: Vector<Hand = Right, Action = BothSided, F: FromReal>,
        T: Connection<P, V>,
        F: Fn(Jet<𝐑𝐞𝐚𝐥::𝒞, <V::F as Interval>::R, N>) -> Tangent<P, V, N>,
        const N: usize,
    >(
        connection: &T,
        curve: F,
        from: <V::F as Interval>::R,
        to: <V::F as Interval>::R,
    ) -> TensorProduct<V, Dual<V>>;
}

impl<V: Vector<Hand = Right, Action = BothSided>> TensorProduct<V, Dual<V>> {
    pub fn mul_v(&self, v: &V) -> V {
        V::from_fn(|i| (0..V::N).fold(V::F::zero(), |sum, j| sum + self[(i, j)] * v[j]))
    }

    pub fn mul_dual_v(&self, v: &Dual<V>) -> Dual<V> {
        Dual::<V>::from_fn(|j| (0..V::N).fold(V::F::zero(), |sum, i| sum + v[i] * self[(i, j)]))
    }

    pub fn compose(&self, rhs: &Self) -> Self {
        TensorProduct::from_fn_ij(|i, j| {
            (0..V::N).fold(V::F::zero(), |sum, k| sum + self[(i, k)] * rhs[(k, j)])
        })
    }

    pub fn identity() -> Self {
        TensorProduct::from_fn_ij(|i, j| if i == j { V::F::one() } else { V::F::zero() })
    }
}

impl<V: Vector<Hand = Left, Action = BothSided>> TensorProduct<Dual<V>, V> {
    pub fn mul_v(&self, v: &V) -> V {
        V::from_fn(|i| (0..V::N).fold(V::F::zero(), |sum, j| sum + v[j] * self[(j, i)]))
    }

    pub fn mul_dual_v(&self, v: &Dual<V>) -> Dual<V> {
        Dual::<V>::from_fn(|j| (0..V::N).fold(V::F::zero(), |sum, i| sum + self[(j, i)] * v[i]))
    }

    pub fn compose(&self, rhs: &Self) -> Self {
        TensorProduct::from_fn_ij(|i, j| {
            (0..V::N).fold(V::F::zero(), |sum, k| sum + rhs[(i, k)] * self[(k, j)])
        })
    }

    pub fn identity() -> Self {
        TensorProduct::from_fn_ij(|i, j| if i == j { V::F::one() } else { V::F::zero() })
    }
}

fn transport_accurate<V, const N: usize>(
    full: &TensorProduct<V, Dual<V>>,
    half: &TensorProduct<V, Dual<V>>,
) -> bool
where
    V: Vector<Hand = Right, Action = BothSided, F: FromReal>,
{
    let epsilon = <V::F as Interval>::R::epsilon();
    let epsilon_squared = epsilon * epsilon;

    let richardson = <V::F as Interval>::R::from_nat((1usize << N) - 1);
    let richardson_squared = richardson * richardson;

    (0..V::N).all(|i| {
        (0..V::N).all(|j| {
            let error = full[(i, j)].interval_squared(&half[(i, j)]).abs();
            let magnitude = half[(i, j)].interval_squared(&V::F::zero()).abs();
            let one = <V::F as Interval>::R::one();
            let scale = if magnitude.exact_lt(one) {
                one
            } else {
                magnitude
            };
            let estimated_error_squared = error / richardson_squared;

            estimated_error_squared.exact_le(epsilon_squared * scale)
        })
    })
}

fn adaptive_parallel_transport<
    V: Vector<Hand = Right, Action = BothSided, F: FromReal>,
    F: Fn(<V::F as Interval>::R, <V::F as Interval>::R) -> Option<TensorProduct<V, Dual<V>>>,
    const N: usize,
>(
    step: F,
    from: <V::F as Interval>::R,
    to: <V::F as Interval>::R,
) -> TensorProduct<V, Dual<V>> {
    let mut t = from;
    let mut transport = TensorProduct::<V, Dual<V>>::identity();
    let mut h = to - from;

    if from.exact_eq(to) {
        return transport;
    }

    let two = <V::F as Interval>::R::one() + <V::F as Interval>::R::one();

    loop {
        let half_h = h / two;
        let midpoint = t + half_h;
        let next = midpoint + half_h;
        let full_h = next - t;

        let full = step(t, full_h);
        let half = step(t, half_h)
            .and_then(|first| step(midpoint, half_h).map(|second| second.compose(&first)));

        let Some(half) = half else {
            h = half_h;
            continue;
        };

        let Some(full) = full else {
            h = half_h;
            continue;
        };

        if !transport_accurate::<V, N>(&full, &half) {
            h = half_h;
            continue;
        }

        // The two-half-step operator is the better local approximation.  A
        // later transport acts on the left, so accumulate in path order.
        transport = half.compose(&transport);

        let progresses = (to - next).abs().exact_lt((to - t).abs());

        if !progresses {
            return transport;
        }

        t = next;

        if t.exact_eq(to) {
            return transport;
        }

        let doubled = h * two;
        let remaining = to - t;

        h = if h.is_sign_negative() {
            if remaining.exact_lt(doubled) {
                doubled
            } else {
                remaining
            }
        } else if doubled.exact_lt(remaining) {
            doubled
        } else {
            remaining
        };
    }
}

fn parallel_transport_taylor<
    P: Point,
    V: Vector<Hand = Right, Action = BothSided, F: FromReal>,
    T: Connection<P, V>,
    F: Fn(Jet<𝐑𝐞𝐚𝐥::𝒞, <V::F as Interval>::R, N>) -> Tangent<P, V, N>,
    const N: usize,
>(
    connection: &T,
    curve: F,
    from: <V::F as Interval>::R,
    to: <V::F as Interval>::R,
) -> TensorProduct<V, Dual<V>> {
    const {
        assert!(N > 0, "parallel transport requires a positive Taylor order");
    }

    let step = |t: <V::F as Interval>::R,
                h: <V::F as Interval>::R|
     -> Option<TensorProduct<V, Dual<V>>> {
        let time = Jet::<𝐑𝐞𝐚𝐥::𝒞, <V::F as Interval>::R, N>::new(
            t,
            core::array::from_fn(|i| {
                if i == 0 {
                    <V::F as Interval>::R::one()
                } else {
                    <V::F as Interval>::R::zero()
                }
            }),
        );

        let path = curve(time);
        let point = LiftedTM::<P, V, T, N>::new(path.0.clone(), path.1.clone());
        let connection = Prolongation::<P, V, T, N>::new(
            connection.base_point(),
            JetVectorIn::<𝐅𝐥𝐝::𝒞, V, N>::zero(),
        );
        let velocity = TensorOver(V::Array::from_fn(|i| path.1[i].derivative()), PhantomData);
        let christoffel = connection.christoffel_symbols(point)?;

        let a = -TensorProduct::pure(christoffel, Sinister(velocity))
            .reassociate::<Right>()
            .contract::<OnRight<ThroughSinister<Here>>>();

        let compose =
            |lhs: &TensorProduct<JetVector<V, N>, Dual<JetVector<V, N>>>,
             rhs: &TensorProduct<JetVector<V, N>, Dual<JetVector<V, N>>>| {
                TensorProduct::<JetVector<V, N>, Dual<JetVector<V, N>>>::from_fn_ij(|i, j| {
                    (0..V::N).fold(Jet::<𝐅𝐥𝐝::𝒞, V::F, N>::zero(), |sum, k| {
                        sum + lhs[(i, k)].clone() * rhs[(k, j)].clone()
                    })
                })
            };

        // Solve the fundamental equation X' = A X, X(0) = I.  This computes
        // the transport itself rather than re-solving the same linear ODE for
        // each vector to which it is later applied.
        let mut x = TensorProduct::<JetVector<V, N>, Dual<JetVector<V, N>>>::from_fn_ij(|i, j| {
            Jet::from_parts(
                if i == j { V::F::one() } else { V::F::zero() },
                core::array::from_fn(|_| V::F::zero()),
            )
        });

        for _ in 0..N {
            let derivative = compose(&a, &x);

            x = TensorProduct::<JetVector<V, N>, Dual<JetVector<V, N>>>::from_fn_ij(|i, j| {
                Jet::integrate_from(
                    if i == j { V::F::one() } else { V::F::zero() },
                    derivative[(i, j)].clone(),
                )
            });
        }

        let h = V::F::from_real(h);

        Some(TensorProduct::<V, Dual<V>>::from_fn_ij(|i, j| {
            let coefficient = &x[(i, j)];
            let mut value = coefficient[N];

            for n in (0..N).rev() {
                value = value * h + coefficient[n];
            }

            value
        }))
    };

    adaptive_parallel_transport::<V, _, N>(step, from, to)
}

impl<C> TransportRegion<𝐓𝐞𝐧𝐬::𝒞> for C
where
    C: Ⱶ<𝐓𝐞𝐧𝐬::𝒞> + Ⱶ<𝐅𝐨𝐫𝐦::𝒞, Absent>,
{
    fn parallel_transport<
        P: Point,
        V: Vector<Hand = Right, Action = BothSided, F: FromReal>,
        T: Connection<P, V>,
        F: Fn(Jet<𝐑𝐞𝐚𝐥::𝒞, <V::F as Interval>::R, N>) -> Tangent<P, V, N>,
        const N: usize,
    >(
        connection: &T,
        curve: F,
        from: <V::F as Interval>::R,
        to: <V::F as Interval>::R,
    ) -> TensorProduct<V, Dual<V>> {
        parallel_transport_taylor(connection, curve, from, to)
    }
}

// Form-bearing region: solve the Magnus equation for Ω and return exp(Ω), the
// transport operator itself.  Vector transport is only an application of this
// result and therefore never needs to repeat the connection solve.
impl<C> TransportRegion<𝐅𝐨𝐫𝐦::𝒞> for C
where
    C: Ⱶ<𝐅𝐨𝐫𝐦::𝒞>,
{
    fn parallel_transport<
        P: Point,
        V: Vector<Hand = Right, Action = BothSided, F: FromReal>,
        T: Connection<P, V>,
        F: Fn(Jet<𝐑𝐞𝐚𝐥::𝒞, <V::F as Interval>::R, N>) -> Tangent<P, V, N>,
        const N: usize,
    >(
        connection: &T,
        curve: F,
        from: <V::F as Interval>::R,
        to: <V::F as Interval>::R,
    ) -> TensorProduct<V, Dual<V>> {
        const {
            assert!(N > 0, "parallel transport requires a positive Taylor order");
        }

        let step = |t: <V::F as Interval>::R,
                    h: <V::F as Interval>::R|
         -> Option<TensorProduct<V, Dual<V>>> {
            let time = Jet::<𝐑𝐞𝐚𝐥::𝒞, <V::F as Interval>::R, N>::new(
                t,
                core::array::from_fn(|i| {
                    if i == 0 {
                        <V::F as Interval>::R::one()
                    } else {
                        <V::F as Interval>::R::zero()
                    }
                }),
            );

            let path = curve(time);
            let point = LiftedTM::<P, V, T, N>::new(path.0.clone(), path.1.clone());
            let connection = Prolongation::<P, V, T, N>::new(
                connection.base_point(),
                JetVectorIn::<𝐅𝐥𝐝::𝒞, V, N>::zero(),
            );
            let velocity = TensorOver(V::Array::from_fn(|i| path.1[i].derivative()), PhantomData);
            let christoffel = connection.christoffel_symbols(point)?;

            let a = -TensorProduct::pure(christoffel, Sinister(velocity))
                .reassociate::<Right>()
                .contract::<OnRight<ThroughSinister<Here>>>();

            let compose =
                |lhs: &TensorProduct<JetVector<V, N>, Dual<JetVector<V, N>>>,
                 rhs: &TensorProduct<JetVector<V, N>, Dual<JetVector<V, N>>>| {
                    TensorProduct::<JetVector<V, N>, Dual<JetVector<V, N>>>::from_fn_ij(|i, j| {
                        (0..V::N).fold(Jet::<𝐅𝐥𝐝::𝒞, V::F, N>::zero(), |sum, k| {
                            sum + lhs[(i, k)].clone() * rhs[(k, j)].clone()
                        })
                    })
                };

            let mut bernoulli = [<V::F as Interval>::R::zero(); N];
            bernoulli[0] = <V::F as Interval>::R::one();

            for n in 1..N {
                let mut sum = <V::F as Interval>::R::zero();
                let mut factorial = <V::F as Interval>::R::one();

                for k in 1..=n {
                    factorial =
                        factorial * <<V::F as Interval>::R as NumCast>::from(k + 1).unwrap();
                    sum = sum + bernoulli[n - k] / factorial;
                }

                bernoulli[n] = -sum;
            }

            let mut omega =
                TensorProduct::<JetVector<V, N>, Dual<JetVector<V, N>>>::from_fn_ij(|_, _| {
                    Jet::from_parts(V::F::zero(), core::array::from_fn(|_| V::F::zero()))
                });

            for _ in 0..N {
                let mut rhs = a.clone();
                let mut ad = a.clone();

                for k in 1..N {
                    ad = compose(&omega, &ad) - compose(&ad, &omega);

                    let coefficient = Jet::<𝐅𝐥𝐝::𝒞, V::F, N>::from_parts(
                        V::F::from_real(bernoulli[k]),
                        core::array::from_fn(|_| V::F::zero()),
                    );

                    rhs = rhs + ad.clone() * coefficient;
                }

                omega =
                    TensorProduct::<JetVector<V, N>, Dual<JetVector<V, N>>>::from_fn_ij(|i, j| {
                        Jet::integrate_from(V::F::zero(), rhs[(i, j)].clone())
                    });
            }

            let h = V::F::from_real(h);
            let omega = TensorProduct::<V, Dual<V>>::from_fn_ij(|i, j| {
                let coefficient = &omega[(i, j)];
                let mut value = coefficient[N];

                for n in (0..N).rev() {
                    value = value * h + coefficient[n];
                }

                value
            });

            Some(endomorphism_exp(omega))
        };

        adaptive_parallel_transport::<V, _, N>(step, from, to)
    }
}

#[cfg(feature = "testing")]
fn constant_jet_vector<V: Tensor, const N: usize>(v: V) -> JetVector<V, N> {
    JetVectorIn::from_fn(|i| Jet::from_parts(v[i], [V::F::zero(); N]))
}

impl<P: Point, V: Tensor, T: Connection<P, V>, const N: usize>
    Chart<LiftedTM<P, V, T, N>, JetVector<V, N>> for Prolongation<P, V, T, N>
{
    type Global = <T::Global as OptionallyOption<P>>::Mapped<LiftedTM<P, V, T, N>>;

    fn to_local(&self, point: &LiftedTM<P, V, T, N>) -> Option<JetVector<V, N>> {
        T::tangent_to_local(
            TangentElement::new(self.0.clone(), self.1.clone()),
            TangentElement::new(point.0.clone(), point.1.clone()),
        )
    }

    fn to_global(&self, coordinate: JetVector<V, N>) -> Self::Global {
        T::tangent_to_global(
            TangentElement::new(self.0.clone(), self.1.clone()),
            coordinate,
        )
        .cast_option(|(base, jet)| TangentElement::new(base, jet))
    }

    fn chart_at(point: &LiftedTM<P, V, T, N>) -> Self {
        TangentElement::new(point.0.clone(), point.1.clone())
    }
}

impl<P: Point, V: Tensor, T: Connection<P, V>, const N: usize>
    ExpMap<LiftedTM<P, V, T, N>, JetVector<V, N>> for Prolongation<P, V, T, N>
{
}
impl<P: Point, V: Tensor, T: Connection<P, V>, const N: usize>
    TangentBundle<LiftedTM<P, V, T, N>, JetVector<V, N>> for Prolongation<P, V, T, N>
{
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
    pub fn at<𝒞: Cat, Point: DifferentialRegion<𝒞, F, Output, Route>, Output, Route>(
        &self,
        point: Point,
    ) -> Output
    where
        Self: EvaluableAt<𝒞, Point, Output, Route>,
    {
        <Self as EvaluableAt<𝒞, Point, Output, Route>>::evaluate_at(self, point)
    }

    pub fn along<V>(self, direction: V) -> Along<F, V> {
        Along {
            f: self.0,
            direction,
        }
    }
}

impl<F, V> Along<F, V> {
    /// Evaluates the directional derivative at `point`.
    pub fn at<𝒞: Cat, Point, Output, Route>(&self, point: Point) -> Output
    where
        Self: EvaluableAt<𝒞, Point, Output, Route>,
    {
        <Self as EvaluableAt<𝒞, Point, Output, Route>>::evaluate_at(self, point)
    }
}

impl<𝒞, F, P, V, Q, W, JP, JQ> EvaluableAt<𝒞, P, W, ManifoldRoute<Q, JP, JQ>> for Along<F, V>
where
    𝒞: Cat,
    P: Connection<P, V> + ι,
    P::C: Ⱶ<𝐓𝐞𝐧𝐬::𝒞, Absent>,
    V: Tensor<F: ι<C: JetRegion<𝒞>>, Hand = Right, Action: ActionExists>,
    Q: Connection<Q, W>,
    W: Tensor<F = V::F, Hand = Right, Action: TensorProductAction<V::Action>>,
    JP: Point,
    JQ: Point,
    F: ManifoldJetMap<P, V, Q, W, 1, ManifoldRoute<Q, JP, JQ>>,
    Jet<𝒞, V::F>: Field,
{
    fn evaluate_at(&self, point: P) -> W {
        let tangent = JetVector::from_fn(|coordinate| {
            Jet::from_parts(V::F::zero(), [self.direction[coordinate]])
        });

        let output = <F as ManifoldJetMap<P, V, Q, W, 1, ManifoldRoute<Q, JP, JQ>>>::jet_at(
            &self.f,
            Tangent::new(point, tangent),
        );

        W::from_fn(|coordinate| output.1[coordinate][1])
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
pub trait EvaluableAt<𝒞: Cat, Point, Output, Route = Ø> {
    fn evaluate_at(&self, point: Point) -> Output;
}

fn evaluate_derivative_at<𝒞, F, BT, FT>(
    derivative: &d<F>,
    point: BT,
) -> TangentMap<BT, FT, FT, FT>
where
    𝒞: Cat,
    F: JetMap<𝒞, BT, FT, 1, BT::F>,
    BT: Tensor<Hand = Right, Action: ActionExists>,
    FT: Tensor<F = BT::F, Hand = Right, Action: TensorProductAction<BT::Action>>,
    Jet<𝒞, BT::F>: Field,
{
    let columns: BT::Array<FT> = BT::Array::from_fn(|input_coordinate| {
        let input = JetVectorIn::<𝒞, BT>::from_fn(|coordinate| {
            Jet::from_parts(
                point[coordinate],
                [if input_coordinate == coordinate {
                    BT::F::one()
                } else {
                    BT::F::zero()
                }],
            )
        });

        let output = <F as JetMap<𝒞, BT, FT, 1, BT::F, Ø>>::jet_at(&derivative.0, input);

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

impl<
    𝒞: Cat,
    F: JetMap<𝒞, BT, FT, 1, BT::F>,
    BT: Tensor<F: ι<C: JetRegion<𝒞>>, Hand = Right, Action: ActionExists>,
    FT: Tensor<F = BT::F, Hand = Right, Action: TensorProductAction<BT::Action>>,
> EvaluableAt<𝒞, BT, TangentMap<BT, FT, FT, FT>> for d<F>
where
    Jet<𝒞, BT::F>: Field,
{
    fn evaluate_at(&self, point: BT) -> TangentMap<BT, FT, FT, FT> {
        evaluate_derivative_at::<𝒞, _, _, _>(self, point)
    }
}

impl<
    𝒞: Cat,
    F: JetMap<𝒞, BT, FT, 1, BT::F>,
    BT: Vector<F: ι<C: JetRegion<𝒞>>>,
    FT: Tensor<F = BT::F, Hand = Right, Action: TensorProductAction<BT::Action>>,
> EvaluableAt<𝒞, BT, FT> for Along<F, BT>
where
    Jet<𝒞, BT::F>: Field,
{
    fn evaluate_at(&self, point: BT) -> FT {
        let input = JetVectorIn::<𝒞, BT, 1, BT::F>::from_fn(|coordinate| {
            Jet::new(point[coordinate], [self.direction[coordinate]])
        });

        let output: JetVectorIn<𝒞, FT, 1, BT::F> =
            <F as JetMap<𝒞, BT, FT, 1, BT::F, Ø>>::jet_at(&self.f, input);

        FT::from_fn(|coordinate| output[coordinate][1])
    }
}

/// A map that can be evaluated through a selected categorical jet presentation.
///
/// Ordinary generic Rust functions implement this trait through the blanket
/// `Fn(JetVector<𝒞, BT, ..>)` implementation. Differential programs implement it
/// recursively, adding jet layers while `Route` remembers how to inject
/// captured base-field constants into the current scalar type.
pub trait JetMap<𝒞: Cat, BT: Tensor, FT: Tensor<F = BT::F>, const N: usize, S: Field, Route = Ø> {
    /// Evaluates the map without discarding any jet coefficients.
    fn jet_at(&self, input: JetVectorIn<𝒞, BT, N, S>) -> JetVectorIn<𝒞, FT, N, S>;
}

impl<
    𝒞: Cat,
    F: Fn(JetVectorIn<𝒞, BT, N, S>) -> JetVectorIn<𝒞, FT, N, S>,
    BT: Tensor,
    FT: Tensor<F = BT::F>,
    const N: usize,
    S: Field,
    Route,
> JetMap<𝒞, BT, FT, N, S, Route> for F
{
    fn jet_at(&self, input: JetVectorIn<𝒞, BT, N, S>) -> JetVectorIn<𝒞, FT, N, S> {
        self(input)
    }
}

fn evaluate_manifold_derivative_at<F, P, V, Q, W, Route>(
    derivative: &d<F>,
    point: P,
) -> TangentMap<V, Q, W, Q>
where
    P: Connection<P, V>,
    V: Tensor<Hand = Right, Action: ActionExists>,
    Q: Connection<Q, W>,
    W: Tensor<F = V::F, Hand = Right, Action: TensorProductAction<V::Action>>,
    F: ManifoldJetMap<P, V, Q, W, 1, Route>,
{
    let columns: V::Array<W> = V::Array::from_fn(|input_coordinate| {
        let tangent = JetVector::from_fn(|coordinate| {
            Jet::from_parts(
                V::F::zero(),
                [if input_coordinate == coordinate {
                    V::F::one()
                } else {
                    V::F::zero()
                }],
            )
        });

        let output = <F as ManifoldJetMap<P, V, Q, W, 1, Route>>::jet_at(
            &derivative.0,
            Tangent::new(point.clone(), tangent),
        );

        W::from_fn(|output_coordinate| output.1[output_coordinate][1])
    });

    let rows: W::Array<<Dual<V> as Tensor>::Array<V::F>> = W::Array::from_fn(|output_coordinate| {
        <Dual<V> as Tensor>::Array::from_fn(|input_coordinate| {
            columns[input_coordinate][output_coordinate]
        })
    });

    TangentMap::new(TensorProduct(TensorProductArray(rows, PhantomData)))
}

/// A manifold-valued map evaluated by commuting intrinsic tangent jets through
/// the source and target Rust representations.
pub trait ManifoldJetMap<P: Point, V: Tensor, Q: Point, W: Tensor<F = V::F>, const N: usize, Route>
{
    fn jet_at(&self, input: Tangent<P, V, N>) -> Tangent<Q, W, N>;
}

/// Selects the differential evaluator from the canonical category of the
/// point supplied to `d::at`.
#[doc(hidden)]
pub trait DifferentialRegion<𝒞: Cat, F, Output, Route>: Point {}

impl<𝒞, F, BT, FT> DifferentialRegion<𝒞, F, TangentMap<BT, FT, FT, FT>, Ø> for BT
where
    𝒞: Cat,
    F: JetMap<𝒞, BT, FT, 1, BT::F>,
    BT: Tensor<F: ι<C: JetRegion<𝒞>>, Hand = Right, Action: ActionExists>,
    FT: Tensor<F = BT::F, Hand = Right, Action: TensorProductAction<BT::Action>>,
    Jet<𝒞, BT::F>: Field,
{
}

impl<𝒞, F, P, V, Q, W, JP, JQ>
    DifferentialRegion<𝒞, F, TangentMap<V, Q, W, Q>, ManifoldRoute<Q, JP, JQ>> for P
where
    𝒞: Cat,
    P: Point + ι + Connection<P, V>,
    P::C: Ⱶ<𝐓𝐞𝐧𝐬::𝒞, Absent>,
    V: Tensor<F: ι<C: JetRegion<𝒞>>, Hand = Right, Action: ActionExists>,
    Q: Connection<Q, W>,
    W: Tensor<F = V::F, Hand = Right, Action: TensorProductAction<V::Action>>,
    JP: Point,
    JQ: Point,
    F: ManifoldJetMap<P, V, Q, W, 1, ManifoldRoute<Q, JP, JQ>>,
{
}

#[doc(hidden)]
pub struct ManifoldRoute<Q: Point, JP: Point, JQ: Point>(PhantomData<fn(JP) -> (Q, JQ)>);

#[doc(hidden)]
pub struct DifferentiatedManifoldRoute<
    𝒞: Cat,
    JP: Point,
    JV: Tensor,
    JQ: Point,
    JW: Tensor,
    InnerRoute,
>(PhantomData<fn(𝒞, JP, JV, InnerRoute) -> (JQ, JW)>);

impl<𝒞, F, P, V, Q, W, JP, JV, JQ, JW, InnerRoute, const N: usize>
    ManifoldJetMap<
        P,
        V,
        TangentMap<V, Q, W>,
        TangentMap<V, Q, W>,
        N,
        DifferentiatedManifoldRoute<𝒞, JP, JV, JQ, JW, InnerRoute>,
    > for d<F>
where
    𝒞: Cat,
    P: Connection<P, V>,
    V: Tensor<Hand = Right, Action: ActionExists>,
    Q: Connection<Q, W>,
    W: Tensor<F = V::F, Hand = Right, Action: TensorProductAction<V::Action>>,
    JP: Point + Connection<JP, JV> + CommutesJet<P, V, N>,
    JV: Tensor<F = Jet<𝒞, V::F, N>, Hand = Right, Action: ActionExists>,
    JQ: Point + Connection<JQ, JW> + CommutesJet<Q, W, N>,
    JW: Tensor<F = Jet<𝒞, V::F, N>, Hand = Right, Action: TensorProductAction<JV::Action>>,
    F: ManifoldJetMap<JP, JV, JQ, JW, 1, InnerRoute>,
    Jet<𝒞, V::F, N>: Field,
    Jet<𝐅𝐥𝐝::𝒞, V::F, N>: Field,
    TangentMap<V, Q, W>: Tensor<F = V::F, Hand = Right>,
{
    fn jet_at(
        &self,
        input: Tangent<P, V, N>,
    ) -> Tangent<TangentMap<V, Q, W>, TangentMap<V, Q, W>, N> {
        const {
            assert!(JV::N == V::N);
            assert!(JW::N == W::N);
        }

        let outer_point = <JP as CommutesJet<P, V, N>>::commute_jet(input);

        let columns: V::Array<JetVectorIn<𝒞, W, N>> = V::Array::from_fn(|input_coordinate| {
            let inner_tangent: JetVector<JV, 1> = TensorOver::from_fn(|coordinate| {
                Jet::<𝐅𝐥𝐝::𝒞, Jet<𝒞, V::F, N>, 1>::from_parts(
                    Jet::<𝒞, V::F, N>::zero(),
                    [if coordinate == input_coordinate {
                        Jet::<𝒞, V::F, N>::one()
                    } else {
                        Jet::<𝒞, V::F, N>::zero()
                    }],
                )
            });

            let inner_input: Tangent<JP, JV, 1> =
                Tangent::<JP, JV, 1>::new(outer_point.clone(), inner_tangent);

            let output =
                <F as ManifoldJetMap<JP, JV, JQ, JW, 1, InnerRoute>>::jet_at(&self.0, inner_input);

            JetVectorIn::<𝒞, W, N>::from_fn(|output_coordinate| {
                output.1[output_coordinate][1].clone()
            })
        });

        let derivative = JetVectorIn::<𝒞, TangentMap<V, Q, W>, N>::from_fn(|index| {
            let output_coordinate = index / V::N;
            let input_coordinate = index % V::N;

            columns[input_coordinate][output_coordinate].clone()
        });

        derivative.retag::<𝐅𝐥𝐝::𝒞>().into_tangent(|value| value)
    }
}

impl<𝒞, F, P, V, Q, W, JP, JV, JQ, JW, InnerRoute>
    DifferentialRegion<
        𝒞,
        d<F>,
        TangentMap<V, TangentMap<V, Q, W>, TangentMap<V, Q, W>>,
        DifferentiatedManifoldRoute<𝒞, JP, JV, JQ, JW, InnerRoute>,
    > for P
where
    𝒞: Cat,
    P: Point + ι + Connection<P, V>,
    P::C: Ⱶ<𝐓𝐞𝐧𝐬::𝒞, Absent>,
    V: Tensor<Hand = Right, Action: ActionExists>,
    Q: Connection<Q, W>,
    W: Tensor<F = V::F, Hand = Right, Action: TensorProductAction<V::Action>>,
    JP: Point + Connection<JP, JV> + CommutesJet<P, V, 1>,
    JV: Tensor<F = Jet<𝒞, V::F, 1>, Hand = Right, Action: ActionExists>,
    JQ: Point + Connection<JQ, JW> + CommutesJet<Q, W, 1>,
    JW: Tensor<F = Jet<𝒞, V::F, 1>, Hand = Right, Action: TensorProductAction<JV::Action>>,
    F: ManifoldJetMap<JP, JV, JQ, JW, 1, InnerRoute>,
    Jet<𝒞, V::F, 1>: Field,
    Jet<𝐅𝐥𝐝::𝒞, V::F, 1>: Field,
    TangentMap<V, Q, W>: Tensor<F = V::F, Hand = Right, Action: TensorProductAction<V::Action>>,
    d<F>: ManifoldJetMap<
            P,
            V,
            TangentMap<V, Q, W>,
            TangentMap<V, Q, W>,
            1,
            DifferentiatedManifoldRoute<𝒞, JP, JV, JQ, JW, InnerRoute>,
        >,
{
}

impl<𝒞, F, P, V, Q, W, JP, JV, JQ, JW, InnerRoute>
    EvaluableAt<
        𝒞,
        P,
        TangentMap<V, TangentMap<V, Q, W>, TangentMap<V, Q, W>>,
        DifferentiatedManifoldRoute<𝒞, JP, JV, JQ, JW, InnerRoute>,
    > for d<d<F>>
where
    𝒞: Cat,
    P: Connection<P, V>,
    V: Tensor<Hand = Right, Action: ActionExists>,
    Q: Connection<Q, W>,
    W: Tensor<F = V::F, Hand = Right, Action: TensorProductAction<V::Action>>,
    JP: Point + Connection<JP, JV> + CommutesJet<P, V, 1>,
    JV: Tensor<F = Jet<𝒞, V::F, 1>, Hand = Right, Action: ActionExists>,
    JQ: Point + Connection<JQ, JW> + CommutesJet<Q, W, 1>,
    JW: Tensor<F = Jet<𝒞, V::F, 1>, Hand = Right, Action: TensorProductAction<JV::Action>>,
    F: ManifoldJetMap<JP, JV, JQ, JW, 1, InnerRoute>,
    Jet<𝒞, V::F, 1>: Field,
    Jet<𝐅𝐥𝐝::𝒞, V::F, 1>: Field,
    TangentMap<V, Q, W>: Tensor<F = V::F, Hand = Right, Action: TensorProductAction<V::Action>>
        + Connection<TangentMap<V, Q, W>, TangentMap<V, Q, W>>,
    d<F>: ManifoldJetMap<
            P,
            V,
            TangentMap<V, Q, W>,
            TangentMap<V, Q, W>,
            1,
            DifferentiatedManifoldRoute<𝒞, JP, JV, JQ, JW, InnerRoute>,
        >,
{
    fn evaluate_at(&self, point: P) -> TangentMap<V, TangentMap<V, Q, W>, TangentMap<V, Q, W>> {
        evaluate_manifold_derivative_at::<
            d<F>,
            P,
            V,
            TangentMap<V, Q, W>,
            TangentMap<V, Q, W>,
            DifferentiatedManifoldRoute<𝒞, JP, JV, JQ, JW, InnerRoute>,
        >(self, point)
    }
}

impl<𝒞, F, P, V, Q, W, JP, JQ> EvaluableAt<𝒞, P, TangentMap<V, Q, W, Q>, ManifoldRoute<Q, JP, JQ>>
    for d<F>
where
    𝒞: Cat,
    P: Connection<P, V>,
    V: Tensor<F: ι<C: JetRegion<𝒞>>, Hand = Right, Action: ActionExists>,
    Q: Connection<Q, W>,
    W: Tensor<F = V::F, Hand = Right, Action: TensorProductAction<V::Action>>,
    JP: Point,
    JQ: Point,
    F: ManifoldJetMap<P, V, Q, W, 1, ManifoldRoute<Q, JP, JQ>>,
{
    fn evaluate_at(&self, point: P) -> TangentMap<V, Q, W, Q> {
        evaluate_manifold_derivative_at::<F, P, V, Q, W, ManifoldRoute<Q, JP, JQ>>(self, point)
    }
}

impl<P, V, Q, W, F, const N: usize, JP, JQ> ManifoldJetMap<P, V, Q, W, N, ManifoldRoute<Q, JP, JQ>>
    for F
where
    P: Connection<P, V>,
    V: Tensor,
    Q: Connection<Q, W>,
    W: Tensor<F = V::F>,
    JP: CommutesJet<P, V, N>,
    JQ: CommutesJet<Q, W, N>,
    F: Fn(JP) -> JQ,
{
    fn jet_at(&self, input: Tangent<P, V, N>) -> Tangent<Q, W, N> {
        JQ::uncommute_jet(self(JP::commute_jet(input)))
    }
}

/// An isomorphism between a connection's intrinsic split jet and one concrete
/// Rust presentation obtained by commuting jettification through a nominal
/// type constructor.
pub trait CommutesJet<P: Point, V: Tensor, const N: usize>: Point
where
    P: Connection<P, V>,
{
    fn commute_jet(value: Tangent<P, V, N>) -> Self;

    fn uncommute_jet(value: Self) -> Tangent<P, V, N>;
}

impl<P, V, const N: usize> CommutesJet<P, V, N> for Tangent<P, V, N>
where
    P: Connection<P, V>,
    V: Tensor,
{
    fn commute_jet(value: Tangent<P, V, N>) -> Self {
        value
    }

    fn uncommute_jet(value: Self) -> Tangent<P, V, N> {
        value
    }
}

impl<
    𝒞: Cat,
    F: JetMap<𝒞, BT, FT, 1, Jet<𝒞, S, N>, ː<JetLayer<𝒞, N>, Route>>,
    BT: Vector<F = FT::F, Hand = Right>,
    FT: Vector<Hand = Right, Action: TensorProductAction<BT::Action>>,
    const N: usize,
    S: Field,
    Route: ConstantRoute<BT::F, Output = S>,
> JetMap<𝒞, BT, HomOf<BT, FT>, N, S, Route> for d<F>
where
    // The outer presentation.
    JetVectorIn<𝒞, FT, N, S>: Vector<F = Jet<𝒞, S, N>>,
    JetVectorIn<𝒞, BT, N, S>: Tensor<F = Jet<𝒞, S, N>>,
    // One additional derivative layer over the existing outer scalar.
    JetVectorIn<𝒞, BT, 1, Jet<𝒞, S, N>>: Tensor<F = Jet<𝒞, Jet<𝒞, S, N>>>,
    JetVectorIn<𝒞, FT, 1, Jet<𝒞, S, N>>: Tensor<F = Jet<𝒞, Jet<𝒞, S, N>>>,
    Jet<𝒞, S, N>: Field,
{
    fn jet_at(&self, input: JetVectorIn<𝒞, BT, N, S>) -> JetVectorIn<𝒞, HomOf<BT, FT>, N, S> {
        #[allow(type_alias_bounds)]
        type OuterScalar<𝒞: Cat, S, const N: usize> = Jet<𝒞, S, N>;

        let columns: BT::Array<JetVectorIn<𝒞, FT, N, S>> = BT::Array::from_fn(|input_coordinate| {
            let nested_input =
                JetVectorIn::<𝒞, BT, 1, OuterScalar<𝒞, S, N>>::from_fn(|coordinate| {
                    Jet::from_parts(
                        input[coordinate],
                        [if input_coordinate == coordinate {
                            OuterScalar::<𝒞, S, N>::one()
                        } else {
                            OuterScalar::<𝒞, S, N>::zero()
                        }],
                    )
                });

            let nested_output: JetVectorIn<𝒞, FT, 1, OuterScalar<𝒞, S, N>> = <F as JetMap<
                𝒞,
                BT,
                FT,
                1,
                OuterScalar<𝒞, S, N>,
                ː<JetLayer<𝒞, N>, Route>,
            >>::jet_at(
                &self.0,
                nested_input,
            );

            JetVectorIn::<𝒞, FT, N, S>::from_fn(|output_coordinate| {
                nested_output[output_coordinate][1]
            })
        });

        let rows: FT::Array<<Dual<BT> as Tensor>::Array<OuterScalar<𝒞, S, N>>> =
            FT::Array::from_fn(|output_coordinate| {
                <Dual<BT> as Tensor>::Array::from_fn(|input_coordinate| {
                    columns[input_coordinate][output_coordinate]
                })
            });

        TensorOver::<HomOf<BT, FT>, Jet<𝒞, S, N>>(
            TensorProductArray(rows, PhantomData),
            PhantomData,
        )
    }
}

impl<𝒞: Cat, F, BT, FT, const N: usize, S, Route> JetMap<𝒞, BT, FT, N, S, Route> for Along<F, BT>
where
    BT: Vector<F = FT::F>,
    FT: Vector,
    S: Field,
    Route: ConstantRoute<BT::F, Output = S>,
    Jet<𝒞, S, N>: Field,
    JetVectorIn<𝒞, FT, N, S>: Tensor<F = Jet<𝒞, S, N>>,
    JetVectorIn<𝒞, BT, N, S>: Tensor<F = Jet<𝒞, S, N>>,
    JetVectorIn<𝒞, BT, 1, Jet<𝒞, S, N>>: Tensor<F = Jet<𝒞, Jet<𝒞, S, N>>>,
    JetVectorIn<𝒞, FT, 1, Jet<𝒞, S, N>>: Tensor<F = Jet<𝒞, Jet<𝒞, S, N>>>,
    F: JetMap<𝒞, BT, FT, 1, Jet<𝒞, S, N>, ː<JetLayer<𝒞, N>, Route>>,
{
    fn jet_at(&self, input: JetVectorIn<𝒞, BT, N, S>) -> JetVectorIn<𝒞, FT, N, S> {
        #[allow(type_alias_bounds)]
        type OuterScalar<𝒞: Cat, S, const N: usize> = Jet<𝒞, S, N>;

        let nested_input =
            JetVectorIn::<𝒞, BT, 1, OuterScalar<𝒞, S, N>>::from_fn(|coordinate| {
                Jet::from_parts(
                    input[coordinate],
                    [Jet::from_parts(
                        Route::constant(self.direction[coordinate]),
                        [S::zero(); N],
                    )],
                )
            });

        let nested_output: JetVectorIn<𝒞, FT, 1, OuterScalar<𝒞, S, N>> = <F as JetMap<
            𝒞,
            BT,
            FT,
            1,
            OuterScalar<𝒞, S, N>,
            ː<JetLayer<𝒞, N>, Route>,
        >>::jet_at(
            &self.f, nested_input
        );

        JetVectorIn::<𝒞, FT, N, S>::from_fn(|coordinate| nested_output[coordinate][1])
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
