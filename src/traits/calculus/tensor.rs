use num_traits::{Inv, One, Zero};

use core::{
    marker::PhantomData,
    ops::{Deref, DerefMut, Index, IndexMut},
};

use crate::{
    impl_vector_ops,
    traits::{
        ActionExists, Array, Atomic, BothSided, Cat, DivRing, Dual, Field, Form, Handedness,
        Interval, Left, Metric, NonZero, Nondegenerate, Normalize, NormalizeWith, OneSided, Point,
        Rehandable, Right, Sesquilinear, Sidedness, Sinister, TangentBundle, Tensor,
        TensorNormalizer, TensorProductAction, Undecorated, Vector,
        calculus::{FormLift, Jet, JetVector, NondegenerateLift, TangentElement, TensorOver},
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
    pub(crate) DirectSumArray<V::F, U::Array<V::F>, V::Array<V::F>>,
);

impl<U: Tensor, V: Tensor<F = U::F, Hand = U::Hand>> DirectSum<U, V> {
    pub fn join(u: U, v: V) -> Self {
        Self(DirectSumArray(
            u.as_ref().clone(),
            v.as_ref().clone(),
            PhantomData,
        ))
    }

    pub fn split(self) -> (U, V) {
        (
            U::from_iter(self.0.0.into_iter()),
            V::from_iter(self.0.1.into_iter()),
        )
    }

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
pub struct DirectSumArray<T: Point, U: Array<T>, V: Array<T>>(
    pub(crate) U,
    pub(crate) V,
    pub(crate) PhantomData<T>,
);

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

impl<U: Form, V: Form<F = U::F, Hand = U::Hand>> Form for DirectSum<U, V> {
    fn flat(&self) -> Dual<Self> {
        let (u, v) = self.clone().split();

        Self::dual_isomorphism_inverse(DirectSum::join(u.flat(), v.flat()))
    }
}

impl<U: Nondegenerate, V: Nondegenerate<F = U::F, Hand = U::Hand>> Nondegenerate
    for DirectSum<U, V>
{
    fn sharp(v: Dual<Self>) -> Self {
        let (u, v) = Self::dual_isomorphism(v).split();

        Self::join(U::sharp(u), V::sharp(v))
    }
}

impl<U: FormLift, V: FormLift<F = U::F, Hand = U::Hand>> FormLift for DirectSum<U, V> {
    fn jet_flat_array<𝒞: Cat, S: Field, const N: usize>(
        value: &Self::Array<Jet<𝒞, S, N>>,
    ) -> <Dual<Self> as Tensor>::Array<Jet<𝒞, S, N>>
    where
        Jet<𝒞, S, N>: Field,
    {
        DirectSumArray(
            U::jet_flat_array::<𝒞, S, N>(&value.0),
            V::jet_flat_array::<𝒞, S, N>(&value.1),
            PhantomData,
        )
    }
}

impl<U: NondegenerateLift, V: NondegenerateLift<F = U::F, Hand = U::Hand>> NondegenerateLift
    for DirectSum<U, V>
{
    fn jet_sharp_array<𝒞: Cat, S: Field, const N: usize>(
        value: &<Dual<Self> as Tensor>::Array<Jet<𝒞, S, N>>,
    ) -> Self::Array<Jet<𝒞, S, N>>
    where
        Jet<𝒞, S, N>: Field,
    {
        DirectSumArray(
            U::jet_sharp_array::<𝒞, S, N>(&value.0),
            V::jet_sharp_array::<𝒞, S, N>(&value.1),
            PhantomData,
        )
    }
}

impl<U: Sesquilinear, V: Sesquilinear<F = U::F, Hand = U::Hand>> Sesquilinear for DirectSum<U, V> where
    <U::Action as Sidedness>::Meet<V::Action>: ActionExists
{
}

impl<U: Tensor + Interval, V: Tensor<F = U::F, Hand = U::Hand> + Interval<R = U::R>> Interval
    for DirectSum<U, V>
{
    type R = U::R;

    fn interval_squared(&self, other: &Self) -> Self::R {
        let (u1, v1) = self.clone().split();
        let (u2, v2) = other.clone().split();

        u1.interval_squared(&u2) + v1.interval_squared(&v2)
    }
}

impl<U: Tensor + Metric, V: Tensor<F = U::F, Hand = U::Hand> + Metric<R = U::R>> Metric
    for DirectSum<U, V>
{
}

/// The nested array representation used by [`TensorProduct`].
///
/// Its flat iteration and indexing order is outer coordinate first, then inner
/// coordinate. No claim is made that the nested arrays are contiguous.
#[derive(Debug, Copy, Clone)]
pub struct TensorProductArray<T: Point, U: Array<V>, V: Array<T>>(
    pub(crate) U,
    pub(crate) PhantomData<(T, V)>,
);

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
>(pub(crate) TensorProductArray<V::F, U::Array<V::Array<V::F>>, V::Array<V::F>>);

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

pub(crate) type HomOf<BT, FT> = TensorProduct<FT, Dual<BT>>;

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
>(pub HomOf<BT, FT>, PhantomData<fn() -> (FP, Fiber)>);

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

impl<U: Tensor, V: Tensor<F = U::F, Hand = U::Hand>, S: Point> TensorOver<DirectSum<U, V>, S> {
    pub fn split(self) -> (TensorOver<U, S>, TensorOver<V, S>) {
        let DirectSumArray(u, v, _) = self.0;

        (TensorOver(u, PhantomData), TensorOver(v, PhantomData))
    }

    pub fn join(u: TensorOver<U, S>, v: TensorOver<V, S>) -> Self {
        Self(DirectSumArray(u.0, v.0, PhantomData), PhantomData)
    }
}

impl<P: Point, U: Tensor, V: Tensor<F = U::F, Hand = U::Hand>, Tower, const N: usize>
    TangentElement<P, DirectSum<U, V>, Tower, N>
{
    pub fn split(
        self,
    ) -> (
        TangentElement<P, U, Tower, N>,
        TangentElement<P, V, Tower, N>,
    ) {
        let (u, v) = self.1.split();
        (
            TangentElement::new(self.0.clone(), u),
            TangentElement::new(self.0, v),
        )
    }

    pub fn join(p: P, u: JetVector<U, N>, v: JetVector<V, N>) -> Self {
        Self::new(p, TensorOver::join(u, v))
    }
}
