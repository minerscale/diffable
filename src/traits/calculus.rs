use std::{
    marker::PhantomData,
    ops::{Add, Deref, DerefMut, Div, Index, IndexMut, Mul, Neg, Rem, Sub},
};

use num_traits::{Euclid, Inv, Num, NumCast, One, ToPrimitive, Zero};

use crate::{
    coords::Coords,
    impl_vector_ops,
    traits::{
        ActionExists, Array, BothSided, CField, Chart, DivRing, Dual, Euclidean, ExactCmp, ExpMap,
        Field, Form, Handedness, Interval, Left, Metric, NonZero, Nondegenerate, Point, Real,
        Right, Sesquilinear, Sidedness, TangentBundle, Tensor, TensorProductAction,
    },
};

#[derive(Debug, Copy, Clone)]
pub struct DirectSum<U: Tensor<F = V::F>, V: Tensor>(
    DirectSumArray<V::F, U::Array<V::F>, V::Array<V::F>>,
);

impl<F: Field, H: Handedness, U: Tensor<F = F, Hand = H>, V: Tensor<F = F, Hand = H>>
    DirectSum<U, V>
{
    pub fn dual_isomorphism(dual: Dual<Self>) -> DirectSum<Dual<U>, Dual<V>> {
        DirectSum::<Dual<U>, Dual<V>>::from_fn(|i| dual[i])
    }

    pub fn dual_isomorphism_inverse(dual: DirectSum<Dual<U>, Dual<V>>) -> Dual<Self> {
        Dual::<Self>::from_fn(|i| dual[i])
    }
}

impl<F: Field, H: Handedness, U: Tensor<F = F, Hand = H>, V: Tensor<F = F, Hand = H>> Index<usize>
    for DirectSum<U, V>
{
    type Output = F;

    fn index(&self, index: usize) -> &F {
        &self.0[index]
    }
}

impl<F: Field, H: Handedness, U: Tensor<F = F, Hand = H>, V: Tensor<F = F, Hand = H>>
    IndexMut<usize> for DirectSum<U, V>
{
    fn index_mut(&mut self, index: usize) -> &mut F {
        &mut self.0[index]
    }
}

#[derive(Debug, Copy, Clone)]
pub struct DirectSumArray<T: Point, U: Array<T>, V: Array<T>>(U, V, PhantomData<T>);

impl<T: Point, U: Array<T>, V: Array<T>> Array<T> for DirectSumArray<T, U, V> {
    const N: usize = U::N + V::N;

    type Iter<'a>
        = std::iter::Chain<U::Iter<'a>, V::Iter<'a>>
    where
        Self: 'a,
        T: 'a;

    type IterMut<'a>
        = std::iter::Chain<U::IterMut<'a>, V::IterMut<'a>>
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
        Self(
            U::from_fn(|i| f(i)),
            V::from_fn(|i| f(U::N + i)),
            PhantomData,
        )
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

    type IntoIter = std::iter::Chain<U::IntoIter, V::IntoIter>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter().chain(self.1)
    }
}

impl<F: Field, H: Handedness, U: Tensor<F = F, Hand = H>, V: Tensor<F = F, Hand = H>> Tensor
    for DirectSum<U, V>
{
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

#[derive(Debug, Copy, Clone)]
pub struct TensorProductArray<T: Point, U: Array<V>, V: Array<T>>(U, PhantomData<(T, V)>);

fn iter_inner<'a, T: Point, V: Array<T>>(v: &'a V) -> V::Iter<'a> {
    v.iter()
}

fn iter_inner_mut<'a, T: Point, V: Array<T>>(v: &'a mut V) -> V::IterMut<'a> {
    v.iter_mut()
}

impl<T: Point, U: Array<V>, V: Array<T>> Array<T> for TensorProductArray<T, U, V> {
    const N: usize = U::N * V::N;

    type Iter<'a>
        = std::iter::FlatMap<U::Iter<'a>, V::Iter<'a>, fn(&'a V) -> V::Iter<'a>>
    where
        Self: 'a,
        T: 'a;

    type IterMut<'a>
        = std::iter::FlatMap<U::IterMut<'a>, V::IterMut<'a>, fn(&'a mut V) -> V::IterMut<'a>>
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
    type IntoIter = std::iter::Flatten<U::IntoIter>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter().flatten()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TensorProduct<
    U: Tensor<Hand = Right, Action: TensorProductAction<V::Action>>,
    V: Tensor<F = U::F, Hand = Left, Action: ActionExists>,
>(TensorProductArray<V::F, U::Array<V::Array<V::F>>, V::Array<V::F>>);

impl<
    U: Tensor<Hand = Right, Action: TensorProductAction<V::Action>>,
    V: Tensor<F = U::F, Hand = Left, Action: ActionExists>,
> Tensor for TensorProduct<U, V>
{
    type F = V::F;
    type Action = <U::Action as TensorProductAction<V::Action>>::Action;
    type Hand = <U::Action as TensorProductAction<V::Action>>::Hand;

    type Array<T: Point> = TensorProductArray<T, U::Array<V::Array<T>>, V::Array<T>>;

    fn from_fn(f: impl FnMut(usize) -> Self::F) -> Self {
        Self(Self::Array::from_fn(f))
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
> Index<usize> for TensorProduct<U, V>
{
    type Output = V::F;

    fn index(&self, index: usize) -> &V::F {
        &self.0[index]
    }
}

impl<
    U: Tensor<Hand = Right, Action: TensorProductAction<V::Action>>,
    V: Tensor<F = U::F, Hand = Left, Action: ActionExists>,
> IndexMut<usize> for TensorProduct<U, V>
{
    fn index_mut(&mut self, index: usize) -> &mut V::F {
        &mut self.0[index]
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

pub trait Section<
    BP: Point,
    BT: Tensor,
    Base: TangentBundle<BP, BT>,
    FP: Point,
    FT: Tensor<F = BT::F>,
    Fiber: TangentBundle<FP, FT>,
>
{
    fn at(&self, value: Base) -> Fiber;
}

impl<
    BP: Point,
    BT: Tensor,
    Base: TangentBundle<BP, BT>,
    FP: Point,
    FT: Tensor<F = BT::F>,
    Fiber: TangentBundle<FP, FT>,
    F: Fn(Base) -> Fiber,
> Section<BP, BT, Base, FP, FT, Fiber> for F
{
    fn at(&self, value: Base) -> Fiber {
        self(value)
    }
}

type HomOf<BT, FT> = TensorProduct<FT, Dual<BT>>;

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
        Self(self.0.clone(), self.1.clone())
    }
}

impl<
    BT: Tensor<Hand = Right, Action: ActionExists>,
    FP: Point,
    FT: Tensor<F = BT::F, Hand = Right, Action: TensorProductAction<BT::Action>>,
    Fiber: TangentBundle<FP, FT>,
> Index<usize> for TangentMap<BT, FP, FT, Fiber>
{
    type Output = BT::F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<
    BT: Tensor<Hand = Right, Action: ActionExists>,
    FP: Point,
    FT: Tensor<F = BT::F, Hand = Right, Action: TensorProductAction<BT::Action>>,
    Fiber: TangentBundle<FP, FT>,
> IndexMut<usize> for TangentMap<BT, FP, FT, Fiber>
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
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

pub trait JetMode: Copy + Clone + std::fmt::Debug {}

#[derive(Debug, Copy, Clone)]
pub enum Algebraic {}
#[derive(Debug, Copy, Clone)]
pub enum JetReal {}

impl JetMode for Algebraic {}
impl JetMode for JetReal {}

#[derive(Copy, Clone, Debug)]
pub struct JetVector<V: Tensor, Mode: JetMode = Algebraic, const N: usize = 1>(
    V::Array<Jet<V::F, Mode, N>>,
);

impl<V: Tensor, M: JetMode, const N: usize> JetVector<V, M, N> {
    pub fn constant(v: V) -> Self {
        Self(V::Array::<Jet<V::F, M, N>>::from_fn(|i| {
            Jet::from_field(v[i])
        }))
    }
}

impl<V: Tensor, const N: usize> Tensor for JetVector<V, Algebraic, N> {
    type F = Jet<V::F, Algebraic, N>;

    type Array<T: Point> = V::Array<T>;

    type Hand = V::Hand;
    type Action = V::Action;

    fn from_fn(f: impl FnMut(usize) -> Self::F) -> Self {
        Self(Self::Array::from_fn(f))
    }
}

impl<V: Tensor<F: Real>, const N: usize> Tensor for JetVector<V, JetReal, N> {
    type F = Jet<V::F, JetReal, N>;

    type Array<T: Point> = V::Array<T>;

    type Hand = V::Hand;
    type Action = V::Action;

    fn from_fn(f: impl FnMut(usize) -> Self::F) -> Self {
        Self(Self::Array::from_fn(f))
    }
}

impl<V: Tensor, M: JetMode, const N: usize> Index<usize> for JetVector<V, M, N> {
    type Output = Jet<V::F, M, N>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<V: Tensor, M: JetMode, const N: usize> IndexMut<usize> for JetVector<V, M, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<V: Tensor, M: JetMode, const N: usize> AsRef<V::Array<Jet<V::F, M, N>>>
    for JetVector<V, M, N>
{
    fn as_ref(&self) -> &V::Array<Jet<V::F, M, N>> {
        &self.0
    }
}

impl<V: Tensor, M: JetMode, const N: usize> AsMut<V::Array<Jet<V::F, M, N>>>
    for JetVector<V, M, N>
{
    fn as_mut(&mut self) -> &mut V::Array<Jet<V::F, M, N>> {
        &mut self.0
    }
}

impl_vector_ops!(JetVector<V, Algebraic, N>, V: Tensor, const N: usize);
impl_vector_ops!(JetVector<V, JetReal, N>, V: Tensor<F: Real>, const N: usize);

type JetCoords<F, const N: usize> = DirectSum<Coords<F, 1>, Coords<F, N>>;

#[derive(Debug, Copy, Clone)]
pub struct Jet<F: Field, Mode: JetMode, const N: usize = 1>(JetCoords<F, N>, PhantomData<Mode>);

impl<F: Field, M: JetMode, const N: usize> Jet<F, M, N> {
    pub fn from_field(value: F) -> Self {
        Self(
            DirectSum(DirectSumArray([value], [F::zero(); N], PhantomData)),
            PhantomData,
        )
    }

    pub fn new(value: F, coefficients: [F; N]) -> Self {
        Self(
            DirectSum(DirectSumArray([value], coefficients, PhantomData)),
            PhantomData,
        )
    }

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

#[derive(Debug, Copy, Clone)]
pub struct Nil;
#[derive(Debug, Copy, Clone)]
pub struct Cons<Head, Tail>(PhantomData<(Head, Tail)>);

#[derive(Debug, Clone)]
pub struct TangentElement<P: Point, V: Tensor, Tower>(pub P, pub JetVector<V>, PhantomData<Tower>);

impl<P: Point, V: Tensor, Tower> TangentElement<P, V, Tower> {
    pub fn new(p: P, v: JetVector<V>) -> Self {
        Self(p, v, PhantomData)
    }
}

type Prolongation<P, V, T> = TangentElement<P, V, Cons<T, Nil>>;

pub type Tangent<P, V> = TangentElement<P, V, Nil>;
pub type TM<P, V, T, U> = TangentElement<P, V, Cons<T, Cons<U, Nil>>>;
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

pub trait TangentLift<P: Point, V: Tensor>: TangentBundle<P, V> {
    fn tangent_to_local(base: Tangent<P, V>, local: Tangent<P, V>) -> Option<JetVector<V>>;
    fn tangent_to_global(base: Tangent<P, V>, coordinate: JetVector<V>) -> (P, JetVector<V>);
}

pub trait JetTangentBundle<BP: Point, BT: Tensor>: TangentBundle<BP, BT> + Point {
    type JetBundle: TangentBundle<Self::JetBundle, JetVector<BT>>;

    fn lift(base: BP, jet: JetVector<BT>) -> Self::JetBundle;
    fn into_parts(bundle: Self::JetBundle) -> (BP, JetVector<BT>);
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

impl<P: Point, V: Tensor, T: TangentLift<P, V>> JetTangentBundle<P, V> for T {
    type JetBundle = LiftedTM<P, V, T>;

    fn lift(base: P, jet: JetVector<V>) -> Self::JetBundle {
        TangentElement::new(base, jet)
    }

    fn into_parts(bundle: Self::JetBundle) -> (P, JetVector<V>) {
        (bundle.0, bundle.1)
    }
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

pub trait LiftedBundle {
    type Point: Point;
    type Tangent: Tensor;
    type Bundle: JetTangentBundle<Self::Point, Self::Tangent, JetBundle = Self>;

    fn base_point(&self) -> Self::Point;
    fn jet(&self) -> &JetVector<Self::Tangent>;
}

impl<P, V, T> LiftedBundle for LiftedTM<P, V, T>
where
    P: Point,
    V: Tensor,
    T: JetTangentBundle<P, V, JetBundle = Self>,
{
    type Point = P;
    type Tangent = V;
    type Bundle = T;

    fn base_point(&self) -> P {
        self.0.clone()
    }

    fn jet(&self) -> &JetVector<V> {
        &self.1
    }
}

pub struct Differential<F, BT, FT, M: JetMode> {
    function: F,
    marker: PhantomData<(fn(BT) -> FT, M)>,
}

pub fn d<F, BT, FT, M: JetMode>(function: F) -> Differential<F, BT, FT, M>
where
    BT: Tensor,
    FT: Tensor<F = BT::F>,
    F: Fn(JetVector<BT, M>) -> JetVector<FT, M>,
{
    Differential {
        function,
        marker: PhantomData,
    }
}

impl<F, BT, FT, M: JetMode> Differential<F, BT, FT, M>
where
    BT: Tensor<Hand = Right, Action: ActionExists>,
    FT: Tensor<F = BT::F, Hand = Right, Action: TensorProductAction<BT::Action>>,
    F: Fn(JetVector<BT, M>) -> JetVector<FT, M>,
    JetVector<BT, M>: Tensor<F = Jet<BT::F, M>>,
{
    pub fn at(&self, point: BT) -> TangentMap<BT, FT, FT, FT> {
        let columns: BT::Array<FT> = BT::Array::from_fn(|input_coordinate| {
            let input = JetVector::<BT, M>::from_fn(|j| {
                Jet::new(
                    point[j],
                    [if input_coordinate == j {
                        BT::F::one()
                    } else {
                        BT::F::zero()
                    }],
                )
            });

            let output = (self.function)(input);

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

pub trait FormLift: Form {
    fn jet_flat<const N: usize, M: JetMode>(
        value: &JetVector<Self, M, N>,
    ) -> Dual<JetVector<Self, M, N>>
    where
        JetVector<Self, M, N>: Tensor;
}

pub trait NondegenerateLift: Nondegenerate + FormLift {
    fn jet_sharp<const N: usize, M: JetMode>(
        value: Dual<JetVector<Self, M, N>>,
    ) -> JetVector<Self, M, N>
    where
        JetVector<Self, M, N>: Tensor;
}

impl<V: FormLift, const N: usize, M: JetMode> Form for JetVector<V, M, N>
where
    Self: Tensor,
{
    fn flat(&self) -> Dual<Self> {
        V::jet_flat(self)
    }
}

impl<V: NondegenerateLift, const N: usize, M: JetMode> Nondegenerate for JetVector<V, M, N>
where
    Self: Tensor,
{
    fn sharp(v: Dual<Self>) -> Self {
        V::jet_sharp(v)
    }
}

impl<V: Sesquilinear + Interval, const N: usize, M: JetMode> Interval for JetVector<V, M, N>
where
    Self: Sesquilinear<F: Field<Fixed: Real>>,
{
    type R = <<Self as Tensor>::F as Field>::Fixed;

    fn interval_squared(&self, other: &Self) -> Self::R {
        (self.clone() - other.clone()).norm_squared()
    }
}

impl<V: Sesquilinear, const N: usize, M: JetMode> Sesquilinear for JetVector<V, M, N> where
    Self: Nondegenerate
{
}

impl<V: Tensor + Metric, const N: usize, M: JetMode> Metric for JetVector<V, M, N> where
    Self: Interval
{
}

impl<V: Euclidean + NondegenerateLift, const N: usize> Euclidean for JetVector<V, JetReal, N> where
    Self: Tensor<F: Real, Action = BothSided>
{
}

impl<R: Real, const N: usize> PartialOrd for Jet<R, JetReal, N> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
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
        self * NonZero::new(rhs).unwrap().inv().0
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
