use std::{
    marker::PhantomData,
    ops::{Add, Deref, DerefMut, Index, IndexMut, Mul, Neg, Sub},
};

use num_traits::{Inv, One, Zero};

use crate::{
    coords::Coords,
    impl_vector_ops,
    traits::{
        ActionExists, Array, CField, Chart, DivRing, Dual, ExpMap, Field, Handedness, Left,
        NonZero, Point, Right, Sidedness, TangentBundle, TensorProductAction, Vector,
    },
};

#[derive(Debug, Copy, Clone)]
pub struct DirectSum<U: Vector<F = V::F>, V: Vector>(
    DirectSumArray<V::F, U::Array<V::F>, V::Array<V::F>>,
);

impl<F: Field, H: Handedness, U: Vector<F = F, Hand = H>, V: Vector<F = F, Hand = H>>
    DirectSum<U, V>
{
    pub fn dual_isomorphism(dual: Dual<Self>) -> DirectSum<Dual<U>, Dual<V>> {
        DirectSum::<Dual<U>, Dual<V>>::from_fn(|i| dual[i])
    }

    pub fn dual_isomorphism_inverse(dual: DirectSum<Dual<U>, Dual<V>>) -> Dual<Self> {
        Dual::<Self>::from_fn(|i| dual[i])
    }
}

impl<F: Field, H: Handedness, U: Vector<F = F, Hand = H>, V: Vector<F = F, Hand = H>> Index<usize>
    for DirectSum<U, V>
{
    type Output = F;

    fn index(&self, index: usize) -> &F {
        &self.0[index]
    }
}

impl<F: Field, H: Handedness, U: Vector<F = F, Hand = H>, V: Vector<F = F, Hand = H>>
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

impl<F: Field, H: Handedness, U: Vector<F = F, Hand = H>, V: Vector<F = F, Hand = H>> Vector
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

impl<F: Field, H: Handedness, U: Vector<F = F, Hand = H>, V: Vector<F = F, Hand = H>>
    AsRef<DirectSumArray<F, U::Array<F>, V::Array<F>>> for DirectSum<U, V>
{
    fn as_ref(&self) -> &DirectSumArray<F, U::Array<F>, V::Array<F>> {
        &self.0
    }
}

impl<F: Field, H: Handedness, U: Vector<F = F, Hand = H>, V: Vector<F = F, Hand = H>>
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
    U: Vector<F = F, Hand = H>,
    V: Vector<F = F, Hand = H>
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
    U: Vector<Hand = Right, Action: TensorProductAction<V::Action>>,
    V: Vector<F = U::F, Hand = Left, Action: ActionExists>,
>(TensorProductArray<V::F, U::Array<V::Array<V::F>>, V::Array<V::F>>);

impl<
    U: Vector<Hand = Right, Action: TensorProductAction<V::Action>>,
    V: Vector<F = U::F, Hand = Left, Action: ActionExists>,
> Vector for TensorProduct<U, V>
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
    U: Vector<Hand = Right, Action: TensorProductAction<V::Action>>,
    V: Vector<F = U::F, Hand = Left, Action: ActionExists>,
> AsRef<TensorProductArray<V::F, U::Array<V::Array<V::F>>, V::Array<V::F>>>
    for TensorProduct<U, V>
{
    fn as_ref(&self) -> &TensorProductArray<V::F, U::Array<V::Array<V::F>>, V::Array<V::F>> {
        &self.0
    }
}

impl<
    U: Vector<Hand = Right, Action: TensorProductAction<V::Action>>,
    V: Vector<F = U::F, Hand = Left, Action: ActionExists>,
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
    U: Vector<Hand = Right, Action: TensorProductAction<V::Action>>,
    V: Vector<F = U::F, Hand = Left, Action: ActionExists>,
> Index<usize> for TensorProduct<U, V>
{
    type Output = V::F;

    fn index(&self, index: usize) -> &V::F {
        &self.0[index]
    }
}

impl<
    U: Vector<Hand = Right, Action: TensorProductAction<V::Action>>,
    V: Vector<F = U::F, Hand = Left, Action: ActionExists>,
> IndexMut<usize> for TensorProduct<U, V>
{
    fn index_mut(&mut self, index: usize) -> &mut V::F {
        &mut self.0[index]
    }
}

impl<
    U: Vector<Hand = Right, Action: TensorProductAction<V::Action>>,
    V: Vector<F = U::F, Hand = Left, Action: ActionExists>,
> Index<(usize, usize)> for TensorProduct<U, V>
{
    type Output = V::F;

    fn index(&self, index: (usize, usize)) -> &V::F {
        &self.0[index]
    }
}

impl<
    U: Vector<Hand = Right, Action: TensorProductAction<V::Action>>,
    V: Vector<F = U::F, Hand = Left, Action: ActionExists>,
> IndexMut<(usize, usize)> for TensorProduct<U, V>
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut V::F {
        &mut self.0[index]
    }
}

impl_vector_ops!(
    TensorProduct<U, V>,
    U: Vector<Hand = Right, Action: TensorProductAction<V::Action>>,
    V: Vector<F = U::F, Hand = Left, Action: ActionExists>,
);

pub trait Section<
    BP: Point,
    BT: Vector,
    Base: TangentBundle<BP, BT>,
    FP: Point,
    FT: Vector<F = BT::F>,
    Fiber: TangentBundle<FP, FT>,
>
{
    fn at(&self, value: Base) -> Fiber;
}

impl<
    BP: Point,
    BT: Vector,
    Base: TangentBundle<BP, BT>,
    FP: Point,
    FT: Vector<F = BT::F>,
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
    BT: Vector<Hand = Right, Action: ActionExists>,
    FP: Point,
    FT: Vector<F = BT::F, Hand = Right, Action: TensorProductAction<BT::Action>>,
    Fiber: TangentBundle<FP, FT>,
>(HomOf<BT, FT>, PhantomData<fn() -> (FP, Fiber)>);

impl<
    BT: Vector<Hand = Right, Action: ActionExists>,
    FP: Point,
    FT: Vector<F = BT::F, Hand = Right, Action: TensorProductAction<BT::Action>>,
    Fiber: TangentBundle<FP, FT>,
> TangentMap<BT, FP, FT, Fiber>
{
    pub fn new(v: HomOf<BT, FT>) -> Self {
        Self(v, PhantomData)
    }
}

impl<
    BT: Vector<Hand = Right, Action: ActionExists>,
    FP: Point,
    FT: Vector<F = BT::F, Hand = Right, Action: TensorProductAction<BT::Action>>,
    Fiber: TangentBundle<FP, FT>,
> Clone for TangentMap<BT, FP, FT, Fiber>
{
    fn clone(&self) -> Self {
        Self(self.0.clone(), self.1.clone())
    }
}

impl<
    BT: Vector<Hand = Right, Action: ActionExists>,
    FP: Point,
    FT: Vector<F = BT::F, Hand = Right, Action: TensorProductAction<BT::Action>>,
    Fiber: TangentBundle<FP, FT>,
> Index<usize> for TangentMap<BT, FP, FT, Fiber>
{
    type Output = BT::F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<
    BT: Vector<Hand = Right, Action: ActionExists>,
    FP: Point,
    FT: Vector<F = BT::F, Hand = Right, Action: TensorProductAction<BT::Action>>,
    Fiber: TangentBundle<FP, FT>,
> IndexMut<usize> for TangentMap<BT, FP, FT, Fiber>
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<
    BT: Vector<Hand = Right, Action: ActionExists>,
    FP: Point,
    FT: Vector<F = BT::F, Hand = Right, Action: TensorProductAction<BT::Action>>,
    Fiber: TangentBundle<FP, FT>,
>
    AsRef<
        TensorProductArray<
            BT::F,
            FT::Array<<Dual<BT> as Vector>::Array<BT::F>>,
            <Dual<BT> as Vector>::Array<BT::F>,
        >,
    > for TangentMap<BT, FP, FT, Fiber>
{
    fn as_ref(
        &self,
    ) -> &TensorProductArray<
        BT::F,
        FT::Array<<Dual<BT> as Vector>::Array<BT::F>>,
        <Dual<BT> as Vector>::Array<BT::F>,
    > {
        self.0.as_ref()
    }
}

impl<
    BT: Vector<Hand = Right, Action: ActionExists>,
    FP: Point,
    FT: Vector<F = BT::F, Hand = Right, Action: TensorProductAction<BT::Action>>,
    Fiber: TangentBundle<FP, FT>,
>
    AsMut<
        TensorProductArray<
            BT::F,
            FT::Array<<Dual<BT> as Vector>::Array<BT::F>>,
            <Dual<BT> as Vector>::Array<BT::F>,
        >,
    > for TangentMap<BT, FP, FT, Fiber>
{
    fn as_mut(
        &mut self,
    ) -> &mut TensorProductArray<
        BT::F,
        FT::Array<<Dual<BT> as Vector>::Array<BT::F>>,
        <Dual<BT> as Vector>::Array<BT::F>,
    > {
        self.0.as_mut()
    }
}

impl<
    BT: Vector<Hand = Right, Action: ActionExists>,
    FP: Point,
    FT: Vector<F = BT::F, Hand = Right, Action: TensorProductAction<BT::Action>>,
    Fiber: TangentBundle<FP, FT>,
> Vector for TangentMap<BT, FP, FT, Fiber>
{
    type F = <HomOf<BT, FT> as Vector>::F;
    type Array<T: Point> = <HomOf<BT, FT> as Vector>::Array<T>;
    type Hand = <HomOf<BT, FT> as Vector>::Hand;
    type Action = <HomOf<BT, FT> as Vector>::Action;

    fn from_fn(f: impl FnMut(usize) -> Self::F) -> Self {
        Self(HomOf::<BT, FT>::from_fn(f), PhantomData)
    }
}

impl_vector_ops!(TangentMap<BT, FP, FT, Fiber>,
    BT: Vector<Hand = Right, Action: ActionExists>,
    FP: Point,
    FT: Vector<F = BT::F, Hand = Right, Action: TensorProductAction<BT::Action>>,
    Fiber: TangentBundle<FP, FT>,
);

impl<
    BT: Vector<Hand = Right, Action: ActionExists>,
    FP: Point,
    FT: Vector<F = BT::F, Hand = Right, Action: TensorProductAction<BT::Action>>,
    Fiber: TangentBundle<FP, FT>,
> Deref for TangentMap<BT, FP, FT, Fiber>
{
    type Target = HomOf<BT, FT>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<
    BT: Vector<Hand = Right, Action: ActionExists>,
    FP: Point,
    FT: Vector<F = BT::F, Hand = Right, Action: TensorProductAction<BT::Action>>,
    Fiber: TangentBundle<FP, FT>,
> DerefMut for TangentMap<BT, FP, FT, Fiber>
{
    fn deref_mut(&mut self) -> &mut HomOf<BT, FT> {
        &mut self.0
    }
}

#[derive(Copy, Clone, Debug)]
pub struct JetVector<V: Vector, const N: usize = 1>(V::Array<Jet<V::F, N>>);

impl<V: Vector, const N: usize> JetVector<V, N> {
    pub fn constant(v: V) -> Self {
        Self::from_fn(|i| Jet::from_field(v[i]))
    }
}

impl<V: Vector, const N: usize> Vector for JetVector<V, N> {
    type F = Jet<V::F, N>;

    type Array<T: Point> = V::Array<T>;

    type Hand = V::Hand;
    type Action = V::Action;

    fn from_fn(f: impl FnMut(usize) -> Self::F) -> Self {
        Self(Self::Array::from_fn(f))
    }
}

impl<V: Vector, const N: usize> Index<usize> for JetVector<V, N> {
    type Output = Jet<V::F, N>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<V: Vector, const N: usize> IndexMut<usize> for JetVector<V, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<V: Vector, const N: usize> AsRef<V::Array<Jet<V::F, N>>> for JetVector<V, N> {
    fn as_ref(&self) -> &V::Array<Jet<V::F, N>> {
        &self.0
    }
}

impl<V: Vector, const N: usize> AsMut<V::Array<Jet<V::F, N>>> for JetVector<V, N> {
    fn as_mut(&mut self) -> &mut V::Array<Jet<V::F, N>> {
        &mut self.0
    }
}

impl_vector_ops!(JetVector<V, N>, V: Vector, const N: usize);

type JetCoords<F, const N: usize> = DirectSum<Coords<F, 1>, Coords<F, N>>;

#[derive(Debug, Copy, Clone)]
pub struct Jet<F: Field, const N: usize = 1>(pub JetCoords<F, N>);

impl<F: Field, const N: usize> Jet<F, N> {
    pub fn from_field(value: F) -> Self {
        Self(DirectSum(DirectSumArray(
            [value],
            [F::zero(); N],
            PhantomData,
        )))
    }

    pub fn new(value: F, coefficients: [F; N]) -> Self {
        Self(DirectSum(DirectSumArray(
            [value],
            coefficients,
            PhantomData,
        )))
    }

    pub fn from_fn(f: impl FnMut(usize) -> F) -> Self {
        Self(JetCoords::from_fn(f))
    }
}

impl<F: CField, const N: usize> CField for Jet<F, N> {}

impl<F: Field, const N: usize> PartialEq for Jet<F, N> {
    fn eq(&self, other: &Self) -> bool {
        self[0] == other[0]
    }
}

impl<F: Field, const N: usize> Index<usize> for Jet<F, N> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<F: Field, const N: usize> IndexMut<usize> for Jet<F, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<F: Field, const N: usize> Field for Jet<F, N> {
    type Fixed = Jet<F::Fixed, N>;

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

impl<F: Field, const N: usize> Add for Jet<F, N> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl<F: Field, const N: usize> Sub for Jet<F, N> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl<F: Field, const N: usize> Mul for Jet<F, N> {
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

impl<F: Field, const N: usize> Neg for Jet<F, N> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl<F: Field, const N: usize> One for Jet<F, N> {
    fn one() -> Self {
        Self::from_fn(|x| if x == 0 { F::one() } else { F::zero() })
    }
}

impl<F: Field, const N: usize> Zero for Jet<F, N> {
    fn zero() -> Self {
        Self(DirectSum::zero())
    }

    fn is_zero(&self) -> bool {
        self[0].is_zero()
    }
}

impl<F: Field, const N: usize> Inv for NonZero<Jet<F, N>> {
    type Output = NonZero<Jet<F, N>>;

    fn inv(self) -> Self::Output {
        let input = self.0;

        // Spell this using your DivRing::Mul machinery.
        let constant_inverse: F = <F as DivRing>::Mul::from(NonZero::new_unchecked(input[0]))
            .inv()
            .into()
            .0;

        let mut output = Jet::<F, N>::zero();
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
pub struct TangentElement<P: Point, V: Vector, Tower>(pub P, pub JetVector<V>, PhantomData<Tower>);

impl<P: Point, V: Vector, Tower> TangentElement<P, V, Tower> {
    pub fn new(p: P, v: JetVector<V>) -> Self {
        Self(p, v, PhantomData)
    }
}

type Prolongation<P, V, T> = TangentElement<P, V, Cons<T, Nil>>;

pub type Tangent<P, V> = TangentElement<P, V, Nil>;
pub type TM<P, V, T, U> = TangentElement<P, V, Cons<T, Cons<U, Nil>>>;
pub type LiftedTM<P, V, T> = TM<P, V, T, Prolongation<P, V, T>>;

impl<P: Point, V: Vector, T: TangentBundle<P, V>, U: TangentBundle<Self, JetVector<V>>> Chart<P, V>
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
impl<P: Point, V: Vector, T: TangentBundle<P, V>, U: TangentBundle<Self, JetVector<V>>> ExpMap<P, V>
    for TM<P, V, T, U>
{
}
impl<P: Point, V: Vector, T: TangentBundle<P, V>, U: TangentBundle<Self, JetVector<V>>>
    TangentBundle<P, V> for TM<P, V, T, U>
{
}

impl<P: Point, V: Vector, T: TangentBundle<P, V>, U: TangentBundle<Self, JetVector<V>>>
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
impl<P: Point, V: Vector, T: TangentBundle<P, V>, U: TangentBundle<Self, JetVector<V>>>
    ExpMap<Self, JetVector<V>> for TM<P, V, T, U>
{
}

impl<P: Point, V: Vector, T: TangentBundle<P, V>, U: TangentBundle<Self, JetVector<V>>>
    TangentBundle<Self, JetVector<V>> for TM<P, V, T, U>
{
}

impl<P: Point, V: Vector, T: TangentLift<P, V>, U: TangentBundle<Self, JetVector<V>>>
    TangentLift<P, V> for TM<P, V, T, U>
{
    fn tangent_to_local(base: Tangent<P, V>, local: Tangent<P, V>) -> Option<JetVector<V>> {
        T::tangent_to_local(base, local)
    }

    fn tangent_to_global(base: Tangent<P, V>, coordinate: JetVector<V>) -> (P, JetVector<V>) {
        T::tangent_to_global(base, coordinate)
    }
}

pub trait TangentLift<P: Point, V: Vector>: TangentBundle<P, V> {
    fn tangent_to_local(base: Tangent<P, V>, local: Tangent<P, V>) -> Option<JetVector<V>>;
    fn tangent_to_global(base: Tangent<P, V>, coordinate: JetVector<V>) -> (P, JetVector<V>);
}

pub trait JetTangentBundle<BP: Point, BT: Vector>: TangentBundle<BP, BT> + Point {
    type JetBundle: TangentBundle<Self::JetBundle, JetVector<BT>>;

    fn lift(base: BP, jet: JetVector<BT>) -> Self::JetBundle;
    fn into_parts(bundle: Self::JetBundle) -> (BP, JetVector<BT>);
}

impl<P: Point, V: Vector, T: TangentLift<P, V>> Chart<LiftedTM<P, V, T>, JetVector<V>>
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

impl<P: Point, V: Vector, T: TangentLift<P, V>> ExpMap<LiftedTM<P, V, T>, JetVector<V>>
    for Prolongation<P, V, T>
{
}
impl<P: Point, V: Vector, T: TangentLift<P, V>> TangentBundle<LiftedTM<P, V, T>, JetVector<V>>
    for Prolongation<P, V, T>
{
}

impl<P: Point, V: Vector, T: TangentLift<P, V>> JetTangentBundle<P, V> for T {
    type JetBundle = LiftedTM<P, V, T>;

    fn lift(base: P, jet: JetVector<V>) -> Self::JetBundle {
        TangentElement::new(base, jet)
    }

    fn into_parts(bundle: Self::JetBundle) -> (P, JetVector<V>) {
        (bundle.0, bundle.1)
    }
}

impl<V: Vector> TangentLift<V, V> for V {
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
    type Tangent: Vector;
    type Bundle: JetTangentBundle<Self::Point, Self::Tangent, JetBundle = Self>;

    fn base_point(&self) -> Self::Point;
    fn jet(&self) -> &JetVector<Self::Tangent>;
}

impl<P, V, T> LiftedBundle for LiftedTM<P, V, T>
where
    P: Point,
    V: Vector,
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

type ScalarJetBundle<F> = LiftedTM<Coords<F, 1>, Coords<F, 1>, Coords<F, 1>>;

pub fn scalar_jet<F: Field>(bundle: ScalarJetBundle<F>) -> Jet<F> {
    Jet::from_field(bundle.0[0]) + bundle.1[0]
}

pub fn scalar_bundle<F: Field>(mut jet: Jet<F>) -> ScalarJetBundle<F> {
    let base = Coords::from(jet[0]);

    jet[0] = F::zero();

    let tangent = JetVector::<Coords<F, 1>>::from_fn(|_| jet);

    TangentElement::new(base, tangent)
}

pub fn lift_scalar_jet<F, Function>(
    function: Function,
) -> impl Fn(ScalarJetBundle<F>) -> ScalarJetBundle<F>
where
    F: Field,
    Function: Fn(Jet<F>) -> Jet<F>,
{
    move |bundle| scalar_bundle(function(scalar_jet(bundle)))
}

pub struct Differential<F, BT, FT> {
    function: F,
    marker: PhantomData<fn(BT) -> FT>,
}

pub fn d<F, BT, FT>(function: F) -> Differential<F, BT, FT>
where
    BT: Vector,
    FT: Vector<F = BT::F>,
    F: Fn(JetVector<BT>) -> JetVector<FT>,
{
    Differential {
        function,
        marker: PhantomData,
    }
}

impl<F, BT, FT> Differential<F, BT, FT>
where
    BT: Vector<Hand = Right, Action: ActionExists>,
    FT: Vector<F = BT::F, Hand = Right, Action: TensorProductAction<BT::Action>>,
    F: Fn(JetVector<BT>) -> JetVector<FT>,
{
    pub fn at(&self, point: BT) -> TangentMap<BT, FT, FT, FT> {
        let columns: BT::Array<FT> = BT::Array::from_fn(|input_coordinate| {
            let input = JetVector::<BT>::from_fn(|j| {
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

        let rows: FT::Array<<Dual<BT> as Vector>::Array<BT::F>> =
            FT::Array::from_fn(|output_coordinate| {
                <Dual<BT> as Vector>::Array::from_fn(|input_coordinate| {
                    columns[input_coordinate][output_coordinate]
                })
            });

        TangentMap::new(TensorProduct(TensorProductArray(rows, PhantomData)))
    }
}
