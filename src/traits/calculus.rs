use std::{
    marker::PhantomData,
    ops::{Index, IndexMut},
};

use crate::{
    impl_vector_ops,
    traits::{Array, CField, Dual, Field, Handedness, Point, TangentBundle, Vector},
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

impl<T: Point, U: Array<V>, V: Array<T>> IntoIterator for TensorProductArray<T, U, V> {
    type Item = T;
    type IntoIter = std::iter::Flatten<U::IntoIter>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter().flatten()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TensorProduct<U: Vector<F = V::F>, V: Vector<F: CField>>(
    TensorProductArray<V::F, U::Array<V::Array<V::F>>, V::Array<V::F>>,
);

impl<U: Vector<F = V::F>, V: Vector<F: CField>> Vector for TensorProduct<U, V> {
    type F = V::F;

    type Array<T: Point> = TensorProductArray<T, U::Array<V::Array<T>>, V::Array<T>>;

    // just adopt V's handedness, it doesn't matter since it's all abelian anyway.
    type Hand = V::Hand;

    fn from_fn(f: impl FnMut(usize) -> Self::F) -> Self {
        Self(Self::Array::from_fn(f))
    }
}

impl<U: Vector<F = V::F>, V: Vector<F: CField>>
    AsRef<TensorProductArray<V::F, U::Array<V::Array<V::F>>, V::Array<V::F>>>
    for TensorProduct<U, V>
{
    fn as_ref(&self) -> &TensorProductArray<V::F, U::Array<V::Array<V::F>>, V::Array<V::F>> {
        &self.0
    }
}

impl<F: CField, U: Vector<F = F>, V: Vector<F = F>>
    AsMut<TensorProductArray<F, U::Array<V::Array<V::F>>, V::Array<F>>> for TensorProduct<U, V>
{
    fn as_mut(&mut self) -> &mut TensorProductArray<F, U::Array<V::Array<V::F>>, V::Array<F>> {
        &mut self.0
    }
}

impl<F: CField, U: Vector<F = F>, V: Vector<F = F>> Index<usize> for TensorProduct<U, V> {
    type Output = V::F;

    fn index(&self, index: usize) -> &V::F {
        &self.0[index]
    }
}

impl<F: CField, U: Vector<F = F>, V: Vector<F = F>> IndexMut<usize> for TensorProduct<U, V> {
    fn index_mut(&mut self, index: usize) -> &mut V::F {
        &mut self.0[index]
    }
}

impl_vector_ops!(TensorProduct<U, V>, U: Vector<F = V::F>, V: Vector<F: CField>);

pub trait Section {
    type BaseManifold: Point;
    type BaseTangent: Vector;
    type Base: TangentBundle<Self::BaseManifold, Self::BaseTangent>;

    type FiberManifold: Point;
    type FiberTangent: Vector;
    type Fiber: TangentBundle<Self::FiberManifold, Self::FiberTangent>;

    fn at(&self, v: Self::Base) -> Self::Fiber;
}

type HomOf<F> = TensorProduct<<F as Section>::FiberTangent, Dual<<F as Section>::BaseTangent>>;

#[allow(non_camel_case_types)]
pub struct d<F: Section>(pub F);

impl<
    F: Section<
            BaseTangent: Vector<F = <F::FiberTangent as Vector>::F>,
            FiberTangent: Vector<F: CField>,
        >,
> Section for d<F>
{
    type BaseManifold = F::BaseManifold;
    type BaseTangent = F::BaseTangent;
    type Base = F::Base;

    type FiberManifold = HomOf<F>;
    type FiberTangent = HomOf<F>;
    type Fiber = HomOf<F>;

    fn at(&self, _v: Self::Base) -> Self::Fiber {
        todo!()
    }
}
