use std::{
    marker::PhantomData,
    ops::{Add, Neg},
};

use num_traits::Zero;

use crate::{
    impl_group_via_grothendieck,
    traits::{CMonoid, Euclidean, Group, LieGroup},
};

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct N(pub usize);

impl CMonoid for N {
    fn identity() -> Self {
        Self(0)
    }

    fn compose(&self, other: &Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl_group_via_grothendieck!(Z<V>, N, <V: Euclidean>);

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Z<V: Euclidean>(pub isize, PhantomData<V>);

impl<V: Euclidean> Z<V> {
    pub fn new(v: isize) -> Self {
        Self(v, PhantomData)
    }
}

impl<V: Euclidean> Into<(N, N)> for Z<V> {
    fn into(self) -> (N, N) {
        if self.0 < 0 {
            (N::identity(), N(isize::try_into(-self.0).unwrap()))
        } else {
            (N(isize::try_into(self.0).unwrap()), N::identity())
        }
    }
}

impl<V: Euclidean> From<(N, N)> for Z<V> {
    fn from(value: (N, N)) -> Self {
        let pos = value.0.0;
        let neg = value.1.0;
        
        if pos >= neg {
            // The net result is positive, check if it fits in isize
            Self::new(isize::try_from(pos - neg).unwrap())
        } else {
            // The net result is negative, safely cast the absolute difference
            let diff = neg - pos;
            Self::new(-isize::try_from(diff).unwrap())
        }
    }
}

impl<V: Euclidean> Add for Z<V> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.0 + rhs.0)
    }
}

impl<V: Euclidean> Zero for Z<V> {
    fn zero() -> Self {
        Self::new(0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl<V: Euclidean> Neg for Z<V> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0, PhantomData)
    }
}

impl<V: Euclidean> LieGroup<V> for Z<V> {
    fn identity_exp(_: V) -> Self {
        Self::zero()
    }

    fn identity_log(p: &Self) -> Option<V> {
        if p.is_zero() { Some(V::zero()) } else { None }
    }
}
