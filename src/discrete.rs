use std::{
    marker::PhantomData,
    ops::{Add, Mul},
};

use num_traits::{One, Zero};

use crate::{
    impl_group_via_add, impl_ring_via_grothendieck,
    traits::{Euclidean, LieGroup},
};

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct N(pub usize);

impl Zero for N {
    fn zero() -> Self {
        N(0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl One for N {
    fn one() -> Self {
        Self(1)
    }
}

impl Add for N {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Mul for N {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl_ring_via_grothendieck!(Z<V>, N, V: Euclidean);

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Z<V: Euclidean>(pub isize, PhantomData<V>);
impl_group_via_add!(Z<V>, V: Euclidean);

impl<V: Euclidean> Z<V> {
    pub fn new(v: isize) -> Self {
        Self(v, PhantomData)
    }
}

impl<V: Euclidean> From<Z<V>> for (N, N) {
    fn from(val: Z<V>) -> Self {
        if val.0 < 0 {
            (N::zero(), N(isize::try_into(-val.0).unwrap()))
        } else {
            (N(isize::try_into(val.0).unwrap()), N::zero())
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

impl<V: Euclidean> LieGroup<V> for Z<V> {
    fn identity_exp(_: V) -> Self {
        Self::zero()
    }

    fn identity_log(p: &Self) -> Option<V> {
        if p.is_zero() { Some(V::zero()) } else { None }
    }
}
