//! Discrete additive spaces and their group completion.
//!
//! [`N`] supplies the additive natural-number monoid, while [`Z`] is its
//! Grothendieck completion. The latter also acts as the lattice used to form
//! the circle [`S1`](crate::flat::S1) as a quotient of the real line.

use core::{
    marker::PhantomData,
    ops::{Add, Mul},
};

use num_traits::{One, Zero};

use crate::{
    coords::Coords,
    impl_group_via_add, impl_ring_via_grothendieck,
    traits::{
        Euclidean, Group, LieGroup,
        calculus::{CommutesJet, JetVector, JetVectorIn, Tangent},
        𝐑𝐞𝐚𝐥,
    },
};

/// The natural numbers `ℕ` under addition — the free commutative monoid on one
/// generator. Grothendieck-completed to [`Z`] to form the integer lattice.
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

/// The integers `ℤ`, as the Grothendieck completion of [`N`]. Serves as the
/// covering lattice for [`S1`](crate::flat::S1).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Z<V: Euclidean>(pub isize, PhantomData<V>);
impl_group_via_add!(Z<V>, V: Euclidean);

impl<V: Euclidean> Z<V> {
    /// Constructs the integer `v` in the lattice associated with `V`.
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

impl<V: Euclidean> LieGroup<Coords<V::F, 0>> for Z<V> {
    fn compose_jet<const M: usize>(
        lhs: Tangent<Self, Coords<V::F, 0>, M>,
        rhs: Tangent<Self, Coords<V::F, 0>, M>,
    ) -> Tangent<Self, Coords<V::F, 0>, M> {
        Tangent::new(lhs.0.compose(&rhs.0), lhs.1.compose(&rhs.1))
    }

    fn inverse_jet<const M: usize>(
        value: Tangent<Self, Coords<V::F, 0>, M>,
    ) -> Tangent<Self, Coords<V::F, 0>, M> {
        Tangent::new(value.0.inverse(), value.1.inverse())
    }

    fn identity_exp<const M: usize>(
        coordinate: JetVector<Coords<V::F, 0>, M>,
    ) -> Tangent<Self, Coords<V::F, 0>, M> {
        Tangent::new(Self::identity(), coordinate)
    }

    fn identity_log<const M: usize>(
        point: Tangent<Self, Coords<V::F, 0>, M>,
    ) -> Option<JetVector<Coords<V::F, 0>, M>> {
        point.0.is_zero().then_some(point.1)
    }
}

#[allow(type_alias_bounds)]
type ZJet<V: Euclidean, const M: usize> = Z<JetVectorIn<𝐑𝐞𝐚𝐥::𝒞, V, M>>;

impl<V: Euclidean, const M: usize> CommutesJet<Z<V>, Coords<V::F, 0>, M> for ZJet<V, M> {
    fn commute_jet(value: Tangent<Z<V>, Coords<V::F, 0>, M>) -> Self {
        Z::new(value.0.0)
    }

    fn uncommute_jet(value: Self) -> Tangent<Z<V>, Coords<V::F, 0>, M> {
        Tangent::new(Z::new(value.0), JetVector::zero())
    }
}
