#![cfg(feature = "testing")]

#[macro_use]
mod common;

use common::*;
use diffable::{
    coords::Coords,
    discrete::{N, Z},
    epsilon_metric::R64,
    test_cmonoid, test_lie_group, test_ring,
};
use proptest::prelude::*;

test_cmonoid!(
    cmonoid_n,
    N,
    arb_z().prop_map(|x| N(x.0.abs().try_into().unwrap()))
);
test_lie_group!(
    lie_group_z,
    Z<Coords<R64, 0>>,
    Coords<R64, 0>,
    arb_z().prop_map(|z| Z::<Coords<R64, 0>>::new(z.0)),
    arb_vec::<0>(),
    arb_scalar()
);
test_ring!(ring_z, Z<_>, arb_z());
