#![cfg(feature = "testing")]

#[macro_use]
mod common;

use common::*;
use diffable::{
    discrete::{N, Z},
    epsilon_metric::R64,
    test_cmonoid, test_group, test_tangent_bundle,
    traits::Chart,
};
use proptest::prelude::*;

test_cmonoid!(
    cmonoid_n,
    N,
    arb_z().prop_map(|x| N(x.0.abs().try_into().unwrap()))
);
test_group!(group_z, Z<_>, arb_z());
test_tangent_bundle!(tangent_bundle_z, R64, Z<_>, arb_z(), arb_vec::<1>());
