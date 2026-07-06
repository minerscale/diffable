#![cfg(feature = "testing")]

#[macro_use]
mod common;

use common::*;

use diffable::{
    coords::Coords,
    epsilon_metric::R64,
    test_chart, test_euclidean, test_exp_map, test_group, test_inner_product, test_metric,
    test_monoid, test_riemannian, test_tangent_bundle,
    traits::{
        Chart, CMonoid, Euclidean, ExpMap, Group, InnerProduct, Metric, Riemannian,
        TangentBundle,
    },
};

use proptest::prelude::*;

// Ensure that the space is actually euclidean
test_euclidean!(euclidian_v0, R64, Coords<_, _>, arb_vec::<0>(), arb_vec::<0>(), arb_scalar());
test_euclidean!(euclidian_v1, R64, Coords<_, _>, arb_vec::<1>(), arb_vec::<1>(), arb_scalar());
test_euclidean!(euclidian_v2, R64, Coords<_, _>, arb_vec::<2>(), arb_vec::<2>(), arb_scalar());
test_euclidean!(euclidian_v3, R64, Coords<_, _>, arb_vec::<3>(), arb_vec::<3>(), arb_scalar());
