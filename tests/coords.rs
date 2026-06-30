#![cfg(feature = "testing")]

#[macro_use]
mod common;

use common::*;

use diffable::{
    coords::Coords,
    test_chart, test_euclidean, test_exp_map, test_inner_product, test_lie_group, test_metric,
    test_tangent_bundle,
    traits::{Chart, Euclidean, ExpMap, InnerProduct, LieGroup, Metric, TangentBundle},
};

use proptest::prelude::*;

// Ensure that the space is actually euclidean
test_euclidean!(euclidian_v0, Coords<_, _>, arb_vec::<0>(), arb_vec::<0>(), arb_scalar());
test_euclidean!(euclidian_v1, Coords<_, _>, arb_vec::<1>(), arb_vec::<1>(), arb_scalar());
test_euclidean!(euclidian_v2, Coords<_, _>, arb_vec::<2>(), arb_vec::<2>(), arb_scalar());
test_euclidean!(euclidian_v3, Coords<_, _>, arb_vec::<3>(), arb_vec::<3>(), arb_scalar());

// Lie group axioms
test_lie_group!(lie_group_v0, Coords<f64, 0>, arb_vec::<0>());
test_lie_group!(lie_group_v1, Coords<f64, 1>, arb_vec::<1>());
test_lie_group!(lie_group_v2, Coords<f64, 2>, arb_vec::<2>());
test_lie_group!(lie_group_v3, Coords<f64, 3>, arb_vec::<3>());
