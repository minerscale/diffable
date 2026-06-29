#![cfg(feature = "testing")]

#[macro_use]
mod common;

use common::*;

use diffable::{
    coords::Coords,
    test_chart, test_euclidean, test_exp_map, test_exp_map_lie_group, test_lie_group, test_metric,
    test_tangent_bundle,
    traits::{Chart, Euclidean, ExpMap, LeftTranslationChart, LieGroup, Metric, TangentBundle},
};

use proptest::prelude::*;

// Ensure that the space is actually euclidean
test_euclidean!(euclidian_v0, Coords<_, _>, arb_vec::<0>(), arb_vec::<0>());
test_euclidean!(euclidian_v1, Coords<_, _>, arb_vec::<1>(), arb_vec::<1>());
test_euclidean!(euclidian_v2, Coords<_, _>, arb_vec::<2>(), arb_vec::<2>());
test_euclidean!(euclidian_v3, Coords<_, _>, arb_vec::<3>(), arb_vec::<3>());

// Stereographic chart roundtrips
test_chart!(chart_v0, Coords<_, _>, Coords<_, _>, arb_vec::<0>());
test_chart!(chart_v1, Coords<_, _>, Coords<_, _>, arb_vec::<1>());
test_chart!(chart_v2, Coords<_, _>, Coords<_, _>, arb_vec::<2>());
test_chart!(chart_v3, Coords<_, _>, Coords<_, _>, arb_vec::<3>());

// SphereExpMap
test_exp_map_lie_group!(exp_map_v0, Coords<f64, _>, Coords<_, _>, 0, arb_vec::<0>(), arb_vec::<0>());
test_exp_map_lie_group!(exp_map_v1, Coords<f64, _>, Coords<_, _>, 1, arb_vec::<1>(), arb_vec::<1>());
test_exp_map_lie_group!(exp_map_v2, Coords<f64, _>, Coords<_, _>, 2, arb_vec::<2>(), arb_vec::<2>());
test_exp_map_lie_group!(exp_map_v3, Coords<f64, _>, Coords<_, _>, 3, arb_vec::<3>(), arb_vec::<3>());

// LeftTranslationChart as TangentBundle
test_tangent_bundle!(geodesic_chart_v0, LeftTranslationChart<_, Coords<_, _>, Coords<_, _>, Coords<_, _>>, Coords<_, _>, arb_vec::<0>(), arb_vec::<0>());
test_tangent_bundle!(geodesic_chart_v1, LeftTranslationChart<_, Coords<_, _>, Coords<_, _>, Coords<_, _>>, Coords<_, _>, arb_vec::<1>(), arb_vec::<1>());
test_tangent_bundle!(geodesic_chart_v2, LeftTranslationChart<_, Coords<_, _>, Coords<_, _>, Coords<_, _>>, Coords<_, _>, arb_vec::<2>(), arb_vec::<2>());
test_tangent_bundle!(geodesic_chart_v3, LeftTranslationChart<_, Coords<_, _>, Coords<_, _>, Coords<_, _>>, Coords<_, _>, arb_vec::<3>(), arb_vec::<3>());

// Lie group axioms
test_lie_group!(lie_group_v0, Coords<f64, 0>, arb_vec::<0>());
test_lie_group!(lie_group_v1, Coords<f64, 1>, arb_vec::<1>());
test_lie_group!(lie_group_v2, Coords<f64, 2>, arb_vec::<2>());

// Metric axioms
test_metric!(metric_v0, Coords<_, _>, arb_vec::<0>());
test_metric!(metric_v1, Coords<_, _>, arb_vec::<1>());
test_metric!(metric_v2, Coords<_, _>, arb_vec::<2>());
