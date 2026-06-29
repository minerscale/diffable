#![cfg(feature = "testing")]

#[macro_use]
mod common;

use common::*;

use diffable::{
    coords::Coords,
    hypersphere::{Sphere, SphereExpMap, Stereographic},
    test_chart, test_exp_map, test_exp_map_lie_group, test_lie_group, test_metric,
    test_tangent_bundle,
    traits::{Chart, ExpMap, LeftTranslationChart, LieGroup, Metric, TangentBundle},
};

use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Instantiations — adding a new manifold is one line per trait
// ---------------------------------------------------------------------------

// Stereographic chart roundtrips
test_chart!(stereo_s0, Stereographic<_, _>, Sphere<_, _>, arb_sphere0());
test_chart!(stereo_s1, Stereographic<_, _>, Sphere<_, _>, arb_sphere1());
test_chart!(stereo_s3, Stereographic<_, _>, Sphere<_, _>, arb_sphere3());

// SphereExpMap
test_exp_map_lie_group!(sphere_exp_map_s0, SphereExpMap<_, Coords<f64, _>>, Sphere<_, _>, 0, arb_sphere0(), arb_vec0());
test_exp_map_lie_group!(sphere_exp_map_s1, SphereExpMap<_, Coords<f64, _>>, Sphere<_, _>, 1, arb_sphere1(), arb_vec1());
test_exp_map_lie_group!(sphere_exp_map_s3, SphereExpMap<_, Coords<f64, _>>, Sphere<_, _>, 3, arb_sphere3(), arb_vec3());

// GeodesicChart as TangentBundle (includes all ExpMap tests)
test_tangent_bundle!(geodesic_chart_s0, LeftTranslationChart<_,_,_,SphereExpMap<_, Coords<_, _>>>, Sphere<_, _>, arb_sphere0(), arb_vec0());
test_tangent_bundle!(geodesic_chart_s1, LeftTranslationChart<_,_,_,SphereExpMap<_, Coords<_, _>>>, Sphere<_, _>, arb_sphere1(), arb_vec1());
test_tangent_bundle!(geodesic_chart_s3, LeftTranslationChart<_,_,_,SphereExpMap<_, Coords<_, _>>>, Sphere<_, _>, arb_sphere3(), arb_vec3());

// Lie group axioms
test_lie_group!(lie_group_s0, Sphere<_, _>, arb_sphere0());
test_lie_group!(lie_group_s1, Sphere<_, _>, arb_sphere1());
test_lie_group!(lie_group_s3, Sphere<_, _>, arb_sphere3());

// Metric axioms
test_metric!(metric_s0, Sphere<_, _>, arb_sphere0());
test_metric!(metric_s1, Sphere<_, _>, arb_sphere1());
test_metric!(metric_s3, Sphere<_, _>, arb_sphere3());

// ---------------------------------------------------------------------------
// Bespoke tests: properties specific to these manifolds, not general laws
// ---------------------------------------------------------------------------

proptest! {
    // S^1 is abelian, so exp is a group homomorphism: exp(v+w) = exp(v) * exp(w)
    #[test]
    fn s1_exp_homomorphism(v in -1.5f64..1.5f64, w in -1.5f64..1.5f64) {
        let chart = SphereExpMap::new(Stereographic::south_pole());
        let ev: Coords<f64, 1> = [v].into();
        let ew: Coords<f64, 1> = [w].into();
        let evw: Coords<f64, 1> = [v + w].into();
        prop_assert!(
            chart.to_global(ev).compose(&chart.to_global(ew))
                .within(&chart.to_global(evw), EPSILON)
        );
    }

    // S^1 is abelian: composition commutes
    #[test]
    fn s1_commutativity(a in arb_sphere1(), b in arb_sphere1()) {
        prop_assert!(a.compose(&b).within(&b.compose(&a), EPSILON));
    }
}
