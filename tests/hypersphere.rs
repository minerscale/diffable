#![cfg(feature = "testing")]

#[macro_use]
mod common;

use common::*;

use diffable::{
    coords::Coords,
    hypersphere::{So3, Sphere, Stereographic},
    test_chart, test_exp_map, test_lie_group, test_metric, test_tangent_bundle,
    traits::{Chart, ExpMap, LieGroup, Metric, TangentBundle},
};

use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Instantiations — adding a new manifold is one line per trait
// ---------------------------------------------------------------------------

// Stereographic chart roundtrips
test_chart!(stereo_s0, Stereographic<_>, arb_sphere0());
test_chart!(stereo_s1, Stereographic<_>, arb_sphere1());
test_chart!(stereo_s3, Stereographic<_>, arb_sphere3());

// Sphere as TangentBundle (via blanket LieGroup impl; includes all ExpMap tests)
test_tangent_bundle!(tangent_bundle_s0, Sphere<_, _>, Sphere<_, _>, arb_sphere0(), arb_vec0());
test_tangent_bundle!(tangent_bundle_s1, Sphere<_, _>, Sphere<_, _>, arb_sphere1(), arb_vec1());
test_tangent_bundle!(tangent_bundle_s3, Sphere<_, _>, Sphere<_, _>, arb_sphere3(), arb_vec3());

// Lie group axioms
test_lie_group!(lie_group_s0, Sphere<_, _>, arb_sphere0());
test_lie_group!(lie_group_s1, Sphere<_, _>, arb_sphere1());
test_lie_group!(lie_group_s3, Sphere<_, _>, arb_sphere3());

// Metric axioms
test_metric!(metric_s0, Sphere<_, _>, arb_sphere0());
test_metric!(metric_s1, Sphere<_, _>, arb_sphere1());
test_metric!(metric_s3, Sphere<_, _>, arb_sphere3());

// Lie group axioms
test_lie_group!(lie_group_so3, So3<Coords<_, _>>, arb_so3());
test_tangent_bundle!(tangent_bundle_so3, So3<Coords<_, _>>, So3<Coords<_, _>>, arb_so3(), arb_vec3());

// ---------------------------------------------------------------------------
// Bespoke tests: properties specific to these manifolds, not general laws
// ---------------------------------------------------------------------------
proptest! {
    // S^1 is abelian, so exp is a group homomorphism: exp(v+w) = exp(v) * exp(w)
    #[test]
    fn s1_exp_homomorphism(v in -1.5f64..1.5f64, w in -1.5f64..1.5f64) {
        let chart = Sphere::<1, _>::identity();
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
