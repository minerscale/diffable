#![cfg(feature = "testing")]

#[macro_use]
mod common;

use common::*;
use diffable::{
    coords::Coords,
    epsilon_metric::R64,
    flat::{KleinBottle, KleinBottleCover, MyopicTorus, MyopicTorusCover, S1, Torus, TorusCover},
    group_presentation, test_pseudo_riemannian, test_quotient, test_tangent_bundle,
    traits::{Chart, GroupPresentation, InnerProduct, NerveComplex, NerveComplexParameters, Nodes},
};

use proptest::prelude::*;

test_tangent_bundle!(
    tangent_bundle_s1_quotient,
    R64,
    S1<Coords<_, _>>,
    arb_s1_quotient(),
    arb_vec::<1>(),
    arb_scalar()
);
test_quotient!(
    quotient_s1,
    S1<Coords<_, _>>,
    arb_s1_quotient(),
    arb_vec1(),
    arb_z()
);
test_pseudo_riemannian!(
    riemannian_s1,
    S1<Coords<_, _>>,
    arb_s1_quotient(),
    arb_vec1()
);

// ($mod_name:ident, $scalar:ty, $chart:ty, $point:ty, $arb_point:expr, $arb_vec:expr)
test_tangent_bundle!(
    torus_tangent_bundle,
    R64,
    Torus<Coords<R64, 1>, Coords<R64, 2>>,
    (arb_s1_quotient(), arb_s1_quotient()).prop_map(|(a, b)| Torus::new(a, b)),
    arb_vec2(),
    arb_scalar()
);
test_pseudo_riemannian!(
    riemannian_torus,
    Torus<Coords<R64, 1>, Coords<R64, 2>>,
    (arb_s1_quotient(), arb_s1_quotient()).prop_map(|(a, b)| Torus::new(a, b)),
    arb_vec2()
);

test_tangent_bundle!(
    klein_bottle_tangent_bundle,
    R64,
    KleinBottle<Coords<R64, 1>, Coords<R64, 2>>,
    (arb_s1_quotient(), arb_s1_quotient()).prop_map(|(a, b)| KleinBottle::new(a, b)),
    arb_vec2(),
    arb_scalar()
);
// ($mod_name:ident, $chart:ty, $arb_point:expr, $arb_vec:expr)
test_pseudo_riemannian!(
    riemannian_klein_bottle,
    KleinBottle<Coords<R64, 1>, Coords<R64, 2>>,
    (arb_s1_quotient(), arb_s1_quotient()).prop_map(|(a, b)| KleinBottle::new(a, b)),
    arb_vec2()
);

#[test]
fn klein_bottle_fundamental_group() {
    let presentation = KleinBottleCover::<Coords<R64, 1>, Coords<R64, 2>>::fundamental_group();

    group_presentation!(
        KLEIN_BOTTLE,
        n_generators = 2,
        relations = [[(0, false), (0, false), (1, false), (1, false)],]
    );

    assert!(
        presentation.check_exactly_equal(&KLEIN_BOTTLE),
        "Expected: {:?}\nActual: {:?}",
        presentation,
        KLEIN_BOTTLE
    );
}

#[test]
fn torus_fundamental_group() {
    let presentation = TorusCover::<Coords<R64, 1>, Coords<R64, 2>>::fundamental_group();

    group_presentation!(
        TORUS,
        n_generators = 2,
        relations = [[(0, false), (1, false), (0, true), (1, true)],]
    );

    assert!(
        presentation.check_exactly_equal(&TORUS),
        "Expected: {:?}\nActual: {:?}",
        presentation,
        TORUS
    );
}

#[test]
fn myopic_torus_cover_invariants() {
    type T = MyopicTorus<Coords<f64, 1>, Coords<f64, 2>>;
    type Cover = MyopicTorusCover<Coords<f64, 1>, Coords<f64, 2>>;
    let s = T::s() as f64;
    let h = 1.0 / s; // lattice spacing
    let delta_s = 2f64.sqrt() / (2.0 * s); // covering radius
    let rho = (2f64.sqrt() + 2.0) / (4.0 * s); // node radius, from `sdf`
    let big_r = 2.0 / s; // myopia radius

    assert!(
        2.0 * rho < big_r,
        "2ρ < R: adjacent base points must see each other"
    );
    assert!(delta_s < rho, "δ_s < ρ: the cover must actually cover");
    assert!(h < big_r, "adjacent base points within myopia radius");

    // `covering_radius()` inverts `C = (1+κ)·2δ_s`. If they disagree, `C` was
    // hardcoded from a different `S` — the bug that cost 512 geodesic flows.
    let recovered = Cover::covering_radius().unwrap();
    assert_eq!(R64(recovered), R64(delta_s));
}

proptest! {
    #[test]
    fn myopic_torus_geodesic(
        p in (arb_s1_quotient_f64(), arb_s1_quotient_f64()).prop_map(
            |(a, b)| Torus::<Coords<f64, 1>, Coords<f64, 2>>::new(a, b)
        ),
        q in (arb_s1_quotient_f64(), arb_s1_quotient_f64()).prop_map(
            |(a, b)| Torus::<Coords<f64, 1>, Coords<f64, 2>>::new(a, b)
        )) {

        type Cover = MyopicTorusCover::<Coords<f64, 1>, Coords<f64, 2>>;

        let n = Cover::nodes().len();
        for i in 0..n {
            let ns: Vec<_> = Cover::get_neighbors(i).collect();
            println!("node {i}: degree {}", ns.len());
            for j in &ns {
                assert!(Cover::edge_weight(i, *j).is_some(), "{i}-{j} invisible");
            }
        }

        let expected_distance = p.to_local(&q).unwrap().norm();
        let Some(diffable::traits::Geodesic {path: _, length, certificate}) =
            Cover::geodesic_path(&p, &q) else {
                panic!("no geodesic found")
            };

        prop_assert!(certificate.is_global(), "not certified as global minimum {certificate:?}");
        prop_assert_eq!(R64(length), R64(expected_distance));
    }
}
