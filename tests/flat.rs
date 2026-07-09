#![cfg(feature = "testing")]

#[macro_use]
mod common;

use common::*;
use diffable::{
    coords::Coords,
    epsilon_metric::R64,
    flat::{KleinBottle, KleinBottleCover, S1, Torus, TorusCover},
    group_presentation, test_quotient, test_riemannian, test_tangent_bundle,
    traits::{Chart, Group, GroupPresentation, NerveComplex, Quotient},
};
use proptest::prelude::*;

use num_traits::Euclid;

test_tangent_bundle!(
    tangent_bundle_s1_quotient,
    R64,
    S1<Coords<_, _>>,
    arb_s1_quotient(),
    arb_vec::<1>()
);
test_quotient!(
    quotient_s1,
    S1<Coords<_, _>>,
    arb_s1_quotient(),
    arb_vec1(),
    arb_z()
);
test_riemannian!(
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
    arb_vec2()
);
test_riemannian!(
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
    arb_vec2()
);
// ($mod_name:ident, $chart:ty, $arb_point:expr, $arb_vec:expr)
test_riemannian!(
    riemannian_klein_bottle,
    KleinBottle<Coords<R64, 1>, Coords<R64, 2>>,
    (arb_s1_quotient(), arb_s1_quotient()).prop_map(|(a, b)| KleinBottle::new(a, b)),
    arb_vec2()
);

#[test]
fn modulo() {
    let x = R64(12.4);
    let k: S1<Coords<R64, 1>> = S1::new([x].into());
    let sum = k.compose(&k).compose(&k);

    println!("{:?}", sum);

    assert_eq!((sum).lift(), [(x + x + x).rem_euclid(&R64(1.0))].into());
}

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
