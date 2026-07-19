#![cfg(feature = "testing")]

#[macro_use]
mod common;

use common::*;

use diffable::{
    complex::Complex,
    coords::Coords,
    epsilon_metric::R64,
    spacetime::{Lorentz, Minkowski, Sl, Sl2c, SlAlgebra},
    test_group, test_quotient, test_tangent_bundle, test_vector,
    traits::{LieGroup, Quotient},
};

use proptest::prelude::*;

prop_compose! {
    pub fn arb_sl_algebra_2c()(
        e01_re in -6.0f64..6.00f64, e01_im in -6.0f64..6.00f64,
        e10_re in -6.0f64..6.00f64, e10_im in -6.0f64..6.00f64,
        h_re in -6.0f64..6.00f64, h_im in -6.0f64..6.00f64,
    ) -> SlAlgebra<Complex<R64>, 2, 3> {
        let c = |re, im| Complex::from([R64(re), R64(im)]);
        [c(e01_re, e01_im), c(e10_re, e10_im), c(h_re, h_im)].into()
    }
}

pub fn arb_sl2c() -> impl Strategy<Value = Sl2c<R64>> {
    arb_sl_algebra_2c().prop_map(Sl::identity_exp)
}

test_vector!(
    minkowski,
    R64,
    Minkowski<_>,
    arb_vec::<4>().prop_map(|x| <Coords<R64, 4, 0> as Into<[R64; 4]>>::into(x).into()),
    arb_scalar()
);

test_group!(lie_group_sl2c, Sl2c<R64>, arb_sl2c());
test_tangent_bundle!(
    tangent_bundle_sl2c,
    Complex<R64>,
    Sl2c<R64>,
    arb_sl2c(),
    arb_sl_algebra_2c(),
    arb_vec::<2>().prop_map(|x| Complex::<R64>::from(x))
);

test_quotient!(
    quotient_lorentz,
    Lorentz<R64>,
    arb_sl2c().prop_map(|x| Lorentz::new(x)),
    arb_sl2c(),
    arb_root_of_unity()
);
test_tangent_bundle!(
    tangent_bundle_lorentz,
    Complex<R64>,
    Lorentz<R64>,
    arb_sl2c().prop_map(|x| Lorentz::new(x)),
    arb_sl_algebra_2c(),
    arb_vec::<2>().prop_map(|x| Complex::<R64>::from(x))
);
