#![cfg(feature = "testing")]

#[macro_use]
mod common;

use common::*;

use diffable::{
    complex::Complex, epsilon_metric::R64, test_euclidean, test_field, test_pseudo_riemannian,
    test_tangent_bundle, traits::NonZero,
};

use proptest::prelude::*;

test_euclidean!(
    complex_euclidean,
    R64,
    Complex<_>,
    arb_vec::<2>().prop_map(|x| x.into()),
    arb_scalar()
);
test_field!(
    complex_ring,
    Complex<R64>,
    arb_vec::<2>().prop_map(|x| Complex::<R64>::from(x))
);
test_pseudo_riemannian!(
    complex_mul,
    NonZero<Complex<R64>>,
    arb_vec::<2>().prop_filter_map("must be nonzero", |x| NonZero::new(Complex::<R64>::from(x))),
    arb_vec::<2>().prop_map(|x| Complex::<R64>::from(x))
);
test_tangent_bundle!(
    complex_exp_log,
    R64,
    NonZero<Complex<R64>>,
    arb_vec::<2>().prop_filter_map("must be nonzero", |x| NonZero::new(Complex::<R64>::from(x))),
    arb_vec::<2>().prop_map(|x| Complex::<R64>::from(x))
);
