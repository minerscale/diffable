#![cfg(feature = "testing")]

#[macro_use]
mod common;

use common::*;

use diffable::{
    complex::Complex,
    epsilon_metric::R64,
    test_cfield, test_metric, test_pseudo_riemannian, test_tangent_bundle,
    traits::{NonZero, Symmetrized},
};

use num_traits::Zero;
use proptest::prelude::*;

test_cfield!(
    complex_field,
    Complex<R64>,
    arb_vec::<2>().prop_map(|x| Complex::<R64>::from(x)),
    arb_scalar()
);
test_metric!(
    complex_metric,
    Complex<R64>,
    arb_vec::<2>().prop_map(|x| Complex::<R64>::from(x))
);

test_cfield!(
    symmetrized_complex_field,
    Symmetrized<Complex<R64>>,
    arb_vec::<2>().prop_map(|x| Symmetrized(Complex::<R64>::from(x))),
    arb_vec::<2>().prop_map(|x| Symmetrized(Complex::<R64>::from(x)))
);
test_metric!(
    symmetrized_complex_metric,
    Complex<R64>,
    arb_vec::<2>().prop_map(|x| Complex::<R64>::from(x))
);

test_pseudo_riemannian!(
    complex_mul,
    NonZero<Complex<R64>>,
    arb_vec::<2>().prop_filter_map("must be nonzero", |x| NonZero::new(Complex::<R64>::from(x))),
    arb_vec::<2>()
);
test_tangent_bundle!(
    complex_exp_log,
    NonZero<Complex<R64>>,
    arb_vec::<2>().prop_filter_map("must be nonzero", |x| NonZero::new(Complex::<R64>::from(x)))
);

proptest! {
    #[test]
    fn real_embedding_preserves_multiplication(
        x in arb_scalar(),
        y in arb_scalar(),
    ) {
        prop_assert_eq!(
            Complex::from(x * y),
            Complex::from(x) * Complex::from(y),
        );
    }

    #[test]
    fn real_embedding_preserves_addition(
        x in arb_scalar(),
        y in arb_scalar(),
    ) {
        prop_assert_eq!(
            Complex::from(x + y),
            Complex::from(x) + Complex::from(y),
        );
    }

    #[test]
    fn real_embedding_is_injective(r in arb_scalar()) {
        let embedded = Complex::from(r);
        let [re, im] = embedded.into();

        prop_assert_eq!(re, r);
        prop_assert_eq!(im, R64::zero());
    }
}
