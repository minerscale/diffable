#![cfg(feature = "testing")]

#[macro_use]
mod common;

use common::*;

use diffable::{
    complex::Complex,
    coords::Coords,
    epsilon_metric::R64,
    spacetime::{Minkowski, Sl},
    test_pseudo_euclidean,
};
use num_traits::One;

use proptest::prelude::*;

test_pseudo_euclidean!(
    minkowski,
    R64,
    Minkowski<_>,
    arb_vec::<4>().prop_map(|x| <Coords<R64, 4, 0> as Into<[R64; 4]>>::into(x).into()),
    arb_scalar()
);

#[test]
fn sl_mul() {
    let x = Sl::<2, Complex<R64>>::one();

    println!("{:?}", x);

    let _ = x * x.clone();
}
