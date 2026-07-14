#![cfg(feature = "testing")]

#[macro_use]
mod common;

use common::*;

use diffable::{coords::Coords, epsilon_metric::R64, test_euclidean, test_field};

use proptest::prelude::*;

// Ensure the underlying scalar is a field
test_field!(field_r64, R64, arb_scalar());

// Ensure that the space is actually euclidean
test_euclidean!(euclidian_v0, R64, Coords<_, _>, arb_vec::<0>(), arb_scalar());
test_euclidean!(euclidian_v1, R64, Coords<_, _>, arb_vec::<1>(), arb_scalar());
test_euclidean!(euclidian_v2, R64, Coords<_, _>, arb_vec::<2>(), arb_scalar());
test_euclidean!(euclidian_v3, R64, Coords<_, _>, arb_vec::<3>(), arb_scalar());
