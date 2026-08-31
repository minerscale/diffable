#![cfg(feature = "testing")]

#[macro_use]
mod common;

use common::*;

use diffable::{
    complex::Complex, coords::Coords, epsilon_metric::R64, test_cfield, test_euclidean,
    test_pseudo_euclidean, test_vector, test_holonomy, spacetime::Minkowski
};

use proptest::prelude::*;

// Ensure the underlying scalar is a field
test_cfield!(field_r64, R64, arb_scalar(), arb_scalar());

// Ensure that the space is actually euclidean
test_euclidean!(euclidian_v0, Coords<_, _>, arb_vec::<0>(), arb_scalar());
test_euclidean!(euclidian_v1, Coords<_, _>, arb_vec::<1>(), arb_scalar());
test_euclidean!(euclidian_v2, Coords<_, _>, arb_vec::<2>(), arb_scalar());
test_euclidean!(euclidian_v3, Coords<_, _>, arb_vec::<3>(), arb_scalar());

test_pseudo_euclidean!(pseudo_euclidean_v21, Coords<_, _, 1>, arb_vec::<2>().prop_map(|v| {
	let c: [_;_] = v.into();
	c.into()
}), arb_scalar());
test_pseudo_euclidean!(pseudo_euclidean_v31, Coords<_, _, 1>, arb_vec::<3>().prop_map(|v| {
	let c: [_;_] = v.into();
	c.into()
}), arb_scalar());
test_pseudo_euclidean!(pseudo_euclidean_v32, Coords<_, _, 2>, arb_vec::<3>().prop_map(|v| {
	let c: [_;_] = v.into();
	c.into()
}), arb_scalar());

test_vector!(complex_v0, Coords<_, _>, (arb_vec::<0>(), arb_vec::<0>()).prop_map(|(a,b)| {
	let arr_a: [_; _] = a.into();
	let arr_b: [_; _] = b.into();
	Coords::<Complex::<_>, 0>::from_fn(|i| [arr_a[i], arr_b[i]].into())
}), arb_vec::<2>().prop_map(|x| Complex(x)));

test_vector!(complex_v1, Coords<_, _>, (arb_vec::<1>(), arb_vec::<1>()).prop_map(|(a,b)| {
	let arr_a: [_; _] = a.into();
	let arr_b: [_; _] = b.into();
	Coords::<Complex::<_>, 1>::from_fn(|i| [arr_a[i], arr_b[i]].into())
}), arb_vec::<2>().prop_map(|x| Complex(x)));

test_vector!(complex_v2, Coords<_, _>, (arb_vec::<2>(), arb_vec::<2>()).prop_map(|(a,b)| {
	let arr_a: [_; _] = a.into();
	let arr_b: [_; _] = b.into();
	Coords::<Complex::<_>, 2>::from_fn(|i| [arr_a[i], arr_b[i]].into())
}), arb_vec::<2>().prop_map(|x| Complex(x)));

test_vector!(complex_v3, Coords<_, _>, (arb_vec::<3>(), arb_vec::<3>()).prop_map(|(a,b)| {
	let arr_a: [_; _] = a.into();
	let arr_b: [_; _] = b.into();
	Coords::<Complex::<_>, 3>::from_fn(|i| [arr_a[i], arr_b[i]].into())
}), arb_vec::<2>().prop_map(|x| Complex(x)));

test_vector!(complex_v21, Coords<_, _, 1>, (arb_vec::<2>(), arb_vec::<2>()).prop_map(|(a,b)| {
	let arr_a: [_; _] = a.into();
	let arr_b: [_; _] = b.into();
	Coords::<Complex::<_>, 2, _>::from_fn(|i| [arr_a[i], arr_b[i]].into())
}), arb_vec::<2>().prop_map(|x| Complex(x)));

test_vector!(complex_v31, Coords<_, _, 1>, (arb_vec::<3>(), arb_vec::<3>()).prop_map(|(a,b)| {
	let arr_a: [_; _] = a.into();
	let arr_b: [_; _] = b.into();
	Coords::<Complex::<_>, 3, _>::from_fn(|i| [arr_a[i], arr_b[i]].into())
}), arb_vec::<2>().prop_map(|x| Complex(x)));

test_vector!(complex_v32, Coords<_, _, 2>, (arb_vec::<3>(), arb_vec::<3>()).prop_map(|(a,b)| {
	let arr_a: [_; _] = a.into();
	let arr_b: [_; _] = b.into();
	Coords::<Complex::<_>, 3, _>::from_fn(|i| [arr_a[i], arr_b[i]].into())
}), arb_vec::<2>().prop_map(|x| Complex(x)));

test_holonomy!(
    holonomy_euclidean2,
    Coords<R64, 2>,
    Coords<R64, 2>,
    arb_vec2(),
    arb_vec2()
);

test_holonomy!(
    holonomy_minkowski,
    Minkowski<R64>,
    Minkowski<R64>,
    arb_vec4().prop_map(|x| {
        let values: [R64; 4] = x.into();
        Minkowski::from(values)
    }),
    arb_vec4().prop_map(|x| {
        let values: [R64; 4] = x.into();
        values.into()
    })
);
