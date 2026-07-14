#![cfg(feature = "testing")]

#[macro_use]
mod common;

use common::*;

use diffable::{
    complex::Complex, coords::Coords, epsilon_metric::R64, test_euclidean, test_involutive_field,
    test_pseudo_euclidean, test_quadratic, test_real_structure,
};

use proptest::prelude::*;

// Ensure the underlying scalar is a field
test_involutive_field!(field_r64, R64, arb_scalar(), arb_scalar());
test_real_structure!(real_structure_v0, Coords<_, _>, arb_vec::<0>(), arb_scalar());
test_real_structure!(real_structure_v1, Coords<_, _>, arb_vec::<1>(), arb_scalar());
test_real_structure!(real_structure_v2, Coords<_, _>, arb_vec::<2>(), arb_scalar());
test_real_structure!(real_structure_v3, Coords<_, _>, arb_vec::<3>(), arb_scalar());
test_real_structure!(real_structure_v21, Coords<_, _, 1>, arb_vec::<2>().prop_map(|v| {
	let c: [_;_] = v.into();
	c.into()
}), arb_scalar());
test_real_structure!(real_structure_v31, Coords<_, _, 1>, arb_vec::<3>().prop_map(|v| {
	let c: [_;_] = v.into();
	c.into()
}), arb_scalar());
test_real_structure!(real_structure_v32, Coords<_, _, 2>, arb_vec::<3>().prop_map(|v| {
	let c: [_;_] = v.into();
	c.into()
}), arb_scalar());

// Ensure that the space is actually euclidean
test_euclidean!(euclidian_v0, R64, Coords<_, _>, arb_vec::<0>(), arb_scalar());
test_euclidean!(euclidian_v1, R64, Coords<_, _>, arb_vec::<1>(), arb_scalar());
test_euclidean!(euclidian_v2, R64, Coords<_, _>, arb_vec::<2>(), arb_scalar());
test_euclidean!(euclidian_v3, R64, Coords<_, _>, arb_vec::<3>(), arb_scalar());

test_pseudo_euclidean!(pseudo_euclidean_v21, R64, Coords<_, _, 1>, arb_vec::<2>().prop_map(|v| {
	let c: [_;_] = v.into();
	c.into()
}), arb_scalar());
test_pseudo_euclidean!(pseudo_euclidean_v31, R64, Coords<_, _, 1>, arb_vec::<3>().prop_map(|v| {
	let c: [_;_] = v.into();
	c.into()
}), arb_scalar());
test_pseudo_euclidean!(pseudo_euclidean_v32, R64, Coords<_, _, 2>, arb_vec::<3>().prop_map(|v| {
	let c: [_;_] = v.into();
	c.into()
}), arb_scalar());

test_quadratic!(complex_v0, Complex<R64>, Coords<_, _>, (arb_vec::<0>(), arb_vec::<0>()).prop_map(|(a,b)| {
	let arr_a: [_; _] = a.into();
	let arr_b: [_; _] = b.into();
	Coords::<Complex::<_>, 0>::from_fn(|i| [arr_a[i], arr_b[i]].into())
}), arb_vec::<2>().prop_map(|x| Complex(x)));

test_quadratic!(complex_v1, Complex<R64>, Coords<_, _>, (arb_vec::<1>(), arb_vec::<1>()).prop_map(|(a,b)| {
	let arr_a: [_; _] = a.into();
	let arr_b: [_; _] = b.into();
	Coords::<Complex::<_>, 1>::from_fn(|i| [arr_a[i], arr_b[i]].into())
}), arb_vec::<2>().prop_map(|x| Complex(x)));

test_quadratic!(complex_v2, Complex<R64>, Coords<_, _>, (arb_vec::<2>(), arb_vec::<2>()).prop_map(|(a,b)| {
	let arr_a: [_; _] = a.into();
	let arr_b: [_; _] = b.into();
	Coords::<Complex::<_>, 2>::from_fn(|i| [arr_a[i], arr_b[i]].into())
}), arb_vec::<2>().prop_map(|x| Complex(x)));

test_quadratic!(complex_v3, Complex<R64>, Coords<_, _>, (arb_vec::<3>(), arb_vec::<3>()).prop_map(|(a,b)| {
	let arr_a: [_; _] = a.into();
	let arr_b: [_; _] = b.into();
	Coords::<Complex::<_>, 3>::from_fn(|i| [arr_a[i], arr_b[i]].into())
}), arb_vec::<2>().prop_map(|x| Complex(x)));

test_quadratic!(complex_v21, Complex<R64>, Coords<_, _, 1>, (arb_vec::<2>(), arb_vec::<2>()).prop_map(|(a,b)| {
	let arr_a: [_; _] = a.into();
	let arr_b: [_; _] = b.into();
	Coords::<Complex::<_>, 2, _>::from_fn(|i| [arr_a[i], arr_b[i]].into())
}), arb_vec::<2>().prop_map(|x| Complex(x)));

test_quadratic!(complex_v31, Complex<R64>, Coords<_, _, 1>, (arb_vec::<3>(), arb_vec::<3>()).prop_map(|(a,b)| {
	let arr_a: [_; _] = a.into();
	let arr_b: [_; _] = b.into();
	Coords::<Complex::<_>, 3, _>::from_fn(|i| [arr_a[i], arr_b[i]].into())
}), arb_vec::<2>().prop_map(|x| Complex(x)));

test_quadratic!(complex_v32, Complex<R64>, Coords<_, _, 2>, (arb_vec::<3>(), arb_vec::<3>()).prop_map(|(a,b)| {
	let arr_a: [_; _] = a.into();
	let arr_b: [_; _] = b.into();
	Coords::<Complex::<_>, 3, _>::from_fn(|i| [arr_a[i], arr_b[i]].into())
}), arb_vec::<2>().prop_map(|x| Complex(x)));
