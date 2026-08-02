#![cfg(feature = "testing")]

use std::ops::Mul;

use diffable::{
    complex::Complex,
    coords::Coords,
    traits::{Euclidean, Field, Tensor, d},
};
use num_traits::Zero;

#[test]
fn differential_of_scalar_linear_function() {
    fn double<V: Euclidean>(x: V) -> V {
        x * V::F::from_nat(2)
    }

    fn double_v<V: Vector>(x: V) -> V {
        x * V::F::from_nat(2)
    }

    let derivative = d(double).at(0.0.into());
    let derivative_v = d(double_v).at(Complex::zero().into());

    assert_eq!(derivative[0], 2.0);
    assert_eq!(derivative_v[0], Complex::from_fixed(2.0));
}

#[test]
fn differential_computes_a_jacobian() {
    type V = Coords<f64, 2>;

    // f(x, y) = (x² + y, xy)
    fn f<V: Euclidean>(v: V) -> V {
        V::from_iter([v[0] * v[0] + v[1], v[0] * v[1]])
    }

    let jacobian = d(f).at([2.0, 3.0].into());

    // J_f(x,y) = [ 2x  1 ]
    //            [  y  x ]
    //
    // J_f(2,3) = [ 4  1 ]
    //            [ 3  2 ]

    assert_eq!(jacobian[0], 4.0);
    assert_eq!(jacobian[1], 1.0);
    assert_eq!(jacobian[2], 3.0);
    assert_eq!(jacobian[3], 2.0);
}

#[test]
fn jacobian_uses_output_by_input_orientation() {
    type V = Coords<f64, 2>;

    // f(x,y) = (2x + 3y, 5x + 7y)
    fn f<V: Euclidean>(v: V) -> V {
        let to = |n| V::F::from_nat(n);

        V::from_iter([v[0] * to(2) + v[1] * to(3), v[0] * to(5) + v[1] * to(7)])
    }

    let jacobian = d(f).at([11.0, 13.0].into());

    assert_eq!(jacobian[0], 2.0);
    assert_eq!(jacobian[1], 3.0);
    assert_eq!(jacobian[2], 5.0);
    assert_eq!(jacobian[3], 7.0);
}
