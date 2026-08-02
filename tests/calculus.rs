#![cfg(feature = "testing")]

use std::ops::Mul;

use diffable::{
    coords::Coords,
    traits::{Field, Vector, d},
};

#[test]
fn differential_of_scalar_linear_function() {
    fn double<V: Vector + Mul<V::F, Output = V>>(x: V) -> V {
        x * V::F::from_nat(2)
    }

    let derivative = d(double).at(Coords::from(0.0));

    assert_eq!(derivative[0], 2.0);
}

#[test]
fn differential_computes_a_jacobian() {
    type V = Coords<f64, 2>;

    // f(x, y) = (x² + y, xy)
    fn f<V: Vector>(v: V) -> V {
        V::from_iter([v[0] * v[0] + v[1], v[0] * v[1]])
    }

    let jacobian = d(f).at(V::from_iter([2.0, 3.0]));

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
    fn f<V: Vector>(v: V) -> V {
        let to = |n| V::F::from_nat(n);

        V::from_iter([v[0] * to(2) + v[1] * to(3), v[0] * to(5) + v[1] * to(7)])
    }

    let jacobian = d(f).at(V::from_iter([11.0, 13.0]));

    assert_eq!(jacobian[0], 2.0);
    assert_eq!(jacobian[1], 3.0);
    assert_eq!(jacobian[2], 5.0);
    assert_eq!(jacobian[3], 7.0);
}
