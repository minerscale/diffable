#![cfg(feature = "testing")]

use diffable::{
    complex::Complex,
    coords::Coords,
    traits::{Euclidean, Field, Form, Sinister, Tensor, Vector, d},
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

    let derivative = d(double).at(Coords::from(0.0));
    let derivative_v = d(double_v).at(Sinister(Coords::from(Complex::zero()).flat()));

    assert_eq!(derivative[0], 2.0);
    assert_eq!(derivative_v[0], Complex::from_fixed(2.0));
}

#[test]
fn differential_computes_a_multivariate_hessian() {
    // f(x, y) = (x²y, xy²)
    fn f<V: Euclidean>(v: V) -> V {
        let [x, y] = [v[0], v[1]];

        V::from_iter([x * x * y, x * y * y])
    }

    let hessian = d(d(f)).at(Coords([2.0, 3.0]));

    // H(f₀) = [ 2y  2x ] = [ 6  4 ]
    //         [ 2x   0 ]   [ 4  0 ]
    //
    // H(f₁) = [  0  2y ] = [ 0  6 ]
    //         [ 2y  2x ]   [ 6  4 ]
    //
    // The raw tensor layout is:
    //
    //     output × first-input × second-input
    //
    // corresponding to:
    //
    //     (V ⊗ V*) ⊗ V*

    assert!(
        hessian
            .iter()
            .eq(&[6.0, 4.0, 4.0, 0.0, 0.0, 6.0, 6.0, 4.0])
    );
}

#[test]
fn second_derivative_of_square() {
    fn square<V: Euclidean>(x: V) -> V {
        V::from_fn(|_| x[0] * x[0])
    }

    let second = d(d(square)).at(Coords::from(3.0));

    assert_eq!(second[0], 2.0);
}

#[test]
fn third_derivative_of_cube() {
    fn cube<V: Euclidean>(x: V) -> V {
        x.map(|x| x.powi(3))
    }

    let second = d(d(d(cube))).at(Coords::from(-6.0));

    assert_eq!(second[0], 6.0);
}

#[test]
fn differential_computes_a_jacobian() {
    // f(x, y) = (x² + y, xy)
    fn f<V: Euclidean>(v: V) -> V {
        V::from_iter([v[0] * v[0] + v[1], v[0] * v[1]])
    }

    let jacobian = d(f).at(Coords([2.0, 3.0]));

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
    // f(x,y) = (2x + 3y, 5x + 7y)
    fn f<V: Euclidean>(v: V) -> V {
        let to = |n| V::F::from_nat(n);

        V::from_iter([v[0] * to(2) + v[1] * to(3), v[0] * to(5) + v[1] * to(7)])
    }

    let jacobian = d(f).at(Coords([11.0, 13.0]));

    assert_eq!(jacobian[0], 2.0);
    assert_eq!(jacobian[1], 3.0);
    assert_eq!(jacobian[2], 5.0);
    assert_eq!(jacobian[3], 7.0);
}
