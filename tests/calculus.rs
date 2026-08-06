#![cfg(feature = "testing")]

use diffable::{
    complex::Complex,
    coords::Coords,
    traits::{
        Dual, Euclidean, Field, Form, Left, Right, Sinister, Tensor, Vector,
        calculus::{Contract, OnLeft, Reassociate, TensorProduct, d},
    },
};
use num_traits::Zero;

#[test]
fn reassociation_is_selected_by_a_tree_path() {
    type A = Coords<f64, 2>;
    type B = Dual<Coords<f64, 3>>;
    type C = Dual<Coords<f64, 5>>;
    type LeftAssociated = TensorProduct<TensorProduct<A, B>, C>;

    let tensor = LeftAssociated::from_fn(|i| i as f64);
    let right = tensor.reassociate();
    let left = right.reassociate();

    assert!(left.iter().copied().eq((0..30).map(|i| i as f64)));

    type Outer = TensorProduct<LeftAssociated, Dual<Coords<f64, 7>>>;
    let tensor = Outer::from_fn(|i| i as f64);

    let reassociated = tensor.reassociate::<OnLeft<Right>>();

    assert!(reassociated.iter().copied().eq((0..210).map(|i| i as f64)));
}

#[test]
fn contraction_is_selected_by_the_same_tree_paths() {
    type V = Coords<f64, 2>;
    type W = Coords<f64, 3>;
    type X = Coords<f64, 5>;

    let endomorphism = TensorProduct::<V, Dual<V>>::from_iter([1.0, 2.0, 3.0, 4.0]);
    let trace = endomorphism.contract();
    assert_eq!(trace, 5.0);

    type Tensor = TensorProduct<TensorProduct<W, Dual<W>>, Dual<X>>;
    let tensor = Tensor::from_fn(|i| i as f64);
    let contracted = tensor.contract();

    // For each X* coordinate k, sum (i, i, k) over the W/W* pair.
    assert!(
        contracted
            .iter()
            .copied()
            .eq([60.0, 63.0, 66.0, 69.0, 72.0])
    );

    type Opposite = TensorProduct<Dual<Sinister<V>>, Sinister<V>>;
    let tensor = Opposite::from_iter([1.0, 2.0, 3.0, 4.0]);
    let trace = tensor.contract();
    assert_eq!(trace, 5.0);

    type OnTheRight = TensorProduct<W, Sinister<TensorProduct<V, Dual<V>>>>;
    let tensor = OnTheRight::from_fn(|i| i as f64);
    let contracted = tensor.contract();
    assert!(contracted.iter().copied().eq([3.0, 11.0, 19.0]));

    type Deep = TensorProduct<TensorProduct<TensorProduct<V, Dual<V>>, Dual<W>>, Dual<X>>;
    let tensor = Deep::from_fn(|i| i as f64);
    let contracted = tensor.contract();

    assert!(
        contracted
            .iter()
            .copied()
            .eq((0..W::N * X::N).map(|i| 45.0 + 2.0 * i as f64))
    );
}

#[test]
fn matrix_multiplication_is_tensor_contraction() {
    type F = f64;
    type V = Coords<F, 2>;
    type Endomorphism = TensorProduct<V, Dual<V>>;
    type Product = TensorProduct<Endomorphism, Sinister<Endomorphism>>;

    let a = Endomorphism::from_iter([1.0, 2.0, 3.0, 4.0]);
    let b = Sinister(Endomorphism::from_iter([5.0, 6.0, 7.0, 8.0]));

    // Coordinates are A[i, j] B[k, l], flattened as (i, j, k, l).
    let tensor = Product::pure(a, b);

    assert!(tensor.iter().copied().eq([
        5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 15.0, 18.0, 21.0, 24.0, 20.0, 24.0, 28.0, 32.0,
    ]));

    // Reassociate and contract j with k:
    //
    // Aⁱⱼ Bᵏₗ  ↦  Aⁱⱼ Bʲₗ.
    let product: Endomorphism = tensor
        .reassociate::<Left>()
        .reassociate::<OnLeft<Right>>()
        .contract();

    let expected = Endomorphism::from_iter([19.0, 22.0, 43.0, 50.0]);

    assert!(product.iter().eq(expected.iter()));
}

#[test]
fn reassociation_exposes_the_other_contraction() {
    type V = Coords<f64, 2>;
    type Tensor = TensorProduct<TensorProduct<V, Dual<V>>, Sinister<V>>;

    let tensor = Tensor::from_fn(|i| i as f64);

    // Contract (V ⊗ V*) first.
    let first_pair = tensor.contract().collapse();

    // Reassociate to V ⊗ (V* ⊗ V), then contract the second pair.
    let second_pair = tensor.reassociate().contract();

    assert!(first_pair.iter().copied().eq([6.0, 8.0]));
    assert!(second_pair.iter().copied().eq([3.0, 11.0]));
}

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

    assert!(hessian.iter().eq(&[6.0, 4.0, 4.0, 0.0, 0.0, 6.0, 6.0, 4.0]));
}

#[test]
fn differential_along_a_direction() {
    // f(x, y) = (x² + y, xy)
    fn f<V: Euclidean>(v: V) -> V {
        V::from_iter([v[0] * v[0] + v[1], v[0] * v[1]])
    }

    let point = Coords([2.0, 3.0]);
    let direction = Coords([5.0, 7.0]);

    let derivative = d(f).along(direction).at(point);

    // J_f(2,3) = [ 4  1 ]
    //            [ 3  2 ]
    //
    // J_f(2,3) · (5,7) = (27,29)

    assert!(derivative.iter().copied().eq([27.0, 29.0]));
}

#[test]
fn directional_derivative_composes_with_differentiation() {
    fn cube<V: Euclidean>(x: V) -> V {
        x.map(|x| x.powi(3))
    }

    let direction = Coords::from(4.0);
    let derivative = d(d(d(cube).along(direction))).at(Coords::from(7.0));

    // D³(x³)[v] = 6v. This sends the captured base-field direction
    // through two existing jet layers before the directional derivative adds
    // its own layer.
    assert_eq!(derivative[0], 24.0);
}

#[test]
fn direction_and_point_can_vary_together() {
    fn cube<V: Euclidean>(x: V) -> V {
        x.map(|x| x.powi(3))
    }

    let derivative = d(|v| d(cube).along(v).at(v)).at(Coords::from(7.0));

    // d/dv (D(v³)ᵥ(v)) = d/dv (3v³) = 9v².
    assert_eq!(derivative[0], 441.0);
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
