#![cfg(feature = "testing")]

#[macro_use]
mod common;

use common::*;

use diffable::{
    complex::Complex,
    coords::Coords,
    epsilon_metric::R64,
    hypersphere::S3,
    quaternion::Quaternion,
    test_field, test_nondegenerate, test_sesquilinear, test_vector,
    traits::{Dual, Field, Form, Nondegenerate, Vector},
};
use num_traits::{One, Zero};
use proptest::prelude::*;

type H = Quaternion<R64>;
type H1 = Coords<H, 1>;
type H41 = Coords<H, 4, 1>;

fn arb_quaternion() -> impl Strategy<Value = H> {
    arb_vec::<4>().prop_map(Quaternion::from)
}

fn arb_h41() -> impl Strategy<Value = H41> {
    (
        arb_quaternion(),
        arb_quaternion(),
        arb_quaternion(),
        arb_quaternion(),
    )
        .prop_map(|(a, b, c, d)| [a, b, c, d].into())
}

test_field!(quaternion_field, H, arb_quaternion(), arb_scalar());
test_vector!(quaternion_vector, H, H41, arb_h41(), arb_quaternion());
test_sesquilinear!(quaternion_form, H41, arb_h41(), arb_quaternion());
test_nondegenerate!(quaternion_nondegenerate, H41, arb_h41(), arb_quaternion());

test_vector!(
    quaternion_dual_vector,
    H,
    Dual<H41>,
    arb_h41().prop_map(|x| x.flat()),
    arb_quaternion()
);
test_sesquilinear!(
    quaternion_dual_form,
    Dual<H41>,
    arb_h41().prop_map(|x| x.flat()),
    arb_quaternion()
);
test_nondegenerate!(
    quaternion_dual_nondegenerate,
    Dual<H41>,
    arb_h41().prop_map(|x| x.flat()),
    arb_quaternion()
);

#[test]
fn hamilton_basis_products_are_noncommutative() {
    let zero = R64::zero();
    let one = R64::one();
    let i = H::new(zero, one, zero, zero);
    let j = H::new(zero, zero, one, zero);
    let k = H::new(zero, zero, zero, one);

    assert_eq!(i * j, k);
    assert_eq!(j * i, -k);
}

#[test]
fn dual_pairing_distinguishes_musical_and_raw_coordinates() {
    let i = H::i();
    let j = H::j();
    let k = H::k();

    let alpha = H1::from_array([i]).flat();
    let beta = H1::from_array([j]).flat();
    let alpha_direct = Dual::<H1>::from_array([i]);
    let beta_direct = Dual::<H1>::from_array([j]);
    let psi_direct = Dual::<Dual::<H1>>::from_array([j]);

    let psi = beta.flat();

    assert_eq!(H1::sharp(beta).pairing(&alpha), -k);
    assert_eq!(H1::sharp(beta_direct).pairing(&alpha_direct), -k);
    assert_eq!(Dual::<H1>::pairing(&alpha, &psi), -k);
    assert_eq!(Dual::<H1>::pairing(&alpha_direct, &psi_direct), k);
}

proptest! {
    #[test]
    fn s3_and_unit_quaternions_round_trip(q in arb_sphere3()) {
        let quaternion = q.to_quaternion();

        prop_assert_eq!(quaternion.norm_squared(), R64::one());

        let recovered = S3::from_quaternion(quaternion);

        prop_assert_eq!(recovered.to_quaternion(), quaternion);
        prop_assert_eq!(recovered, q);
    }

    #[test]
    fn projecting_a_quaternion_onto_s3_normalises_it(q in arb_quaternion()) {
        prop_assume!(q.norm_squared() != R64::zero());

        let projected = S3::<Coords<R64, 3>>::from_quaternion(q);

        prop_assert_eq!(
            projected.to_quaternion().norm_squared(),
            R64::one(),
        );
    }

    #[test]
    fn s3_quaternion_identification_preserves_multiplication(
        a in arb_sphere3(),
        b in arb_sphere3(),
    ) {
        let quaternion_product =
            a.to_quaternion() * b.to_quaternion();

        let sphere_product =
            (a * b).to_quaternion();

        prop_assert_eq!(sphere_product, quaternion_product);
    }

    #[test]
    fn complex_embedding_preserves_addition(
        z in arb_vec::<2>().prop_map(|x| Complex::<R64>::from(x)),
        w in arb_vec::<2>().prop_map(|x| Complex::<R64>::from(x)),
    ) {
        prop_assert_eq!(
            Quaternion::from(z + w),
            Quaternion::from(z) + Quaternion::from(w),
        );
    }

    #[test]
    fn complex_embedding_preserves_multiplication(
        z in arb_vec::<2>().prop_map(|x| Complex::<R64>::from(x)),
        w in arb_vec::<2>().prop_map(|x| Complex::<R64>::from(x)),
    ) {
        prop_assert_eq!(
            Quaternion::from(z * w),
            Quaternion::from(z) * Quaternion::from(w),
        );
    }

    #[test]
    fn complex_embedding_preserves_conjugation(z in arb_vec::<2>().prop_map(|x| Complex::<R64>::from(x))) {
        prop_assert_eq!(
            Quaternion::from(z.conj()),
            Quaternion::from(z).conj(),
        );
    }

    #[test]
    fn complex_embedding_is_injective(z in arb_vec::<2>().prop_map(|x| Complex::<R64>::from(x))) {
        let embedded = Quaternion::from(z);
        let [re, im, j, k] = embedded.into();

        let [z_re, z_im] = z.into();
        prop_assert_eq!([re, im], [z_re, z_im]);
        prop_assert_eq!(j, R64::zero());
        prop_assert_eq!(k, R64::zero());
    }

    #[test]
    fn scalar_tower_commutes(r in arb_scalar()) {
        let direct = Quaternion::<R64>::from_fixed(r);
        let through_complex =
            Quaternion::from(Complex::from(r));

        prop_assert_eq!(direct, through_complex);
    }
}
