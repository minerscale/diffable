//! Property-test suites for the axioms represented by Diffable's traits.
//!
//! The exported `test_*` macros mirror the trait hierarchy: for example,
//! [`test_euclidean!`](crate::test_euclidean) includes the inherited vector,
//! form, interval, and metric obligations, while
//! [`test_field!`](crate::test_field) includes the corresponding ring and
//! division-ring laws. Enable the `testing` feature to use this module.

// ---------------------------------------------------------------------------
// Trait test macros
// These generate the full suite of invariant tests for any implementation
// of Chart, ExpMap, TangentBundle, LieGroup, and Metric. To test a new
// manifold, just invoke the relevant macro with appropriate generators.
// ---------------------------------------------------------------------------

use num_traits::One;

use crate::traits::{
    Cat, DivRing, Field, FromReal, Interval, Point, Tensor,
    calculus::{Jet, JetVector, Tangent, TensorOver},
};

/// Tests the axioms required by [`Vector`](crate::traits::Vector).
#[macro_export]
macro_rules! test_vector {
    ($mod_name:ident, $space:ty, $arb_point:expr, $arb_scalar:expr) => {
        mod $mod_name {
            use super::*;
            use $crate::{
                test_lie_group,
                traits::{Field, Tensor},
            };

            test_lie_group!(
                lie_group,
                $space,
                $space,
                $arb_point,
                $arb_point,
                $arb_scalar
            );

            proptest! {
                #[test]
                fn global_chart(p in $arb_point, q in $arb_point) {
                    prop_assert!(<$space>::check_global_chart(&p, &q));
                }

                #[test]
                fn global_geodesic_scaling(a in $arb_point, c in $arb_point, k in $arb_scalar) {
                    prop_assert!(<$space>::check_global_geodesic_scaling(&a, c, k.to_fixed()));
                }
            }
        }
    };
}

/// Tests the inherited vector, form, interval, and chart axioms of a
/// pseudo-Euclidean space.
///
/// This composes [`test_vector!`](crate::test_vector),
/// [`test_interval!`](crate::test_interval),
/// [`test_pseudo_riemannian!`](crate::test_pseudo_riemannian),
/// [`test_sesquilinear!`](crate::test_sesquilinear), and
/// [`test_nondegenerate!`](crate::test_nondegenerate).
#[macro_export]
macro_rules! test_pseudo_euclidean {
    ($mod_name:ident, $space:ty, $arb_point:expr, $arb_scalar:expr) => {
        mod $mod_name {
            use super::*;
            use $crate::{
                test_interval, test_nondegenerate, test_pseudo_riemannian, test_sesquilinear,
                test_vector,
            };

            test_vector!(vector, $space, $arb_point, $arb_scalar);
            test_interval!(interval, $space, $arb_point);
            test_pseudo_riemannian!(riemannian, $space, $arb_point, $arb_point);
            test_sesquilinear!(sesquilinear, $space, $arb_point, $arb_scalar);
            test_nondegenerate!(nondegenerate, $space, $arb_point, $arb_scalar);
        }
    };
}

/// Tests the axioms required by [`Euclidean`](crate::traits::Euclidean).
#[macro_export]
macro_rules! test_euclidean {
    ($mod_name:ident, $space:ty, $arb_point:expr, $arb_scalar:expr) => {
        mod $mod_name {
            use super::*;
            use $crate::{
                test_inner_product, test_metric, test_pseudo_euclidean, traits::Euclidean,
            };

            test_pseudo_euclidean!(pseudo_euclidean, $space, $arb_point, $arb_scalar);
            test_inner_product!(inner_product, $space, $arb_point, $arb_scalar);
            test_metric!(metric, $space, $arb_point);

            proptest! {
                #[test]
                fn pythagorean(a in $arb_point, b in $arb_point) {
                    prop_assert!(<$space>::check_pythagorean(&a, &b));
                }
            }
        }
    };
}

/// Tests the roundtrip invariant required by [`Chart`](crate::traits::Chart).
#[macro_export]
macro_rules! test_chart {
    ($mod_name:ident, $chart:ty, $arb_point:expr) => {
        mod $mod_name {
            use $crate::traits::Chart;

            use super::*;
            proptest! {
                #[test]
                fn coverage(p in $arb_point) {
                    prop_assert!(<$chart>::check_local_inverse(&p))
                }
            }
        }
    };
}

/// Tests the universally observable [`ExpMap`] laws.
///
/// This macro checks chart coverage and the origin/centring invariants. It does
/// not test that radial lines in the tangent space are mapped to geodesics with
/// the correct affine scaling, although that law remains part of the contract
/// of [`ExpMap`].
///
/// Testing that law through the generic `ExpMap` interface requires knowing
/// that the sampled tangent vectors lie within an appropriate injectivity
/// domain, or independently observing the resulting geodesic. The interface
/// provides neither certificate. Earlier tests attempted to infer this from
/// the quadratic form, but that inference is not valid for every `ExpMap` and
/// can reject correct implementations.
///
/// Implementations should therefore test the geodesic-scaling law separately
/// whenever they can generate tangent vectors with the necessary
/// implementation-specific guarantees.
///
/// [`ExpMap`]: crate::traits::ExpMap
#[macro_export]
macro_rules! test_exp_map {
    ($mod_name:ident, $chart:ty, $arb_point:expr) => {
        mod $mod_name {
            use super::*;
            use $crate::{
                test_chart,
                traits::{Chart, ExpMap},
            };

            // inherit all Chart tests
            test_chart!(chart, $chart, $arb_point);

            proptest! {
                #[test]
                fn preservation_of_origin(p in $arb_point) {
                    let chart = <$chart>::chart_at(&p);
                    prop_assert!(chart.check_preservation_of_origin());
                }

                #[test]
                fn chart_at_base_point(p in $arb_point) {
                    let chart = <$chart>::chart_at(&p);
                    prop_assert!(chart.check_chart_at_base_point());
                }

                #[test]
                fn base_point_is_origin(p in $arb_point) {
                    let chart = <$chart>::chart_at(&p);
                    prop_assert!(chart.check_base_point_is_origin());
                }
            }
        }
    };
}

/// Tests that [`Interval`](crate::traits::Interval) and
/// [`ExpMap`](crate::traits::ExpMap) agree along exponential coordinates.
#[macro_export]
macro_rules! test_pseudo_riemannian {
    ($mod_name:ident, $chart:ty, $arb_point:expr, $arb_vec:expr) => {
        mod $mod_name {
            use super::*;
            use $crate::traits::{Chart, PseudoRiemannian};

            proptest! {
                #[test]
                fn chart_interval_compatibility(p in $arb_point, v in $arb_vec) {
                    let chart = <$chart>::chart_at(&p);
                    prop_assert!(chart.check_isometry(v));
                }
            }
        }
    };
}

/// Tests [`TangentBundle`](crate::traits::TangentBundle) on top of all
/// [`ExpMap`](crate::traits::ExpMap) invariants.
#[macro_export]
macro_rules! test_tangent_bundle {
    ($mod_name:ident, $chart:ty, $arb_point:expr) => {
        mod $mod_name {
            use super::*;
            use $crate::{test_exp_map, traits::TangentBundle};

            // inherit all ExpMap tests
            test_exp_map!(exp_map, $chart, $arb_point);

            proptest! {
                // The TangentFibre invariant: chart_at(&p).to_global(zero) == p
                #[test]
                fn check_universal_centring(p in $arb_point) {
                    prop_assert!(<$chart>::check_universal_centring(p));
                }
            }
        }
    };
}

/// Tests the axioms required by [`CMonoid`](crate::traits::CMonoid).
#[macro_export]
macro_rules! test_cmonoid {
    ($mod_name:ident, $point:ty, $arb_point:expr) => {
        mod $mod_name {
            use super::*;
            use $crate::traits::CMonoid;

            proptest! {
                #[test]
                fn left_identity(p in $arb_point) {
                    prop_assert!(
                        <$point as CMonoid>::check_left_identity(&p)
                    );
                }

                #[test]
                fn right_identity(p in $arb_point) {
                    prop_assert!(<$point as CMonoid>::check_right_identity(&p));
                }

                #[test]
                fn associativity(a in $arb_point, b in $arb_point, c in $arb_point) {
                    prop_assert!(<$point as CMonoid>::check_associativity(a, b, c));
                }

                #[test]
                fn commutativity(a in $arb_point, b in $arb_point) {
                    prop_assert!(<$point as CMonoid>::check_commutativity(a, b));
                }
            }
        }
    };
}

/// Tests the axioms required by [`Group`](crate::traits::Group).
#[macro_export]
macro_rules! test_group {
    ($mod_name:ident, $point:ty, $arb_point:expr) => {
        mod $mod_name {
            use super::*;
            use $crate::traits::Group;

            proptest! {
                #[test]
                fn left_identity(p in $arb_point) {
                    prop_assert!(
                        <$point as Group>::check_left_identity(&p)
                    );
                }

                #[test]
                fn right_identity(p in $arb_point) {
                    prop_assert!(<$point as Group>::check_right_identity(&p));
                }

                #[test]
                fn associativity(a in $arb_point, b in $arb_point, c in $arb_point) {
                    prop_assert!(<$point as Group>::check_associativity(a, b, c));
                }

                #[test]
                fn left_inverse(p in $arb_point) {
                    prop_assert!(<$point as Group>::check_left_inverse(&p));
                }

                #[test]
                fn right_inverse(p in $arb_point) {
                    prop_assert!(<$point as Group>::check_right_inverse(&p));
                }
            }
        }
    };
}

/// Tests the axioms required by [`Monoid`](crate::traits::Monoid).
#[macro_export]
macro_rules! test_monoid {
    ($mod_name:ident, $point:ty, $arb_point:expr) => {
        mod $mod_name {
            use super::*;
            use $crate::traits::Monoid;

            proptest! {
                #[test]
                fn left_identity(p in $arb_point) {
                    prop_assert!(
                        <$point as Monoid>::check_left_identity(&p)
                    );
                }

                #[test]
                fn right_identity(p in $arb_point) {
                    prop_assert!(<$point as Monoid>::check_right_identity(&p));
                }

                #[test]
                fn associativity(a in $arb_point, b in $arb_point, c in $arb_point) {
                    prop_assert!(<$point as Monoid>::check_associativity(a, b, c));
                }
            }
        }
    };
}

/// Tests the [`CGroup`](crate::traits::CGroup) axioms: everything
/// [`test_cmonoid!`](crate::test_cmonoid) checks, plus
/// additive inverses.
#[macro_export]
macro_rules! test_cgroup {
    ($mod_name:ident, $point:ty, $arb_point:expr) => {
        mod $mod_name {
            use super::*;
            use $crate::{test_cmonoid, traits::CGroup};

            test_cmonoid!(monoid, $point, $arb_point);
            proptest! {
                #[test]
                fn left_inverse(p in $arb_point) {
                    prop_assert!(<$point as CGroup>::check_left_inverse(&p));
                }

                #[test]
                fn right_inverse(p in $arb_point) {
                    prop_assert!(<$point as CGroup>::check_right_inverse(&p));
                }

                #[test]
                fn sub_agrees_with_neg(a in $arb_point, b in $arb_point) {
                    prop_assert!(<$point>::check_sub_agrees_with_neg(&a, &b))
                }
            }
        }
    };
}

/// Tests the [`MulGroup`](crate::traits::MulGroup) axioms: everything
/// [`test_monoid!`](crate::test_monoid) checks, plus
/// multiplicative inverses.
#[macro_export]
macro_rules! test_mul_group {
    ($mod_name:ident, $point:ty, $arb_point:expr) => {
        mod $mod_name {
            use super::*;
            use $crate::{test_monoid, traits::MulGroup};

            test_monoid!(monoid, $point, $arb_point);
            proptest! {
                #[test]
                fn left_inverse(p in $arb_point) {
                    prop_assert!(<$point as MulGroup>::check_left_inverse(&p));
                }

                #[test]
                fn right_inverse(p in $arb_point) {
                    prop_assert!(<$point as MulGroup>::check_right_inverse(&p));
                }
            }
        }
    };
}

/// Tests the axioms required by [`Metric`](crate::traits::Metric).
#[macro_export]
macro_rules! test_metric {
    ($mod_name:ident, $point:ty, $arb_point:expr) => {
        mod $mod_name {
            use super::*;
            use $crate::{test_interval, traits::Metric};

            test_interval!(interval, $point, $arb_point);

            proptest! {
                #[test]
                fn non_negative(a in $arb_point, b in $arb_point) {
                    prop_assert!(<$point>::check_non_negative(a, b));
                }

                #[test]
                fn distance_agrees_with_interval(a in $arb_point, b in $arb_point) {
                    prop_assert!(<$point>::check_distance_agrees_with_interval(a, b))
                }
            }
        }
    };
}

/// Tests the axioms required by [`Interval`](crate::traits::Interval).
#[macro_export]
macro_rules! test_interval {
    ($mod_name:ident, $point:ty, $arb_point:expr) => {
        mod $mod_name {
            use super::*;
            use $crate::traits::Interval;

            proptest! {
                #[test]
                fn interval_symmetry(a in $arb_point, b in $arb_point) {
                    prop_assert!(<$point>::check_interval_symmetry(a, b));
                }

                #[test]
                fn self_interval_zero(p in $arb_point) {
                    prop_assert!(<$point>::check_self_interval_zero(p))
                }

                #[test]
                fn interval_squared_agrees_with_interval(a in $arb_point, b in $arb_point) {
                    prop_assert!(<$point>::check_interval_squared_agrees_with_interval(&a, &b))
                }
            }
        }
    };
}

/// Tests the axioms required by [`Form`](crate::traits::Form): dot/pairing
/// agreement and translation invariance
#[macro_export]
macro_rules! test_form {
    ($mod_name:ident, $point:ty, $arb_point:expr, $arb_scalar:expr) => {
        mod $mod_name {
            use super::*;
            use $crate::traits::{Field, Form};

            proptest! {
                #[test]
                fn dot_agrees_with_pairing(a in $arb_point, b in $arb_point) {
                    prop_assert!(<$point>::check_dot_agrees_with_pairing(&a, &b));
                }

                #[test]
                fn translation_invariance(a in $arb_point, b in $arb_point, c in $arb_point) {
                    prop_assert!(<$point>::check_translation_invariance(&a, &b, &c));
                }
            }
        }
    };
}

/// Tests that [`Nondegenerate::sharp`](crate::traits::Nondegenerate::sharp) is
/// exactly the inverse of [`Form::flat`](crate::traits::Form::flat).
#[macro_export]
macro_rules! test_nondegenerate {
    ($mod_name:ident, $point:ty, $arb_point:expr, $arb_scalar:expr) => {
        mod $mod_name {
            use super::*;
            use $crate::{
                test_form,
                traits::{Dual, Nondegenerate},
            };

            test_form!(form, $point, $arb_point, $arb_scalar);

            proptest! {
                #[test]
                fn isomorphism(a in $arb_point) {
                    prop_assert!(<$point>::check_isomorphism(&a));
                }
            }
        }
    };
}

/// Tests the [`Sesquilinear`](crate::traits::Sesquilinear) axioms: Hermitian
/// symmetry, additivity, and
/// scalar linearity in the first argument.
#[macro_export]
macro_rules! test_sesquilinear {
    ($mod_name:ident, $point:ty, $arb_point:expr, $arb_scalar:expr) => {
        mod $mod_name {
            use super::*;
            use $crate::traits::Sesquilinear;

            proptest! {
                #[test]
                fn hermitian_symmetry(a in $arb_point, b in $arb_point) {
                    prop_assert!(<$point>::check_hermitian_symmetry(a, b));
                }

                #[test]
                fn additivity(a in $arb_point, b in $arb_point, c in $arb_point) {
                    prop_assert!(<$point>::check_additivity(a, b, c));
                }

                #[test]
                fn scalar_linearity(a in $arb_point, c in $arb_point, k in $arb_scalar) {
                    prop_assert!(<$point>::check_scalar_linearity(a, c, k));
                }
            }
        }
    };
}

/// Tests the axioms required by [`InnerProduct`](crate::traits::InnerProduct).
#[macro_export]
macro_rules! test_inner_product {
    ($mod_name:ident, $point:ty, $arb_point:expr, $arb_scalar:expr) => {
        mod $mod_name {
            use super::*;
            use $crate::test_sesquilinear;
            use $crate::traits::InnerProduct;

            test_sesquilinear!(bilinear, $point, $arb_point, $arb_scalar);

            proptest! {
                #[test]
                fn positive_definite(a in $arb_point) {
                    prop_assert!(<$point>::check_positive_definite(a));
                }

                #[test]
                fn check_metric_compatibility(a in $arb_point, b in $arb_point) {
                    prop_assert!(<$point>::check_metric_compatibility(a, b));
                }
            }
        }
    };
}

/// Tests the [`Quotient`](crate::traits::Quotient) axioms: that canonical
/// respects cosets, and the
/// inherited LieGroup axioms which follow from the quotient structure.
#[macro_export]
macro_rules! test_quotient {
    (
        $mod_name:ident,
        $quotient:ty,
        $vector:ty,
        $arb_quotient:expr,
        $arb_vector:expr,
        $arb_scalar:expr,
        $arb_g:expr,
        $arb_h:expr
    ) => {
        mod $mod_name {
            use super::*;

            use $crate::{test_lie_group, traits::Quotient};

            test_lie_group!(
                lie_group,
                $quotient,
                $vector,
                $arb_quotient,
                $arb_vector,
                $arb_scalar
            );

            proptest! {
                #[test]
                fn new_respects_coset(
                    g in $arb_g,
                    h in $arb_h,
                ) {
                    prop_assert!(
                        <$quotient>::check_new_respects_coset(g, h)
                    );
                }
            }
        }
    };
}

/// Tests the axioms required by [`DivRing`](crate::traits::DivRing).
#[macro_export]
macro_rules! test_div_ring {
    ($mod_name:ident, $point:ty, $arb_point:expr) => {
        mod $mod_name {
            use super::*;
            use $crate::{
                test_mul_group, test_ring,
                traits::{DivRing, NonZero},
            };

            test_ring!(ring, $point, $arb_point);
            test_mul_group!(
                mul_group,
                NonZero<$point>,
                $arb_point.prop_filter_map("was zero", |x| NonZero::new(x))
            );
        }
    };
}

/// Tests the axioms required by [`CField`](crate::traits::CField).
#[macro_export]
macro_rules! test_cfield {
    ($mod_name:ident, $point:ty, $arb_point:expr, $arb_fixed:expr) => {
        mod $mod_name {
            use super::*;
            use $crate::{test_field, traits::CField};

            test_field!(field, $point, $arb_point, $arb_fixed);

            proptest! {
                #[test]
                fn commutativity(a in $arb_point, b in $arb_point) {
                    prop_assert!(<$point>::check_commutativity(a, b));
                }
            }
        }
    };
}

/// Tests the axioms required by [`Field`](crate::traits::Field).
#[macro_export]
macro_rules! test_field {
    ($mod_name:ident, $point:ty, $arb_point:expr, $arb_fixed:expr) => {
        mod $mod_name {
            use super::*;
            use $crate::{test_div_ring, traits::Field};

            test_div_ring!(div_ring, $point, $arb_point);
            proptest! {
                #[test]
                fn conj_additive(a in $arb_point, b in $arb_point) {
                    prop_assert!(<$point>::check_conj_additive(a, b));
                }

                #[test]
                fn conj_multiplicative(a in $arb_point, b in $arb_point) {
                    prop_assert!(<$point>::check_conj_multiplicative(a, b));
                }

                #[test]
                fn conj_involution(a in $arb_point) {
                    prop_assert!(<$point>::check_conj_involution(a));
                }

                #[test]
                fn from_fixed_additive(x in $arb_fixed, y in $arb_fixed) {
                    prop_assert!(<$point>::check_from_fixed_additive(x, y));
                }

                #[test]
                fn from_fixed_multiplicative(x in $arb_fixed, y in $arb_fixed) {
                    prop_assert!(<$point>::check_from_fixed_multiplicative(x, y));
                }

                #[test]
                fn descent(x in $arb_point) {
                    prop_assert!(<$point>::check_descent(x));
                }

                #[test]
                fn norm_squared_self_adjoint(x in $arb_point) {
                    prop_assert!(<$point>::check_norm_squared_self_adjoint(x));
                }

                #[test]
                fn from_fixed_is_fixed(x in $arb_fixed) {
                    prop_assert!(<$point>::check_from_fixed_is_fixed(x));
                }

                #[test]
                fn fixed_field_is_central(x in $arb_fixed, y in $arb_point) {
                    prop_assert!(<$point>::check_fixed_field_is_central(x, y));
                }
            }

            #[test]
            fn characteristic() {
                assert!(<$point>::check_characteristic_up_to(256))
            }

            #[test]
            fn conj_unit() {
                assert!(<$point>::check_conj_unit());
            }

            #[test]
            fn from_fixed_unit() {
                assert!(<$point>::check_from_fixed_unit());
            }
        }
    };
}

/// Tests the axioms required by [`Ring`](crate::traits::Ring).
#[macro_export]
macro_rules! test_ring {
    ($mod_name:ident, $point:ty, $arb_point:expr) => {
        mod $mod_name {
            use super::*;
            use $crate::{test_cgroup, test_rig, traits::Ring};

            test_cgroup!(group, $point, $arb_point);
            test_rig!(rig, $point, $arb_point);
        }
    };
}

/// Tests the [`Rig`](crate::traits::Rig) axioms: everything
/// [`test_cmonoid!`](crate::test_cmonoid) and
/// [`test_monoid!`](crate::test_monoid)
/// check, plus distributivity and multiplicative annihilation by zero.
#[macro_export]
macro_rules! test_rig {
    ($mod_name:ident, $point:ty, $arb_point:expr) => {
        mod $mod_name {
            use super::*;
            use $crate::{test_cmonoid, test_monoid, traits::Rig};

            test_cmonoid!(cmonoid, $point, $arb_point);
            test_monoid!(monoid, $point, $arb_point);

            proptest! {
                #[test]
                fn left_distributivity(a in $arb_point, b in $arb_point, c in $arb_point) {
                    prop_assert!(<$point>::check_left_distributivity(a, b, c));
                }

                #[test]
                fn right_distributivity(a in $arb_point, b in $arb_point, c in $arb_point) {
                    prop_assert!(<$point>::check_right_distributivity(a, b, c));
                }

                #[test]
                fn left_annihilation(g in $arb_point) {
                    prop_assert!(g.check_left_annihilation());
                }

                #[test]
                fn right_annihilation(g in $arb_point) {
                    prop_assert!(g.check_right_annihilation());
                }
            }
        }
    };
}

/// Tests the axioms required by [`Connection`](crate::traits::Connection).
///
/// The lifted chart laws are exercised at jet order two, while truncation
/// coherence is checked from order two to order one. The supplied generators
/// must produce points in `P`, vectors in `V`, and scalars in `V::F`.
#[macro_export]
macro_rules! test_connection {
    (
        $mod_name:ident,
        $connection:ty,
        $point:ty,
        $vector:ty,
        $arb_point:expr,
        $arb_vector:expr,
        $arb_scalar:expr
    ) => {
        mod $mod_name {
            use super::*;

            use num_traits::Zero;

            use $crate::{
                test_tangent_bundle,
                traits::{
                    Chart, Tensor,
                    calculus::{Connection, Jet, JetVector, Tangent, TensorOver},
                    𝐅𝐥𝐝,
                },
            };

            test_tangent_bundle!(tangent_bundle, $connection, $arb_point);

            proptest! {
                #[test]
                fn quadratic_geodesic_acceleration(
                    observer in $arb_point,
                    point in $arb_point,
                    u in $arb_vector,
                    v in $arb_vector,
                    scalar in $arb_scalar,
                ) {
                    let chart =
                        <$connection as Chart<$point, $vector>>::chart_at(
                            &observer,
                        );

                    prop_assert!(
                        <$connection as Connection<$point, $vector>>::
                            check_quadratic_geodesic_acceleration(
                                &chart,
                                point,
                                u,
                                v,
                                scalar,
                            )
                    );
                }

                #[test]
                fn tangent_to_local_agrees_with_chart(
                    base in $arb_point,
                    point in $arb_point,
                ) {
                    prop_assert!(
                        <$connection as Connection<$point, $vector>>::
                            check_tangent_to_local_agrees_with_chart(
                                base,
                                point,
                            )
                    );
                }

                #[test]
                fn tangent_to_global_agrees_with_chart(
                    base in $arb_point,
                    coordinate in $arb_vector,
                ) {
                    prop_assert!(
                        <$connection as Connection<$point, $vector>>::
                            check_tangent_to_global_agrees_with_chart(
                                base,
                                coordinate,
                            )
                    );
                }

                #[test]
                fn truncation_coherence(
                    base_point in $arb_point,
                    base_d1 in $arb_vector,
                    base_d2 in $arb_vector,
                    coordinate_0 in $arb_vector,
                    coordinate_1 in $arb_vector,
                    coordinate_2 in $arb_vector,
                ) {
                    let base =
                        $crate::traits::testing::tangent_from_coefficients(
                            base_point,
                            [base_d1, base_d2],
                        );

                    let coordinate =
                        $crate::traits::testing::jet_vector_from_coefficients(
                            coordinate_0,
                            [coordinate_1, coordinate_2],
                        );

                    prop_assert!(
                        <$connection as Connection<$point, $vector>>::
                            check_truncation_coherence::<1, 2>(
                                base,
                                coordinate,
                            )
                    );
                }

                #[test]
                fn tangent_isomorphism(
                    base_point in $arb_point,
                    base_d1 in $arb_vector,
                    base_d2 in $arb_vector,

                    local_point in $arb_point,
                    local_d1 in $arb_vector,
                    local_d2 in $arb_vector,

                    coordinate_0 in $arb_vector,
                    coordinate_1 in $arb_vector,
                    coordinate_2 in $arb_vector,
                ) {
                    let base =
                        $crate::traits::testing::tangent_from_coefficients(
                            base_point,
                            [base_d1, base_d2],
                        );

                    let local =
                        $crate::traits::testing::tangent_from_coefficients(
                            local_point,
                            [local_d1, local_d2],
                        );

                    let coordinate =
                        $crate::traits::testing::jet_vector_from_coefficients(
                            coordinate_0,
                            [coordinate_1, coordinate_2],
                        );

                    prop_assert!(
                        <$connection as Connection<$point, $vector>>::
                            check_tangent_isomorphism::<2>(
                                base,
                                local,
                                coordinate,
                            )
                    );
                }
            }
        }
    };
}

/// Tests the axioms required by [`LieGroup`](crate::traits::LieGroup).
///
/// A Lie group must satisfy the ordinary group laws, and the connection
/// generated by left translation of its identity exponential must satisfy the
/// complete lifted tangent-chart contract.
#[macro_export]
macro_rules! test_lie_group {
    (
        $mod_name:ident,
        $group:ty,
        $vector:ty,
        $arb_group:expr,
        $arb_vector:expr,
        $arb_scalar:expr
    ) => {
        mod $mod_name {
            use super::*;

            use $crate::{test_connection, test_group};

            test_group!(group, $group, $arb_group);

            test_connection!(
                connection,
                $group,
                $group,
                $vector,
                $arb_group,
                $arb_vector,
                $arb_scalar
            );
        }
    };
}

#[doc(hidden)]
pub fn jet_vector_from_coefficients<V: Tensor, const N: usize>(
    primal: V,
    coefficients: [V; N],
) -> JetVector<V, N> {
    TensorOver::from_fn(|coordinate| {
        Jet::from_parts(
            primal[coordinate].clone(),
            core::array::from_fn(|order| coefficients[order][coordinate].clone()),
        )
    })
}

#[doc(hidden)]
pub fn tangent_from_coefficients<P: Point, V: Tensor, const N: usize>(
    point: P,
    coefficients: [V; N],
) -> Tangent<P, V, N> {
    Tangent::new(point, jet_vector_from_coefficients(V::zero(), coefficients))
}

/// Constructs a coefficient-wise constant jet without requiring a categorical
/// admission proof for `Jet::constant`.
///
/// This is the raw representation-level embedding:
///
/// x ↦ (x, 0, ..., 0).
#[doc(hidden)]
pub fn constant_jet<𝒞: Cat, F: Field, const N: usize>(value: F) -> Jet<𝒞, F, N> {
    Jet::from_parts(value, [F::zero(); N])
}

/// Generates a small closed polynomial curve from arbitrary Chebyshev
/// coefficients.
///
/// With
///
/// z = 2t - 1,
///
/// this evaluates
///
/// c(t) = (1 / 16)(1 - z²) Σₖ aₖ Tₖ(z).
///
/// Consequently `c(0) = c(1) = 0`. The curve is smooth on `[0, 1]` and
/// continuous, though not necessarily smooth, at the swippity.
#[doc(hidden)]
pub fn closed_chebyshev_curve<V, const N: usize>(
    t: Jet<crate::traits::𝐑𝐞𝐚𝐥::𝒞, <V::F as Interval>::R, N>,
    coefficients: &[V],
) -> JetVector<V, N>
where
    V: Tensor<F: FromReal>,
    Jet<crate::traits::𝐅𝐥𝐝::𝒞, V::F, N>: Field,
{
    assert!(!coefficients.is_empty());

    type FieldJet<F, const N: usize> = Jet<crate::traits::𝐅𝐥𝐝::𝒞, F, N>;

    let t = FieldJet::<V::F, N>::from_parts(
        V::F::from_real(t[0]),
        core::array::from_fn(|i| V::F::from_real(t[i + 1])),
    );

    let one = FieldJet::<V::F, N>::one();
    let two = one.clone() + one.clone();
    let four = two.clone() + two.clone();
    let sixteen = four.clone() * four;

    // Map [0,1] to [-1,1].
    let z = two.clone() * t - one.clone();

    // 1 - z² = 4t(1-t), so this vanishes exactly at both endpoints.
    let envelope = (one.clone() - z.clone() * z.clone()).div(sixteen);

    JetVector::<V, N>::from_fn(|i| {
        let mut sum =
            constant_jet::<crate::traits::𝐅𝐥𝐝::𝒞, V::F, N>(coefficients[0][i]);

        if coefficients.len() > 1 {
            let mut previous = one.clone();
            let mut current = z.clone();

            sum = sum
                + constant_jet::<crate::traits::𝐅𝐥𝐝::𝒞, V::F, N>(coefficients[1][i])
                    * current.clone();

            for coefficient in coefficients.iter().skip(2) {
                let next = two.clone() * z.clone() * current.clone() - previous;

                sum = sum
                    + constant_jet::<crate::traits::𝐅𝐥𝐝::𝒞, V::F, N>(coefficient[i]) * next.clone();

                previous = current;
                current = next;
            }
        }

        envelope.clone() * sum
    })
}

/// Tests that restricted holonomy preserves the model-space form.
///
/// Arbitrary Chebyshev coefficients are used to generate small closed
/// polynomial loops in the tangent space at each sampled base point. Those
/// loops are lifted through the connection before parallel transport is
/// evaluated.
#[macro_export]
macro_rules! test_holonomy {
    (
        $mod_name:ident,
        $point:ty,
        $space:ty,
        $arb_point:expr,
        $arb_vector:expr
    ) => {
        mod $mod_name {
            use super::*;

            use num_traits::Zero;
            use $crate::traits::{
                Chart, ExpMap, OptionallyOption, Tensor,
                calculus::{
                    Connection, Jet, JetVector, ParallelTransport, TRANSPORT_ORDER, Tangent,
                },
                testing::closed_chebyshev_curve,
            };

            proptest! {
                #![proptest_config(ProptestConfig::with_cases(10))]
                #[test]
                fn preserves_form_around_small_closed_loops(
                    base in $arb_point,
                    coefficients in proptest::collection::vec(
                        $arb_vector,
                        1..=TRANSPORT_ORDER + 1,
                    ),
                    u in $arb_vector,
                    v in $arb_vector,
                ) {
                    let loop_base = base.clone();

                    let closed_curve = move |t| {
                        let coordinate =
                            closed_chebyshev_curve::<
                                $space,
                                TRANSPORT_ORDER,
                            >(
                                t,
                                &coefficients,
                            );

                        let tangent_base =
                            Tangent::<
                                $point,
                                $space,
                                TRANSPORT_ORDER,
                            >::new(
                                loop_base.base_point(),
                                JetVector::<
                                    $space,
                                    TRANSPORT_ORDER,
                                >::zero(),
                            );

                        let (point, tangent) =
                            <$point as Connection<
                                $point,
                                $space,
                            >>::tangent_to_global(
                                tangent_base,
                                coordinate,
                            )
                            .into_option()
                            .unwrap();

                        Tangent::new(point, tangent)
                    };

                    prop_assert!(
                        base.check_holonomy_preserves_form::<
                            TRANSPORT_ORDER
                        >(
                            <$space>::zero(),
                            closed_curve,
                            u,
                            v,
                        )
                    );
                }
            }
        }
    };
}
