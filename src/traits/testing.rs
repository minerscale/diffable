// ---------------------------------------------------------------------------
// Trait test macros
// These generate the full suite of invariant tests for any implementation
// of Chart, ExpMap, TangentBundle, LieGroup, and Metric. To test a new
// manifold, just invoke the relevant macro with appropriate generators.
// ---------------------------------------------------------------------------

#[macro_export]
macro_rules! test_vector {
    ($mod_name:ident, $scalar:ty, $space:ty, $arb_point:expr, $arb_scalar:expr) => {
        mod $mod_name {
            use super::*;
            use $crate::{
                test_group, test_tangent_bundle,
                traits::{Field, Form, Nondegenerate, Vector},
            };

            test_tangent_bundle!(
                tangent_bundle,
                $scalar,
                $space,
                $arb_point,
                $arb_point,
                $arb_scalar
            );

            test_group!(group, $space, $arb_point);

            proptest! {
                #[test]
                fn global_chart(p in $arb_point, q in $arb_point) {
                    prop_assert!(<$space>::check_global_chart(&p, &q));
                }

                #[test]
                fn translation_invariance(
                    a in $arb_point,
                    b in $arb_point,
                    c in $arb_point,
                ) {
                    prop_assert!(<$space>::check_translation_invariance(&a, &b, &c));
                }

                #[test]
                fn global_geodesic_scaling(
                    p in $arb_point,
                    v in $arb_point,
                    t in $arb_scalar,
                ) {
                    prop_assert!(<$space>::check_global_geodesic_scaling(&p, v, t));
                }

                #[test]
                fn check_isomorphism(
                    p in $arb_point,
                ) {
                    prop_assert!(<$space>::check_isomorphism(&p))
                }
            }
        }
    };
}

/// Tests that a space claiming to be a pseudo-Euclidean space is a pseudo-Euclidean space
#[macro_export]
macro_rules! test_pseudo_euclidean {
    ($mod_name:ident, $scalar:ty, $space:ty, $arb_point:expr, $arb_scalar:expr) => {
        mod $mod_name {
            use super::*;
            use $crate::{test_interval, test_pseudo_riemannian, test_sesquilinear, test_vector};

            test_vector!(quadratic, $scalar, $space, $arb_point, $arb_scalar);
            test_interval!(interval, $space, $arb_point);
            test_pseudo_riemannian!(riemannian, $space, $arb_point, $arb_point);
            test_sesquilinear!(sesquilinear, $space, $arb_point, $arb_scalar);
        }
    };
}

/// Tests that a space claiming to be a euclidean space is a euclidean space
#[macro_export]
macro_rules! test_euclidean {
    ($mod_name:ident, $scalar:ty, $space:ty, $arb_point:expr, $arb_scalar:expr) => {
        mod $mod_name {
            use super::*;
            use $crate::{
                test_inner_product, test_metric, test_pseudo_euclidean, traits::Euclidean,
            };

            test_pseudo_euclidean!(pseudo_euclidean, $scalar, $space, $arb_point, $arb_scalar);
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

/// Tests the chart roundtrip invariant: to_global(to_local(p)) == p
/// for any chart type and point generator.
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

/// Tests the ExpMap invariants: preservation of origin, geodesic symmetry,
/// geodesic scaling, and first-order isometry. The chart is constructed
/// via chart_at on a generated base point.
#[macro_export]
macro_rules! test_exp_map {
    ($mod_name:ident, $scalar:ty, $chart:ty, $arb_point:expr, $arb_vec:expr, $arb_scalar:expr) => {
        mod $mod_name {
            use super::*;
            use $crate::{
                test_chart,
                traits::{Chart, ExpMap, Field},
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

                #[test]
                fn geodesic_symmetry(p in $arb_point, v in $arb_vec) {
                    let chart = <$chart>::chart_at(&p);
                    prop_assert!(chart.check_geodesic_symmetry(v));
                }

                #[test]
                fn geodesic_scaling(p in $arb_point, v in $arb_vec, t in $arb_scalar) {
                    let chart = <$chart>::chart_at(&p);
                    prop_assert!(chart.check_geodesic_scaling(v, t.to_fixed()));
                }
            }
        }
    };
}

/// Tests that `Metric` and `ExpMap` agree: `d(p, exp_p(v)) == |log_p(exp_p(v))|`.
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

/// Tests the TangentBundle invariant on top of all ExpMap invariants.
#[macro_export]
macro_rules! test_tangent_bundle {
    ($mod_name:ident, $scalar:ty, $chart:ty, $arb_point:expr, $arb_vec:expr, $arb_scalar: expr) => {
        mod $mod_name {
            use super::*;
            use $crate::{test_exp_map, traits::TangentBundle};

            // inherit all ExpMap tests
            test_exp_map!(exp_map, $scalar, $chart, $arb_point, $arb_vec, $arb_scalar);

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

/// Tests the `CMonoid` axioms: identity, associativity, commutativity.
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

/// Tests the `Group` axioms: identity, associativity, inverses.
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

/// Tests the `Monoid` axioms: identity, associativity (no commutativity).
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

/// Tests the `CGroup` axioms: everything `test_cmonoid!` checks, plus
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

/// Tests the `MulGroup` axioms: everything `test_monoid!` checks, plus
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

/// Tests the Metric axioms: non-negativity, symmetry, self-distance is zero.
#[macro_export]
macro_rules! test_metric {
    ($mod_name:ident, $point:ty, $arb_point:expr) => {
        mod $mod_name {
            use super::*;
            use $crate::{test_interval, traits::{Metric}};

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

/// Tests the Interval axioms: Symmetry and self-interval is zero.
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

/// Tests the `Sesquilinear` axioms: Hermitian symmetry, additivity, and
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

/// Tests the InnerProduct axioms: symmetry, bilinearity, positive-definiteness.
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

/// Tests the Quotient axioms: that canonical respects cosets, and the
/// inherited LieGroup axioms which follow from the quotient structure.
#[macro_export]
macro_rules! test_quotient {
    ($mod_name:ident, $quotient:ty, $arb_quotient:expr, $arb_g:expr, $arb_h:expr) => {
        mod $mod_name {
            use super::*;
            use $crate::{test_group, traits::Quotient};

            // A quotient group is a Lie group — inherit all LieGroup axioms.
            test_group!(lie_group, $quotient, $arb_quotient);

            proptest! {
                #[test]
                fn new_respects_coset(g in $arb_g, h in $arb_h) {
                    prop_assert!(<$quotient>::check_new_respects_coset(g, h));
                }
            }
        }
    };
}

/// Tests the `DivRing` axioms: that the Inverse is properly implemented.
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

/// Tests the `Field` axioms: that we have a commutative division ring.
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
                fn commutativity(a in $arb_point, b in $arb_point) {
                    prop_assert!(<$point>::check_commutativity(a, b));
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
        }
    };
}

/// Tests the `Ring` axioms: everything `test_cgroup!` and `test_rig!` check.
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

/// Tests the `Rig` axioms: everything `test_cmonoid!` and `test_monoid!`
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
