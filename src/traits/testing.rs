// ---------------------------------------------------------------------------
// Trait test macros
// These generate the full suite of invariant tests for any implementation
// of Chart, ExpMap, TangentBundle, LieGroup, and Metric. To test a new
// manifold, just invoke the relevant macro with appropriate generators.
// ---------------------------------------------------------------------------

/// Tests that a space claiming to be a euclidean space is a euclidean space
#[macro_export]
macro_rules! test_euclidean {
    ($mod_name:ident, $scalar:ty, $space:ty, $arb_point:expr, $arb_vec:expr, $arb_scalar:expr) => {
        mod $mod_name {
            use super::*;

            // inherit all TangentFibre tests
            test_tangent_bundle!(tangent_bundle, $scalar, $space, $arb_point, $arb_vec);
            test_metric!(metric, $space, $arb_vec);
            test_inner_product!(inner_product, $space, $arb_point, $arb_scalar);
            test_group!(group, $space, $arb_vec);
            test_riemannian!(riemannian, $space, $arb_point, $arb_vec);

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
                    v in $arb_vec,
                    t in $arb_scalar, // unbounded t, flat space has no injectivity radius
                ) {
                    prop_assert!(<$space>::check_global_geodesic_scaling(&p, v, t));
                }

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
    ($mod_name:ident, $scalar:ty, $chart:ty, $arb_point:expr, $arb_vec:expr) => {
        mod $mod_name {
            use super::*;
            use num_traits::NumCast;

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
                fn geodesic_scaling(p in $arb_point, v in $arb_vec, t in 0.0f64..1.0f64) {
                    let chart = <$chart>::chart_at(&p);
                    prop_assert!(chart.check_geodesic_scaling(v, <$scalar as NumCast>::from(t).unwrap()));
                }
            }
        }
    };
}

#[macro_export]
macro_rules! test_riemannian {
    ($mod_name:ident, $chart:ty, $arb_point:expr, $arb_vec:expr) => {
        mod $mod_name {
            use super::*;
            proptest! {
                #[test]
                fn chart_metric_compatibility(p in $arb_point, v in $arb_vec) {
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
    ($mod_name:ident, $scalar:ty, $chart:ty, $arb_point:expr, $arb_vec:expr) => {
        mod $mod_name {
            use super::*;

            // inherit all ExpMap tests
            test_exp_map!(exp_map, $scalar, $chart, $arb_point, $arb_vec);

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

#[macro_export]
macro_rules! test_group {
    ($mod_name:ident, $point:ty, $arb_point:expr) => {
        mod $mod_name {
            use super::*;
            proptest! {
                #[test]
                fn left_identity(p in $arb_point) {
                    prop_assert!(
                        <$point>::check_left_identity(&p)
                    );
                }

                #[test]
                fn right_identity(p in $arb_point) {
                    prop_assert!(<$point>::check_right_identity(&p));
                }

                #[test]
                fn left_inverse(p in $arb_point) {
                    prop_assert!(<$point>::check_left_inverse(&p));
                }

                #[test]
                fn right_inverse(p in $arb_point) {
                    prop_assert!(<$point>::check_right_inverse(&p));
                }

                #[test]
                fn associativity(a in $arb_point, b in $arb_point, c in $arb_point) {
                    prop_assert!(<$point>::check_associativity(a, b, c));
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
            proptest! {
                #[test]
                fn non_negative(a in $arb_point, b in $arb_point) {
                    prop_assert!(<$point>::check_non_negative(a, b));
                }

                #[test]
                fn symmetry(a in $arb_point, b in $arb_point) {
                    prop_assert!(<$point>::check_metric_symmetry(a, b));
                }

                #[test]
                fn self_distance_zero(p in $arb_point) {
                    prop_assert!(<$point>::check_self_distance_zero(p))
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
            proptest! {
                #[test]
                fn symmetry(a in $arb_point, b in $arb_point) {
                    prop_assert!(<$point>::check_inner_product_symmetry(a, b));
                }

                #[test]
                fn additivity(a in $arb_point, b in $arb_point, c in $arb_point) {
                    prop_assert!(<$point>::check_additivity(a, b, c));
                }

                #[test]
                fn scalar_linearity(a in $arb_point, c in $arb_point, k in $arb_scalar) {
                    prop_assert!(<$point>::check_scalar_linearity(a, c, k));
                }

                #[test]
                fn positive_definite(a in $arb_point) {
                    prop_assert!(<$point>::check_positive_definite(a));
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
