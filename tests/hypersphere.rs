#![cfg(feature = "testing")]

#[macro_use]
mod common;

use common::*;

use diffable::{
    coords::Coords,
    epsilon_metric::R64,
    hypersphere::{S0, S1Cover, S3, So3, So3Cover, Sphere, Stereographic, UnitComplex},
    test_chart, test_exp_map, test_group, test_metric, test_monoid, test_quotient, test_riemannian,
    test_tangent_bundle,
    traits::{
        Chart, CMonoid, ExpMap, Group, GroupPresentation, InnerProduct, LieGroup, Metric,
        NerveComplex, Quotient, Riemannian, TangentBundle,
    },
};

use num_traits::{One, Zero};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Instantiations — adding a new manifold is one line per trait
// ---------------------------------------------------------------------------

// Stereographic chart roundtrips
test_chart!(stereo_s0, Stereographic<_>, arb_sphere0().prop_map(|x| x.0));
test_chart!(stereo_s1, Stereographic<_>, arb_sphere1().prop_map(|x| x.0));
test_chart!(stereo_s2, Stereographic<_>, arb_sphere2());
test_chart!(stereo_s3, Stereographic<_>, arb_sphere3().prop_map(|x| x.0));

// Metric axioms
test_metric!(metric_sphere0, Sphere<_, _>, arb_sphere0().prop_map(|x| x.0));
test_metric!(metric_sphere1, Sphere<_, _>, arb_sphere1().prop_map(|x| x.0));
test_metric!(metric_sphere2, Sphere<_, _>, arb_sphere2());
test_metric!(metric_sphere3, Sphere<_, _>, arb_sphere3().prop_map(|x| x.0));

// Metric + ExpMap compatibility
test_riemannian!(riemannian_sphere0, Sphere<_, _>, arb_sphere0().prop_map(|x| x.0), arb_vec::<0>());
test_riemannian!(riemannian_sphere1, Sphere<_, _>, arb_sphere1().prop_map(|x| x.0), arb_vec::<1>());
test_riemannian!(riemannian_sphere2, Sphere<_, _>, arb_sphere2(), arb_vec::<2>());
test_riemannian!(riemannian_sphere3, Sphere<_, _>, arb_sphere3().prop_map(|x| x.0), arb_vec::<3>());

test_riemannian!(
    riemannian_s0,
    S0<Coords<_, _>>,
    arb_sphere0(),
    arb_vec::<0>()
);
test_riemannian!(
    riemannian_s1,
    UnitComplex<Coords<_, _>>,
    arb_sphere1(),
    arb_vec::<1>()
);
test_riemannian!(
    riemannian_s3,
    S3<Coords<_, _>>,
    arb_sphere3(),
    arb_vec::<3>()
);

test_tangent_bundle!(tangent_bundle_sphere0, R64, Sphere<_, _>, arb_sphere0().prop_map(|x| x.0), arb_vec0());
test_tangent_bundle!(tangent_bundle_sphere1, R64, Sphere<_, _>, arb_sphere1().prop_map(|x| x.0), arb_vec1());
test_tangent_bundle!(tangent_bundle_sphere2, R64, Sphere<_, _>, arb_sphere2(), arb_vec2());
test_tangent_bundle!(tangent_bundle_sphere3, R64, Sphere<_, _>, arb_sphere3().prop_map(|x| x.0), arb_vec3());
test_tangent_bundle!(tangent_bundle_sphere4, R64, Sphere<_, _>, arb_sphere4(), arb_vec4());

// Sphere as TangentBundle (via blanket LieGroup impl; includes all ExpMap tests)
test_tangent_bundle!(
    tangent_bundle_s0,
    R64,
    S0<Coords<_, _>>,
    arb_sphere0(),
    arb_vec0()
);
test_tangent_bundle!(
    tangent_bundle_s1,
    R64,
    UnitComplex<Coords<_, _>>,
    arb_sphere1(),
    arb_vec1()
);
test_tangent_bundle!(
    tangent_bundle_s3,
    R64,
    S3<Coords<_, _>>,
    arb_sphere3(),
    arb_vec3()
);
test_tangent_bundle!(
    tangent_bundle_so3,
    R64,
    So3<Coords<_, _>>,
    arb_so3(),
    arb_vec3()
);

// Lie group axioms
test_group!(lie_group_s0, S0<_>, arb_sphere0());
test_group!(lie_group_s1, UnitComplex<_>, arb_sphere1());
test_group!(lie_group_s3, S3<_>, arb_sphere3());
// ($mod_name:ident, $quotient:ty, $arb_quotient:expr, $arb_g:expr, $arb_h:expr)
test_quotient!(
    lie_group_so3,
    So3<Coords<_, _>>,
    arb_so3(),
    arb_sphere3(),
    arb_sphere0().prop_map(|v| S0(Sphere::new(v.0.real(), Coords::zero())))
);

test_exp_map!(so3_cover, R64, So3Cover, arb_so3(), arb_vec3());

// ---------------------------------------------------------------------------
// Bespoke tests: properties specific to these manifolds, not general laws
// ---------------------------------------------------------------------------
proptest! {
    // S^1 is abelian, so exp is a group homomorphism: exp(v+w) = exp(v) * exp(w)
    #[test]
    fn s1_exp_homomorphism(v in -1.5f64..1.5f64, w in -1.5f64..1.5f64) {
        let (v, w) = (R64(v), R64(w));
        let chart = UnitComplex::identity();
        let ev: Coords<R64, 1> = [v].into();
        let ew: Coords<R64, 1> = [w].into();
        let evw: Coords<R64, 1> = [v + w].into();
        prop_assert!(
            chart.to_global(ev).compose(&chart.to_global(ew)) == chart.to_global(evw)
        );
    }

    // S^1 is abelian: composition commutes
    #[test]
    fn s1_commutativity(a in arb_sphere1(), b in arb_sphere1()) {
        prop_assert!(a.compose(&b) == b.compose(&a));
    }
}

proptest! {
    // Equality on SO(3) is equality of cosets: [q] == [-q] for every q,
    // no matter which representative Quotient::new keeps.
    #[test]
    fn so3_antipodal_lifts_are_equal(g in arb_sphere3()) {
        let neg = S3(Sphere::new(-g.0.real(), -g.0.imag()));
        prop_assert!(So3::new(g) == So3::new(neg));
    }
}

#[test]
fn so3_equator_equality() {
    // 180° rotations lift to quaternions with real part exactly 0, where the
    // canonical-representative rule (real >= 0) accepts both lifts. Coset
    // equality must still hold there.
    let q = S3(Sphere::<3, Coords<R64, 3>>::new(
        R64::zero(),
        [R64::one(), R64::zero(), R64::zero()].into(),
    ));
    let neg_q = S3(Sphere::new(-q.0.real(), -q.0.imag()));
    assert!(So3::new(q.clone()) == So3::new(neg_q));

    // and a 180° rotation about an arbitrary-ish axis
    let axis: Coords<R64, 3> = [R64(1.0), R64(2.0), R64(-0.5)].into();
    let axis = axis * (R64(1.0) / axis.norm());
    let half_turn = S3::identity_exp(axis * R64(std::f64::consts::PI));
    let neg = S3(Sphere::new(-half_turn.0.real(), -half_turn.0.imag()));
    assert!(So3::new(half_turn) == So3::new(neg));
}

#[test]
fn dirac_belt_trick() {
    let axis: Coords<R64, 3> = [R64::one(), R64::zero(), R64::zero()].into();
    let su2_identity = S3::identity();
    let so3_identity = So3::<Coords<R64, 3>>::identity();

    let half_period = R64(std::f64::consts::PI);
    let full_period = R64(std::f64::consts::TAU);

    // Half period: back to SO(3) identity, but NOT SU(2) identity
    let half_su2 = S3::identity_exp(axis * half_period);
    assert!(
        So3::new(half_su2.clone()) == so3_identity,
        "360° rotation should be identity in SO(3)"
    );
    assert!(
        half_su2 != su2_identity,
        "360° rotation should NOT be identity in SU(2) — the belt trick"
    );

    // Full period: back to SU(2) identity
    let full_su2 = S3::identity_exp(axis * full_period);
    assert!(
        full_su2 == su2_identity,
        "720° rotation should be identity in SU(2)"
    );
}

#[test]
fn s1_fundamental_group() {
    let cover = S1Cover::chart_at(&UnitComplex::identity());

    let pi1 = cover.fundamental_group();
    println!("generators: {}", pi1.n_generators());
    for (i, rel) in pi1.relations().into_iter().enumerate() {
        println!("relation {}: {:?}", i, rel);
    }

    for i in 0..S1Cover::nodes().len() {
        let neighbors: Vec<_> = S1Cover::get_neighbors(i).collect();
        println!(
            "node {}: {} neighbors {:?}",
            i,
            neighbors.len(),
            neighbors.iter().collect::<Vec<_>>()
        );
    }

    // debug: check what neighbours each node sees
    for i in 0..S1Cover::nodes().len() {
        let neighbors: Vec<_> = S1Cover::get_neighbors(i).collect();
        println!(
            "node {}: {:?} -> {} neighbors",
            i,
            S1Cover::nodes()[i],
            neighbors.len()
        );
    }

    let pi1 = cover.fundamental_group();
    println!(
        "generators: {}, relations: {}",
        pi1.n_generators(),
        pi1.relations().into_iter().count()
    );

    assert_eq!(pi1.n_generators(), 1);
    assert_eq!(pi1.relations().into_iter().count(), 0);
}

#[test]
fn so3_fundamental_group() {
    let cover = So3Cover::chart_at(&So3::identity());

    for i in 0..11 {
        let tv = So3::<Coords<R64, 3>>::identity_log(&So3Cover::nodes()[i].base_point());
        println!("node {}: {:?}", i, tv);
    }

    let pi1 = cover.fundamental_group();

    let n = pi1.n_generators();
    let relations: Vec<_> = pi1.relations().into_iter().collect();

    println!("f := FreeGroup({});", n);
    println!("x := List([1..{}], i -> GeneratorsOfGroup(f)[i]);;", n);
    print!("rels := [");

    for (i, &rel) in relations.iter().enumerate() {
        if i > 0 {
            print!(",");
        }
        print!("\n  ");

        let word: Vec<(usize, bool)> = rel.clone().into_iter().collect();
        if word.is_empty() {
            print!("x[1]*x[1]^-1"); // identity relation
        } else {
            for (j, &(gens, inv)) in word.iter().enumerate() {
                if j > 0 {
                    print!("*");
                }
                if inv {
                    print!("x[{}]^-1", gens + 1);
                } else {
                    print!("x[{}]", gens + 1);
                }
            }
        }
    }

    println!("\n];;");
    println!("G := f/rels;;");
    println!("Size(G);");

    let nodes = So3Cover::nodes();
    for i in 0..11 {
        for j in (i + 1)..11 {
            if let Some(d) = nodes[i].local_distance(&nodes[j].base_point()) {
                println!("dist {}-{}: {:.6}", i, j, d);
            }
        }
    }

    for i in 0..So3Cover::nodes().len() {
        let neighbors: Vec<_> = So3Cover::get_neighbors(i).collect();
        println!(
            "node {}: {} neighbors {:?}",
            i,
            neighbors.len(),
            neighbors.iter().collect::<Vec<_>>()
        );
    }

    let pi1 = cover.fundamental_group();
    println!("generators: {}", pi1.n_generators());
    for (i, rel) in pi1.relations().into_iter().enumerate() {
        println!("relation {}: {:?}", i, rel);
    }

    assert_eq!(pi1.n_generators(), 1);
    let relations: Vec<_> = pi1.relations().into_iter().collect();
    assert_eq!(relations.len(), 1);
    let expected = vec![(0usize, false), (0usize, false)];
    assert!(relations[0].clone().into_iter().eq(expected.into_iter()));
}

#[test]
fn so3_check_graph_structure() {
    // The nerve of the So3Cover ball cover is the hemi-600-cell: the
    // 60-vertex vertex-transitive triangulation of RP^3 obtained from the
    // boundary complex of the 600-cell by identifying antipodal vertices.
    // Its f-vector is (60, 360, 600, 300); here we verify the 1- and
    // 2-skeleton, which is what fundamental_group consumes.
    //
    // (Why not Walkup's 11-vertex RP^3_11? Its graph is K_11 minus 4 edges,
    // which contains ~129 triangles while the complex has only 80 2-faces —
    // so no cover whose 2-simplices are detected as mutually-overlapping
    // triples can ever reproduce it. The hemi-600-cell is "flag" in the
    // relevant sense: mutually overlapping triples of balls all genuinely
    // share a point, and every such triple is a 2-face.)
    let nodes = So3Cover::nodes();
    let n = nodes.len();
    assert_eq!(n, 60);

    let neighbors: Vec<Vec<usize>> = (0..n)
        .map(|i| So3Cover::get_neighbors(i).collect())
        .collect();

    // vertex-transitive: every node has exactly 12 neighbours,
    // at distance pi/5 (the 600-cell edge length)
    for (i, nbrs) in neighbors.iter().enumerate() {
        assert_eq!(nbrs.len(), 12, "node {} has wrong degree", i);
        for &j in nbrs {
            let d = nodes[i].local_distance(&nodes[j].base_point()).unwrap();
            assert!(
                d == R64(std::f64::consts::PI / 5.0),
                "edge {}-{} has length {} != pi/5",
                i,
                j,
                d
            );
        }
    }

    // adjacency is symmetric
    for i in 0..n {
        for &j in &neighbors[i] {
            assert!(neighbors[j].contains(&i), "asymmetric edge {}-{}", i, j);
        }
    }

    let mut edges = std::collections::HashSet::new();
    let mut triangles = std::collections::HashSet::new();
    for i in 0..n {
        for &j in &neighbors[i] {
            if j > i {
                edges.insert((i, j));
            }
            for &k in &neighbors[i] {
                if k <= j {
                    continue;
                }
                if j > i && neighbors[j].contains(&k) {
                    triangles.insert((i, j, k));
                }
            }
        }
    }

    println!("edges: {}, triangles: {}", edges.len(), triangles.len());
    assert_eq!(edges.len(), 360);
    assert_eq!(triangles.len(), 600);
}
