#![cfg(feature = "testing")]

#[macro_use]
mod common;

use common::*;

use diffable::{
    coords::Coords,
    epsilon_metric::R64,
    hypersphere::{S1Cover, So3, So3Cover, Sphere, Stereographic},
    test_chart, test_exp_map, test_lie_group, test_metric, test_tangent_bundle,
    traits::{
        Chart, ExpMap, Group, GroupPresentation, LieGroup, Metric, NerveComplex, Quotient,
        TangentBundle,
    },
};

use num_traits::{One, Zero};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Instantiations — adding a new manifold is one line per trait
// ---------------------------------------------------------------------------

// Stereographic chart roundtrips
test_chart!(stereo_s0, Stereographic<_>, arb_sphere0());
test_chart!(stereo_s1, Stereographic<_>, arb_sphere1());
test_chart!(stereo_s3, Stereographic<_>, arb_sphere3());

// Sphere as TangentBundle (via blanket LieGroup impl; includes all ExpMap tests)
test_tangent_bundle!(tangent_bundle_s0, Sphere<_, _>, Sphere<_, _>, arb_sphere0(), arb_vec0());
test_tangent_bundle!(tangent_bundle_s1, Sphere<_, _>, Sphere<_, _>, arb_sphere1(), arb_vec1());
test_tangent_bundle!(tangent_bundle_s3, Sphere<_, _>, Sphere<_, _>, arb_sphere3(), arb_vec3());
test_tangent_bundle!(
    tangent_bundle_so3,
    So3<Coords<_, _>>,
    So3<Coords<_, _>>,
    arb_so3(),
    arb_vec3()
);

// Lie group axioms
test_lie_group!(lie_group_s0, Sphere<_, _>, arb_sphere0());
test_lie_group!(lie_group_s1, Sphere<_, _>, arb_sphere1());
test_lie_group!(lie_group_s3, Sphere<_, _>, arb_sphere3());
test_lie_group!(lie_group_so3, So3<Coords<_, _>>, arb_so3());

// Metric axioms
test_metric!(metric_s0, Sphere<_, _>, arb_sphere0());
test_metric!(metric_s1, Sphere<_, _>, arb_sphere1());
test_metric!(metric_s3, Sphere<_, _>, arb_sphere3());

// ---------------------------------------------------------------------------
// Bespoke tests: properties specific to these manifolds, not general laws
// ---------------------------------------------------------------------------
proptest! {
    // S^1 is abelian, so exp is a group homomorphism: exp(v+w) = exp(v) * exp(w)
    #[test]
    fn s1_exp_homomorphism(v in -1.5f64..1.5f64, w in -1.5f64..1.5f64) {
        let (v, w) = (R64(v), R64(w));
        let chart = Sphere::<1, _>::identity();
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

#[test]
fn dirac_belt_trick() {
    let axis: Coords<R64, 3> = [R64::one(), R64::zero(), R64::zero()].into();
    let su2_identity = Sphere::<3, Coords<R64, 3>>::identity();
    let so3_identity = So3::<Coords<R64, 3>>::identity();

    let half_period = R64(std::f64::consts::PI);
    let full_period = R64(std::f64::consts::TAU);

    // Half period: back to SO(3) identity, but NOT SU(2) identity
    let half_su2 = Sphere::<3, Coords<R64, 3>>::identity_exp(axis * half_period);
    assert!(
        So3::new(half_su2.clone()) == so3_identity,
        "360° rotation should be identity in SO(3)"
    );
    assert!(
        half_su2 != su2_identity,
        "360° rotation should NOT be identity in SU(2) — the belt trick"
    );

    // Full period: back to SU(2) identity
    let full_su2 = Sphere::<3, Coords<R64, 3>>::identity_exp(axis * full_period);
    assert!(
        full_su2 == su2_identity,
        "720° rotation should be identity in SU(2)"
    );
}

#[test]
fn s1_fundamental_group() {
    let cover = S1Cover::chart_at(&Sphere::identity());

    let pi1 = cover.fundamental_group();
    println!("generators: {}", pi1.n_generators());
    for (i, rel) in pi1.relations().into_iter().enumerate() {
        println!("relation {}: {:?}", i, rel);
    }

    for i in 0..S1Cover::nodes().len() {
        let chart = S1Cover::chart_at(&S1Cover::nodes()[i].base_point());
        let neighbors: Vec<_> = chart.get_neighbors().collect();
        println!(
            "node {}: {} neighbors {:?}",
            i,
            neighbors.len(),
            neighbors.iter().collect::<Vec<_>>()
        );
    }

    // debug: check what neighbours each node sees
    for i in 0..S1Cover::nodes().len() {
        let chart = S1Cover::chart_at(&S1Cover::nodes()[i].base_point());
        let neighbors: Vec<_> = chart.get_neighbors().collect();
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
        let chart = So3Cover::chart_at(&So3Cover::nodes()[i].base_point());
        let neighbors: Vec<_> = chart.get_neighbors().collect();
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
    // verify we get exactly the Walkup triangulation
    let walkup_facets: &[[usize; 4]] = &[
        [0, 1, 2, 3],
        [0, 1, 2, 4],
        [0, 1, 3, 5],
        [0, 1, 4, 5],
        [0, 2, 3, 6],
        [0, 2, 4, 6],
        [0, 3, 5, 6],
        [0, 4, 5, 6],
        [1, 2, 3, 7],
        [1, 2, 4, 8],
        [1, 2, 7, 8],
        [1, 3, 5, 9],
        [1, 3, 7, 9],
        [1, 4, 5, 10],
        [1, 4, 8, 10],
        [1, 5, 9, 10],
        [1, 6, 7, 8],
        [1, 6, 7, 9],
        [1, 6, 8, 10],
        [1, 6, 9, 10],
        [2, 3, 6, 10],
        [2, 3, 7, 10],
        [2, 4, 6, 9],
        [2, 4, 8, 9],
        [2, 5, 7, 8],
        [2, 5, 7, 10],
        [2, 5, 8, 9],
        [2, 5, 9, 10],
        [2, 6, 9, 10],
        [3, 4, 7, 9],
        [3, 4, 7, 10],
        [3, 4, 8, 9],
        [3, 4, 8, 10],
        [3, 5, 6, 8],
        [3, 5, 8, 9],
        [3, 6, 8, 10],
        [4, 5, 6, 7],
        [4, 5, 7, 10],
        [4, 6, 7, 8],
        [5, 6, 7, 8],
    ];

    let mut walkup_edges = std::collections::HashSet::new();
    let mut walkup_triangles = std::collections::HashSet::new();

    for facet in walkup_facets {
        for i in 0..4 {
            for j in (i + 1)..4 {
                let a = facet[i].min(facet[j]);
                let b = facet[i].max(facet[j]);
                walkup_edges.insert((a, b));
                for k in (j + 1)..4 {
                    let mut t = [facet[i], facet[j], facet[k]];
                    t.sort();
                    walkup_triangles.insert((t[0], t[1], t[2]));
                }
            }
        }
    }

    let nodes = So3Cover::nodes();
    let n = nodes.len();
    let neighbors: Vec<Vec<usize>> = (0..n)
        .map(|i| {
            So3Cover::chart_at(&nodes[i].base_point())
                .get_neighbors()
                .collect()
        })
        .collect();

    let mut our_edges = std::collections::HashSet::new();
    let mut our_triangles = std::collections::HashSet::new();

    for i in 0..n {
        for &j in &neighbors[i] {
            if j > i {
                our_edges.insert((i, j));
            }
            for &k in &neighbors[i] {
                if k <= j {
                    continue;
                }
                if neighbors[j].contains(&k) {
                    our_triangles.insert((i, j, k));
                }
            }
        }
    }

    println!(
        "walkup edges: {}, our edges: {}",
        walkup_edges.len(),
        our_edges.len()
    );
    println!(
        "walkup triangles: {}, our triangles: {}",
        walkup_triangles.len(),
        our_triangles.len()
    );
    println!(
        "missing edges: {:?}",
        walkup_edges.difference(&our_edges).collect::<Vec<_>>()
    );
    println!(
        "extra edges: {:?}",
        our_edges.difference(&walkup_edges).collect::<Vec<_>>()
    );
    println!(
        "missing triangles: {}",
        walkup_triangles.difference(&our_triangles).count()
    );
    println!(
        "extra triangles: {}",
        our_triangles.difference(&walkup_triangles).count()
    );

    assert_eq!(our_edges, walkup_edges);
    assert_eq!(our_triangles, walkup_triangles);
    todo!()
}
