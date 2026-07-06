use num_traits::{One, Zero, real::Real};

use super::{Euclidean, ExpMap, Point, TangentBundle};

/// A presentation of a group by generators and relations.
///
/// A group presentation `⟨S | R⟩` consists of a set of generators `S`
/// and a set of relations `R` — words in the generators that evaluate to
/// the identity. The presented group is the free group on `S` quotiented
/// by the normal closure of `R`.
///
/// By implementing this trait, you certify that your type faithfully
/// represents such a presentation — the generators are indexed `0..n_generators`,
/// and the relations are words in those generators (pairs of generator index
/// and whether it appears inverted).
///
/// The specific storage container is immaterial — only the mathematical
/// content (the generators and relations as iterable sequences) matters.
pub trait GroupPresentation {
    type Word: IntoIterator<Item = (usize, bool)> + Clone + std::fmt::Debug;
    type Relations<'a>: IntoIterator<Item = &'a Self::Word> + std::fmt::Debug
    where
        <Self as GroupPresentation>::Word: 'a,
        Self: 'a;

    /// The number of generators in the presentation.
    fn n_generators(&self) -> usize;

    /// The relations — words in the generators that evaluate to the identity.
    fn relations(&self) -> Self::Relations<'_>;
}

/// A finite collection of [`TangentBundle`] charts whose injectivity domains
/// together cover a manifold `P`, forming the nerve of the cover as a
/// simplicial complex.
///
/// # What makes this special
/// Every atlas covers its manifold by definition — that is not what
/// distinguishes `NerveComplex`. What is special is fourfold:
///
/// - **Finiteness**: the charts can be explicitly enumerated via [`Self::nodes`]
/// - **Geodesic structure**: each node is a [`TangentBundle`], so distances
///   within each injectivity domain are exact, not merely approximate
/// - **Centring**: each node is centred at its own base point, so the graph
///   of overlapping injectivity domains faithfully represents the manifold's
///   geometry
/// - **Simplicial structure**: overlapping charts form a simplicial complex —
///   0-simplices (nodes), 1-simplices (overlapping pairs), 2-simplices
///   (mutually overlapping triples), and so on — whose homotopy type matches
///   the manifold by the nerve theorem
///
/// Together these properties reduce global geodesic distance to an exact
/// graph search problem, and make the full homotopy type of the manifold
/// — including `π₁(M)`, `π₂(M)`, and higher — recoverable from the
/// intersection pattern of the cover.
///
/// # Compactness
/// When implemented with [`Bounded`] nodes (charts with explicitly bounded
/// domains), `NerveComplex` provides a constructive proof that `P` is
/// compact — finitely many bounded open sets cover `P` if and only if `P`
/// is compact. With unbounded nodes (e.g. on flat manifolds where the exp
/// map is globally defined), `NerveComplex` can be implemented for
/// non-compact manifolds and makes no compactness claim.
///
/// # The covering invariant
/// The implementor certifies that for every point `p: P`, at least one
/// node `n` in `Self::nodes()` satisfies `n.to_local(p).is_some()` — i.e.
/// `p` lies within `n`'s injectivity domain. This invariant is not a
/// separate requirement: it is automatically certified by the [`Chart`]
/// contract inherited via [`ExpMap`]. Specifically, `chart_at(p)` must
/// return a chart covering `p`, and since `chart_at` finds its chart from
/// `Self::nodes()`, the covering invariant follows directly from
/// `check_local_inverse` passing in `test_chart!`. No additional tests
/// are needed beyond those already required by the trait hierarchy.
///
/// # The nerve theorem
/// Since injectivity domains are star-shaped (hence contractible), the
/// nerve theorem guarantees that the simplicial complex formed by the
/// cover has the same homotopy type as `M`. This makes `π₁(M)` recoverable
/// from the spanning tree of the 1-skeleton (the overlap graph), with
/// relations arising from 2-simplices (triangles — triple intersections),
/// and higher homotopy groups `πₙ(M)` recoverable from `n`-simplices.
/// The `'static` lifetime on `nodes()` is load-bearing: it guarantees that
/// `nodes()` returns the same slice on every call, making `chart_at`
/// always search the same fixed set and the covering invariant follow
/// from `check_local_inverse`.
///
/// # Implementing
/// Nodes should be spaced such that every point lies within the injectivity
/// domain of at least one node. For principled node spacing, use the Rauch
/// bound `π / √κ_max` (computable via [`TangentBundle::max_sectional_curvature`])
/// as the cover radius at each node — this guarantees the radius stays
/// within the injectivity domain. Sampling density must additionally satisfy
/// `d < 2π / √κ_max` (twice the Rauch bound) to ensure adjacent nodes
/// overlap and the nerve faithfully captures the topology. Near high-curvature
/// regions, both the radius and the required sampling density shrink
/// proportionally — the cover automatically adapts to the geometry.
///
/// [`Chart`]: crate::traits::Chart
pub trait NerveComplex<
    P: Point,
    V: Euclidean,
    T: TangentBundle<P, V> + Point,
    B: Bounded<T, V> + 'static,
>: ExpMap<T, V>
{
    /// Returns the fixed set of [`TangentBundle`] charts that cover the manifold.
    ///
    /// This function must be *effectively pure* — it must return the same nodes
    /// on every call, since the nodes are a property of the type, not of any
    /// particular instance. The idiomatic way to enforce this is via
    /// [`std::sync::LazyLock`] or [`std::sync::OnceLock`], which guarantee the
    /// initialiser runs exactly once regardless of how many times `nodes()` is
    /// called:
    ///
    /// ```rust,ignore
    /// fn nodes() -> &'static [MyNode] {
    ///     static NODES: LazyLock<Vec<MyNode>> = LazyLock::new(|| {
    ///         // compute nodes here, runs exactly once
    ///     });
    ///     &NODES
    /// }
    /// ```
    ///
    /// The `'static` lifetime is load-bearing: it guarantees that `nodes()`
    /// returns the *same* slice on every call — not merely an equal one, but
    /// the identical allocation. This makes `nodes()` effectively a pure
    /// function at the memory level, which in turn means that `chart_at(p)`
    /// always searches the same fixed set of nodes. Since [`check_local_inverse`]
    /// verifies that `chart_at(p).to_local(p).is_some()` for arbitrary `p`,
    /// and `chart_at` finds its chart from this fixed `nodes()`, the covering
    /// invariant — every point is covered by at least one node — is automatically
    /// certified by the existing [`Chart`] test infrastructure. No additional
    /// `check_*` methods or `test_*` macros are needed.
    ///
    /// [`Chart`]: crate::traits::Chart
    /// [`check_local_inverse`]: crate::traits::Chart::check_local_inverse
    fn nodes() -> &'static [B];

    /// Returns the indices of nodes whose bounded domains overlap the
    /// bounded domain of this node — the 1-skeleton of the nerve.
    ///
    /// # The overlap test
    /// Two domains are declared to overlap when the *geodesic midpoint* of
    /// the two base points lies strictly inside both domains (as measured by
    /// each node's own [`Bounded::sdf`] in its own chart). This is sound for
    /// any star-shaped domains — a common point is a common point — and it
    /// is *exact* when the domains are geodesic balls of equal radius `ρ`:
    /// two such balls intersect iff `d(p_i, p_j) < 2ρ`, iff the midpoint
    /// (at distance `d/2` from each centre) lies in both.
    ///
    /// Note that testing whether each centre lies inside the *other* domain
    /// is not the same thing: balls of radius `ρ` already overlap at centre
    /// separation `2ρ`, but their centres only see each other at separation
    /// `ρ`. The midpoint test reports the true intersection, which is what
    /// the nerve theorem needs.
    ///
    /// The default implementation is an `O(n)` linear scan over all nodes.
    /// Override this for better performance if your cover has additional
    /// structure (e.g. a spatial index or precomputed adjacency list).
    fn get_neighbors(i: usize) -> impl Iterator<Item = usize> {
        let nodes = Self::nodes();
        let inode = &nodes[i];
        let base = inode.base_point();
        nodes.iter().enumerate().filter_map(move |(j, jnode)| {
            if j == i {
                return None;
            }
            let half = (V::F::one() + V::F::one()).recip();
            match (
                inode.inner().to_local(&jnode.base_point().base_point()),
                jnode.inner().to_local(&base.base_point()),
            ) {
                (Some(v_ij), Some(v_ji))
                    if inode.sdf(&(v_ij * half)) < V::F::zero()
                        && jnode.sdf(&(v_ji * half)) < V::F::zero() =>
                {
                    Some(j)
                }
                _ => None,
            }
        })
    }

    /// Computes the fundamental group π₁(M) of the manifold from the
    /// graph structure of this cover via the spanning tree construction.
    ///
    /// By the nerve theorem, since the domains are contractible and cover
    /// the manifold (with contractible intersections — a *good* cover),
    /// the nerve of this cover has the same homotopy type as `M`. The
    /// fundamental group is therefore recoverable purely from the graph of
    /// overlapping domains (generators) and the triangles of the nerve
    /// (relations).
    ///
    /// The returned presentation is Tietze-simplified: generators are
    /// eliminated using the relations wherever possible, so the presentation
    /// returned is a small — often minimal — presentation of π₁(M) rather
    /// than the raw one-generator-per-non-tree-edge presentation, which for
    /// interesting covers can have hundreds of generators.
    fn fundamental_group(&self) -> impl GroupPresentation {
        let nodes = Self::nodes();
        let n = nodes.len();

        // BFS spanning tree
        let mut parent: Vec<Option<usize>> = vec![None; n];
        let mut visited: Vec<bool> = vec![false; n];
        let mut queue = std::collections::VecDeque::new();
        visited[0] = true;
        queue.push_back(0usize);
        while let Some(idx) = queue.pop_front() {
            for neighbour_idx in Self::get_neighbors(idx) {
                if !visited[neighbour_idx] {
                    visited[neighbour_idx] = true;
                    parent[neighbour_idx] = Some(idx);
                    queue.push_back(neighbour_idx);
                }
            }
        }

        // generators: non-tree edges (i < j to avoid duplicates)
        let generators: Vec<(usize, usize)> = (0..n)
            .flat_map(|i| {
                let parent = &parent;
                Self::get_neighbors(i)
                    .filter_map(move |j| {
                        if i < j && parent[j] != Some(i) && parent[i] != Some(j) {
                            Some((i, j))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        let path_to_root = |mut idx: usize| -> Vec<usize> {
            let mut path = vec![idx];
            while let Some(p) = parent[idx] {
                path.push(p);
                idx = p;
            }
            path.reverse();
            path
        };

        let path_to_word = |path: Vec<usize>| -> Vec<(usize, bool)> {
            path.windows(2)
                .filter_map(|w| {
                    let (a, b) = (w[0], w[1]);
                    generators
                        .iter()
                        .enumerate()
                        .find_map(|(gen_idx, &(x, y))| {
                            if (x, y) == (a, b) {
                                Some((gen_idx, false))
                            } else if (x, y) == (b, a) {
                                Some((gen_idx, true))
                            } else {
                                None
                            }
                        })
                })
                .collect()
        };

        fn reduce_word(word: Vec<(usize, bool)>) -> Vec<(usize, bool)> {
            let mut reduced: Vec<(usize, bool)> = Vec::new();
            for letter in word {
                if let Some(&last) = reduced.last() {
                    if last == (letter.0, !letter.1) {
                        reduced.pop();
                        continue;
                    }
                }
                reduced.push(letter);
            }
            reduced
        }

        // relations come from triangles (triple intersections) in the nerve —
        // not from non-tree edges directly. For each triple (i,j,k) where all
        // three pairs are neighbours, the triangle boundary gives a relation:
        // the word formed by the cycle i→j→k→i expressed in generators.
        // π₁ of a graph is always free (no relations from edges alone);
        // relations only arise from 2-simplices (filled triangles) in the nerve.
        let edge_word = |a: usize, b: usize| -> Vec<(usize, bool)> {
            let mut path = path_to_root(a);
            path.extend(path_to_root(b).into_iter().rev());
            path_to_word(path)
        };

        let neighbors: Vec<Vec<usize>> = (0..n).map(|i| Self::get_neighbors(i).collect()).collect();

        let triangles: Vec<(usize, usize, usize)> = (0..n)
            .flat_map(|i| {
                let neighbors = &neighbors;
                neighbors[i].iter().flat_map(move |&j| {
                    neighbors[j].iter().filter_map(move |&k| {
                        (i < j && j < k && neighbors[i].contains(&k)).then_some((i, j, k))
                    })
                })
            })
            .collect();

        let relations: Vec<Vec<(usize, bool)>> = triangles
            .into_iter()
            .map(|(i, j, k)| {
                let mut w = edge_word(i, j);
                w.extend(edge_word(j, k));
                w.extend(edge_word(k, i));
                reduce_word(w)
            })
            .filter(|w| !w.is_empty())
            .collect();

        // -------------------------------------------------------------------
        // Tietze simplification.
        //
        // The raw presentation has one generator per non-tree edge and one
        // relation per triangle. For any good cover of a manifold with
        // non-trivial H₃ (e.g. any closed orientable 3-manifold) the nerve
        // must contain 3-simplices, whose 1-skeletons are K₄s, so the raw
        // presentation *necessarily* has ≥ 3 generators — no cover exists
        // whose raw presentation is minimal. Simplification is therefore
        // part of the computation, not a cosmetic afterthought.
        //
        // The moves used are classical Tietze transformations, which
        // preserve the isomorphism type of the presented group:
        //   - free + cyclic reduction of relators,
        //   - deletion of duplicate relators (up to rotation and inversion),
        //   - elimination of a generator that occurs exactly once in some
        //     relator, by solving that relator for it and substituting.
        // -------------------------------------------------------------------
        fn invert(w: &[(usize, bool)]) -> Vec<(usize, bool)> {
            w.iter().rev().map(|&(g, inv)| (g, !inv)).collect()
        }

        fn cyclic_reduce(mut w: Vec<(usize, bool)>) -> Vec<(usize, bool)> {
            w = reduce_word(w);
            while w.len() >= 2 {
                let (f, l) = (w[0], *w.last().unwrap());
                if f.0 == l.0 && f.1 != l.1 {
                    w.remove(0);
                    w.pop();
                    w = reduce_word(w);
                } else {
                    break;
                }
            }
            w
        }

        // canonical form of a relator up to cyclic rotation and inversion,
        // for duplicate detection
        fn canonical_relator(w: &[(usize, bool)]) -> Vec<(usize, bool)> {
            let mut best: Option<Vec<(usize, bool)>> = None;
            for cand in [w.to_vec(), invert(w)] {
                for r in 0..cand.len().max(1) {
                    let mut rot = cand.clone();
                    rot.rotate_left(r % cand.len().max(1));
                    if best.as_ref().is_none_or(|b| rot < *b) {
                        best = Some(rot);
                    }
                }
            }
            best.unwrap_or_default()
        }

        fn substitute(
            w: &[(usize, bool)],
            g: usize,
            replacement: &[(usize, bool)],
        ) -> Vec<(usize, bool)> {
            let inv_rep = invert(replacement);
            let mut out = Vec::new();
            for &(x, inv) in w {
                if x == g {
                    out.extend(if inv {
                        inv_rep.clone()
                    } else {
                        replacement.to_vec()
                    });
                } else {
                    out.push((x, inv));
                }
            }
            reduce_word(out)
        }

        let mut alive: Vec<bool> = vec![true; generators.len()];
        let mut rels: Vec<Vec<(usize, bool)>> = relations
            .into_iter()
            .map(cyclic_reduce)
            .filter(|w| !w.is_empty())
            .collect();

        loop {
            let mut seen = std::collections::HashSet::new();
            rels.retain(|w| seen.insert(canonical_relator(w)));
            rels.sort_by_key(|w| w.len());

            // find a relator in which some generator occurs exactly once
            let mut action: Option<(usize, Vec<(usize, bool)>, usize)> = None;
            'search: for (ri, r) in rels.iter().enumerate() {
                let mut counts = std::collections::HashMap::new();
                for &(g, _) in r {
                    *counts.entry(g).or_insert(0usize) += 1;
                }
                for (pos, &(g, inv)) in r.iter().enumerate() {
                    if counts[&g] == 1 {
                        // rotate r to start at g: r = g^e · rest == 1,
                        // so g^e = rest⁻¹
                        let mut rest: Vec<(usize, bool)> = Vec::new();
                        rest.extend_from_slice(&r[pos + 1..]);
                        rest.extend_from_slice(&r[..pos]);
                        let repl = if inv {
                            reduce_word(rest)
                        } else {
                            invert(&rest)
                        };
                        action = Some((g, repl, ri));
                        break 'search;
                    }
                }
            }

            match action {
                Some((g, repl, ri)) => {
                    rels.remove(ri);
                    alive[g] = false;
                    rels = rels
                        .iter()
                        .map(|w| cyclic_reduce(substitute(w, g, &repl)))
                        .filter(|w| !w.is_empty())
                        .collect();
                }
                None => break,
            }
        }

        // renumber the surviving generators to 0..k
        let mut remap = std::collections::HashMap::new();
        for (g, &a) in alive.iter().enumerate() {
            if a {
                let idx = remap.len();
                remap.insert(g, idx);
            }
        }
        let relations: Vec<Vec<(usize, bool)>> = rels
            .iter()
            .map(|w| {
                let w: Vec<(usize, bool)> = w.iter().map(|&(g, i)| (remap[&g], i)).collect();
                // prefer the mostly-uninverted form of each relator
                // (x·x rather than x⁻¹·x⁻¹)
                let inv_count = w.iter().filter(|&&(_, i)| i).count();
                if inv_count * 2 > w.len() {
                    invert(&w)
                } else {
                    w
                }
            })
            .collect();
        let n_generators = remap.len();

        #[derive(Debug, PartialEq)]
        struct FundamentalGroupPresentation {
            n_generators: usize,
            relations: Vec<Vec<(usize, bool)>>,
        }

        impl GroupPresentation for FundamentalGroupPresentation {
            type Word = Vec<(usize, bool)>;
            type Relations<'a> = &'a [Vec<(usize, bool)>];

            fn n_generators(&self) -> usize {
                self.n_generators
            }

            fn relations(&self) -> Self::Relations<'_> {
                &self.relations
            }
        }

        FundamentalGroupPresentation {
            n_generators,
            relations,
        }
    }
}

/// Restricts the domain of [`Chart::to_local`] to some subset
/// defined by a signed distance function in the tangent space
/// at each point on a [`TangentBundle`].
///
/// [`Chart::to_local`]: crate::traits::Chart::to_local
/// [`TangentBundle`]: crate::traits::TangentBundle
pub trait Bounded<P: Point, V: Euclidean>: TangentBundle<P, V> {
    /// The signed distance field in the tangent
    /// space of the chart centered at &self.
    fn sdf(&self, v: &V) -> V::F;
    fn new(p: P) -> Self;
    fn inner(&self) -> &P;
}

/// Implements [`Chart`], [`ExpMap`], and [`TangentBundle`] for `$type` by
/// delegating to `$type`'s [`Bounded::inner`] chart, restricting `to_local`
/// to the region where [`Bounded::sdf`] is negative.
///
/// A `Bounded` chart is not automatically a `Chart` -- its domain is
/// smaller than the chart it wraps -- so this macro does the one
/// mechanical thing every `Bounded` implementor needs: reject points
/// outside the signed-distance-field boundary, and defer everything else
/// to the inner chart.
///
/// [`Chart`]: crate::traits::Chart
/// [`ExpMap`]: crate::traits::ExpMap
/// [`TangentBundle`]: crate::traits::TangentBundle
#[macro_export]
macro_rules! impl_tangent_bundle_via_bounded {
    ($type:ty, $point:ty, $v:ty) => {
        impl $crate::traits::Chart<$point, $v> for $type {
            fn to_local(&self, point: &$point) -> Option<$v> {
                self.inner()
                    .to_local(point)
                    .filter(|v| self.sdf(v) < <$v as $crate::traits::Euclidean>::F::zero())
            }
            fn to_global(&self, coord: $v) -> $point {
                self.inner().to_global(coord)
            }

            fn chart_at(p: &$point) -> Self {
                Self::new(<$point>::chart_at(p))
            }
        }

        impl $crate::traits::ExpMap<$point, $v> for $type {}
        impl $crate::traits::TangentBundle<$point, $v> for $type {}
    };
}
