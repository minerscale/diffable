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
pub trait GroupPresentation: std::fmt::Debug {
    type Word: IntoIterator<Item = (usize, bool), IntoIter: ExactSizeIterator>
        + Clone
        + std::fmt::Debug;
    type Relations<'a>: IntoIterator<Item = &'a Self::Word, IntoIter: ExactSizeIterator>
        + std::fmt::Debug
    where
        <Self as GroupPresentation>::Word: 'a,
        Self: 'a;

    /// The number of generators in the presentation.
    fn n_generators(&self) -> usize;

    /// The relations — words in the generators that evaluate to the identity.
    fn relations(&self) -> Self::Relations<'_>;

    fn check_exactly_equal(&self, other: &impl GroupPresentation) -> bool {
        if self.n_generators() != other.n_generators() {
            return false;
        }
        let self_iter = self.relations().into_iter();
        let other_iter = other.relations().into_iter();
        if self_iter.len() != other_iter.len() {
            return false;
        }
        self_iter.zip(other_iter).all(|(a, b)| {
            let a_iter = a.clone().into_iter();
            let b_iter = b.clone().into_iter();
            a_iter.len() == b_iter.len() && a_iter.zip(b_iter).all(|(x, y)| x == y)
        })
    }
}

/// A single relation, borrowed as a sub-slice of some
/// [`StaticGroupPresentation`]'s single flat backing array.
#[derive(Debug, Clone, Copy)]
pub struct StaticWord(pub &'static [(usize, bool)]);

impl IntoIterator for StaticWord {
    type Item = (usize, bool);
    type IntoIter = std::iter::Copied<std::slice::Iter<'static, (usize, bool)>>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter().copied()
    }
}

#[derive(Debug)]
pub struct StaticGroupPresentation {
    pub n_generators: usize,
    /// Every relation's (usize, bool) pairs, concatenated in order,
    /// as one contiguous array. This is the field that actually
    /// carries the cache-locality guarantee.
    pub flat_words: &'static [(usize, bool)],
    /// One handle per relation, each a sub-slice of `flat_words`.
    /// Exists to satisfy `GroupPresentation::Relations`'s
    /// `Item = &'a Word` shape (an iterator over *references* to
    /// stored words) — `flat_words` alone can't be handed out this
    /// way, since the trait wants one addressable `Word` per
    /// relation, not just a flat stream of pairs with no relation
    /// boundaries.
    pub words: &'static [StaticWord],
}

impl StaticGroupPresentation {
    pub const fn n_relations(&self) -> usize {
        self.words.len()
    }
}

impl GroupPresentation for StaticGroupPresentation {
    type Word = StaticWord;
    type Relations<'a> = &'a [StaticWord];

    fn n_generators(&self) -> usize {
        self.n_generators
    }

    fn relations(&self) -> Self::Relations<'_> {
        self.words
    }
}

// -------------------------------------------------------------------
// static_presentation!(NAME, n_generators = N, relations = [
//     [(gen, inverted), ...],
//     ...
// ]);
//
// Builds `flat_words` as one array literal containing every pair
// from every relation, in order (trivial — just concatenation via
// macro repetition). Builds `words` by const-slicing that same flat
// array at each relation's boundary, using `const fn` array
// arithmetic evaluated entirely at compile time — no runtime
// computation, and critically, both arrays end up backed by the
// same underlying `flat_words` allocation, which is what makes the
// contiguity real rather than incidental.
// -------------------------------------------------------------------
#[macro_export]
macro_rules! group_presentation {
    (
        $vis:vis $name:ident,
        n_generators = $n:expr,
        relations = [ $( [ $( ($g:expr, $inv:expr) ),* $(,)? ] ),* $(,)? ]
    ) => {
        $vis static $name: $crate::traits::StaticGroupPresentation = {
            // one flat array: every relation's pairs, concatenated
            const FLAT: &[(usize, bool)] = &[
                $( $( ($g, $inv) ),* ),*
            ];

            // length of each individual relation, in order — each
            // inner array literal's `.len()` is const-evaluable
            const LENS: &[usize] = &[ $( [ $( ($g, $inv) ),* ].len() ),* ];

            const fn offsets() -> [usize; LENS.len() + 1] {
                let mut out = [0usize; LENS.len() + 1];
                let mut i = 0;
                while i < LENS.len() {
                    out[i + 1] = out[i] + LENS[i];
                    i += 1;
                }
                out
            }
            const OFFSETS: [usize; LENS.len() + 1] = offsets();

            const fn build_words() -> [$crate::traits::StaticWord; LENS.len()] {
                let mut out = [$crate::traits::StaticWord(&[]); LENS.len()];
                let mut i = 0;
                while i < LENS.len() {
                    let start = OFFSETS[i];
                    let end = OFFSETS[i + 1];
                    let (_, rest) = FLAT.split_at(start);
                    let (word, _) = rest.split_at(end - start);
                    out[i] = $crate::traits::StaticWord(word);
                    i += 1;
                }
                out
            }
            static WORDS: [$crate::traits::StaticWord; LENS.len()] = build_words();

            $crate::traits::StaticGroupPresentation {
                n_generators: $n,
                flat_words: FLAT,
                words: &WORDS,
            }
        };
    };
}

// ============================================================
// module: nodes_cache
//
// Provides "exactly one covering per type, globally, for the
// life of the process" — the covering is a fact ABOUT the type,
// not per-instance or per-thread state, so this storage must be
// reachable from anywhere, hence Send + Sync. That bound is not
// incidental plumbing: a type whose covering cannot be safely
// shared across threads is a type that cannot have a single
// canonical global covering, which would contradict the "type
// IS its set of nodes" philosophy anyway.
// ============================================================
mod nodes_cache {
    use std::any::{Any, TypeId};
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    fn registry() -> &'static Mutex<HashMap<(TypeId, TypeId), &'static (dyn Any + Send + Sync)>> {
        static REGISTRY: OnceLock<
            Mutex<HashMap<(TypeId, TypeId), &'static (dyn Any + Send + Sync)>>,
        > = OnceLock::new();
        REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
    }

    fn slot_for<Caller: 'static + ?Sized, B: 'static + Send + Sync>()
    -> &'static OnceLock<&'static [B]> {
        let key = (TypeId::of::<Caller>(), TypeId::of::<B>());
        let mut map = registry().lock().unwrap();
        let entry = map.entry(key).or_insert_with(|| {
            let slot: &'static OnceLock<&'static [B]> = Box::leak(Box::new(OnceLock::new()));
            slot as &'static (dyn Any + Send + Sync)
        });
        entry
            .downcast_ref::<OnceLock<&'static [B]>>()
            .expect("TypeId collision — should be impossible")
    }

    /// Runs `build` at most once, ever, for this specific
    /// (Caller, B) pair. Two different `Caller` types building
    /// the same `B` get independent, non-colliding cache slots —
    /// this is intentional: `NerveComplex` allows multiple distinct
    /// cover types over the same node type `B`.
    pub fn get_or_build<Caller: 'static + ?Sized, B: 'static + Send + Sync>(
        build: impl FnOnce() -> Vec<B>,
    ) -> &'static [B] {
        *slot_for::<Caller, B>().get_or_init(|| Box::leak(build().into_boxed_slice()))
    }
}

/// The one-time recipe for producing a fixed node set `Vec<B>`.
///
/// Implement this — never [`Nodes::nodes`] directly, which is
/// blanket-implemented and cannot be overridden (see [`Nodes`]).
/// `build_nodes` is invoked at most once, ever, for a given
/// implementing type; its result is memoized process-wide by
/// [`Nodes::nodes`], keyed on the pair `(Self, B)`.
///
/// Distinct implementing types are permitted to build the same `B`
/// independently, and a single implementing type is permitted to
/// build several different `B`s — each `(Self, B)` pair gets its
/// own separately-memoized node set, with no collision between them.
/// This is intentional: it lets multiple distinct cover types exist
/// over the same underlying bounded chart type.
pub trait BuildNodes<B: 'static + Send + Sync> {
    fn build_nodes() -> Vec<B>;
}

/// Sealed accessor for a memoized, canonical node set.
///
/// `nodes()` is guaranteed — by construction, not by convention —
/// to return the exact same `'static` slice (same allocation, same
/// pointer, same length) on every call, from any thread, for the
/// life of the process. [`BuildNodes::build_nodes`] is invoked at
/// most once per implementing type to produce it; every subsequent
/// call, from anywhere, receives the identical slice rather than a
/// freshly computed one.
///
/// # Why this cannot be overridden
/// This trait is never implemented directly. It is blanket-implemented
/// for every `T: BuildNodes<B>`, and Rust's coherence rules forbid a
/// second, competing `impl Nodes<B> for T` from ever existing for the
/// same `T`. This makes "exactly one canonical node set per
/// `(implementing type, B)` pair" a fact enforced by the trait
/// hierarchy itself — not a discipline implementors must remember to
/// uphold. There is no method named `nodes` for an implementor to
/// accidentally shadow; the only thing an implementor ever writes is
/// [`BuildNodes::build_nodes`].
///
/// # Why `Send + Sync`
/// The node set is a property of the *type*, not of any particular
/// call site, instance, or thread — it is a fact about what the type
/// *is*, not state owned by any one caller. Because of this, the
/// canonical storage backing `nodes()` must be safely reachable from
/// any thread that might ask for it, which is exactly what
/// `Send + Sync` certifies. A type that cannot satisfy `Send + Sync`
/// cannot have a single global canonical node set, which would
/// contradict the premise that the node set is inherent to the type
/// rather than to any particular thread or instance.
pub trait Nodes<B: 'static + Send + Sync>: BuildNodes<B> + 'static {
    fn nodes() -> &'static [B] {
        nodes_cache::get_or_build::<Self, B>(Self::build_nodes)
    }
}

// Blanket impl guarantees that nodes() can never be overridden: any
// second `impl Nodes<B> for T` would conflict with this one under
// Rust's coherence rules and fail to compile. Caching is keyed on
// the pair (T, B) — see `nodes_cache` — so distinct implementing
// types sharing the same B, or a single type building several
// different Bs, are all independently memoized with no collisions.
impl<B: 'static + Send + Sync, T: BuildNodes<B> + 'static + Send + Sync> Nodes<B> for T {}

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
///
/// [`Self::nodes`] (via [`Nodes`]) is guaranteed, not merely conventionally
/// expected, to return the identical `'static` slice — same allocation,
/// same pointer — on every call. This is enforced structurally by
/// [`Nodes`]'s sealing (see its documentation), not left to implementor
/// discipline. This makes `chart_at` always search the same fixed set,
/// so the covering invariant follows directly from `check_local_inverse`
/// passing in `test_chart!`.
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
/// Implementors provide the node set via [`BuildNodes::build_nodes`], not
/// via [`Nodes::nodes`] directly — the latter is sealed and derived
/// automatically once [`BuildNodes`] is implemented.
///
/// [`Chart`]: crate::traits::Chart
/// [`Self::nodes`]: Nodes::nodes
pub trait NerveComplex<
    P: Point,
    V: Euclidean,
    T: TangentBundle<P, V> + Point,
    B: Bounded<T, V> + 'static + Send + Sync,
>: ExpMap<T, V> + Nodes<B>
{
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
                inode.as_ref().to_local(&jnode.base_point().base_point()),
                jnode.as_ref().to_local(&base.base_point()),
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
    fn fundamental_group() -> impl GroupPresentation {
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

        // -------------------------------------------------------------------
        // Canonicalization, up to generator relabeling.
        //
        // The Tietze loop above produces *a* minimal presentation of the
        // correct group, but the generator numbering it lands on depends on
        // incidental choices (elimination order, original relation order,
        // hash-set iteration order for duplicate detection) — it is a valid
        // simplifier, not yet a canonicalizer. Two presentations of
        // isomorphic groups can emerge from that loop looking different.
        //
        // This pass closes that gap: it searches over every relabeling of
        // the surviving generators (every permutation composed with every
        // choice of inverting each generator or not — the only relabelings
        // that preserve "being a presentation of the same group" without
        // doing a full, potentially-undecidable Nielsen/automorphism search)
        // and keeps the lexicographically smallest resulting relator list.
        //
        // Two presentations that are equal up to relabeling will, after this
        // pass, be BYTE-FOR-BYTE identical: same n_generators, same relators,
        // same order. Equality is then just `==` on the output.
        // -------------------------------------------------------------------
        fn canonical_relator_form(w: &[(usize, bool)]) -> Vec<(usize, bool)> {
            // identical logic to `canonical_relator` above, factored out so
            // it can be re-applied after relabeling too
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

        fn canonicalize_presentation(
            n_generators: usize,
            relators: &[Vec<(usize, bool)>],
        ) -> (usize, Vec<Vec<(usize, bool)>>) {
            // n_generators is typically very small after Tietze reduction
            // (2 for a surface group, rarely more than a handful in
            // practice for nerve-derived presentations), so brute-forcing
            // n! * 2^n relabelings is cheap. If this ever needs to scale to
            // larger n, this is the place to swap in a smarter search
            // (e.g. canonical-form-via-refinement, as in graph isomorphism
            // canonicalizers), but it is not needed at this size.
            let mut perm: Vec<usize> = (0..n_generators).collect();
            let mut best: Option<Vec<Vec<(usize, bool)>>> = None;

            loop {
                for invert_mask in 0u32..(1 << n_generators) {
                    let relabel_one = |g: usize, inv: bool| -> (usize, bool) {
                        let new_g = perm[g];
                        let flip = (invert_mask >> g) & 1 == 1;
                        (new_g, inv ^ flip)
                    };

                    let mut relabeled: Vec<Vec<(usize, bool)>> = relators
                        .iter()
                        .map(|w| {
                            let relabeled_word: Vec<(usize, bool)> =
                                w.iter().map(|&(g, inv)| relabel_one(g, inv)).collect();
                            canonical_relator_form(&relabeled_word)
                        })
                        .collect();
                    relabeled.sort();

                    if best.as_ref().is_none_or(|b| relabeled < *b) {
                        best = Some(relabeled);
                    }
                }

                // advance to next permutation of generator indices
                if !next_permutation(&mut perm) {
                    break;
                }
            }

            (n_generators, best.unwrap_or_default())
        }

        // standard next-lexicographic-permutation; returns false when perm
        // is already the last (fully descending) permutation
        fn next_permutation(perm: &mut [usize]) -> bool {
            if perm.len() < 2 {
                return false;
            }
            let mut i = perm.len() - 1;
            while i > 0 && perm[i - 1] >= perm[i] {
                i -= 1;
            }
            if i == 0 {
                return false;
            }
            let mut j = perm.len() - 1;
            while perm[j] <= perm[i - 1] {
                j -= 1;
            }
            perm.swap(i - 1, j);
            perm[i..].reverse();
            true
        }

        let (n_generators, relations) = canonicalize_presentation(n_generators, &relations);

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
pub trait Bounded<P: Point, V: Euclidean>: TangentBundle<P, V> + From<P> + AsRef<P> {
    /// The signed distance field in the tangent
    /// space of the chart centered at &self.
    fn sdf(&self, v: &V) -> V::F;
}

/// Implements [`Chart`], [`ExpMap`], and [`TangentBundle`] for `$target` by
/// delegating to `$target`'s [`AsRef::as_ref`] chart, restricting `to_local`
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
    ($target:ty, $point:ty, $v:ty, $($generics:tt)*) => {
        impl<$($generics)*> $crate::traits::Chart<$point, $v> for $target {
            fn to_local(&self, point: &$point) -> Option<$v> {
                <$point as $crate::traits::Chart<$point, $v>>::to_local(
                    <$target as AsRef<$point>>::as_ref(self), point
                ).filter(|v| self.sdf(v) < <$v as $crate::traits::Euclidean>::F::zero())
            }
            fn to_global(&self, coord: $v) -> $point {
                <$point as $crate::traits::Chart<$point, $v>>::to_global(
                    <$target as AsRef<$point>>::as_ref(self), coord
                )
            }

            fn chart_at(p: &$point) -> Self {
                <$target as From<$point>>::from(<$point as $crate::traits::Chart<$point, $v>>::chart_at(p))
            }
        }

        impl<$($generics)*> $crate::traits::ExpMap<$point, $v> for $target {}
        impl<$($generics)*> $crate::traits::TangentBundle<$point, $v> for $target {}
    };
}
