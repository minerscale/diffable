use std::collections::BinaryHeap;

use num_traits::{NumCast, One, Zero, real::Real};

use super::{Euclidean, Point, TangentBundle};

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
        type Registry =
            OnceLock<Mutex<HashMap<(TypeId, TypeId), &'static (dyn Any + Send + Sync)>>>;
        static REGISTRY: Registry = OnceLock::new();
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
        slot_for::<Caller, B>().get_or_init(|| Box::leak(build().into_boxed_slice()))
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

/// Lift a `usize` into the scalar field.
#[inline]
fn scalar<F: Real>(x: usize) -> F {
    <F as NumCast>::from(x).expect("usize is representable in the scalar field")
}

#[inline]
fn half<F: Real>() -> F {
    (F::one() + F::one()).recip()
}

#[derive(Debug, Clone)]
pub enum Geodesic<P, F> {
    /// Exhaustive enumeration under a positive bound: this is `d_M(p, q)`.
    Global { path: Vec<P>, length: F },
    /// An exact geodesic, but not certified globally minimal — either the
    /// bound was zero, or the candidate set was truncated for space.
    Local { path: Vec<P>, length: F },
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
    B: Bounded<T, P, V> + 'static + Send + Sync,
>: Nodes<B>
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
        let ibase = inode.base_point();
        nodes.iter().enumerate().filter_map(move |(j, jnode)| {
            if j == i {
                return None;
            }

            let half = (V::F::one() + V::F::one()).recip();
            match (
                inode.as_ref().to_local(&jnode.base_point()),
                jnode.as_ref().to_local(&ibase),
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
                if let Some(&last) = reduced.last()
                    && last == (letter.0, !letter.1)
                {
                    reduced.pop();
                    continue;
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
            type Action = Option<(usize, Vec<(usize, bool)>, usize)>;
            let mut action: Action = None;
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

    // -------------------------------------------------------------------
    // Exact global geodesic distance.
    //
    // # Why this needs no curvature bound
    //
    // A geodesic that is not the globally shortest path in its homotopy
    // class need not be unstable: a single homotopy class can contain
    // several distinct index-0 geodesics, separated by index-1 saddles.
    // Consider `R²` with a tall smooth bump between `p` and `q` — the
    // domain is contractible, so *every* path is homotopic to every other,
    // yet there are two locally-minimizing geodesics (left of the bump,
    // right of the bump) of different lengths. Homotopy cannot see the
    // bump, because the bump is metric, not topological. Descending from
    // an arbitrary starting path lands in whichever basin you started in.
    //
    // Basins are carved by conjugate points, and conjugate points are
    // manufactured by positive curvature (Rauch). One is therefore tempted
    // to demand a curvature bound `κ`, derive the Rauch radius `π/√κ`, and
    // check that the cover fits inside it — thereby *proving* that every
    // `log` the algorithm takes is defined.
    //
    // That proof is unnecessary. [`Chart::to_local`] returns `Option`: it
    // reports, at the call site, exactly the fact the curvature bound was
    // being asked to establish. `edge_weight(i, j) == Some(_)` *is* the
    // statement "`x_j` lies inside `x_i`'s injectivity domain", verified
    // rather than asserted. Where a `log` fails, the algorithm refines and
    // retries; it never needs to know why. A curvature bound would be an
    // unverifiable assertion standing in for an observable fact.
    //
    // # Why this needs no metric
    //
    // The 1-skeleton is *already* exactly weighted. Every node is a
    // [`TangentBundle`], and the [`ExpMap`] contract certifies that
    // coordinate distances from the origin equal arc lengths along
    // geodesics. So `‖log_{x_i}(x_j)‖` is the exact geodesic arc length
    // between adjacent base points, with no [`Metric`] impl required and no
    // approximation anywhere.
    //
    // Note this is arc-length correctness, not minimality. Minimality would
    // need [`Riemannian`], and the algorithm does not want it: every
    // straightened candidate below is a genuine geodesic, hence has length
    // `≥ d_M(p, q)`, whether or not any individual `log` happened to pick a
    // minimizing branch. Globality is supplied by taking the minimum over
    // basins, not by trusting any single `log`.
    //
    // # The algorithm
    //
    // 1. **Dijkstra on the weighted 1-skeleton selects the basin.** The
    //    resulting edge-path is a combinatorial object: it says which
    //    valley to descend into, not how long the valley's floor is.
    //
    // 2. **Curve shortening computes the exact length.** The edge-path is
    //    lifted to a polyline in `M` and relaxed by discrete geodesic flow
    //    until every interior vertex is straight. A polyline with no kinks
    //    *is* a geodesic — each segment is a geodesic segment and they join
    //    smoothly — so its length is that geodesic's exact arc length, for
    //    any vertex count. No corner-cutting error survives into the answer.
    //
    // Keeping these apart is what makes "exact" defensible. Stage-1 path
    // lengths systematically overestimate (chords cut corners), and the
    // overestimate scales with hop count, so a basin resolved by twelve
    // nodes and one resolved by five cannot be ranked against each other.
    // Stage 2 removes this entirely: each candidate's reported length is the
    // true arc length of a real geodesic from `p` to `q`, hence `≥ d_M(p,q)`,
    // with equality for the globally minimizing one.
    //
    // The single consequence worth internalising: a loose
    // [`Self::overestimation_bound`] cannot corrupt the result, only widen
    // the search. That bound is used *only* to decide how many basins to
    // inspect — never to compute a distance.
    //
    // [`Metric`]: crate::traits::Metric
    // [`Riemannian`]: crate::traits::Riemannian
    // -------------------------------------------------------------------

    /// The one irreducible assumption: graph distance on the 1-skeleton
    /// overestimates true geodesic distance by at most a factor `1 + e`.
    ///
    /// This is the sole thing the cover cannot discover about itself. A good
    /// cover is a *topological* condition — contractible intersections — and
    /// topology is blind to metric structure. An arbitrarily narrow, arbitrarily
    /// short bump can hide entirely inside one chart, splitting a homotopy class
    /// into two basins that no amount of inspecting the nerve will reveal. For
    /// any cover you fix, there exists a metric on the same manifold, admitting
    /// the same good cover, whose basin structure that cover cannot resolve.
    ///
    /// Asserting `e` rules this out, because "`d_G ≤ (1+e)·d_M` for all pairs"
    /// applied to the global minimizer says its basin *is* shadowed by an
    /// edge-path of graph length `≤ (1+e)·d_M`. So the enumeration budget
    /// below provably contains a representative of it. The classical sampling
    /// bound gives `e = 4·δ_s/ε` for an `ε`-net of covering radius `δ_s`; for
    /// a cover of geodesic balls of radius `ρ` with base points at covering
    /// radius `δ_s`, that is `e = 2·δ_s/ρ`. Oversampling — many small nodes,
    /// `δ_s ≪ ρ` — is what makes `e` small.
    ///
    /// # The default is None, and None is not a certificate
    ///
    /// With `e = 0` only the Dijkstra-optimal basin is inspected. The returned
    /// length is then the *exact* arc length of a real geodesic from `p` to `q`
    /// — never an approximation — but its globality is unwarranted. This is the
    /// fast, uncertified mode. Override to opt into the global guarantee.
    ///
    /// This bound is asserted, not checked, and nothing inside can catch a
    /// false one. Garbage bound in, confident garbage out.
    fn overestimation_bound() -> Option<V::F> {
        None
    }

    /// Hard cap on how many candidate edge-paths are straightened. Raising it
    /// strengthens the guarantee; lowering it trades certification for speed.
    fn max_candidate_paths() -> usize {
        256
    }

    /// Midpoint-insertion passes before relaxation begins.
    ///
    /// Affects tightness, never soundness: a converged polyline is an exact
    /// geodesic at any vertex count, but a coarse one may converge to a
    /// *longer* geodesic in the same basin, which merely loses the minimum
    /// rather than corrupting it. Also caps the on-demand refinement used
    /// when a `log` fails mid-flow.
    fn refinement_passes() -> usize {
        3
    }

    /// Relative tolerance on *polyline length* at which the discrete geodesic
    /// flow is declared converged.
    ///
    /// The stopping test is on the length, not on the residual kink. Two
    /// reasons. The kink `‖(log_b a + log_b c)/2‖` is a near-cancellation of
    /// two vectors of magnitude `h` (the segment length), so its floating-point
    /// noise floor is `~ε·h`, not `~ε` — a kink threshold scaled to machine
    /// epsilon can never be met and the flow silently exits on the iteration
    /// cap instead. And length is the quantity actually wanted, so watching it
    /// converge is both scale-free and directly meaningful.
    ///
    /// Length is *stationary* at a geodesic: a first-order residual `τ` in
    /// vertex position produces an `O(τ²)` error in length. So a length that
    /// has stopped moving to relative `ε` is correct to relative `ε`, and the
    /// underlying vertices are correct to `√ε` — which is all they need to be.
    fn straightening_tolerance() -> V::F {
        V::F::epsilon() * scalar(8)
    }

    /// Iteration cap for the flow, as a function of vertex count.
    ///
    /// The flow is a discrete heat equation on the polyline, so convergence is
    /// linear with rate `1 − O(1/n²)`: a fixed cap that suffices at twenty
    /// vertices is badly short at five hundred. Scaling quadratically keeps the
    /// cap a safety net rather than the de facto stopping rule.
    fn max_straightening_iterations(n: usize) -> usize {
        (64 + 4 * n.saturating_mul(n)).min(100_000)
    }

    /// Safety cap on polyline length, so a pathological path cannot explode
    /// the subdivision.
    fn max_polyline_points() -> usize {
        4096
    }

    // ---------------------------------------------------------------
    // The weighted 1-skeleton. Already exact; nothing to supply.
    // ---------------------------------------------------------------

    /// The base point of node `i`, as a point of `M`.
    fn base_point_of(i: usize) -> P {
        Self::nodes()[i].base_point()
    }

    /// The **exact** geodesic arc length between the base points of two
    /// adjacent nodes, or `None` if `x_j` lies outside `x_i`'s injectivity
    /// domain — which is precisely the condition a curvature bound would
    /// otherwise have to assert.
    ///
    /// Deliberately uses the *unbounded* chart (`as_ref()`), not the
    /// `sdf`-restricted one: adjacent base points sit at separation up to `2ρ`
    /// and so generally lie outside each other's *bounded* domains even though
    /// their domains overlap.
    fn edge_weight(i: usize, j: usize) -> Option<V::F> {
        let target = Self::base_point_of(j);
        Self::nodes()[i]
            .as_ref()
            .to_local(&target)
            .map(|v| v.norm())
    }

    /// The symmetrised, exactly-weighted adjacency list of the 1-skeleton.
    fn adjacency() -> Vec<Vec<(usize, V::F)>> {
        let n = Self::nodes().len();
        let mut adj: Vec<Vec<(usize, V::F)>> = vec![Vec::new(); n];
        for i in 0..n {
            for j in Self::get_neighbors(i) {
                let Some(w) = Self::edge_weight(i, j) else {
                    debug_assert!(
                        false,
                        "adjacent nodes cannot see each other: 2ρ exceeds injectivity radius"
                    );
                    continue;
                };
                if !adj[i].iter().any(|&(k, _)| k == j) {
                    adj[i].push((j, w));
                }
                if !adj[j].iter().any(|&(k, _)| k == i) {
                    adj[j].push((i, w));
                }
            }
        }
        adj
    }

    /// The node whose bounded domain contains `p`, nearest base point first.
    /// `None` only if the covering invariant fails.
    fn locate(p: &P) -> Option<usize> {
        let mut best: Option<(usize, V::F)> = None;
        for (i, node) in Self::nodes().iter().enumerate() {
            let Some(v) = node.as_ref().to_local(p) else {
                continue;
            };
            if node.sdf(&v) >= V::F::zero() {
                continue;
            }
            let d = v.norm();
            if best.is_none_or(|(_, bd)| d < bd) {
                best = Some((i, d));
            }
        }
        best.map(|(i, _)| i)
    }

    // ---------------------------------------------------------------
    // Stage 1 — graph search selects the basin.
    // ---------------------------------------------------------------

    /// Dense `O(n²)` Dijkstra. Dense rather than heap-based because
    /// `V::F: Real` supplies only `PartialOrd`, and node counts are small.
    fn dijkstra(adj: &[Vec<(usize, V::F)>], src: usize) -> Vec<Option<V::F>> {
        let n = adj.len();
        let inf = None;
        let mut dist = vec![inf; n];
        let mut done = vec![false; n];
        if src >= n {
            return dist;
        }
        dist[src] = Some(V::F::zero());

        for _ in 0..n {
            let mut chosen: Option<(usize, V::F)> = None;
            for v in 0..n {
                if done[v] {
                    continue;
                }
                let Some(d) = dist[v] else { continue };
                if chosen.is_none_or(|(_, best)| d < best) {
                    chosen = Some((v, d));
                }
            }
            let Some((u, du)) = chosen else { break };
            done[u] = true;

            for &(w, wt) in &adj[u] {
                let relaxed = du + wt;
                if dist[w].is_none_or(|dw| relaxed < dw) {
                    dist[w] = Some(relaxed);
                }
            }
        }
        dist
    }

    /// Every simple edge-path from `src` to `dst` whose graph length fits inside
    /// `budget`, returned in **nondecreasing order of length**, together with a
    /// flag reporting whether the enumeration was exhaustive.
    ///
    /// This is uniform-cost search over path prefixes, keyed on
    /// `f = acc + to_dst[u]`. Because `to_dst` is the exact Dijkstra distance to
    /// the target it is an admissible *and consistent* heuristic, so `f` never
    /// decreases along a prefix and complete paths pop in nondecreasing total
    /// length. Three consequences that a depth-first search does not give:
    ///
    /// - **The first path returned is the Dijkstra optimum.** A DFS returns
    ///   whatever adjacency-list order it stumbles into, and can miss the
    ///   optimum entirely under a candidate cap — which would be absurd, given
    ///   that finding the optimum is the whole point of stage 1.
    /// - **Termination is principled.** The moment a popped `f` exceeds
    ///   `budget`, every remaining path also exceeds it. Enumeration inside the
    ///   budget is then provably complete, and the cap never fired.
    /// - **The cap degrades to a memory backstop.** Hitting it means "I ran out
    ///   of room", reported through `exhaustive = false`, rather than silently
    ///   truncating the candidate set and quietly weakening the guarantee.
    ///
    /// Simple paths suffice throughout: a globally minimizing geodesic never
    /// self-intersects, so its node sequence can always be chosen simple.
    ///
    /// # Why the budget test appears exactly once, and carries slack
    ///
    /// `budget` derives from `to_dst[src]`, which Dijkstra accumulated
    /// *backwards* from `dst`. This search accumulates `acc` *forwards* from
    /// `src`. Floating-point addition is not associative, so along the optimal
    /// path `acc + to_dst[v]` and `budget` are the same real number reached by
    /// two different summation orders, and can differ by an ulp.
    ///
    /// With `overestimation_bound() == None` the budget is exactly `graph_opt`,
    /// so the optimal path sits *on* the boundary. One ulp of disagreement then
    /// prunes the very path the search exists to find, and the function returns
    /// an empty set while cheerfully reporting `exhaustive = true`. A positive
    /// bound hides the bug behind slack it never needed.
    ///
    /// Two defences. The comparison lives in exactly one place — pruning on
    /// push would evaluate the same predicate against a differently-rounded
    /// quantity, which is how one gets a prefix that passes one test and fails
    /// its twin. And it carries an explicit relative tolerance, sized to the
    /// accumulated rounding of a few dozen additions. Admitting a handful of
    /// marginally-over-budget paths is free: each is straightened into a real
    /// geodesic, and the caller takes a minimum.
    fn candidate_paths(
        adj: &[Vec<(usize, V::F)>],
        src: usize,
        dst: usize,
        budget: V::F,
        to_dst: &[Option<V::F>],
    ) -> (Vec<Vec<usize>>, bool) {
        /// A search frontier entry, ordered so `BinaryHeap` — a max-heap — pops
        /// the smallest `f` first.
        ///
        /// `Ord` goes through `f64::total_cmp` rather than `V::F::partial_cmp`
        /// because a heap requires a **total, transitive** order, and a
        /// tolerance-based scalar (see [`R64`]) supplies neither: `a ≈ b` and
        /// `b ≈ c` with `a < c` is permitted by design. Sift-down then makes
        /// mutually contradictory decisions and the heap silently stops being a
        /// heap — which would void this function's whole ordering guarantee,
        /// and with it the meaning of `exhaustive`.
        ///
        /// [`R64`]: crate::epsilon_metric::R64
        struct Entry<F> {
            f: F,
            acc: F,
            path: Vec<usize>,
        }

        impl<F: Real> Entry<F> {
            fn key(&self) -> f64 {
                self.f.to_f64().expect("geodesic lengths are finite reals")
            }
        }

        impl<F: Real> PartialEq for Entry<F> {
            fn eq(&self, other: &Self) -> bool {
                self.cmp(other) == std::cmp::Ordering::Equal
            }
        }
        impl<F: Real> Eq for Entry<F> {}
        impl<F: Real> PartialOrd for Entry<F> {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl<F: Real> Ord for Entry<F> {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                // reversed: min-heap on `f`
                other.key().total_cmp(&self.key())
            }
        }

        let cap = Self::max_candidate_paths();
        let mut out: Vec<Vec<usize>> = Vec::new();
        let mut heap: BinaryHeap<Entry<V::F>> = BinaryHeap::new();

        if src >= adj.len() || dst >= adj.len() {
            return (out, true);
        }

        // No route from `src` to `dst`: the empty enumeration is complete.
        let Some(h_src) = to_dst[src] else {
            return (out, true);
        };

        // Relative slack absorbing the rounding difference between `budget`
        // (summed backwards by Dijkstra) and `f` (summed forwards here).
        // Eight ulps covers a few dozen hops; scale with path length if your
        // covers get deep.
        let ceiling = budget + budget * V::F::epsilon() * scalar(8);

        heap.push(Entry {
            f: h_src,
            acc: V::F::zero(),
            path: vec![src],
        });

        // Prefixes vastly outnumber completed paths, so the frontier — not
        // `out` — is what actually threatens memory.
        let frontier_cap = cap.saturating_mul(64);

        while let Some(Entry { f, acc, path }) = heap.pop() {
            // `f` is nondecreasing across pops, so every remaining entry also
            // exceeds the ceiling. Enumeration inside budget is complete.
            if f > ceiling {
                return (out, true);
            }

            let u = *path.last().expect("path is never empty");
            if u == dst {
                out.push(path);
                if out.len() >= cap {
                    return (out, false); // out of room, not out of paths
                }
                continue; // a completed path is never extended
            }

            for &(v, w) in &adj[u] {
                if path.contains(&v) {
                    continue; // simple paths only
                }
                let Some(h_v) = to_dst[v] else {
                    continue; // `v` cannot reach `dst` at all
                };
                if heap.len() >= frontier_cap {
                    return (out, false);
                }

                // No budget test here. The pop-side check is the single
                // authority, and it sees the value actually stored in the heap.
                let acc2 = acc + w;
                let mut next = Vec::with_capacity(path.len() + 1);
                next.extend_from_slice(&path);
                next.push(v);
                heap.push(Entry {
                    f: acc2 + h_v,
                    acc: acc2,
                    path: next,
                });
            }
        }

        (out, true) // heap drained: everything inside budget was seen
    }

    // ---------------------------------------------------------------
    // Stage 2 — curve shortening computes the exact length.
    //
    // Every geometric precondition here is *observed*, never assumed:
    // each `log` returns `Option`, and a `None` triggers refinement.
    // ---------------------------------------------------------------

    /// Exact geodesic arc length between two nearby points, trying either
    /// endpoint's chart. `None` if neither sees the other.
    fn hop(a: &P, b: &P) -> Option<V::F> {
        let ca = T::chart_at(a);
        if let Some(v) = ca.to_local(b) {
            return Some(v.norm());
        }
        let cb = T::chart_at(b);
        cb.to_local(a).map(|v| v.norm())
    }

    /// The geodesic midpoint of two nearby points.
    fn midpoint(a: &P, b: &P) -> Option<P> {
        let ca = T::chart_at(a);
        if let Some(v) = ca.to_local(b) {
            return Some(ca.to_global(v * half()));
        }
        let cb = T::chart_at(b);
        cb.to_local(a).map(|v| cb.to_global(v * half()))
    }

    /// Total length, as a sum of exact geodesic hops.
    ///
    /// Once the polyline has converged its vertices lie *on* a geodesic, and
    /// consecutive geodesic distances along a geodesic sum to its arc length
    /// exactly. This is what makes the final number exact rather than a
    /// chordal underestimate.
    fn polyline_length(pts: &[P]) -> Option<V::F> {
        let mut total = V::F::zero();
        for w in pts.windows(2) {
            total = total + Self::hop(&w[0], &w[1])?;
        }
        Some(total)
    }

    /// Insert the geodesic midpoint of every segment, halving spacing.
    fn refine(pts: Vec<P>) -> Option<Vec<P>> {
        if pts.len() < 2 {
            return Some(pts);
        }
        let mut out = Vec::with_capacity(pts.len() * 2);
        for w in pts.windows(2) {
            out.push(w[0].clone());
            out.push(Self::midpoint(&w[0], &w[1])?);
        }
        out.push(pts.last()?.clone());
        Some(out)
    }

    /// Run the discrete geodesic flow at the current resolution until the
    /// polyline length stops moving. `None` if a `log` failed — the caller's
    /// cue to refine, not to give up.
    ///
    /// # Why the stopping test extrapolates
    ///
    /// The flow is a discrete heat equation on the polyline, so the length
    /// converges *linearly* with ratio `r = 1 − O(1/n²)`. When successive
    /// lengths differ by `d`, the remaining distance to the limit is
    /// `d·r/(1−r)`, which is `~n²·d`. A stopping test on the step size `d`
    /// therefore overshoots the true error by the condition number — at 64
    /// vertices, a step of `10⁻¹⁵` still hides an error of `10⁻¹¹`.
    ///
    /// Aitken's estimate of that tail costs two scalars and turns the tolerance
    /// into what its name claims. It is used only to decide *when* to stop,
    /// never to compute *what* is returned: the length reported is always
    /// `polyline_length` of an actual polyline, so the invariant "every number
    /// this algorithm returns is the true arc length of a real geodesic"
    /// survives.
    fn relax_to_convergence(pts: &mut Vec<P>) -> Option<V::F> {
        let tol = Self::straightening_tolerance();
        let eps = V::F::epsilon();
        let iters = Self::max_straightening_iterations(pts.len());
 
        let mut prev2: Option<V::F> = None;
        let mut prev = Self::polyline_length(pts)?;
 
        for _ in 0..iters {
            Self::relax_sweep(pts)?;
            let len = Self::polyline_length(pts)?;
            let d1 = prev - len; // ≥ 0: the flow is a descent
 
            // Arithmetic has run out; no extrapolation is meaningful.
            if eps * len >= d1.abs() {
                return Some(len);
            }
 
            if let Some(p2) = prev2 {
                let d0 = p2 - prev;
                // A healthy linear tail: steps positive and shrinking.
                if V::F::zero() < d0 && d1 < d0 {
                    let r = d1 / d0;
                    let remaining = d1 * r / (V::F::one() - r);
                    if tol * len >= remaining {
                        return Some(len);
                    }
                }
            }
 
            prev2 = Some(prev);
            prev = len;
        }
 
        Some(prev) // iteration cap; accept the current state
    }


    /// One relaxation at an interior vertex `b` between `a` and `c`.
    ///
    /// The discrete energy `E(b) = d(a,b)² + d(b,c)²` has gradient
    /// `∇E = -2(log_b a + log_b c)`, so descending with step `1/2` sends `b`
    /// to the geodesic midpoint. The returned magnitude is the residual kink:
    /// zero exactly when `log_b a` and `log_b c` are antiparallel, i.e. when
    /// `b` sits straight on the geodesic through `a` and `c`.
    fn relax(a: &P, b: &P, c: &P) -> Option<(P, V::F)> {
        let chart = T::chart_at(b);
        let va = chart.to_local(a)?;
        let vc = chart.to_local(c)?;
        let delta = (va + vc) * half();
        let moved = chart.to_global(delta);
        Some((moved, delta.norm()))
    }

    /// One Gauss–Seidel sweep. Returns the worst residual kink, or `None` if
    /// some `log` failed — the caller's cue to refine, not to give up.
    fn relax_sweep(pts: &mut [P]) -> Option<V::F> {
        if pts.len() < 3 {
            return Some(V::F::zero());
        }
        let mut worst = V::F::zero();
        for i in 1..pts.len() - 1 {
            let (a, b, c) = (pts[i - 1].clone(), pts[i].clone(), pts[i + 1].clone());
            let (moved, kink) = Self::relax(&a, &b, &c)?;
            pts[i] = moved;
            if kink > worst {
                worst = kink;
            }
        }
        Some(worst)
    }

    /// Relax a polyline to the geodesic at the bottom of its basin, and return
    /// that geodesic with its exact length.
    ///
    /// Endpoints are pinned. The flow is the negative gradient of the discrete
    /// energy `Σ d(pᵢ, pᵢ₊₁)²`, so it descends monotonically and cannot cross
    /// the index-1 saddle out of its basin — which is exactly why stage 1's
    /// basin selection is respected rather than undone.
    ///
    /// # Coarse-to-fine
    ///
    /// Gauss–Seidel propagates information one vertex per sweep, so relaxing an
    /// `n`-vertex polyline directly costs `O(n²)` sweeps. Relaxing at the
    /// coarsest resolution first, then refining and re-relaxing, costs `O(n)`
    /// total: each refinement doubles the vertex count but starts from a state
    /// that is already converged one level down, so the fine flow has only
    /// local, high-frequency error left to remove — which is precisely the
    /// error Gauss–Seidel kills fastest.
    ///
    /// This is a multigrid V-cycle, and the ladder was already here: the old
    /// code refined `refinement_passes` times *up front* and then ran the flow
    /// once, at the finest level, paying the full `O(n²)`. Moving the flow
    /// inside the loop is the entire change. Expect two orders of magnitude.
    ///
    /// Refinement also serves a second, unrelated purpose: when a `log` fails
    /// mid-flow, the polyline is locally too coarse for its own charts, and
    /// halving the spacing fixes it. The two uses share a vertex cap but not a
    /// counter — a geometric failure should not consume the accuracy budget.
    fn straighten(pts: Vec<P>) -> Option<(Vec<P>, V::F)> {
        let cap = Self::max_polyline_points();
        let passes = Self::refinement_passes();
 
        let mut pts = pts;
        let mut level = 0usize; // refinements taken for accuracy
        let mut rescues = 0usize; // refinements taken to rescue a failed `log`
 
        let length = loop {
            match Self::relax_to_convergence(&mut pts) {
                Some(len) => {
                    // Converged at this resolution. Refine and descend, unless
                    // we are out of levels or out of room.
                    if level >= passes || pts.len() * 2 > cap {
                        break len;
                    }
                    level += 1;
                    pts = Self::refine(pts)?;
                }
                None => {
                    // A `log` failed. Halve the spacing and retry this level.
                    if rescues >= passes || pts.len() * 2 > cap {
                        return None;
                    }
                    rescues += 1;
                    pts = Self::refine(pts)?;
                }
            }
        };
 
        Some((pts, length))
    }

    // ---------------------------------------------------------------
    // Entry points.
    // ---------------------------------------------------------------

    /// The global geodesic from `p` to `q`, and its exact arc length.
    ///
    /// `None` if either point escapes the cover, if they lie in different
    /// connected components of the nerve, or if refinement cannot make the
    /// charts cooperate.
    ///
    /// # Guarantee
    ///
    /// The returned length is always the *exact* arc length of a real geodesic
    /// from `p` to `q` — never an approximation, up to `O(tol²)` in the
    /// straightening tolerance. Whether it is the *globally* shortest such
    /// geodesic depends on [`Self::overestimation_bound`]: if that bound holds
    /// on `M`, the result is `d_M(p, q)`, the minimum across all homotopy
    /// classes and all basins within each class. At the default bound of zero,
    /// only the Dijkstra-selected basin is inspected.
    fn geodesic_path(p: &P, q: &P) -> Option<Geodesic<P, V::F>> {
        let mut best: Option<(Vec<P>, V::F)> = None;

        // Where `log` is defined it already parametrises by arc length, so
        // this is a real geodesic from `p` to `q` and can never underestimate.
        // Admitted as a *candidate*, not a short-circuit: minimality would
        // need `Riemannian`, and we decline to require it. If it happens to be
        // minimal — as on a sphere or a flat torus — it simply wins.
        let direct = T::chart_at(p);
        if let Some(v) = direct.to_local(q) {
            best = Some((vec![p.clone(), q.clone()], v.norm()));
        }

        let src = Self::locate(p)?;
        let dst = Self::locate(q)?;
        let adj = Self::adjacency();

        // Heuristic and optimum in one sweep: Dijkstra *from* the target.
        let to_dst = Self::dijkstra(&adj, dst);

        let Some(graph_opt) = to_dst[src] else {
            return None;
        };

        let (overestimation_bound, bound_set) = match Self::overestimation_bound() {
            Some(b) => (b, true),
            None => (V::F::zero(), false),
        };

        // Any basin whose true length `L` satisfies `L ≤ d_M(p,q)` has graph
        // length `≤ (1+e)·L ≤ (1+e)·d_M ≤ (1+e)·graph_opt`. So this budget
        // provably admits an edge-path from the global minimizer's basin.
        // A loose `e` widens the search; it cannot lose the answer.
        let budget = graph_opt * (V::F::one() + overestimation_bound);
        let (paths, exhaustive) = Self::candidate_paths(&adj, src, dst, budget, &to_dst);

        let mut complete_straighten = true;
        for path in paths {
            let mut poly = Vec::with_capacity(path.len() + 2);
            poly.push(p.clone());
            poly.extend(path.iter().map(|&k| Self::base_point_of(k)));
            poly.push(q.clone());

            // Each straightened candidate is a genuine geodesic from `p` to
            // `q`, so its length is `≥ d_M(p,q)`, with equality for the global
            // minimizer. The min over candidates is therefore exact.

            match Self::straighten(poly) {
                Some((pts, len)) => {
                    if best.as_ref().is_none_or(|(_, b)| len < *b) {
                        best = Some((pts, len));
                    }
                }
                None => complete_straighten = false,
            }
        }

        let certified = bound_set && exhaustive && complete_straighten;

        println!("best: {:?} | bound_set: {bound_set} | exhaustive: {exhaustive} | complete_straighten: {complete_straighten}", best.is_some());
        match (best, certified) {
            (Some((path, length)), true) => Some(Geodesic::Global { path, length }),
            (Some((path, length)), false) => Some(Geodesic::Local { path, length }),
            _ => None,
        }
    }

    /// The global geodesic distance `d_M(p, q)`. See [`Self::geodesic_path`]
    /// for the guarantee and its one precondition.
    fn geodesic_distance(p: &P, q: &P) -> Option<V::F> {
        Self::geodesic_path(p, q).and_then(|g| match g {
            Geodesic::Global { path: _, length } => Some(length),
            Geodesic::Local { path: _, length: _ } => None,
        })
    }

    /// The best-effort geodesic distance. This gives an exact locally minimal
    /// geodesic but does not guarantee that it is the minimal geodesic.
    fn geodesic_distance_uncertified(p: &P, q: &P) -> Option<V::F> {
        Self::geodesic_path(p, q).map(|g| match g {
            Geodesic::Global { path: _, length } | Geodesic::Local { path: _, length } => length,
        })
    }
}

/// Restricts the domain of [`Chart::to_local`] to some subset
/// defined by a signed distance function in the tangent space
/// at each point on a [`TangentBundle`].
///
/// [`Chart::to_local`]: crate::traits::Chart::to_local
/// [`TangentBundle`]: crate::traits::TangentBundle
pub trait Bounded<T: TangentBundle<P, V>, P: Point, V: Euclidean>:
    TangentBundle<P, V> + From<T> + AsRef<T>
{
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
    ($chart:ty, $ambient:ty, $manifold:ty, $v:ty, $($generics:tt)*) => {
        impl<$($generics)*> Chart<$manifold, $v> for $chart {
            fn to_local(&self, p: &$manifold) -> Option<$v> {
                <$ambient as Chart<$manifold, $v>>::to_local(self.as_ref(), p)
                    .filter(|v| self.sdf(v) < <$v as $crate::traits::Euclidean>::F::zero())
            }
            fn to_global(&self, c: $v) -> $manifold {
                <$ambient as Chart<$manifold, $v>>::to_global(self.as_ref(), c)
            }
            fn chart_at(p: &$manifold) -> Self {
                Self::from(<$ambient as Chart<$manifold, $v>>::chart_at(p))
            }
        }

        impl<$($generics)*> $crate::traits::ExpMap<$manifold, $v> for $chart {
            fn base_point(&self) -> $manifold {
                <$ambient as $crate::traits::ExpMap<$manifold, $v>>::base_point(
                    <$chart as AsRef<$ambient>>::as_ref(self)
                )
            }
        }
        impl<$($generics)*> $crate::traits::TangentBundle<$manifold, $v> for $chart {}
    };
}
