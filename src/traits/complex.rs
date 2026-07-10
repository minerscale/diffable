use std::collections::{BinaryHeap, HashMap};

use num_traits::{NumCast, One, ToPrimitive, Zero, real::Real};

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

/// Everything the nerve's 1- and 2-skeleta determine, computed once per type.
///
/// `fundamental_group` and the geodesic search were building the same spanning
/// tree, the same non-tree-edge generators, and the same triangles, twice, from
/// scratch, on every call. They are facts about the type.
pub struct NerveTopology {
    /// One generator per non-tree edge of a BFS spanning tree.
    pub n_generators: usize,
    /// `(i, j) -> (generator, inverted)` for non-tree edges; absent for tree
    /// edges, which contract to the basepoint through the tree.
    pub edge_gen: HashMap<(usize, usize), (usize, bool)>,
    /// One relator per triangle: the boundary word `∂(i,j,k)`, freely reduced.
    ///
    /// Note `edge_word(a, b)` in the old code — walk `root → a`, then `b → root`
    /// — yields *exactly one letter*, the generator of `(a, b)`, whenever that
    /// edge is non-tree: the junction of the two tree paths is the pair `(a, b)`,
    /// and every other window is a tree edge, which `path_to_word` discards. So
    /// the tree paths were never contributing anything, and the whole
    /// `path_to_root` / `path_to_word` apparatus was an `O(diam · |E|)` way to
    /// look up one entry of `edge_gen`.
    pub relations: Vec<Vec<(usize, bool)>>,
    /// `H₁ = π₁ / [π₁, π₁]`, the prefix-deduplication key.
    pub abel: Abelianisation,
}

/// The abelianisation `G^ab = G/[G,G]` of a finitely presented group,
/// together with the change of basis that puts it in canonical form.
///
/// By Hurewicz this is `H₁(M; ℤ)` when `G = π₁(M)`, which is the point: homology
/// is computable and homotopy is not. Two edge-paths whose images here differ are
/// *provably* in different homotopy classes, and that implication — sound, cheap,
/// one-directional — is exactly what prefix deduplication needs.
///
/// # Why this is a filter and never a decision
///
/// The kernel is `[G,G]`. Two prefixes differing by a commutator collide, and on
/// a group of exponential growth — free groups, surface groups — they collide
/// exponentially often. Worse, on a **contractible** manifold `π₁` is trivial, so
/// *every* prefix has the same image and the key distinguishes nothing at all.
///
/// That last case is the mountain on `ℝ²`: one homotopy class, two index-0
/// geodesics separated by an index-1 saddle. Pruning on key equality alone would
/// silently discard the second valley — on the very manifold the algorithm exists
/// to handle. Basins are geometric; the bump contributes nothing to `π₁`.
///
/// So: **different key ⟹ different class ⟹ keep both.** Same key ⟹ *maybe* the
/// same basin ⟹ compare the straightened prefixes pointwise. The key decides
/// which pairs are worth the geometric check. It never prunes alone.
#[derive(Debug, Clone)]
pub struct Abelianisation {
    /// Coordinates with `dᵢ ≠ 1`, in order. The others are identically zero.
    live: Vec<usize>,
    /// `invariants[live[i]]`, hoisted. `0` for free, `> 1` for torsion.
    live_invariants: Vec<i64>,
    /// `gen_images[g]` restricted to the live coordinates. Length `live.len()`.
    gen_images: Vec<Vec<i64>>,
    /// Kept for reporting only.
    invariants: Vec<i64>,
}

impl Abelianisation {
    /// `ℤⁿ / im(A)`, where `A`'s columns are the relation vectors.
    pub fn from_relations(n: usize, relations: Vec<Vec<i64>>) -> Self {
        let m = relations.len();
        let mut mat = vec![vec![0i64; m]; n];
        for (j, col) in relations.iter().enumerate() {
            for (i, row) in mat.iter_mut().enumerate().take(n) {
                row[j] = col[i];
            }
        }
        let mut row_transform: Vec<Vec<i64>> = (0..n)
            .map(|i| (0..n).map(|j| <i64 as From<_>>::from(i == j)).collect())
            .collect();
        diagonalise(&mut mat, &mut row_transform);

        let mut invariants: Vec<i64> = (0..n)
            .map(|i| if i < m { mat[i][i].abs() } else { 0 })
            .collect();
        make_divisibility_chain(&mut invariants);

        let live: Vec<usize> = (0..n).filter(|&i| invariants[i] != 1).collect();
        let live_invariants: Vec<i64> = live.iter().map(|&i| invariants[i]).collect();

        // `R·e_g`, restricted to the live coordinates and reduced.
        let gen_images: Vec<Vec<i64>> = (0..n)
            .map(|g| {
                let mut w: Vec<i64> = live.iter().map(|&i| row_transform[i][g]).collect();
                for (wi, &d) in w.iter_mut().zip(&live_invariants) {
                    if d != 0 {
                        *wi = wi.rem_euclid(d);
                    }
                }
                w
            })
            .collect();

        Self {
            live,
            live_invariants,
            gen_images,
            invariants,
        }
    }

    /// The key of the empty prefix.
    pub fn identity(&self) -> Vec<i64> {
        vec![0; self.live.len()]
    }

    /// `key(prefix · e) = key(prefix) + key(e)` — abelianisation is a
    /// homomorphism. Tree edges pass `None` and contribute nothing, being
    /// contractible to the basepoint through the tree.
    pub fn extend(&self, key: &[i64], edge: Option<(usize, bool)>) -> Vec<i64> {
        let mut out = key.to_vec();
        let Some((idx, inverted)) = edge else {
            return out;
        };
        for ((o, g), &d) in out
            .iter_mut()
            .zip(&self.gen_images[idx])
            .zip(&self.live_invariants)
        {
            *o += if inverted { -g } else { *g };
            if d != 0 {
                *o = o.rem_euclid(d);
            }
        }
        out
    }

    /// `H₁ ≅ ℤ^rank ⊕ torsion`.
    pub fn free_rank(&self) -> usize {
        self.invariants.iter().filter(|&&d| d == 0).count()
    }

    /// The elementary divisors `d₁ | d₂ | … | d_r`, each `> 1`.
    pub fn torsion(&self) -> Vec<i64> {
        self.invariants.iter().copied().filter(|&d| d > 1).collect()
    }

    /// Whether `H₁` is finite. By Bonnet–Myers, curvature bounded below by a
    /// positive constant forces `π₁` finite, hence `H₁` finite — so `true` here
    /// is weak evidence the key will be a good one, finite groups having bounded
    /// growth and therefore no room to collide exponentially.
    pub fn is_finite(&self) -> bool {
        self.free_rank() == 0
    }
}

/// Reduce `mat` to a diagonal matrix by unimodular row and column operations,
/// accumulating the row operations into `row_transform`.
///
/// This is Smith normal form *without* the divisibility chain, which is all the
/// quotient structure needs: any diagonal `D = R·A·C` with `R` unimodular gives
/// `ℤⁿ/im(A) ≅ ⊕ ℤ/dᵢ` through `R`. The chain is imposed afterwards, and only so
/// that the reported multiset of divisors is canonical.
///
/// The inner loop is the classical one: pick the smallest nonzero pivot, clear
/// its row and column by division with remainder, and repeat — each pass strictly
/// decreases `|pivot|` unless the row and column are already clear, so it
/// terminates.
fn diagonalise(mat: &mut [Vec<i64>], row_transform: &mut [Vec<i64>]) {
    let n = mat.len();
    if n == 0 {
        return;
    }
    let m = mat[0].len();

    for t in 0..n.min(m) {
        loop {
            // Smallest nonzero entry in the remaining submatrix.
            let Some((pi, pj)) = (t..n)
                .flat_map(|i| (t..m).map(move |j| (i, j)))
                .filter(|&(i, j)| mat[i][j] != 0)
                .min_by_key(|&(i, j)| mat[i][j].abs())
            else {
                return; // submatrix is zero: done
            };

            mat.swap(t, pi);
            row_transform.swap(t, pi);
            for row in mat.iter_mut() {
                row.swap(t, pj);
            }

            let pivot = mat[t][t];
            let mut dirty = false;

            for i in (t + 1)..n {
                let q = mat[i][t] / pivot;
                if q != 0 {
                    for j in t..m {
                        mat[i][j] -= q * mat[t][j];
                    }
                    for j in 0..n {
                        row_transform[i][j] -= q * row_transform[t][j];
                    }
                }
                if mat[i][t] != 0 {
                    dirty = true; // remainder survived; re-pivot on it
                }
            }

            for j in (t + 1)..m {
                let q = mat[t][j] / pivot;
                if q != 0 {
                    for row in mat.iter_mut().take(n).skip(t) {
                        row[j] -= q * row[t];
                    }
                }
                if mat[t][j] != 0 {
                    dirty = true;
                }
            }

            if !dirty {
                break;
            }
        }
    }
}

/// Impose `d₁ | d₂ | … | d_r` on the diagonal, leaving zeros (free coordinates)
/// at the end.
///
/// Only the *multiset* of divisors changes, never the group: `ℤ/2 ⊕ ℤ/3 ≅ ℤ/6`.
/// The transform is not tracked, because [`Abelianisation::key`] reduces each
/// coordinate independently and does not care which order the divisors arrive in
/// — the chain exists so `torsion()` reports something canonical.
fn make_divisibility_chain(inv: &mut [i64]) {
    let r = inv.iter().filter(|&&d| d != 0).count();
    let nonzero = &mut inv[..r];

    let mut changed = true;
    while changed {
        changed = false;
        for i in 0..r.saturating_sub(1) {
            let (a, b) = (nonzero[i], nonzero[i + 1]);
            if b % a != 0 {
                let g = gcd(a, b);
                nonzero[i] = g;
                nonzero[i + 1] = a / g * b; // lcm
                changed = true;
            }
        }
    }
    // zeros already sit at the tail
}

fn gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
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

    fn slot_for<Caller: 'static + ?Sized, T: 'static + Send + Sync>()
    -> &'static OnceLock<&'static T> {
        let key = (TypeId::of::<Caller>(), TypeId::of::<T>());
        let mut map = registry().lock().unwrap();
        let entry = map.entry(key).or_insert_with(|| {
            let slot: &'static OnceLock<&'static T> = Box::leak(Box::new(OnceLock::new()));
            slot as &'static (dyn Any + Send + Sync)
        });
        entry
            .downcast_ref::<OnceLock<&'static T>>()
            .expect("TypeId collision — should be impossible")
    }

    /// Runs `build` at most once, ever, for this `(Caller, T)` pair.
    ///
    /// The node set was the first fact about a type worth memoizing this way.
    /// It is not the only one: the weighted adjacency of the 1-skeleton and the
    /// homology of the nerve are equally facts *about the type*, computed from
    /// nothing but `nodes()`, and equally wasteful to recompute per query.
    pub fn get_or_build<Caller: 'static + ?Sized, T: 'static + Send + Sync>(
        build: impl FnOnce() -> T,
    ) -> &'static T {
        slot_for::<Caller, T>().get_or_init(|| Box::leak(Box::new(build())))
    }

    /// Slice form, for `nodes()`. `Vec<B>` and `B` key differently, so this
    /// cannot collide with a `get_or_build::<Caller, Vec<B>>`.
    pub fn get_or_build_slice<Caller: 'static + ?Sized, B: 'static + Send + Sync>(
        build: impl FnOnce() -> Vec<B>,
    ) -> &'static [B] {
        get_or_build::<Caller, Box<[B]>>(|| build().into_boxed_slice())
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
        nodes_cache::get_or_build_slice::<Self, B>(Self::build_nodes)
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

/// A distinct valley of the length functional: one geodesic from `p` to `q`
/// that is a strict local minimum of length among nearby paths.
///
/// A homotopy class may contain several. Two edge-paths lie in the same basin
/// exactly when the discrete geodesic flow carries them to the same geodesic —
/// which is what [`NerveComplex::basins`] uses as the label, rather than any
/// combinatorial invariant of the edge-paths themselves. Homotopy cannot
/// distinguish basins; only the flow can.
#[derive(Debug, Clone)]
pub struct Basin<P, F> {
    /// The converged geodesic, as a polyline whose vertices lie on it.
    pub path: Vec<P>,
    /// Its exact arc length.
    pub length: F,
    /// The edge-path through the nerve that descended into this basin.
    pub witness: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeodesicCertificate {
    /// An `overestimation_bound` was asserted. Without one, the ceiling is
    /// `graph_opt` and exactly one basin is inspected. Fix: assert a bound.
    pub bound_asserted: bool,
    /// The search terminated by clearing its ceiling, not by exhausting
    /// `max_candidate_paths` or the frontier. Fix: raise the caps.
    pub search_exhaustive: bool,
    /// Every candidate examined straightened. A failure here is the rescue
    /// budget running out — `2ρ` reaching for the injectivity radius.
    /// Fix: raise `max_rescues`.
    pub straightening_result: StraighteningResult,
}

impl GeodesicCertificate {
    pub fn is_global(&self) -> bool {
        self.bound_asserted
            && self.search_exhaustive
            && matches!(self.straightening_result, StraighteningResult::Success)
    }
}

#[derive(Debug, Clone)]
pub struct Geodesic<P, F> {
    pub path: Vec<P>,
    pub length: F,
    pub certificate: GeodesicCertificate,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum StraighteningResult {
    Success,
    Stalled(usize),
    ArithmeticFloor,
    NotConverged,
    NotConnected,
    MaxRescues,
}

/// A polyline with its cumulative arc length, so sampling is `O(log n)`.
#[derive(Debug, Clone)]
pub struct ArcPoly<P, F> {
    pub pts: Vec<P>,
    /// `cum[i]` is the arc length from `pts[0]` to `pts[i]`. `cum[0] == 0`.
    pub cum: Vec<F>,
}

impl<P, F: Real> ArcPoly<P, F> {
    pub fn total(&self) -> F {
        self.cum.last().copied().unwrap_or_else(F::zero)
    }
}

fn free_reduce(word: Vec<(usize, bool)>) -> Vec<(usize, bool)> {
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

pub trait NerveComplexParameters<
    P: Point,
    V: Euclidean<F: 'static + Send + Sync>,
    T: TangentBundle<P, V> + Point,
    B: Bounded<T, P, V> + 'static + Send + Sync,
>: Nodes<B>
{
    // ========================================================================
    // LIMITS
    // ========================================================================

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
    /// `d_G ≤ κ·d_M + C`. Returns `(κ, C)`.
    ///
    /// A pure multiplicative bound is unsatisfiable: `p` and `q` in one cell with
    /// `d_M → 0` still pay both legs, so `d_G/d_M → ∞`. The additive term is not
    /// slack, it is the covering radius making itself felt.
    fn overestimation_bound() -> Option<(V::F, V::F)> {
        None
    }

    /// Hard cap on how many candidate edge-paths are straightened. Raising it
    /// strengthens the guarantee; lowering it trades certification for speed.
    fn max_candidate_paths() -> usize {
        512
    }

    /// Cap on heap entries. Prefixes vastly outnumber completions, and this is
    /// the quantity that threatens memory. Independent of the above: no ratio
    /// between prefixes-in-flight and completions exists.
    fn max_frontier() -> usize {
        1 << 20
    }

    /// Cap on local insertions before declaring the charts unusable.
    /// when flowing a polyline, sometimes a point might go out of the
    /// injectivity radius of its neighbors, in that case, we insert
    /// a point between them to try to rescue the polyline.
    fn max_rescues() -> usize {
        64
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

    /// Cap on samples per `same_basin` comparison. Exceeding it means the
    /// polyline is longer than `max_samples · δ_s`, and no comparison at that
    /// spacing could prove two prefixes share a basin — so the prune is
    /// declined rather than performed on insufficient evidence.
    fn max_samples() -> usize {
        64
    }

    /// Generator count above which `fundamental_group` returns a correct but
    /// non-canonical presentation.
    ///
    /// Canonicalisation brute-forces `n! · 2ⁿ` relabelings, each costing
    /// `O(|R|·|w|²)`. At `n = 6` that is milliseconds; at `n = 8`, a minute;
    /// at `n = 9`, hours. Genus-`g` surfaces need `n = 2g`, so this admits
    /// genus 3 and rejects genus 4.
    ///
    /// Above the bound the presentation is still *correct* — Tietze is sound
    /// regardless — it simply is not comparable to another presentation of the
    /// same group by `==`. Use `check_exactly_equal` only below it.
    fn max_canonical_generators() -> usize {
        7
    }

    /// Sweeps of the discrete geodesic flow applied to a prefix before it is
    /// compared for basin membership.
    ///
    /// # Sound at any value
    ///
    /// The flow is a monotone descent, so `A ⇝ σ(A)` for *any* number of
    /// sweeps — `σ(A)` need not be a geodesic, only a descent image of `A`.
    /// The basin argument then runs:
    ///
    ///   1. `A ⇝ σ(A)` and `B ⇝ σ(B)` by descent.
    ///   2. `σ(A)`, `σ(B)` within `δ_s` pointwise ⟹ the region swept between
    ///      them is narrower than `δ_s` ⟹ by [`Self::overestimation_bound`]
    ///      no metric feature lives there ⟹ no saddle ⟹ they are joined by a
    ///      length-nonincreasing homotopy.
    ///   3. Hence `A` and `B` lie in one basin.
    ///
    /// Nothing in that chain mentions convergence. **The sweep count is a pure
    /// tuning knob**: more sweeps prune more often, fewer prune less, neither
    /// can prune wrongly. Zero recovers the raw-polyline test, which never
    /// prunes at all.
    ///
    /// # Why two is enough
    ///
    /// Gauss–Seidel is a *smoother*: it annihilates high-frequency error
    /// fastest, and lattice wiggle is exactly the wavelength-`2h` mode, damped
    /// by roughly `1/3` per sweep. Starting from amplitude `~h/2 ≈ 0.7·δ_s`,
    /// two sweeps leaves `~0.08·δ_s` — well inside the threshold, at `O(n)`
    /// chart operations rather than the `O(n³)` of full convergence.
    fn prefix_smoothing_sweeps() -> usize {
        2
    }

    fn max_basins_per_class() -> usize {
        8
    }

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
}

impl<
    P: Point,
    V: Euclidean<F: 'static + Send + Sync>,
    T: TangentBundle<P, V> + Point,
    B: Bounded<T, P, V> + 'static + Send + Sync,
    C: NerveComplexParameters<P, V, T, B>,
> NerveComplex<P, V, T, B> for C
{
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
    V: Euclidean<F: 'static + Send + Sync>,
    T: TangentBundle<P, V> + Point,
    B: Bounded<T, P, V> + 'static + Send + Sync,
>: NerveComplexParameters<P, V, T, B>
{
    /// The nerve's topology, computed once per type. See [`NerveTopology`].
    fn topology() -> &'static NerveTopology {
        nodes_cache::get_or_build::<Self, NerveTopology>(Self::build_topology)
    }

    fn build_topology() -> NerveTopology {
        let n = Self::nodes().len();
        let neighbors: Vec<Vec<usize>> = (0..n).map(|i| Self::get_neighbors(i).collect()).collect();

        // BFS spanning tree.
        let mut parent: Vec<Option<usize>> = vec![None; n];
        let mut seen = vec![false; n];
        let mut queue = std::collections::VecDeque::from([0usize]);
        seen[0] = true;
        while let Some(u) = queue.pop_front() {
            for &v in &neighbors[u] {
                if !seen[v] {
                    seen[v] = true;
                    parent[v] = Some(u);
                    queue.push_back(v);
                }
            }
        }
        debug_assert!(
            seen.iter().all(|&s| s),
            "nerve is disconnected: `fundamental_group` would report π₁ of \
             component zero, silently"
        );

        // Generators: the non-tree edges.
        let mut edge_gen: HashMap<(usize, usize), (usize, bool)> = HashMap::new();
        let mut n_generators = 0usize;
        for i in 0..n {
            for &j in &neighbors[i] {
                if i < j && parent[j] != Some(i) && parent[i] != Some(j) {
                    edge_gen.insert((i, j), (n_generators, false));
                    edge_gen.insert((j, i), (n_generators, true));
                    n_generators += 1;
                }
            }
        }

        // Relations: one per triangle. `π₁` of a graph is free — relations arise
        // only from 2-simplices.
        let mut relations: Vec<Vec<(usize, bool)>> = Vec::new();
        let mut columns: Vec<Vec<i64>> = Vec::new();
        for i in 0..n {
            for &j in &neighbors[i] {
                for &k in &neighbors[j] {
                    if !(i < j && j < k && neighbors[i].contains(&k)) {
                        continue;
                    }
                    let word: Vec<(usize, bool)> = [(i, j), (j, k), (k, i)]
                        .iter()
                        .filter_map(|e| edge_gen.get(e).copied())
                        .collect();
                    let word = free_reduce(word);
                    if word.is_empty() {
                        continue;
                    }
                    // The abelianised relator is the same word, read additively.
                    let mut col = vec![0i64; n_generators];
                    for &(g, inverted) in &word {
                        col[g] += if inverted { -1 } else { 1 };
                    }
                    if col.iter().any(|&x| x != 0) {
                        columns.push(col);
                    }
                    relations.push(word);
                }
            }
        }

        let abel = Abelianisation::from_relations(n_generators, columns);
        NerveTopology {
            n_generators,
            edge_gen,
            relations,
            abel,
        }
    }

    /// The symmetrised, exactly-weighted adjacency of the 1-skeleton.
    ///
    /// Depends only on `Self`, costs `O(n²)` chart operations, and was being
    /// rebuilt on every query. It is a fact about the type, like `nodes()`.
    fn adjacency() -> &'static Vec<Vec<(usize, V::F)>> {
        nodes_cache::get_or_build::<Self, Vec<Vec<(usize, V::F)>>>(Self::build_adjacency)
    }

    fn build_adjacency() -> Vec<Vec<(usize, V::F)>> {
        let n = Self::nodes().len();
        let mut adj: Vec<Vec<(usize, V::F)>> = vec![Vec::new(); n];
        for i in 0..n {
            for j in Self::get_neighbors(i) {
                let Some(w) = Self::edge_weight(i, j) else {
                    panic!("adjacent nodes cannot see each other: 2ρ exceeds injectivity radius");
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

    /// `H₁` and the edge→generator map, for prefix deduplication.
    fn homology() -> (
        &'static Abelianisation,
        &'static HashMap<(usize, usize), (usize, bool)>,
    ) {
        let t = Self::topology();
        (&t.abel, &t.edge_gen)
    }

    // ========================================================================
    // TYPE-LEVEL TOPOLOGY
    // ========================================================================

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
        let topology = Self::topology();
        let n_generators = topology.n_generators;

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
            w = free_reduce(w);
            while w.len() >= 2 {
                let (f, l) = (w[0], w[w.len() - 1]);
                if f.0 == l.0 && f.1 != l.1 {
                    w.remove(0);
                    w.pop();
                    w = free_reduce(w);
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
            free_reduce(out)
        }

        let mut alive: Vec<bool> = vec![true; n_generators];
        let mut rels: Vec<Vec<(usize, bool)>> = topology
            .relations
            .iter()
            .cloned()
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
                            free_reduce(rest)
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

        // Tietze eliminates a generator only when it occurs exactly once in some
        // relator. Nothing guarantees that ever happens, so nothing bounds the
        // survivor count — on a cover whose triangles give no such generator, this
        // is still one per non-tree edge. `n! · 2ⁿ` is fine at 2 and a hang at 12,
        // and `1u32 << n` is a shift overflow at 32.
        if n_generators > Self::max_canonical_generators() {
            return FundamentalGroupPresentation {
                n_generators,
                relations,
            };
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
                            canonical_relator(&relabeled_word)
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

    // ========================================================================
    // LENGTH CALCULATION GUTS
    // ========================================================================

    /// Build a polyline with its cumulative arc length. `O(len)` hops, once.
    fn arc_poly(pts: Vec<P>) -> Option<ArcPoly<P, V::F>> {
        let mut cum = Vec::with_capacity(pts.len());
        cum.push(V::F::zero());
        for w in pts.windows(2) {
            let d = Self::hop(&w[0], &w[1])?;
            cum.push(*cum.last().expect("nonempty") + d);
        }
        Some(ArcPoly { pts, cum })
    }

    /// The point a fraction `t` along, by arc length. `O(log len)` plus one `log`.
    fn sample(ap: &ArcPoly<P, V::F>, t: V::F) -> Option<P> {
        let total = ap.total();
        if !(total > V::F::zero()) {
            return ap.pts.first().cloned();
        }
        let target = total * t;

        let key = target;
        let (mut lo, mut hi) = (0usize, ap.cum.len() - 1);
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            if ap.cum[mid] <= key {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        let seg = ap.cum[hi] - ap.cum[lo];
        let s = if seg > V::F::zero() {
            (target - ap.cum[lo]) / seg
        } else {
            V::F::zero()
        };
        let chart = T::chart_at(&ap.pts[lo]);
        let v = chart.to_local(&ap.pts[hi])?;
        Some(chart.to_global(v * s))
    }

    /// Sample spacing must be finer than the covering radius: a bump narrower
    /// than `δ_s` is what `Φ` forbids, and a bump wider than `δ_s` cannot hide
    /// between two samples spaced `δ_s` apart. A fixed count silently fails on
    /// long polylines.
    fn n_samples(total: V::F, rho: V::F) -> Option<usize> {
        let n = (total / rho).ceil().to_usize()?;
        (n <= Self::max_samples()).then(|| n.max(2))
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

    // ---------------------------------------------------------------
    // Stage 1 — graph search selects the basin.
    // ---------------------------------------------------------------

    /// Dense `O(n²)` Dijkstra from a set of sources, each carrying an initial
    /// distance.
    ///
    /// Multi-source with offsets is what lets the endpoints `p` and `q` enter
    /// the graph rather than being glued on afterwards. Seeding `dist[j]` with
    /// `‖log_{x_j}(q)‖` for every node containing `q` makes `dist[v]` the true
    /// shortest distance from `x_v` all the way to `q`, legs included.
    ///
    /// Dense rather than heap-based because node counts are small — and,
    /// incidentally, because a `BinaryHeap` keyed on `V::F` would need a total
    /// order the scalar does not supply.
    fn dijkstra(adj: &[Vec<(usize, V::F)>], sources: &[(usize, V::F)]) -> Vec<Option<V::F>> {
        let n = adj.len();
        let mut dist: Vec<Option<V::F>> = vec![None; n];
        let mut done = vec![false; n];

        for &(s, d0) in sources {
            if s < n && dist[s].is_none_or(|d| d0 < d) {
                dist[s] = Some(d0);
            }
        }

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

    /// One relaxation at an interior vertex `b` between `a` and `c`.
    ///
    /// The discrete energy `E(b) = d(a,b)² + d(b,c)²` has gradient
    /// `∇E = −2(log_b a + log_b c)`, so descending with step `1/2` sends `b` to
    /// the geodesic midpoint. The returned magnitude is the residual kink: zero
    /// exactly when `log_b a` and `log_b c` are antiparallel, i.e. when `b` sits
    /// straight on the geodesic through `a` and `c`.
    ///
    /// Note this needs both `log`s *from `b`'s chart* — unlike [`Self::hop`],
    /// which may borrow either endpoint's. `relax` is therefore strictly the
    /// more demanding operation, and it is where a too-coarse polyline first
    /// fails.
    fn relax(a: &P, b: &P, c: &P) -> Option<(P, V::F)> {
        let chart = T::chart_at(b);
        let va = chart.to_local(a)?;
        let vc = chart.to_local(c)?;
        let delta = (va + vc) * half();
        Some((chart.to_global(delta), delta.norm()))
    }

    /// One Gauss–Seidel sweep. Returns `(worst kink, lagged length)`.
    fn relax_sweep(pts: &mut [P]) -> Result<(V::F, V::F), StraighteningResult> {
        if pts.len() < 3 {
            let len = Self::polyline_length(pts).ok_or(StraighteningResult::NotConnected)?;
            return Ok((V::F::zero(), len));
        }
        let mut worst = V::F::zero();
        let mut total = V::F::zero();
        let last = pts.len() - 2;

        for i in 1..pts.len() - 1 {
            let (left, rest) = pts.split_at_mut(i);
            let (mid, right) = rest.split_at_mut(1);
            let chart = T::chart_at(&mid[0]);
            let va = chart
                .to_local(&left[i - 1])
                .ok_or(StraighteningResult::Stalled(i))?;
            let vc = chart
                .to_local(&right[0])
                .ok_or(StraighteningResult::Stalled(i))?;

            // `‖va‖` is the segment `(i-1, i)`; the final vertex also contributes
            // its right segment. Every segment is counted exactly once.
            total = total + va.norm();
            if i == last {
                total = total + vc.norm();
            }

            let delta = (va + vc) * half();
            let kink = delta.norm();
            mid[0] = chart.to_global(delta);
            if kink > worst {
                worst = kink;
            }
        }
        Ok((worst, total))
    }

    /// Halve the two segments adjacent to vertex `i`, by inserting their
    /// geodesic midpoints.
    ///
    /// Local, because the defect is local: one vertex's chart failed to reach
    /// one neighbour. Subdividing the whole polyline to repair one neighbourhood
    /// permanently doubles the cost of every subsequent sweep, for every
    /// candidate, to fix a problem confined to two segments.
    ///
    /// [`Self::midpoint`] tries both endpoints' charts, so it can succeed where
    /// [`Self::relax`] failed — the failure was `b` not seeing `a`, and `a` may
    /// well see `b`. When even that fails the polyline is `Disconnected`.
    fn rescue(mut pts: Vec<P>, i: usize) -> Vec<P> {
        let mid_bc = Self::midpoint(&pts[i], &pts[i + 1])
            .expect("polyline_length succeeded, so adjacent vertices are mutually visible");
        let mid_ab = Self::midpoint(&pts[i - 1], &pts[i])
            .expect("polyline_length succeeded, so adjacent vertices are mutually visible");
        pts.insert(i + 1, mid_bc);
        pts.insert(i, mid_ab);
        pts
    }

    /// Run the flow until every interior vertex is straight.
    ///
    /// # The convergence test is on the kink, not the length
    ///
    /// A polyline is a geodesic precisely when `log_b a = −log_b c` at every
    /// interior vertex: the two adjacent geodesic segments then meet with zero
    /// turning, and their concatenation is a single geodesic. That residual —
    /// the kink — is what [`Self::relax_sweep`] already returns, and it is a
    /// direct, scale-explicit measurement of the property being sought.
    ///
    /// Length is only a *proxy* for it, and a biased one. Gauss–Seidel on a
    /// chain is not a scalar iteration: the error is a sum of modes decaying at
    /// rates `rₖ ≈ cos(kπ/n)`, so the observed step `d` falls steeply while the
    /// fast modes die, then levels off as the slow mode `r₁` takes over. A ratio
    /// `d₁/d₀` sampled during that transient badly underestimates `r₁`, and the
    /// geometric tail estimate `d·r/(1−r)` built from it underestimates the
    /// remaining error by a factor of `(1−r₁)/(1−r_sampled)` — tens, at `n = 6`.
    /// The flow then stops while `~n²·d` of error is still outstanding. Longer
    /// polylines have `r₁` nearer one, longer transients, and worse
    /// underestimates: two candidates converging into the *same* basin from
    /// different hop counts come to rest at different lengths, which is
    /// impossible for two polylines on one curve and is the tell that the test,
    /// not the flow, was wrong.
    ///
    /// The kink has none of this structure. Since length is *stationary* at a
    /// geodesic, a positional residual `τ` costs `O(τ²/h)` in length — so
    /// driving the kink to `√ε·h` buys length correct to `ε·h`, full machine
    /// precision, with no extrapolation and no model of the spectrum.
    fn relax_to_convergence(pts: &mut Vec<P>) -> Result<V::F, StraighteningResult> {
        let eps = V::F::epsilon();
        let iters = Self::max_straightening_iterations(pts.len());
        let segments = scalar::<V::F>(pts.len().saturating_sub(1).max(1));

        let mut prev = Self::polyline_length(pts).ok_or(StraighteningResult::NotConnected)?;

        let mut converged = false;
        for _ in 0..iters {
            let (kink, len) = Self::relax_sweep(pts)?;

            // `len` is lagged — Gauss–Seidel moves each vertex as it goes, so a
            // segment is measured just before its right endpoint moves. Fine for
            // both uses: `h` sets the kink threshold's scale, and the floor test
            // compares successive lengths.
            let h = len / segments;
            if kink < eps.sqrt() * h {
                converged = true;
                break;
            }

            // Arithmetic has run out. Sixteen ulps, not one — at the rounding floor
            // `d` is noise of magnitude `~ε·len` and random sign, so a one-ulp test
            // is a coin flip that keeps the loop alive until the iteration cap.
            if !(eps * len * scalar(16) < (prev - len).abs()) {
                return Err(StraighteningResult::ArithmeticFloor);
            }
            prev = len;
        }

        if !converged {
            return Err(StraighteningResult::NotConverged);
        }
        // The returned length is always `polyline_length` of an actual polyline:
        // exact hops along a converged geodesic. The lagged sum never escapes.
        Self::polyline_length(pts).ok_or(StraighteningResult::NotConnected)
    }

    /// Relax a polyline to the geodesic at the bottom of its basin, and return
    /// that geodesic with its exact length.
    ///
    /// Endpoints are pinned. The flow is the negative gradient of the discrete
    /// energy `Σ d(pᵢ, pᵢ₊₁)²`, so it descends monotonically and cannot cross
    /// the index-1 saddle out of its basin — which is why the graph search's
    /// basin selection is respected rather than undone.
    ///
    /// # A converged polyline is exact at any vertex count
    ///
    /// When every interior vertex is straight — `log_b a = −log_b c` — the two
    /// adjacent geodesic segments meet with zero kink, so their concatenation
    /// *is* a single geodesic. Its length is the sum of exact hops along it,
    /// which is that geodesic's exact arc length. Inserting further vertices
    /// places them *on* a curve that is already straight: the next sweep moves
    /// nothing and the length does not change.
    ///
    /// So there is no accuracy ladder, and none is needed. Subdivision exists
    /// for exactly one reason: to keep every `log` defined, so that the flow is
    /// *unconstrained* and can actually reach its critical point. A vertex whose
    /// chart cannot see a neighbour has stalled, not converged — it would move
    /// further if it could.
    ///
    /// Consequently the flow runs at the coarsest resolution the charts permit.
    /// Gauss–Seidel needs `O(n²)` sweeps of `O(n)` work, so keeping `n` at the
    /// hop count of the edge-path — rather than eight times it — is worth two
    /// orders of magnitude, and costs nothing in exactness.
    fn straighten(mut pts: Vec<P>) -> Result<(Vec<P>, V::F), StraighteningResult> {
        let mut rescues = 0;
        let length = loop {
            match Self::relax_to_convergence(&mut pts) {
                Ok(len) => break len,
                Err(i) => {
                    rescues += 1;
                    if rescues > Self::max_rescues() {
                        return Err(StraighteningResult::MaxRescues);
                    }
                    pts = Self::rescue(
                        pts,
                        match i {
                            StraighteningResult::Stalled(x) => x,
                            e => Err(e)?,
                        },
                    );
                }
            }
        };
        Ok((pts, length))
    }

    /// Every node whose bounded domain contains `p`, paired with the exact
    /// geodesic distance from that node's base point to `p`.
    ///
    /// A point typically lies in three or four overlapping domains. The old
    /// `locate` returned only the nearest, which silently discarded the fact
    /// that the shortest route out of `p` need not begin at the nearest base
    /// point — the second-nearest may sit squarely in the right direction.
    fn locate_all(p: &P) -> Vec<(usize, V::F)> {
        Self::nodes()
            .iter()
            .enumerate()
            .filter_map(|(i, node)| {
                let v = node.as_ref().to_local(p)?;
                (node.sdf(&v) < V::F::zero()).then(|| (i, v.norm()))
            })
            .collect()
    }

    /// Relative tolerance at which two converged geodesics are declared to be
    /// the same basin.
    ///
    /// Basins are separated by index-1 saddles, so their minimizers' lengths
    /// typically differ by far more than this. The exception is symmetry: a
    /// bump placed symmetrically between `p` and `q` yields two distinct
    /// geodesics of *identical* length. Merging those is harmless for
    /// [`Self::geodesic_distance`] — they are equally good answers — but
    /// [`Self::basins`] will report one where there are two. If you need the
    /// paths rather than the distance, compare vertices, not lengths.
    ///
    /// `√ε` rather than `ε`: the converged lengths of two edge-paths in the
    /// same basin agree only to the accuracy of the flow, and length is
    /// stationary at a geodesic, so vertex error `√ε` gives length error `ε`.
    /// Discriminating at `ε` would split one basin into many.
    fn basin_tolerance() -> V::F {
        V::F::epsilon().sqrt()
    }

    /// The covering radius `δ_s`, recovered from the asserted bound.
    ///
    /// `C = (1 + κ)·2δ_s` is where the additive term comes from: two endpoints,
    /// each paying up to `δ_s` to reach a base point, each amplified by `κ`. So
    /// `δ_s = C / (2(1+κ))`, and asserting `Φ` has already asserted the scale
    /// below which the metric holds no features.
    ///
    /// That is precisely what the separation test below needs, so it introduces
    /// no assumption of its own.
    fn covering_radius() -> Option<V::F> {
        let (kappa, c) = Self::overestimation_bound()?;
        Some(c / (scalar::<V::F>(2) * (V::F::one() + kappa)))
    }

    /// Apply `sweeps` descent sweeps with both endpoints pinned.
    ///
    /// `None` only when the charts cannot be made to reach — the same failure
    /// [`Self::straighten`] reports as `MaxRescues`. A failure here declines the
    /// prune rather than losing a basin.
    fn smooth(mut pts: Vec<P>, sweeps: usize) -> Option<Vec<P>> {
        let mut rescues = 0usize;
        let mut done = 0usize;
        while done < sweeps {
            match Self::relax_sweep(&mut pts) {
                Ok(_) => done += 1,
                Err(StraighteningResult::Stalled(i)) => {
                    rescues += 1;
                    if rescues > Self::max_rescues() {
                        return None;
                    }
                    pts = Self::rescue(pts, i);
                }
                Err(_) => return None,
            }
        }
        Some(pts)
    }

    /// The smoothed prefix `[p, x_{i₀}, …, x_u]`, with its arc length, ready to
    /// be compared against others at the same `(node, class)`.
    ///
    /// Both endpoints are pinned: `p`, and the tip base point `x_u`. Two
    /// prefixes at the same tip therefore smooth toward the *same* endpoints,
    /// and comparing them is comparing two candidate geodesics from `p` to
    /// `x_u` in one homotopy class.
    fn smoothed_prefix(p: &P, nodes: &[usize]) -> Option<ArcPoly<P, V::F>> {
        let pts: Vec<P> = std::iter::once(p.clone())
            .chain(nodes.iter().map(|&k| Self::base_point_of(k)))
            .collect();
        let pts = Self::smooth(pts, Self::prefix_smoothing_sweeps())?;
        Self::arc_poly(pts)
    }

    /// Whether two smoothed curves with common endpoints provably lie in one
    /// basin.
    ///
    /// The threshold is `δ_s` and not a fudge of it, because both arguments are
    /// descent images: whatever lattice-scale wiggle their raw polylines
    /// carried has been smoothed away, and what remains is genuine geometric
    /// divergence. `Φ` forbids a saddle in a region narrower than `δ_s`, so two
    /// such curves are joined by a length-nonincreasing homotopy.
    ///
    /// Sample density is a property of the *comparison*, not of either curve:
    /// the longer one sets `n`, so both are read at the same fractions and a
    /// bump wider than `δ_s` cannot slip between samples on either. Comparing
    /// two curves at their own densities — different `n`, different fractions —
    /// makes the test vacuously false whenever their lengths differ, which is
    /// almost always.
    ///
    /// Short-circuits on the first divergence: curves on opposite sides of a
    /// bump part company immediately.
    ///
    /// `None` when the curve is longer than `max_samples · δ_s` and cannot be
    /// sampled finely enough to *prove* anything. Callers must read that as
    /// "not proven", never as "false".
    fn same_basin(a: &ArcPoly<P, V::F>, b: &ArcPoly<P, V::F>, rho: V::F) -> Option<bool> {
        let longer = if a.total() > b.total() {
            a.total()
        } else {
            b.total()
        };
        let n = Self::n_samples(longer, rho)?;
        for k in 1..n {
            let t = scalar::<V::F>(k) / scalar::<V::F>(n);
            let d = Self::hop(&Self::sample(a, t)?, &Self::sample(b, t)?)?;
            if !(d < rho) {
                return Some(false);
            }
        }
        Some(true)
    }

    /// Same basin, with "cannot prove" folded to "no".
    ///
    /// The asymmetry is the whole point. Failing to prune costs search time;
    /// pruning wrongly costs a basin, and therefore the answer.
    fn provably_same_basin(a: &ArcPoly<P, V::F>, b: &ArcPoly<P, V::F>, rho: V::F) -> bool {
        Self::same_basin(a, b, rho).unwrap_or(false)
    }

    /// Every distinct basin of geodesics from `p` to `q` that the cover can
    /// reach, together with two completeness flags: whether the search
    /// terminated by clearing its ceiling (rather than a cap), and whether
    /// every candidate it examined straightened successfully.
    ///
    /// # The endpoints are graph nodes, not decorations
    ///
    /// `p` and `q` are not base points. Their distances to the base points they
    /// sit near — the *legs* — are bounded by the covering radius, but a leg
    /// traversed in the useful direction shortens a path while the same leg
    /// traversed backwards lengthens it. The swing between two classes is
    /// therefore up to twice the covering radius, which on any lattice cover is
    /// large enough to invert the ranking of classes whose true lengths differ
    /// by less than that.
    ///
    /// Worse, a regular lattice manufactures *exact* ties: two base points four
    /// steps apart in either direction have identical graph distance both ways,
    /// and a base-point-only Dijkstra picks between the two homotopy classes by
    /// coin flip. The legs are the entire tiebreaker, and they were invisible.
    ///
    /// So: the target Dijkstra is seeded at *every* node containing `q`, with
    /// initial distance `‖log(q)‖`; the search is seeded at every node
    /// containing `p`, with `acc = ‖log(p)‖`; and a prefix completes when it
    /// reaches any node containing `q`, paying that node's leg. The heuristic
    /// `f = acc + to_dst[u]` is then a lower bound on the *whole polyline*,
    /// endpoints included — which is what the ceiling `(1+e)·best` is compared
    /// against, and what makes that comparison dimensionally sound.
    ///
    /// # Why this is a basin search and not a path search
    ///
    /// Simple edge-paths inside the budget grow exponentially with node count.
    /// Basins do not — they are the index-0 critical points of the length
    /// functional on path space, `O(1)` for any manifold you would cover.
    /// Straightening is both the expensive operation and the projection onto
    /// `π₀` of the sublevel path space: two edge-paths flow to the same
    /// geodesic exactly when they lie in the same component. Run it once per
    /// basin, not once per path.
    ///
    /// # The ceiling shrinks as the search proceeds
    ///
    /// Straightening only shortens, so the first converged length is already an
    /// upper bound on `d_M(p,q)`, and a better one than `graph_opt`, which pays
    /// corner-cutting at every hop. Recall [`Self::overestimation_bound`] in its
    /// per-geodesic form: *every* geodesic `γ` from `p` to `q` is shadowed by an
    /// edge-path of graph length `≤ (1+e)·len(γ)`. An unreached basin's shortest
    /// edge-path has graph length `≥ f`, so `len(γ) ≥ f/(1+e)`. Once
    /// `f > (1+e)·best`, no unreached basin can beat `best`.
    fn basins(p: &P, q: &P) -> Option<(Vec<Basin<P, V::F>>, bool, StraighteningResult)> {
        /// Frontier entry. `tip` indexes the prefix arena; `key` indexes the
        /// key table. Both are `u32`, so an entry is copyable and a push
        /// allocates nothing.
        ///
        /// `fk` caches `f` as an `f64` so `Ord` need not re-convert on every
        /// heap comparison. It goes through `total_cmp` because a heap requires
        /// a **total, transitive** order, and a tolerance-based scalar (see
        /// `R64`) supplies neither: `a ≈ b` and `b ≈ c` with `a < c` is
        /// permitted by design, and sift-down then makes mutually contradictory
        /// decisions.
        #[derive(Clone, Copy)]
        struct Entry<F: Real> {
            f: F,
            acc: F,
            tip: u32,
            key: u32,
            complete: bool,
        }
        impl<F: Real> PartialEq for Entry<F> {
            fn eq(&self, o: &Self) -> bool {
                self.cmp(o).is_eq()
            }
        }
        impl<F: Real> Eq for Entry<F> {}
        impl<F: Real> PartialOrd for Entry<F> {
            fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(o))
            }
        }
        impl<F: Real> Ord for Entry<F> {
            fn cmp(&self, o: &Self) -> std::cmp::Ordering {
                o.f.partial_cmp(&self.f).unwrap() // reversed: min-heap on `f`
            }
        }

        /// Prefix arena. `(node, parent)`; the root has no parent.
        struct Arena(Vec<(u32, Option<u32>)>);
        impl Arena {
            fn push(&mut self, node: usize, parent: Option<u32>) -> u32 {
                self.0.push((node as u32, parent));
                (self.0.len() - 1) as u32
            }
            fn node(&self, tip: u32) -> usize {
                self.0[tip as usize].0 as usize
            }
            fn contains(&self, mut tip: u32, v: usize) -> bool {
                loop {
                    let (node, parent) = self.0[tip as usize];
                    if node as usize == v {
                        return true;
                    }
                    match parent {
                        Some(par) => tip = par,
                        None => return false,
                    }
                }
            }
            fn path(&self, mut tip: u32) -> Vec<usize> {
                let mut out = Vec::new();
                loop {
                    let (node, parent) = self.0[tip as usize];
                    out.push(node as usize);
                    match parent {
                        Some(par) => tip = par,
                        None => break,
                    }
                }
                out.reverse();
                out
            }
        }

        /// Interns `H₁` keys to `u32`, and memoizes the group action of the
        /// generators on them.
        ///
        /// Keys form `ℤ^r ⊕ torsion` and abelianisation is a homomorphism, so
        /// extending a prefix by an edge is a translation. There are at most
        /// `|classes inside the ceiling| × n_generators` distinct `(key, generator)`
        /// pairs — a few thousand — while the search examines hundreds of thousands
        /// of edges. Memoizing the transition turns two `Vec<i64>` allocations per
        /// edge into one hash lookup.
        ///
        /// Tree edges are the identity and never reach the table at all.
        struct Keys<'a> {
            abel: &'a Abelianisation,
            tab: Vec<Vec<i64>>,
            ids: HashMap<Vec<i64>, u32>,
            /// `(key, generator·2 + inverted) -> key`
            trans: HashMap<(u32, u32), u32>,
        }

        impl<'a> Keys<'a> {
            fn new(abel: &'a Abelianisation) -> Self {
                let id = abel.identity();
                Self {
                    abel,
                    tab: vec![id.clone()],
                    ids: HashMap::from([(id, 0u32)]),
                    trans: HashMap::new(),
                }
            }

            fn intern(&mut self, k: Vec<i64>) -> u32 {
                if let Some(&id) = self.ids.get(&k) {
                    return id;
                }
                let id = self.tab.len() as u32;
                self.tab.push(k.clone());
                self.ids.insert(k, id);
                id
            }

            /// The key of `prefix · e`. `edge == None` for tree edges.
            fn step(&mut self, key: u32, edge: Option<(usize, bool)>) -> u32 {
                let Some((g, inverted)) = edge else {
                    return key;
                };
                let tag = (g as u32) * 2 + <u32 as From<_>>::from(inverted);
                if let Some(&next) = self.trans.get(&(key, tag)) {
                    return next;
                }
                let extended = self.abel.extend(&self.tab[key as usize], edge);
                let next = self.intern(extended);
                self.trans.insert((key, tag), next);
                next
            }
        }

        let sources = Self::locate_all(p);
        let targets = Self::locate_all(q);
        if sources.is_empty() || targets.is_empty() {
            debug_assert!(
                false,
                "covering invariant violated: point outside every domain"
            );
            return None;
        }

        let adj = Self::adjacency();
        let to_dst = Self::dijkstra(adj, &targets);

        let mut leg_q: Vec<Option<V::F>> = vec![None; adj.len()];
        for &(j, d) in &targets {
            leg_q[j] = Some(d);
        }

        let graph_opt = sources
            .iter()
            .filter_map(|&(i, leg)| to_dst[i].map(|d| leg + d))
            .reduce(|a, b| if b < a { b } else { a });

        let Some(graph_opt) = graph_opt else {
            // Both endpoints lie in the cover. If a geodesic joins them, the
            // nerve must not disconnect them — the nerve theorem in degree zero.
            debug_assert!(
                T::chart_at(p).to_local(q).is_none(),
                "nerve disconnects points joined by a geodesic"
            );
            return None;
        };

        let (kappa, c) = Self::overestimation_bound().unwrap_or((V::F::one(), V::F::zero()));
        // `rho == None` means no bound was asserted, so nothing may be pruned:
        // without `Φ` there is no argument that a saddle needs room.
        let rho = Self::covering_radius();

        // Rounding slack. `graph_opt` and `to_dst` were summed backwards from the
        // targets; `acc` is summed forwards. Float addition is not associative, so
        // along the optimal path these reach the same real number by different
        // orders and can differ by an ulp — enough, at `κ = 1, C = 0`, to prune
        // the very path the search exists to find.
        let fudge = V::F::one() + V::F::epsilon() * scalar(8);
        let static_budget = graph_opt * kappa * fudge + c;

        let (abel, edge_gen) = Self::homology();
        let mut keys = Keys::new(abel);
        let mut arena = Arena(Vec::new());

        // Dominance tables. Prefixes: keyed on `(node, key)`. Completions: keyed
        // on `key` alone, since every completion ends at `q`.
        let mut visited: HashMap<(u32, u32), Vec<ArcPoly<P, V::F>>> = HashMap::new();
        // Completions all end at `q`. Grouping them by `H₁` class fragments the
        // dedupe: two completions in one basin that reach different target nodes
        // traverse different final edges and land in different classes. There are
        // `O(basins)` completions, so scan the flat list.
        let mut completed: Vec<ArcPoly<P, V::F>> = Vec::new();

        let mut basins: Vec<Basin<P, V::F>> = Vec::new();
        let mut best: Option<V::F> = None;
        let mut straighten_result = StraighteningResult::Success;
        let mut straightened = 0usize;
        let mut exhaustive = true;

        // Where `log` is defined it already parametrises by arc length, so this is
        // a real geodesic and can never underestimate. It seeds `best`, which
        // immediately tightens the ceiling. A candidate, not a short-circuit:
        // minimality would need `Riemannian`, and we decline to require it.
        if let Some(v) = T::chart_at(p).to_local(q) {
            let length = v.norm();
            best = Some(length);
            basins.push(Basin {
                path: vec![p.clone(), q.clone()],
                length,
                witness: Vec::new(),
            });
        }

        let mut heap: BinaryHeap<Entry<V::F>> = BinaryHeap::new();
        for &(i, leg) in &sources {
            let Some(h) = to_dst[i] else { continue };
            let f = leg + h;
            heap.push(Entry {
                f,
                acc: leg,
                tip: arena.push(i, None),
                key: 0,
                complete: false,
            });
        }

        'search: while let Some(Entry {
            f,
            acc,
            tip,
            key,
            complete,
        }) = heap.pop()
        {
            // Static until a geodesic is in hand, then driven by the best length found.
            // `f` is nondecreasing across pops and lower-bounds any completion of this
            // prefix, so once it clears the ceiling no unreached basin can beat `best`.
            let ceiling = match best {
                Some(b) => {
                    let dynamic = b * kappa * fudge + c;
                    if dynamic < static_budget {
                        dynamic
                    } else {
                        static_budget
                    }
                }
                None => static_budget,
            };
            if ceiling < f {
                break; // exhaustive: everything remaining exceeds the ceiling
            }

            let u = arena.node(tip);
            let nodes = arena.path(tip);

            if complete {
                let raw: Vec<P> = std::iter::once(p.clone())
                    .chain(nodes.iter().map(|&k| Self::base_point_of(k)))
                    .chain(std::iter::once(q.clone()))
                    .collect();

                // Smooth before comparing. A raw polyline through base points carries
                // lattice-scale wiggle of amplitude `~h = √2·δ_s`, which has nothing to
                // do with the geometry; a `δ_s` threshold can never pass on it, and no
                // larger constant is sound (the endpoint-leg term `C ≈ 4.2·δ_s` swamps
                // the manifold). Two sweeps of the flow — a descent, hence sound at any
                // count — remove it, and what remains is genuine divergence.
                let smoothed = Self::smooth(raw.clone(), Self::prefix_smoothing_sweeps())
                    .and_then(Self::arc_poly);

                // Deduplicate BEFORE the flow. Same valley ⟹ the flow converges to the
                // geodesic we already have, and straightening is the expensive operation:
                // running it once per basin rather than once per edge-path is the entire
                // point of a basin search.
                //
                // Flat list, not keyed on `H₁`. Completions all end at `q`, which lies in
                // several target nodes; two completions in one valley arriving via
                // different target nodes traverse different final edges and land in
                // different classes, so a keyed table fragments the dedupe. With
                // `O(basins)` entries the key bought nothing anyway.
                let seen = match (&smoothed, rho) {
                    (Some(ap), Some(r)) => completed
                        .iter()
                        .any(|old| Self::provably_same_basin(old, ap, r)),
                    _ => false, // cannot prove: straighten it rather than lose a basin
                };
                if seen {
                    continue;
                }

                // The smoothed completion is already a descent image of `raw`, so it is a
                // free warm start: `straighten` merely finishes it.
                let start = smoothed.as_ref().map_or(raw, |ap| ap.pts.clone());

                match Self::straighten(start) {
                    Ok((pts, length)) => {
                        if let Some(ap) = smoothed {
                            completed.push(ap);
                        }
                        let tol = Self::basin_tolerance() * length;
                        if !basins.iter().any(|b| tol >= (b.length - length).abs()) {
                            basins.push(Basin {
                                path: pts,
                                length,
                                witness: nodes,
                            });
                        }
                        if best.is_none_or(|b| length < b) {
                            best = Some(length);
                        }
                    }
                    Err(e) => straighten_result = e,
                }

                straightened += 1;
                if straightened >= Self::max_candidate_paths() {
                    exhaustive = false;
                    break;
                }
                continue;
            }

            // Pop-time prefix dominance. Pops at `u` come in nondecreasing `acc` — every
            // entry with tip-node `u` carries the same `to_dst[u]` — so whatever is
            // stored is shorter and `old_acc <= acc` needs no test.
            //
            // The prune is licensed by *continuations*, not by the final geodesic: if
            // `σ(A)` and `σ(B)` coincide then for any continuation `r` the paths `A·r`
            // and `B·r` are joined by descent through `σ(A)·r` — the prefix flow leaves
            // `r` untouched and lowers the total energy. Every completion reachable
            // through `B` has a no-longer counterpart through `A`.
            //
            // Without `Φ` there is no covering radius, hence no argument that a saddle
            // needs room, hence nothing may be pruned.
            if let Some(r) = rho {
                let Some(ap) = Self::smoothed_prefix(p, &nodes) else {
                    continue; // charts unusable here; decline the prune and the expansion
                };
                let slot = visited.entry((u as u32, key)).or_default();
                if slot
                    .iter()
                    .any(|old| Self::provably_same_basin(old, &ap, r))
                {
                    continue; // same class, same valley, longer: dead
                }
                // Cap is a memory guard. Failing to store is always sound.
                if slot.len() < Self::max_basins_per_class() {
                    slot.push(ap);
                }
            }

            // Reaching a node that contains `q` completes a candidate — but does not end
            // the prefix. A longer walk may reach a different target node whose leg is
            // shorter, so completion is its own entry carrying the exact total, and the
            // prefix goes on being extended. Making it an entry rather than straightening
            // here is what preserves nondecreasing pop order, which is the sole
            // justification for the early exit.
            if let Some(leg) = leg_q[u] {
                let total = acc + leg;
                heap.push(Entry {
                    f: total,
                    acc: total,
                    tip,
                    key,
                    complete: true,
                });
            }

            for &(v, w) in &adj[u] {
                if arena.contains(tip, v) {
                    continue; // simple paths only
                }
                let Some(h_v) = to_dst[v] else { continue };
                if heap.len() >= Self::max_frontier() {
                    exhaustive = false;
                    break 'search;
                }

                let acc2 = acc + w;
                let f2 = acc2 + h_v;

                // Slack-guarded, and strictly looser than the pop-side test: anything the
                // pop would accept is pushed. The pop-side check remains the single
                // authority — a second, differently-rounded evaluation of the same
                // predicate is how a prefix comes to pass one test and fail its twin.
                if f2 > ceiling * (V::F::one() + V::F::epsilon() * scalar(64)) {
                    continue;
                }

                heap.push(Entry {
                    f: f2,
                    acc: acc2,
                    tip: arena.push(v, Some(tip)),
                    key: keys.step(key, edge_gen.get(&(u, v)).copied()),
                    complete: false,
                });
            }
        }

        if let Some(b) = best {
            debug_assert!(
                !(graph_opt > kappa * b + c),
                "overestimation bound violated: graph_opt {graph_opt:?} > κ·{best:?} + C"
            );
        }

        if basins.is_empty() {
            return None;
        }
        Some((basins, exhaustive, straighten_result))
    }

    // ---------------------------------------------------------------
    // Entry points.
    // ---------------------------------------------------------------

    /// The global geodesic from `p` to `q`, and its exact arc length.
    ///
    /// # Guarantee
    ///
    /// The returned length is always the *exact* arc length of a real geodesic
    /// from `p` to `q` — never an approximation, up to `O(tol²)` in the
    /// straightening tolerance. Whether it is the *globally* shortest such
    /// geodesic depends on three independent facts: an
    /// [`Self::overestimation_bound`] was asserted; the basin search terminated
    /// by clearing its ceiling rather than a cap; and every candidate examined
    /// straightened successfully. `Geodesic::Global` requires all three.
    fn geodesic_path(p: &P, q: &P) -> Option<Geodesic<P, V::F>> {
        let (basins, exhaustive, straightening_result) = Self::basins(p, q)?;

        let winner = basins
            .into_iter()
            .reduce(|a, b| if b.length < a.length { b } else { a })?;

        let certificate = GeodesicCertificate {
            bound_asserted: Self::overestimation_bound().is_some(),
            search_exhaustive: exhaustive,
            straightening_result,
        };

        let Basin { path, length, .. } = winner;
        Some(Geodesic {
            path,
            length,
            certificate,
        })
    }

    /// The global geodesic distance `d_M(p, q)`. See [`Self::geodesic_path`]
    /// for the guarantee and its one precondition.
    fn geodesic_distance(p: &P, q: &P) -> Option<V::F> {
        Self::geodesic_path(p, q).and_then(|g| {
            if g.certificate.is_global() {
                Some(g.length)
            } else {
                None
            }
        })
    }

    /// The best-effort geodesic distance. This gives an exact locally minimal
    /// geodesic but does not guarantee that it is the minimal geodesic.
    fn geodesic_distance_uncertified(p: &P, q: &P) -> Option<V::F> {
        Self::geodesic_path(p, q).map(|g| g.length)
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
