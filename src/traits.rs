use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

use itertools::Itertools;
use num_traits::{One, Zero, real::Real};

/// A point on a manifold.
///
/// The space of all values of a type `P: Point` is interpreted as a
/// smooth manifold `M`. The mathematical structure of `M` is expressed
/// through the associated traits [`Chart`], [`TangentBundle`], [`Metric`] etc.
///
/// The only meaningful requirement is equality — a point is defined by
/// its identity, the ability to ask whether two points are the same.
/// This should rightly be [`Eq`], but [`PartialEq`] is used in practice
/// because `f64`, the typical scalar type, implements only [`PartialEq`]
/// due to `NaN` — an IEEE 754 artifact with no geometric meaning.
pub trait Point: Clone + PartialEq {}

impl<T: Clone + PartialEq> Point for T {}

/// A chart in an atlas of a manifold.
///
/// The space of all values of a type `C: Chart<P, V>` is interpreted
/// as an atlas of the manifold `M` (the space of `P: Point`), covering
/// it with local coordinate neighbourhoods modelled on `R^N` (`Coords`).
/// The atlas axiom — that every point is covered — is expressed by
/// `chart_at`, which must produce a valid chart in a neighbourhood of
/// any given point.
///
/// `to_local` and `to_global` are the coordinate maps, with `to_local`
/// returning `None` at the singularities of the chart.
pub trait Chart<P: Point, V: Euclidean>: Sized {
    fn to_local(&self, point: &P) -> Option<V>;
    fn to_global(&self, coord: V) -> P;
    fn chart_at(p: &P) -> Self;

    /// Calculates the distance between `self` and `other`
    /// in local coordinates, based at &self.
    fn local_distance(&self, other: &P) -> Option<V::F> {
        self.to_local(other).map(|v| v.norm())
    }

    #[cfg(feature = "testing")]
    fn check_local_inverse(p: &P) -> bool {
        let chart = Self::chart_at(p);
        match chart.to_local(p) {
            Some(local) => p == &chart.to_global(local),
            None => false,
        }
    }
}

/// A scalar field for use as the coordinate type of a Euclidean space.
///
/// Bundles the requirements that a scalar must satisfy to be usable
/// throughout diffable — real arithmetic and debuggability.
pub trait Scalar: Real + std::fmt::Debug {}

impl<R: Real + std::fmt::Debug> Scalar for R {}

/// A finite-dimensional Euclidean space.
///
/// The space of all values of a type `E: Euclidean<N>` is interpreted
/// as R^N (with R := E::F) — the canonical flat Euclidean space of dimension `N`
/// over the field `R`. This is the space in which all local coordinate charts take
/// their values, and in which tangent vectors live.
///
/// Beyond the algebraic structure of a vector space (`Add`, `Sub`, `Mul`,
/// `Neg`, `Zero`), a Euclidean space carries an inner product (`InnerProduct`)
/// which induces a norm and metric, and a canonical tangent bundle
/// (`TangentBundle`) whose charts are globally defined with infinite
/// injectivity radius — reflecting the flatness of the space.
///
/// # Flatness
/// Unlike a general Riemannian manifold, a Euclidean space is flat: geodesics
/// are straight lines, parallel transport is path-independent, and the
/// exponential map is a global isomorphism rather than merely a local one.
/// These properties are verified by the `check_*` methods inherited from
/// `TangentBundle` and by the additional checks in `check_global_chart`,
/// `check_global_geodesic_scaling`, `check_translation_invariance`, and `check_pythagorean`.
///
/// # Implementing
/// Use the `test_euclidean!` macro to verify that your implementation
/// satisfies the Euclidean axioms.
pub trait Euclidean:
    InnerProduct<Self::F>
    + TangentBundle<Self, Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Self::F, Output = Self>
    + Neg<Output = Self>
    + Zero
    + Index<usize, Output = Self::F>
    + IndexMut<usize>
    + Copy
{
    type F: Scalar;
    const N: usize;

    type Iter<'a>: Iterator<Item = &'a Self::F>
    where
        Self: 'a;
    fn iter(&self) -> Self::Iter<'_>;

    fn from_fn(f: impl Fn(usize) -> Self::F) -> Self;
    fn from_array<const N: usize>(arr: [Self::F; N]) -> Self;
    fn to_array<const N: usize>(self) -> [Self::F; N] {
        std::array::from_fn(|i| self[i])
    }

    // Flat space has no singularities — to_local is always Some
    #[cfg(feature = "testing")]
    fn check_global_chart(p: &Self, q: &Self) -> bool {
        let chart = Self::chart_at(p);
        chart.to_local(q).is_some()
    }

    // Translation invariance: d(a + c, b + c) == d(a, b)
    #[cfg(feature = "testing")]
    fn check_translation_invariance(a: &Self, b: &Self, c: &Self) -> bool
    where
        Self: Add<Output = Self> + Clone,
    {
        let dist_ab = a.distance(b);
        let dist_translated = (a.clone() + c.clone()).distance(&(b.clone() + c.clone()));
        dist_ab == dist_translated
    }

    // Geodesic scaling holds globally (infinite injectivity radius):
    // to_global(v * t) is parallel to to_global(v) AND scaled by t exactly
    #[cfg(feature = "testing")]
    fn check_global_geodesic_scaling(p: &Self, v: Self, t: Self::F) -> bool {
        let chart = Self::chart_at(p);
        match (
            chart.to_local(&chart.to_global(v * t)),
            chart.to_local(&chart.to_global(v)),
        ) {
            (Some(tv_local), Some(v_local)) => tv_local == v_local * t,
            _ => false,
        }
    }

    // Pythagorean theorem: d(a, b)² == |a - b|²
    #[cfg(feature = "testing")]
    fn check_pythagorean(a: &Self, b: &Self) -> bool
    where
        Self: Sub<Output = Self> + Clone,
    {
        let dist_sq = a.distance(b);
        let dist_sq = dist_sq * dist_sq;
        let diff = a.clone() - b.clone();
        let norm_sq = diff.norm_squared();
        dist_sq == norm_sq
    }
}

pub trait Group: Point {
    fn identity() -> Self;
    fn compose(&self, other: &Self) -> Self;
    fn inverse(&self) -> Self;

    #[cfg(feature = "testing")]
    fn check_left_identity(&self) -> bool
    where
        Self: PartialEq,
    {
        Self::identity().compose(self) == *self
    }

    #[cfg(feature = "testing")]
    fn check_right_identity(&self) -> bool
    where
        Self: PartialEq,
    {
        self.compose(&Self::identity()) == *self
    }

    #[cfg(feature = "testing")]
    fn check_left_inverse(&self) -> bool
    where
        Self: PartialEq,
    {
        self.inverse().compose(self) == Self::identity()
    }

    #[cfg(feature = "testing")]
    fn check_right_inverse(&self) -> bool
    where
        Self: PartialEq,
    {
        self.compose(&self.inverse()) == Self::identity()
    }

    #[cfg(feature = "testing")]
    fn check_associativity(a: Self, b: Self, c: Self) -> bool
    where
        Self: PartialEq,
    {
        a.compose(&b).compose(&c) == a.compose(&b.compose(&c))
    }
}

/// A Lie group structure on a manifold.
///
/// The space of all values of a type `G: LieGroup<V>` is interpreted as
/// a Lie group — a manifold that is also a group, where the group operations
/// are smooth maps. `V` is the Euclidean space coordinatising the group's
/// tangent space at the identity.
///
/// # Group axioms
/// - **Identity**: there exists an element `e` such that `e * g = g * e = g`
/// - **Inverses**: for every `g` there exists `g⁻¹` such that `g * g⁻¹ = g⁻¹ * g = e`
/// - **Associativity**: `(a * b) * c = a * (b * c)`
///
/// These are not enforced by the type system but are certified by implementing
/// this trait, and verified empirically by the `test_lie_group!` macro.
///
/// # Exponential map at the identity
/// `identity_exp` and `identity_log` are the exponential and logarithm maps
/// centred at the group identity — they witness that `V`, the tangent space
/// at the identity, genuinely linearises the group there. They are not
/// required to work, or have any particular meaning, at any other base point.
///
/// # Automatic tangent bundle
/// Implementing `LieGroup` automatically certifies [`Chart`], [`ExpMap`], and
/// [`TangentBundle`] for `Self` via a blanket implementation: a chart centred
/// at any base point `p` is constructed by left translation — `to_global(v) =
/// p * identity_exp(v)` and `to_local(q) = identity_log(p⁻¹ * q)`. This works
/// because a Lie group is homogeneous: left translation by `p` is a smooth
/// isometry carrying the geometry at the identity to every other point, so
/// the exponential map at the identity alone is sufficient to generate a
/// full tangent bundle over the entire group, with no separate wrapper type
/// needed.
pub trait LieGroup<V: Euclidean>: Group {
    fn identity_exp(v: V) -> Self;
    fn identity_log(p: &Self) -> Option<V>;
}

/// A notion of distance on a manifold.
///
/// The space of all values of a type `P: Metric<R>` is interpreted as
/// a metric space — a set `M` equipped with a distance function
/// `d: M × M → R` satisfying:
/// - **Non-negativity**: `d(a, b) >= 0`
/// - **Identity of indiscernibles**: `d(a, a) = 0`
/// - **Symmetry**: `d(a, b) = d(b, a)`
/// - **Triangle inequality**: `d(a, c) <= d(a, b) + d(b, c)`
///
/// These are not enforced by the type system but are certified by
/// implementing this trait. The first three are verified empirically by
/// the `test_metric!` macro; the triangle inequality is omitted from
/// automated testing since it is numerically fragile to check near-degenerate,
/// nearly-collinear triples without a carefully tuned tolerance.
///
/// A metric is independent of any coordinate structure — it requires
/// neither a [`Chart`] nor a [`Euclidean`] tangent space, only the ability
/// to measure distance between two points directly.
pub trait Metric<R: Real>: Point {
    fn distance(&self, other: &Self) -> R;

    #[cfg(feature = "testing")]
    fn check_self_distance_zero(a: Self) -> bool {
        a.distance(&a) == R::zero()
    }

    #[cfg(feature = "testing")]
    fn check_non_negative(a: Self, b: Self) -> bool {
        a.distance(&b) >= R::zero()
    }

    #[cfg(feature = "testing")]
    fn check_metric_symmetry(a: Self, b: Self) -> bool {
        a.distance(&b) == b.distance(&a)
    }
}

/// An inner product structure on a manifold.
///
/// The space of all values of a type `P: InnerProduct<R>` is interpreted
/// as an inner product space — a vector space equipped with a bilinear,
/// symmetric, positive-definite pairing `⟨·,·⟩: P × P → R`.
///
/// An inner product induces a norm, `|v| = sqrt(⟨v,v⟩)`, and that norm in
/// turn induces a metric, `d(a, b) = |a - b|` — which is why `InnerProduct`
/// is a refinement of [`Metric`] rather than an independent trait. `norm`
/// and `norm_squared` are provided as default methods derived purely from
/// `dot`.
///
/// Not every `Metric` is an `InnerProduct` — the sphere's geodesic distance,
/// for instance, is a perfectly good metric that does not arise from any
/// inner product, since the sphere is not a vector space.
pub trait InnerProduct<R: Real>: Metric<R> {
    fn dot(&self, other: &Self) -> R;

    fn norm(&self) -> R {
        self.norm_squared().sqrt()
    }

    fn norm_squared(&self) -> R {
        self.dot(self)
    }

    #[cfg(feature = "testing")]
    fn check_inner_product_symmetry(a: Self, b: Self) -> bool {
        a.dot(&b) == b.dot(&a)
    }

    #[cfg(feature = "testing")]
    fn check_additivity(a: Self, b: Self, c: Self) -> bool
    where
        Self: Add<Output = Self> + Clone,
    {
        let lhs = (a.clone() + b.clone()).dot(&c);
        let rhs = a.dot(&c) + b.dot(&c);
        lhs == rhs
    }

    #[cfg(feature = "testing")]
    fn check_scalar_linearity(a: Self, c: Self, k: R) -> bool
    where
        Self: Mul<R, Output = Self> + Clone,
    {
        let lhs = (a.clone() * k).dot(&c);
        let rhs = k * a.dot(&c);
        lhs == rhs
    }

    #[cfg(feature = "testing")]
    fn check_positive_definite(a: Self) -> bool
    where
        Self: Zero,
    {
        a == Self::zero() || a.norm() > R::zero()
    }
}

/// By implementing ExpMap you certify that for C<P, V>: ExpMap<P, V> that
/// straight lines through the origin in R^N map to geodesics on M, and
/// that distances from the origin equal arc lengths along those geodesics.
///
/// Additionally, you certify that Self::chart_at(&self.base_point()) == self
pub trait ExpMap<P: Point, V: Euclidean>: Chart<P, V> {
    fn base_point(&self) -> P {
        self.to_global(V::zero())
    }

    // Tests that base_point() is consistent with to_local.
    // Meaningful only when base_point() is overridden, since
    // the default impl makes this trivially true by construction.
    #[cfg(feature = "testing")]
    fn check_base_point_is_origin(&self) -> bool {
        self.to_local(&self.base_point())
            .map_or(false, |c| c.norm() == V::F::zero())
    }

    // Tests that log(exp(0)) == 0, i.e. that the
    // round trip at the origin is the identity.
    #[cfg(feature = "testing")]
    fn check_preservation_of_origin(&self) -> bool {
        let zero = V::zero();
        let exp_zero = self.to_global(zero);
        self.to_local(&exp_zero)
            .map_or(false, |c| c.norm() == V::F::zero())
    }

    /// If a chart centred at `p` exists, `chart_at(p)` returns it.
    /// Formally: `chart_at(p).base_point() == p` whenever `p` is
    /// the base point of some valid chart in this atlas.
    ///
    /// This is weaker than the [`TangentBundle`] centring invariant,
    /// which requires this to hold for *all* `p`. Here it is only
    /// required when `p` is already a base point of some chart —
    /// i.e. `chart_at` correctly identifies the chart when queried
    /// at a known base point.
    #[cfg(feature = "testing")]
    fn check_chart_at_base_point(&self) -> bool {
        Self::chart_at(&self.base_point()).check_preservation_of_origin()
    }

    // geodesics are reversible: log(exp(v)) == -log(exp(-v))
    #[cfg(feature = "testing")]
    fn check_geodesic_symmetry(&self, v: V) -> bool {
        match (
            self.to_local(&self.to_global(v)),
            self.to_local(&self.to_global(-v)),
        ) {
            (Some(fwd), Some(bwd)) => fwd == -bwd,
            _ => true, // at singularity, vacuously true
        }
    }

    // geodesics are straight lines: exp(tv) lies on the same geodesic as exp(v),
    // i.e. log(exp(tv)) and log(exp(v)) are parallel in local coords.
    #[cfg(feature = "testing")]
    fn check_geodesic_scaling(&self, v: V, t: V::F) -> bool {
        match (
            self.to_local(&self.to_global(v * t)),
            self.to_local(&self.to_global(v)),
        ) {
            (Some(tv_local), Some(v_local)) => {
                // tv_local and v_local should be parallel:
                // tv_local × v_local == 0, i.e. dot(tv_local, v_local)² == |tv_local|² * |v_local|²
                let dot = tv_local.dot(&v_local);
                let lhs = dot * dot;
                let rhs = tv_local.norm_squared() * v_local.norm_squared();
                lhs == rhs
            }
            _ => true,
        }
    }

    // isometry to first order: |exp(epsilon * v) - base| / epsilon → |v|
    #[cfg(feature = "testing")]
    fn check_first_order_isometry(&self, v: V, coef: V::F) -> bool {
        let small_v = v * coef;
        self.to_local(&self.to_global(small_v))
            .map_or(true, |local| local.norm() == v.norm() * coef)
    }
}

/// A tangent bundle structure on a manifold.
///
/// The space of all values of a type `C: TangentBundle<P, V>` is
/// interpreted as the tangent bundle `TM` of the manifold `M` (the space
/// of `P: Point`). Each instance is a single tangent space `T_p M` at
/// the base point `p`, coordinatised by `V`.
///
/// By implementing `TangentBundle` you certify that for all `p: P`:
/// `C::chart_at(&p).to_global(V::zero()) == p`
///
/// That is, the chart produced at any point is centred at that point —
/// the origin of the local coordinate system corresponds to the base point
/// on the manifold. This is what distinguishes a tangent bundle from a
/// bare [`Chart`] or [`ExpMap`].
///
/// Use the `test_tangent_bundle!` macro to verify this invariant.
pub trait TangentBundle<P: Point, V: Euclidean>: ExpMap<P, V> {
    fn sectional_curvature(&self, v: V, w: V, epsilon: V::F) -> V::F {
        let p2 = self.to_global(v + w * epsilon);
        let actual = (self.to_local(&p2).unwrap() - v).norm();
        let flat = (w * epsilon).norm();

        let two = V::F::one() + V::F::one();
        (two / (epsilon * epsilon)) * (V::F::one() - actual / flat)
    }

    fn max_sectional_curvature(&self, epsilon: V::F) -> V::F {
        (0..V::N)
            .array_combinations::<2>()
            .map(|[i, j]| {
                let v = V::from_fn(|k| if k == i { V::F::one() } else { V::F::zero() });
                let w = V::from_fn(|k| if k == j { V::F::one() } else { V::F::zero() });
                self.sectional_curvature(v, w, epsilon)
            })
            .fold(V::F::zero(), |max, k| if k > max { k } else { max })
    }

    // p is the point on the manifold which is the base point.
    #[cfg(feature = "testing")]
    fn check_universal_centring(p: P) -> bool {
        let chart = Self::chart_at(&p);
        chart.check_preservation_of_origin() && chart.check_base_point_is_origin()
    }
}

/// A quotient of a Lie group by a central subgroup.
///
/// The space of all values of a type `Q: Quotient<G, H, V>` is interpreted
/// as the quotient group `G/H` — the set of cosets `gH`, with the group
/// operation inherited from `G`. This requires `H` to be central in `G`
/// (so the quotient is well-defined and the cosets `gH` and `Hg` coincide),
/// which in particular makes `H` automatically normal.
///
/// # The lift/canonical pattern
/// Rather than representing a coset abstractly, `Quotient` requires a
/// concrete representation via two operations:
///
/// - [`Quotient::new`] maps a value `g: G` to the `Quotient` value
///   representing its coset `gH`. It must satisfy `canonical(g) ==
///   canonical(h.compose(g))` for every `h: H` (acting on `g` via `G`'s own
///   composition) — i.e. it must not distinguish between elements of the
///   same coset. Beyond that one algebraic requirement, `canonical` is free
///   to be any deterministic, even discontinuous, choice function; it need
///   not be smooth or continuous, since it carries no geometric content of
///   its own. For `S³ / {±1} → SO(3)`, `canonical` is a sign comparison on
///   the real component; for `(R\{0}, ×) / {±1} → (R⁺, ×)`, it is `|x|`.
///
/// - [`Quotient::lift`] recovers *some* representative `g: G` of the coset,
///   satisfying `canonical(self.lift()) == self` for every `self: Q`. Which
///   representative is returned is unspecified beyond that round-trip
///   property — only one of possibly several valid choices needs to be
///   produced.
///
/// All group structure on `Q` — composition, inverse, the exponential map
/// at the identity — is defined generically in terms of `G`'s own structure
/// by lifting, operating in `G`, and re-applying `canonical`:
/// `a.compose(b) = canonical(a.lift().compose(&b.lift()))`. This works
/// because all the differential structure lives in `G`, which is already
/// known to be smooth; `canonical` is purely a bookkeeping step applied
/// after the smooth operation completes, never a smoothness-bearing
/// operation in its own right. The map `G → G/H` being a covering map (a
/// local diffeomorphism) is what makes `G/H` itself a smooth manifold, even
/// though `canonical` — being a *global* choice of representative — is
/// typically forced to be discontinuous somewhere, an unavoidable
/// topological obstruction rather than evidence that `canonical` was chosen
/// poorly.
///
/// # Why `H` must be central
/// Centrality (`h.compose(g) == g.compose(h)` for all `g: G`, `h: H`) is
/// what makes left cosets and right cosets coincide, which is what makes
/// `G/H` a group rather than merely a set of cosets with no induced
/// operation. `Sphere<0, V>` — `{1, -1}` under the relevant composition —
/// is central in every `Sphere<N, V>` for `N ∈ {0, 1, 3}` precisely
/// because `-1` commutes with everything (it is, after all, just a scalar
/// multiple of the identity), which is what makes `S³/{±1} → SO(3)` and
/// `(R\{0}, ×)/{±1} → (R⁺, ×)` both legitimate instances of this trait.
pub trait Quotient<G: LieGroup<V>, H: LieGroup<V>, V: Euclidean>: Point {
    /// Maps `g` to the `Quotient` value representing its coset `gH`.
    fn new(g: G) -> Self;

    /// Recovers some representative of `self`'s coset, satisfying
    /// `new(self.lift()) == self`.
    fn lift(&self) -> G;

    /// the subgroup inclusion H ↪ G
    fn embed(h: H) -> G;

    fn quotient_identity() -> Self {
        Self::new(G::identity())
    }

    fn quotient_compose(&self, other: &Self) -> Self {
        Self::new(self.lift().compose(&other.lift()))
    }

    fn quotient_inverse(&self) -> Self {
        Self::new(self.lift().inverse())
    }

    fn quotient_identity_exp(v: V) -> Self {
        Self::new(G::identity_exp(v))
    }

    fn quotient_identity_log(p: &Self) -> Option<V> {
        G::identity_log(&p.lift())
    }

    /// The sole independent Quotient axiom: new must not
    /// distinguish elements of the same coset. Everything else
    /// (group structure, differential structure) follows from this
    /// plus the inherited LieGroup axioms.
    #[cfg(feature = "testing")]
    fn check_new_respects_coset(g: G, h: H) -> bool
    where
        Self: Metric<V::F>,
        Self: PartialEq,
    {
        Self::new(Self::embed(h).compose(&g)) == Self::new(g)
    }
}

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
pub trait NerveComplex<
    P: Point,
    V: Euclidean,
    T: TangentBundle<P, V> + Point,
    B: Bounded<T, V> + PartialEq,
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
    /// always searches the same fixed set of nodes. Since [`Chart::check_local_inverse`]
    /// verifies that `chart_at(p).to_local(p).is_some()` for arbitrary `p`,
    /// and `chart_at` finds its chart from this fixed `nodes()`, the covering
    /// invariant — every point is covered by at least one node — is automatically
    /// certified by the existing [`Chart`] test infrastructure. No additional
    /// `check_*` methods or `test_*` macros are needed.
    fn nodes() -> &'static [B]
    where
        B: 'static;

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
    fn get_neighbors<'a>(&'a self) -> impl Iterator<Item = usize> + 'a
    where
        B: 'static,
        T: 'a,
    {
        let base = self.base_point();
        let inode = B::chart_at(&base);
        Self::nodes()
            .iter()
            .enumerate()
            .filter_map(move |(j_idx, jnode)| {
                let half = (V::F::one() + V::F::one()).recip();
                // both directions must agree
                // Get the non-restricted domain
                let i_sees_j = inode.inner().to_local(&jnode.base_point().base_point());
                let j_sees_i = jnode.inner().to_local(&base.base_point());
                match (i_sees_j, j_sees_i) {
                    (Some(v_ij), Some(v_ji)) if *jnode != inode => {
                        // The same manifold point — the midpoint of the
                        // geodesic joining the two base points — expressed
                        // in each node's own exponential chart. (ExpMap
                        // guarantees radial geodesics are parametrised by
                        // arc length, so halving the log halves the arc.)
                        if inode.sdf(&(v_ij * half)) < V::F::zero()
                            && jnode.sdf(&(v_ji * half)) < V::F::zero()
                        {
                            Some(j_idx)
                        } else {
                            None
                        }
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
    fn fundamental_group(&self) -> impl GroupPresentation
    where
        B: 'static,
    {
        let nodes = Self::nodes();
        let n = nodes.len();

        // BFS spanning tree
        let mut parent: Vec<Option<usize>> = vec![None; n];
        let mut visited: Vec<bool> = vec![false; n];
        let mut queue = std::collections::VecDeque::new();
        visited[0] = true;
        queue.push_back(0usize);
        while let Some(idx) = queue.pop_front() {
            let chart = Self::chart_at(&nodes[idx].base_point());
            for neighbour_idx in chart.get_neighbors() {
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
                Self::chart_at(&nodes[i].base_point())
                    .get_neighbors()
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

        let neighbors: Vec<Vec<usize>> = (0..n)
            .map(|i| {
                Self::chart_at(&nodes[i].base_point())
                    .get_neighbors()
                    .collect()
            })
            .collect();

        let relations: Vec<Vec<(usize, bool)>> = (0..n)
            .flat_map(|i| {
                let neighbors = &neighbors;
                let edge_word = &edge_word;
                let neighbors_i = &neighbors[i];
                neighbors_i
                    .iter()
                    .flat_map(move |&j| {
                        let neighbors = neighbors;
                        let edge_word = edge_word;
                        if j <= i {
                            return vec![];
                        }
                        let neighbors_j = &neighbors[j];
                        neighbors_j
                            .iter()
                            .filter_map(move |&k| {
                                if k <= j {
                                    return None;
                                }
                                if neighbors[i].contains(&k) {
                                    let mut word = edge_word(i, j);
                                    word.extend(edge_word(j, k));
                                    word.extend(edge_word(k, i));
                                    let reduced = reduce_word(word);
                                    if reduced.is_empty() {
                                        None
                                    } else {
                                        Some(reduced)
                                    }
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
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

pub trait Bounded<P: Point, V: Euclidean>: TangentBundle<P, V> {
    /// The signed distance field in the tangent
    /// space of the chart centered at &self.
    fn sdf(&self, v: &V) -> V::F;
    fn new(p: P) -> Self;
    fn inner(&self) -> &P;
}

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

impl<E: Euclidean> Group for E {
    fn identity() -> Self {
        Self::zero()
    }

    fn compose(&self, other: &Self) -> Self {
        *self + *other
    }

    fn inverse(&self) -> Self {
        -*self
    }
}

impl<E: Euclidean> LieGroup<E> for E {
    fn identity_exp(v: E) -> Self {
        v
    }

    fn identity_log(p: &Self) -> Option<E> {
        Some(*p)
    }
}

impl<V: Euclidean, L: LieGroup<V>> Chart<Self, V> for L {
    fn to_local(&self, point: &Self) -> Option<V> {
        let translated = self.inverse().compose(point);
        Self::identity_log(&translated)
    }

    fn to_global(&self, coord: V) -> Self {
        let translated = Self::identity_exp(coord);
        self.compose(&translated)
    }

    fn chart_at(p: &Self) -> Self {
        p.clone()
    }
}

impl<V: Euclidean, L: LieGroup<V>> ExpMap<Self, V> for L {
    // optimisation
    fn base_point(&self) -> Self {
        self.clone()
    }
}

impl<V: Euclidean, L: LieGroup<V>> TangentBundle<Self, V> for L {}

#[macro_export]
macro_rules! impl_lie_group_via_quotient {
    ($type:ty, $g:ty, $h:ty) => {
        impl<V: Euclidean> Group for $type {
            fn identity() -> Self {
                <Self as crate::traits::Quotient<$g, $h, V>>::quotient_identity()
            }
            fn compose(&self, other: &Self) -> Self {
                <Self as crate::traits::Quotient<$g, $h, V>>::quotient_compose(self, other)
            }
            fn inverse(&self) -> Self {
                <Self as crate::traits::Quotient<$g, $h, V>>::quotient_inverse(self)
            }
        }

        impl<V: Euclidean> crate::traits::LieGroup<V> for $type {
            fn identity_exp(v: V) -> Self {
                <Self as crate::traits::Quotient<$g, $h, V>>::quotient_identity_exp(v)
            }
            fn identity_log(p: &Self) -> Option<V> {
                <Self as crate::traits::Quotient<$g, $h, V>>::quotient_identity_log(p)
            }
        }
    };
}

#[cfg(feature = "testing")]
pub mod testing {
    // ---------------------------------------------------------------------------
    // Trait test macros
    // These generate the full suite of invariant tests for any implementation
    // of Chart, ExpMap, TangentBundle, LieGroup, and Metric. To test a new
    // manifold, just invoke the relevant macro with appropriate generators.
    // ---------------------------------------------------------------------------

    /// Tests that a space claiming to be a euclidean space is a euclidean space
    #[macro_export]
    macro_rules! test_euclidean {
        ($mod_name:ident, $space:ty, $arb_point:expr, $arb_vec:expr, $arb_scalar:expr) => {
            mod $mod_name {
                use super::*;

                // inherit all TangentFibre tests
                test_tangent_bundle!(tangent_bundle, $space, $space, $arb_point, $arb_vec);
                test_metric!(metric, $space, $arb_vec);
                test_inner_product!(inner_product, $space, $arb_point, $arb_scalar);

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
        ($mod_name:ident, $chart:ty, $arb_point:expr, $arb_vec:expr) => {
            mod $mod_name {
                use super::*;

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
                        prop_assert!(chart.check_geodesic_scaling(v, R64(t)));
                    }

                    #[test]
                    fn first_order_isometry(p in $arb_point, v in $arb_vec) {
                        let chart = <$chart>::chart_at(&p);
                        prop_assert!(chart.check_first_order_isometry(v, R64(1e-5)));
                    }
                }
            }
        };
    }

    /// Tests the TangentBundle invariant on top of all ExpMap invariants.
    #[macro_export]
    macro_rules! test_tangent_bundle {
        ($mod_name:ident, $chart:ty, $point:ty, $arb_point:expr, $arb_vec:expr) => {
            mod $mod_name {
                use super::*;

                // inherit all ExpMap tests
                test_exp_map!(exp_map, $chart, $arb_point, $arb_vec);

                proptest! {
                    // The TangentFibre invariant: chart_at(&p).to_global(zero) == p
                    #[test]
                    fn tangent_fibre_invariant(p in $arb_point) {
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
                            <$point as Group<_>>::check_left_identity(&p)
                        );
                    }

                    #[test]
                    fn right_identity(p in $arb_point) {
                        prop_assert!(<$point as Group<_>>::check_right_identity(&p));
                    }

                    #[test]
                    fn left_inverse(p in $arb_point) {
                        prop_assert!(<$point as Group<_>>::check_left_inverse(&p));
                    }

                    #[test]
                    fn right_inverse(p in $arb_point) {
                        prop_assert!(<$point as Group<_>>::check_right_inverse(&p));
                    }

                    #[test]
                    fn associativity(a in $arb_point, b in $arb_point, c in $arb_point) {
                        prop_assert!(<$point as Group<_>>::check_associativity(a, b, c));
                    }
                }
            }
        };
    }

    /// Tests the LieGroup axioms: identity, inverses, associativity.
    #[macro_export]
    macro_rules! test_lie_group {
        ($mod_name:ident, $point:ty, $arb_point:expr) => {
            mod $mod_name {
                use super::*;
                proptest! {
                    #[test]
                    fn left_identity(p in $arb_point) {
                        prop_assert!(
                            p.check_left_identity()
                        );
                    }

                    #[test]
                    fn right_identity(p in $arb_point) {
                        prop_assert!(p.check_right_identity());
                    }

                    #[test]
                    fn left_inverse(p in $arb_point) {
                        prop_assert!(p.check_left_inverse());
                    }

                    #[test]
                    fn right_inverse(p in $arb_point) {
                        prop_assert!(p.check_right_inverse());
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
                test_lie_group!(lie_group, $quotient, $arb_quotient);

                proptest! {
                    #[test]
                    fn canonical_respects_coset(g in $arb_g, h in $arb_h) {
                        prop_assert!(<$quotient>::check_canonical_respects_coset(g, h));
                    }
                }
            }
        };
    }

    #[macro_export]
    macro_rules! test_geodesic_cover {
        ($mod_name:ident, $cover:ty, $arb_cover:expr, $arb_point:expr) => {
            mod $mod_name {
                use super::*;
                proptest! {
                    #[test]
                    fn covering(cover in $arb_cover, p in $arb_point) {
                        prop_assert!(cover.check_covering(&p));
                    }
                }
            }
        };
    }
}
