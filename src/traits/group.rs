use std::ops::Neg;

use super::{Euclidean, Point, Smooth};
use num_traits::{Inv, One, Zero};

/// A commutative monoid, in additive notation.
///
/// The space of all values of a type `M: CMonoid` is interpreted as a
/// commutative monoid -- a set equipped with an associative, commutative
/// composition (`+`) and an identity element (`zero`). Unlike [`CGroup`],
/// a `CMonoid` need not have inverses: `N` (the naturals) under addition
/// is the paradigm example, and it is exactly the lack of inverses that
/// makes `N` worth distinguishing from `Z`.
///
/// - **Identity**: `0 + m = m + 0 = m`
/// - **Associativity**: `(a + b) + c = a + (b + c)`
/// - **Commutativity**: `a + b = b + a`
///
/// See [`Monoid`] for the multiplicative-notation counterpart used for
/// monoids that are not assumed to commute. The two are independent
/// traits, not one a supertrait of the other, precisely so that a type
/// needing both an (abelian) additive structure and an (unrelated,
/// possibly non-abelian) multiplicative structure -- a [`Rig`] or
/// [`Ring`] -- can implement both without its `Add` and `Mul` colliding
/// or entailing one another.
///
/// Certified by implementing this trait; verified by `test_cmonoid!`,
/// which includes a commutativity check absent from `Monoid`'s tests.
pub trait CMonoid: Point + Zero {
    #[cfg(feature = "testing")]
    fn check_left_identity(&self) -> bool
    where
        Self: PartialEq,
    {
        Self::zero() + self.clone() == *self
    }

    #[cfg(feature = "testing")]
    fn check_right_identity(&self) -> bool
    where
        Self: PartialEq,
    {
        self.clone() + Self::zero() == *self
    }

    #[cfg(feature = "testing")]
    fn check_associativity(a: Self, b: Self, c: Self) -> bool
    where
        Self: PartialEq,
    {
        (a.clone() + b.clone()) + c.clone() == a + (b + c)
    }

    #[cfg(feature = "testing")]
    fn check_commutativity(a: Self, b: Self) -> bool
    where
        Self: PartialEq,
    {
        a.clone() + b.clone() == b + a
    }
}

impl<M: Point + Zero> CMonoid for M {}

/// A monoid, in multiplicative notation, with no commutativity assumed.
///
/// The space of all values of a type `M: Monoid` is interpreted as a
/// monoid -- a set equipped with an associative composition (`*`) and an
/// identity element (`one`). Composition is *not* required to commute,
/// which is the entire reason this trait exists separately from
/// [`CMonoid`]: it is the multiplicative-notation home for structures that
/// may be non-abelian, most importantly the multiplicative half of a
/// [`Rig`]/[`Ring`] and the non-abelian [`MulGroup`]s (`SO(3)`, unit
/// quaternions) that this crate's Lie groups are built from.
///
/// - **Identity**: `1 * m = m * 1 = m`
/// - **Associativity**: `(a * b) * c = a * (b * c)`
///
/// Certified by implementing this trait; verified by `test_monoid!`.
pub trait Monoid: Point + One {
    #[cfg(feature = "testing")]
    fn check_left_identity(&self) -> bool
    where
        Self: PartialEq,
    {
        Self::one() * self.clone() == *self
    }

    #[cfg(feature = "testing")]
    fn check_right_identity(&self) -> bool
    where
        Self: PartialEq,
    {
        self.clone() * Self::one() == *self
    }

    #[cfg(feature = "testing")]
    fn check_associativity(a: Self, b: Self, c: Self) -> bool
    where
        Self: PartialEq,
    {
        (a.clone() * b.clone()) * c.clone() == a * (b * c)
    }
}

impl<M: Point + One> Monoid for M {}

/// An abelian group, in additive notation.
///
/// The space of all values of a type `G: CGroup` is interpreted as a
/// commutative group: a [`CMonoid`] in which every element additionally
/// has an additive inverse. This is the additive-notation counterpart to
/// [`MulGroup`]; both are operator-flavoured presentations that a concrete
/// type can bridge to the spelling-agnostic [`Group`] in one line via
/// [`impl_group_via_add`]/[`impl_group_via_mul`].
///
/// - **Inverses**: `(-g) + g = g + (-g) = 0`
///
/// Certified by implementing this trait; verified by `test_cgroup!`.
///
/// [`impl_group_via_add`]: crate::impl_group_via_add
/// [`impl_group_via_mul`]: crate::impl_group_via_mul
pub trait CGroup: CMonoid + Neg<Output = Self> {
    #[cfg(feature = "testing")]
    fn check_left_inverse(&self) -> bool
    where
        Self: PartialEq,
    {
        -self.clone() + self.clone() == Self::zero()
    }

    #[cfg(feature = "testing")]
    fn check_right_inverse(&self) -> bool
    where
        Self: PartialEq,
    {
        self.clone() + -self.clone() == Self::zero()
    }
}
impl<G: CMonoid + Neg<Output = Self>> CGroup for G {}

/// Bridges a `+`/`-`-flavoured type into the spelling-agnostic [`Group`]
/// by delegating `identity`/`compose`/`inverse` to its `Zero`/`Add`/`Neg`.
///
/// This exists because `Group` cannot be reached by a single blanket impl
/// from both [`CMonoid`]`+Neg` and [`Monoid`]`+Inv` types at once (the two
/// blanket impls would overlap in the eyes of Rust's coherence checker,
/// which cannot see that no type implements both). Instead, every
/// additively-flavoured `Group` implementor invokes this macro once; see
/// [`impl_group_via_mul`] for the multiplicative counterpart.
///
/// [`impl_group_via_mul`]: crate::impl_group_via_mul
#[macro_export]
macro_rules! impl_group_via_add {
    ($target:ty, $($generics:tt)*) => {
        impl<$($generics)*> $crate::traits::Group for $target {
            fn identity() -> Self {
                <Self as num_traits::Zero>::zero()
            }
            fn compose(&self, other: &Self) -> Self {
                self.clone() + other.clone()
            }
            fn inverse(&self) -> Self {
                -*self
            }
        }
    };
}

/// Implements [`Zero`], [`Add`], and [`Neg`] for `$target` via Grothendieck
/// group completion of the commutative monoid `$monoid`.
///
/// Group completion is the universal way to manufacture an abelian group
/// from a commutative monoid that may lack inverses: represent an element
/// as a formal difference `(a, b)` meaning "a - b", with `(a,b) ~ (c,d)`
/// iff `a+d = b+c` (an honest equivalence relation only because `$monoid`
/// is commutative -- see [`CMonoid`]). Addition is componentwise, the
/// identity is `(0,0)`, and negation swaps the pair: `-(a,b) = (b,a)`,
/// since `-(a-b) = b-a`.
///
/// Unlike quotienting a group by a subgroup ([`Quotient`]), this
/// construction is parameter-free: given `$monoid`, the congruence, the
/// group operations, and the resulting group are all forced -- there is no
/// choice of subgroup to make. It is entirely determined by the input
/// type, which is why it is expressed as a macro deriving trait impls
/// rather than a trait with a method to implement.
///
/// `$target` must be losslessly convertible `Into`/`From` `($monoid,
/// $monoid)`; this macro does not require that representation to be the
/// literal storage of `$target` -- a packed, reduced representation (as
/// [`Z`](crate::discrete::Z) uses, storing a signed integer rather than a
/// pair of naturals) is fine, so long as the conversions round-trip
/// through the formal-difference meaning.
///
/// Completing an already-complete group returns something isomorphic to
/// the original: this construction is idempotent (up to isomorphism) on
/// its own output, since a group has nothing left to complete.
///
/// This produces a [`CGroup`], not a [`Group`]; pair it with
/// [`impl_group_via_add`] to also obtain `Group`.
///
/// [`impl_group_via_add`]: crate::impl_group_via_add
/// [`Zero`]: num_traits::Zero
/// [`Add`]: std::ops::Add
/// [`Neg`]: std::ops::Neg
#[macro_export]
macro_rules! impl_abelian_group_via_grothendieck {
    ($target:ty, $monoid:ty, $($generics:tt)*) => {
        impl<$($generics)*> num_traits::Zero for $target {
            fn zero() -> Self {
                (<$monoid as num_traits::Zero>::zero(), <$monoid as num_traits::Zero>::zero()).into()
            }
            fn is_zero(&self) -> bool {
                let (a, b) = self.clone().into();
                a == b
            }
        }

        impl<$($generics)*> std::ops::Add for $target {
            type Output = Self;
            fn add(self, other: Self) -> Self {
                let (a, b) = self.into();
                let (c, d) = other.into();
                (a + c, b + d).into()
            }
        }

        impl<$($generics)*> std::ops::Neg for $target {
            type Output = Self;
            fn neg(self) -> Self {
                let (a, b) = self.into();
                (b, a).into()
            }
        }
    };
}

/// Implements [`Zero`], [`Add`], [`Neg`], [`One`], and [`Mul`] for
/// `$target` via Grothendieck completion of the commutative semiring
/// (["rig"](Rig)) `$rig`.
///
/// Extends [`impl_abelian_group_via_grothendieck`] with a multiplication
/// compatible with the formal-difference representation, via the usual
/// expansion of a product of differences: `(a-b)(c-d) = (ac+bd) - (ad+bc)`.
/// The additive structure is delegated verbatim; this macro adds only the
/// multiplicative half needed to reach a full [`Ring`].
///
/// As with the additive completion, this is parameter-free: `$rig` alone
/// determines the resulting ring, with no independent choice involved.
///
/// [`Zero`]: num_traits::Zero
/// [`One`]: num_traits::One
/// [`Add`]: std::ops::Add
/// [`Neg`]: std::ops::Neg
/// [`Mul`]: std::ops::Mul
#[macro_export]
macro_rules! impl_ring_via_grothendieck {
    ($target:ty, $rig:ty, $($generics:tt)*) => {
        $crate::impl_abelian_group_via_grothendieck!($target, $rig, $($generics)*);

        impl<$($generics)*> num_traits::One for $target {
            fn one() -> Self {
                (<$rig as num_traits::One>::one(), <$rig as num_traits::Zero>::zero()).into()
            }
        }

        impl<$($generics)*> std::ops::Mul for $target {
            type Output = Self;
            fn mul(self, other: Self) -> Self {
                let (a, b) = self.into();
                let (c, d) = other.into();
                let pos = (a.clone() * c.clone()) + (b.clone() * d.clone());
                let neg = (a * d) + (b * c);
                (pos, neg).into()
            }
        }
    }
}

/// A commutative semiring ("rig" -- a **r**ing without negat**i**on).
///
/// The space of all values of a type `R: Rig` is interpreted as a
/// commutative semiring: a [`CMonoid`] under addition together with a
/// [`Monoid`] under multiplication, connected by distributivity, with
/// `zero` absorbing under multiplication. `N` (the naturals) under `+`/`*`
/// is the paradigm example: it is exactly the missing additive inverses
/// that make it a rig rather than a [`Ring`].
///
/// - **Distributivity**: `a * (b + c) = (a*b) + (a*c)`, and symmetrically
/// - **Annihilation**: `0 * r = r * 0 = 0`
///
/// (The multiplicative axioms -- identity, associativity -- are already
/// certified by [`Monoid`]; `Rig` adds only what connects `+` and `*`.)
///
/// Certified by implementing this trait; verified by `test_rig!`.
pub trait Rig: CMonoid + Monoid {
    #[cfg(feature = "testing")]
    fn check_left_distributivity(a: Self, b: Self, c: Self) -> bool
    where
        Self: PartialEq,
    {
        a.clone() * (b.clone() + c.clone()) == (a.clone() * b) + (a * c)
    }

    #[cfg(feature = "testing")]
    fn check_right_distributivity(a: Self, b: Self, c: Self) -> bool
    where
        Self: PartialEq,
    {
        (a.clone() + b.clone()) * c.clone() == (a * c.clone()) + (b * c)
    }

    #[cfg(feature = "testing")]
    fn check_left_annihilation(&self) -> bool
    where
        Self: PartialEq,
    {
        Self::zero() * self.clone() == Self::zero()
    }

    #[cfg(feature = "testing")]
    fn check_right_annihilation(&self) -> bool
    where
        Self: PartialEq,
    {
        self.clone() * Self::zero() == Self::zero()
    }
}

impl<R: CMonoid + One> Rig for R {}

/// A ring.
///
/// The space of all values of a type `R: Ring` is interpreted as a ring: a
/// [`Rig`] (`+` and `*`, connected by distributivity) whose addition
/// additionally has inverses, making it a [`CGroup`]. This trait adds no
/// methods of its own; it names the join of `CGroup` and `Rig` because the
/// two together are what "ring" means, and having the name available is
/// more useful than always spelling out both bounds. `Z`, built by
/// [`impl_ring_via_grothendieck`] from the rig `N`, is the paradigm
/// instance -- and, like most rings, it has no multiplicative inverses
/// (`2` is not invertible in `Z`), which is exactly why `Ring` is bounded
/// on `CGroup`, not [`MulGroup`]: requiring multiplicative inverses would
/// make ordinary rings like `Z` unable to implement it.
pub trait Ring: CGroup + Rig {}

impl<R: CGroup + Rig> Ring for R {}

/// A (possibly non-abelian) group, in multiplicative notation.
///
/// The space of all values of a type `G: MulGroup` is interpreted as a
/// group in the usual, possibly non-commutative, sense: a [`Monoid`] in
/// which every element additionally has a multiplicative inverse. This is
/// the multiplicative-notation counterpart to [`CGroup`], used for groups
/// such as `SO(3)` and the unit quaternions that are not, and should not
/// be forced to pretend to be, abelian.
///
/// - **Inverses**: `g.inv() * g == g * g.inv() == 1`
///
/// Certified by implementing this trait; verified by `test_mul_group!`.
pub trait MulGroup: Monoid + Inv<Output = Self> {
    #[cfg(feature = "testing")]
    fn check_left_inverse(&self) -> bool
    where
        Self: PartialEq,
    {
        self.clone().inv() * self.clone() == Self::one()
    }

    #[cfg(feature = "testing")]
    fn check_right_inverse(&self) -> bool
    where
        Self: PartialEq,
    {
        self.clone() * self.clone().inv() == Self::one()
    }
}

impl<G: Monoid + Inv<Output = Self>> MulGroup for G {}

/// Bridges a `*`/`Inv`-flavoured type into the spelling-agnostic [`Group`]
/// by delegating `identity`/`compose`/`inverse` to its `One`/`Mul`/`Inv`.
///
/// The multiplicative counterpart to [`impl_group_via_add`]; see its docs
/// for why this exists as a macro rather than a blanket impl. Used for
/// this crate's non-abelian Lie groups (`SO(3)`, unit quaternions), so
/// that they never need to expose an `Add` that wouldn't mean anything.
///
/// [`impl_group_via_add`]: crate::impl_group_via_add
#[macro_export]
macro_rules! impl_group_via_mul {
    ($target:ty, $($generics:tt)*) => {
        impl<$($generics)*> $crate::traits::Group for $target {
            fn identity() -> Self {
                <Self as num_traits::One>::one()
            }
            fn compose(&self, other: &Self) -> Self {
                self.clone() * other.clone()
            }
            fn inverse(&self) -> Self {
                <Self as num_traits::Inv>::inv(self.clone())
            }
        }
    };
}

/// A group, spelled with operator-agnostic named methods.
///
/// The space of all values of a type `G: Group` is interpreted as a group —
/// a set equipped with an associative composition, an identity element, and
/// inverses. This is the purely algebraic layer: `Group` carries no topology,
/// no smoothness, and no coordinate structure. A `Group` need not be a
/// manifold at all — that structure appears only at [`LieGroup`], which
/// refines `Group` with an exponential map and the differential structure of
/// a smooth manifold.
///
/// - **Identity**: `identity().compose(&g) == g.compose(&identity()) == g`
/// - **Inverses**: `g.inverse().compose(&g) == g.compose(&g.inverse()) == identity()`
/// - **Associativity**: `a.compose(&b).compose(&c) == a.compose(&b.compose(&c))`
///
/// Certified by implementing this trait; verified by `test_group!`.
///
/// # Why `compose`/`inverse`/`identity`, not `Mul`/`Neg`/`Add`
/// `Group` deliberately has no operator-trait bound and no commutativity
/// requirement, so that it can describe both abelian groups (this crate's
/// [`Euclidean`] spaces, [`Z`](crate::discrete::Z), [`S1`](crate::flat::S1))
/// and non-abelian ones (`SO(3)`, unit quaternions) uniformly. Real groups
/// split into two genuinely different notations depending on whether they
/// commute — `+` for abelian, `*` otherwise — and a single trait cannot
/// require both `Add` and `Mul` on `Self` without every non-commutative
/// group also being forced to expose a nonsensical, unused `+`. `Group`
/// sidesteps the choice entirely with method names that carry no notational
/// assumption; [`CMonoid`]/[`CGroup`] (additive) and [`Monoid`]/[`MulGroup`]
/// (multiplicative) are the two operator-flavoured presentations, and a
/// concrete type built on either can obtain `Group` in one line via
/// [`impl_group_via_add`] or [`impl_group_via_mul`], which simply delegate
/// `identity`/`compose`/`inverse` to whichever operators the type already
/// has. This is also why `Group` cannot be reached by a single blanket
/// impl over `CMonoid`/`Monoid`: Rust's coherence checker cannot see that
/// no type implements both flavours at once, so the two bridges are
/// supplied as macros invoked per concrete type instead.
///
/// [`Euclidean`]: crate::traits::Euclidean
/// [`impl_group_via_add`]: crate::impl_group_via_add
/// [`impl_group_via_mul`]: crate::impl_group_via_mul
pub trait Group: Point {
    fn identity() -> Self;
    fn compose(&self, other: &Self) -> Self;
    fn inverse(&self) -> Self;

    #[cfg(feature = "testing")]
    fn check_left_identity(&self) -> bool
    where
        Self: PartialEq,
    {
        Self::identity().compose(&self) == *self
    }

    #[cfg(feature = "testing")]
    fn check_right_identity(&self) -> bool
    where
        Self: PartialEq,
    {
        self.clone().compose(&Self::identity()) == *self
    }

    #[cfg(feature = "testing")]
    fn check_associativity(a: Self, b: Self, c: Self) -> bool
    where
        Self: PartialEq,
    {
        a.compose(&b).compose(&c) == a.compose(&b.compose(&c))
    }

    #[cfg(feature = "testing")]
    fn check_left_inverse(&self) -> bool
    where
        Self: PartialEq,
    {
        (self.inverse()).compose(self) == Self::identity()
    }

    #[cfg(feature = "testing")]
    fn check_right_inverse(&self) -> bool
    where
        Self: PartialEq,
    {
        self.compose(&self.inverse()) == Self::identity()
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
///
/// [`Chart`]: crate::traits::Chart
/// [`ExpMap`]: crate::traits::ExpMap
/// [`TangentBundle`]: crate::traits::TangentBundle
pub trait LieGroup<V: Euclidean>: Group {
    fn identity_exp(v: V) -> Self;
    fn identity_log(p: &Self) -> Option<V>;
}

// left translation
impl<V: Euclidean, L: LieGroup<V>> Smooth<V> for L {
    fn exp(&self, coord: V) -> Self {
        let translated = Self::identity_exp(coord);
        self.compose(&translated)
    }

    fn log(&self, point: &Self) -> Option<V> {
        let translated = self.clone().inverse().compose(point);
        Self::identity_log(&translated)
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
        Self: PartialEq,
    {
        Self::new(Self::embed(h).compose(&g)) == Self::new(g)
    }
}

/// Implements [`Group`] and [`LieGroup`] for `$type` by routing every
/// operation through its [`Quotient`]`<$g, $h, V>` implementation.
///
/// `Quotient` supplies default bodies for all of these
/// (`quotient_identity`, `quotient_compose`, `quotient_inverse`,
/// `quotient_identity_exp`, `quotient_identity_log`) in terms of `new` and
/// `lift` alone; this macro is purely the mechanical step of wiring those
/// defaults up to `Group`/`LieGroup`, so that a `Quotient` implementor gets
/// a genuine [`LieGroup`] -- and, through it, [`Chart`], [`ExpMap`], and
/// [`TangentBundle`] via `LieGroup`'s own blanket impl -- without restating
/// any of `Quotient`'s logic.
///
/// [`Chart`]: crate::traits::Chart
/// [`ExpMap`]: crate::traits::ExpMap
/// [`TangentBundle`]: crate::traits::TangentBundle
#[macro_export]
macro_rules! impl_lie_group_via_quotient {
    ($type:ty, $g:ty, $h:ty $(, $bound:path)*) => {
        impl<V: $crate::traits::Euclidean + $($bound +)*> $crate::traits::Group for $type {
            fn identity() -> Self {
                <Self as $crate::traits::Quotient<$g, $h, V>>::quotient_identity()
            }
            fn compose(&self, rhs: &Self) -> Self {
                <Self as $crate::traits::Quotient<$g, $h, V>>::quotient_compose(&self, &rhs)
            }
            fn inverse(&self) -> Self {
                <Self as $crate::traits::Quotient<$g, $h, V>>::quotient_inverse(&self)
            }
        }

        impl<V: $crate::traits::Euclidean + $($bound +)*> $crate::traits::LieGroup<V> for $type {
            fn identity_exp(v: V) -> Self {
                <Self as $crate::traits::Quotient<$g, $h, V>>::quotient_identity_exp(v)
            }
            fn identity_log(p: &Self) -> Option<V> {
                <Self as $crate::traits::Quotient<$g, $h, V>>::quotient_identity_log(p)
            }
        }
    };
}
