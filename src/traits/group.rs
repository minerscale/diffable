use std::ops::{Add, Mul, Neg};

use super::{Euclidean, Point, Smooth};
use num_traits::{One, Zero};

pub trait CMonoid: Point + Zero + Add<Output = Self> {
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
}

impl<M: Point + Zero + Add<Output = Self>> CMonoid for M {}

#[macro_export]
macro_rules! impl_group_via_grothendieck {
    ($target:ty, $monoid:ty, <$param:ident $(: $bounds:path)?>) => {
        impl<$param $(: $bounds)?> Zero for $target {
            fn zero() -> Self {
                (<$monoid>::zero(), <$monoid>::zero()).into()
            }

            fn is_zero(&self) -> bool {
                let (a, b) = self.clone().into();
                a == b
            }
        }

        impl<$param $(: $bounds)?> Add for $target {
            type Output = Self;

            fn add(self, other: Self) -> Self {
                let (a, b) = self.into();
                let (c, d) = other.into();
                (a + c, b + d).into()
            }
        }

        impl<$param $(: $bounds)?> Neg for $target {
            type Output = Self;

            fn neg(self) -> Self {
                let (a, b) = self.into();
                (b, a).into()
            }
        }
    };
}

#[macro_export]
macro_rules! impl_ring_via_grothendieck {
    ($target:ty, $rig:ty, <$param:ident $(: $bounds:path)?>) => {

        crate::impl_group_via_grothendieck!($target, $rig, <$param $(: $bounds)?>);

        impl<$param $(: $bounds)?> One for $target {
            fn one() -> Self {
                // 1 represents (1 - 0)
                (<$rig>::one(), <$rig>::zero()).into()
            }
        }

        impl<$param $(: $bounds)?> Mul for $target {
            type Output = Self;

            fn mul(self, other: Self) -> Self {
                let (a, b) = self.into();
                let (c, d) = other.into();
                
                // Using the expansion: (a - b)(c - d) = (ac + bd) - (ad + bc)
                let pos = (a.clone() * c.clone()) + (b.clone() * d.clone());
                let neg = (a * d) + (b * c);
                
                (pos, neg).into()
            }
        }
    }
}

pub trait Rig: CMonoid + One + Mul<Output = Self> {
    #[cfg(feature = "testing")]
    fn check_mul_left_identity(&self) -> bool
    where
        Self: PartialEq,
    {
        Self::one() * self.clone() == *self
    }

    #[cfg(feature = "testing")]
    fn check_mul_right_identity(&self) -> bool
    where
        Self: PartialEq,
    {
        self.clone() * Self::one() == *self
    }

    #[cfg(feature = "testing")]
    fn check_mul_associativity(a: Self, b: Self, c: Self) -> bool
    where
        Self: PartialEq,
    {
        (a.clone() * b.clone()) * c.clone() == a * (b * c)
    }

    #[cfg(feature = "testing")]
    fn check_mul_commutativity(a: Self, b: Self) -> bool
    where
        Self: PartialEq,
    {
        a.clone() * b.clone() == b * a
    }

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

impl<R: CMonoid + Mul<Output = Self> + One> Rig for R {}

pub trait Ring: Group + Rig {}

impl<R: Group + Rig> Ring for R {}

/// A group.
///
/// The space of all values of a type `G: Group` is interpreted as a group —
/// a set equipped with an associative composition, an identity element, and
/// inverses. This is the purely algebraic layer: `Group` carries no topology,
/// no smoothness, and no coordinate structure. A `Group` need not be a
/// manifold at all — that structure appears only at [`LieGroup`], which
/// refines `Group` with an exponential map and the differential structure of
/// a smooth manifold.
///
/// - **Identity**: there exists `e` with `e * g = g * e = g`
/// - **Inverses**: every `g` has `g⁻¹` with `g * g⁻¹ = g⁻¹ * g = e`
/// - **Associativity**: `(a * b) * c = a * (b * c)`
///
/// Certified by implementing this trait; verified by `test_group!`.
pub trait Group: CMonoid + Neg<Output = Self> {
    #[cfg(feature = "testing")]
    fn check_left_inverse(&self) -> bool
    where
        Self: PartialEq,
    {
        (-self.clone()) + self.clone() == Self::zero()
    }

    #[cfg(feature = "testing")]
    fn check_right_inverse(&self) -> bool
    where
        Self: PartialEq,
    {
        self.clone() + -self.clone() == Self::zero()
    }
}

impl<G: CMonoid + Neg<Output = Self>> Group for G {}

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
        Self::new(G::zero())
    }

    fn quotient_compose(&self, other: &Self) -> Self {
        Self::new(self.lift() + other.lift())
    }

    fn quotient_inverse(&self) -> Self {
        Self::new(-self.lift())
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
        Self::new(Self::embed(h) + g.clone()) == Self::new(g)
    }
}

// left translation
impl<V: Euclidean, L: LieGroup<V>> Smooth<V> for L {
    fn exp(&self, coord: V) -> Self {
        let translated = Self::identity_exp(coord);
        self.clone() + translated
    }

    fn log(&self, point: &Self) -> Option<V> {
        let translated = -self.clone() + point.clone();
        Self::identity_log(&translated)
    }
}

#[macro_export]
macro_rules! impl_lie_group_via_quotient {
    ($type:ty, $g:ty, $h:ty $(, $bound:path)*) => {
        impl<V: Euclidean + $($bound +)*> num_traits::Zero for $type {
            fn zero() -> Self {
                <Self as crate::traits::Quotient<$g, $h, V>>::quotient_identity()
            }

            fn is_zero(&self) -> bool {
                self.lift().is_zero()
            }
        }

        impl<V: Euclidean + $($bound +)*> std::ops::Add for $type {
            type Output = Self;

            fn add(self, rhs: Self) -> Self {
                <Self as crate::traits::Quotient<$g, $h, V>>::quotient_compose(&self, &rhs)
            }
        }

        impl<V: Euclidean + $($bound +)*> std::ops::Neg for $type {
            type Output = Self;

            fn neg(self) -> Self {
                <Self as crate::traits::Quotient<$g, $h, V>>::quotient_inverse(&self)
            }
        }

        impl<V: Euclidean + $($bound +)*> crate::traits::LieGroup<V> for $type {
            fn identity_exp(v: V) -> Self {
                <Self as crate::traits::Quotient<$g, $h, V>>::quotient_identity_exp(v)
            }
            fn identity_log(p: &Self) -> Option<V> {
                <Self as crate::traits::Quotient<$g, $h, V>>::quotient_identity_log(p)
            }
        }
    };
}
