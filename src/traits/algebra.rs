//! Algebraic structures and the blanket theorems connecting them.
//!
//! Additive structure progresses through [`CMonoid`] and [`CGroup`], while
//! multiplicative structure progresses through [`Monoid`] and [`MulGroup`].
//! Their joins produce [`Ring`], [`DivRing`], and [`Field`]; [`CField`] adds
//! commutative multiplication. [`Group`] and [`LieGroup`] carry the same
//! structural approach into geometry.

use crate::{
    impl_group_via_mul,
    traits::{ExactCmp, FromReal, Interval, Metric, Real, Tensor},
};
use core::ops::{Add, Mul, Neg, Sub};
use num_traits::{Inv, NumCast, One, Zero, real::Real as _};

use super::{Point, Smooth};

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
pub trait CGroup: CMonoid + Sub<Output = Self> + Neg<Output = Self> {
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

    #[cfg(feature = "testing")]
    fn check_sub_agrees_with_neg(a: &Self, b: &Self) -> bool
    where
        Self: PartialEq,
    {
        a.clone() - b.clone() == a.clone() + -(b.clone())
    }
}
impl<G: CMonoid + Sub<Output = Self> + Neg<Output = Self>> CGroup for G {}

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
                -self.clone()
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
/// [`Add`]: core::ops::Add
/// [`Neg`]: core::ops::Neg
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

        impl<$($generics)*> core::ops::Add for $target {
            type Output = Self;
            fn add(self, other: Self) -> Self {
                let (a, b) = self.into();
                let (c, d) = other.into();
                (a + c, b + d).into()
            }
        }

        impl<$($generics)*> core::ops::Sub for $target {
            type Output = Self;
            fn sub(self, other: Self) -> Self {
                self + -other
            }
        }

        impl<$($generics)*> core::ops::Neg for $target {
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
/// [`Add`]: core::ops::Add
/// [`Neg`]: core::ops::Neg
/// [`Mul`]: core::ops::Mul
#[macro_export]
macro_rules! impl_ring_via_grothendieck {
    ($target:ty, $rig:ty, $($generics:tt)*) => {
        $crate::impl_abelian_group_via_grothendieck!($target, $rig, $($generics)*);

        impl<$($generics)*> num_traits::One for $target {
            fn one() -> Self {
                (<$rig as num_traits::One>::one(), <$rig as num_traits::Zero>::zero()).into()
            }
        }

        impl<$($generics)*> core::ops::Mul for $target {
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

/// A newtype which certifies the value is non-zero
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct NonZero<T: Zero>(pub T);

impl<T: Zero> NonZero<T> {
    /// Certifies that `value` is nonzero.
    ///
    /// Returns `None` when `value` is zero.
    pub fn new(value: T) -> Option<Self> {
        if !value.is_zero() {
            Some(Self(value))
        } else {
            None
        }
    }

    /// Constructs a certified nonzero value without checking the invariant.
    ///
    /// Callers must establish by construction that `value` is not zero. This
    /// is a logical unchecked constructor rather than an unsafe memory
    /// operation; violating the invariant can nevertheless invalidate the
    /// algebraic assumptions of [`DivRing`] and [`Field`].
    pub fn new_unchecked(value: T) -> Self {
        Self(value)
    }
}

impl<T: Zero + One> Mul<Self> for NonZero<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl<T: Zero + One> One for NonZero<T> {
    fn one() -> Self {
        Self(T::one())
    }
}

impl<T: Zero + One + Inv<Output = T>> Inv for NonZero<T> {
    type Output = Self;

    fn inv(self) -> Self::Output {
        Self(self.0.inv())
    }
}

impl<T: Zero + One + Point> Group for NonZero<T>
where
    NonZero<T>: Inv<Output = Self>,
{
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

/// A division ring.
///
/// A [`Ring`] whose nonzero elements form a [`MulGroup`], so every nonzero
/// value is multiplicatively invertible. [`Field`] adds the involution, fixed
/// field, and characteristic used by the tensor hierarchy.
pub trait DivRing: Ring {
    /// Divides by a nonzero `rhs` using its multiplicative inverse.
    ///
    /// # Panics
    ///
    /// Panics when `rhs` is zero.
    fn div(self, rhs: Self) -> Self {
        self * Self::Mul::from(NonZero::new(rhs).expect("division by zero"))
            .inv()
            .into()
            .0
    }

    /// The multiplicative group corresponding to [`NonZero`].
    type Mul: MulGroup + From<NonZero<Self>> + Into<NonZero<Self>>;
}

impl<R: Ring> DivRing for R
where
    NonZero<Self>: MulGroup,
{
    type Mul = NonZero<Self>;
}

/// A field.
///
/// A division ring equipped with the scalar structure used by the library:
/// an elected central involution, fixed field, characteristic, etc.
///
/// “Field” here includes skew/noncommutative fields.
pub trait Field: DivRing + Copy + PartialEq + core::fmt::Debug {
    /// A distinguished central subfield fixed pointwise by [`Self::conj`].
    ///
    /// Its embedding through [`Self::from_fixed`] must preserve the field
    /// operations, commute with every element of `Self`, and satisfy
    /// `Self::from_fixed(r).conj() == Self::from_fixed(r)`.
    type Fixed: CField<Fixed = Self::Fixed>;

    /// The conjugation operation.
    fn conj(&self) -> Self;

    /// The field's characteristic, as a type-level [`Nat`]. `NatZero` means
    /// characteristic zero (ℚ embeds). Callers that need `1/k` — the matrix
    /// exponential, [`from_nat`](Field::from_nat) — bound on `Characteristic =
    /// NatZero` so that finite-characteristic fields are rejected at compile
    /// time rather than dividing by a zero that only appears at runtime.
    type Characteristic: Nat;

    fn powi(&self, mut s: usize) -> Self {
        let mut base = *self;
        let mut result = Self::one();

        while s != 0 {
            if s & 1 != 0 {
                result = result * base;
            }

            base = base * base;
            s >>= 1;
        }

        result
    }

    fn from_nat(mut n: usize) -> Self {
        if Self::Characteristic::N != 0 {
            debug_assert!(n < Self::Characteristic::N);
        }

        let mut result = Self::zero();
        let mut current = Self::one();

        while n != 0 {
            if n & 1 == 1 {
                result = result + current;
            }

            current = current + current;
            n >>= 1;
        }

        result
    }

    fn norm_squared(self) -> Self::Fixed {
        Self::to_fixed(self * self.conj())
    }

    // Forces a proof that a self-adjoint element can safely drop down
    // into the invariant sub-field.
    fn to_fixed(self) -> Self::Fixed;
    fn from_fixed(x: Self::Fixed) -> Self;

    // conj respects addition.
    #[cfg(feature = "testing")]
    fn check_conj_additive(a: Self, b: Self) -> bool {
        (a + b).conj() == a.conj() + b.conj()
    }

    // conj respects multiplication.
    #[cfg(feature = "testing")]
    fn check_conj_multiplicative(a: Self, b: Self) -> bool {
        (a * b).conj() == b.conj() * a.conj()
    }

    #[cfg(feature = "testing")]
    fn check_conj_unit() -> bool {
        Self::one().conj() == Self::one()
    }

    // conj∘conj = id. Not derivable from the automorphism properties plus
    // descent — see the earlier discussion attempting and failing to build
    // a counterexample.
    #[cfg(feature = "testing")]
    fn check_conj_involution(a: Self) -> bool {
        a.conj().conj() == a
    }

    // from_fixed respects addition.
    #[cfg(feature = "testing")]
    fn check_from_fixed_additive(x: Self::Fixed, y: Self::Fixed) -> bool {
        Self::from_fixed(x + y) == Self::from_fixed(x) + Self::from_fixed(y)
    }

    // from_fixed respects multiplication.
    #[cfg(feature = "testing")]
    fn check_from_fixed_multiplicative(x: Self::Fixed, y: Self::Fixed) -> bool {
        Self::from_fixed(x * y) == Self::from_fixed(x) * Self::from_fixed(y)
    }

    // x + conj(x) is self-adjoint by conj's additivity and involution alone,
    // with no reference to Fixed — this is what actually cashes out the
    // promise that a self-adjoint element "safely drops" into Fixed.
    #[cfg(feature = "testing")]
    fn check_descent(x: Self) -> bool {
        let s = x + x.conj();
        Self::from_fixed(s.to_fixed()) == s
    }

    // x * conj(x) is fixed for any x, not just self-adjoint x — the fact
    // norm_squared relies on to call to_fixed at all.
    #[cfg(feature = "testing")]
    fn check_norm_squared_self_adjoint(x: Self) -> bool {
        let n = x * x.conj();
        n.conj() == n
    }

    // from_fixed's image lands inside conj's fixed points.
    #[cfg(feature = "testing")]
    fn check_from_fixed_is_fixed(x: Self::Fixed) -> bool {
        let y = Self::from_fixed(x);
        y.conj() == y
    }

    #[cfg(feature = "testing")]
    fn check_fixed_field_is_central(x: Self::Fixed, y: Self) -> bool {
        let x = Self::from_fixed(x);
        x * y == y * x
    }

    /// Checks that [`Self::from_fixed`] preserves the multiplicative identity
    /// whenever `Self::Fixed` is nondegenerate.
    ///
    /// A field of characteristic one has `zero() == one()`, so its unique map into
    /// a nondegenerate field cannot preserve both identities: preservation of zero
    /// forces its sole element to map to `Self::zero()`. Such a degenerate fixed
    /// field is intentionally permitted, so the unit law is waived in that case.
    ///
    /// For every nondegenerate fixed field, preserving the unit ensures that
    /// `from_fixed` is nonzero and hence, together with its homomorphism laws,
    /// injective.
    #[cfg(feature = "testing")]
    fn check_from_fixed_unit() -> bool {
        <Self::Fixed as Field>::Characteristic::N == 1
            || Self::from_fixed(Self::Fixed::one()) == Self::one()
    }

    // Audits the declared characteristic against the
    // field's actual arithmetic, as far as `bound` allows.
    #[cfg(feature = "testing")]
    fn check_characteristic_up_to(bound: usize) -> bool {
        let mut acc = Self::zero();

        let bound = match Self::Characteristic::N {
            0 => bound,
            n => bound.min(n),
        };
        for _ in 1..bound {
            acc = acc + Self::one();
            if acc == Self::zero() {
                return false;
            }
        }

        if bound != 0 && bound == Self::Characteristic::N {
            // one more add should send it to zero
            acc + Self::one() == Self::zero()
        } else {
            // didn't probe far enough / characteristic is 0
            acc + Self::one() != Self::zero()
        }
    }
}

/// A marker that asserts that a [`Field`](crate::traits::Field)
/// is multiplicatively commutative.
pub trait CField: Field {
    #[cfg(feature = "testing")]
    fn check_commutativity(a: Self, b: Self) -> bool {
        a * b == b * a
    }
}

/// A characteristic-zero field equipped with an exponential map.
///
/// `FieldExp` certifies that the field provides an elected implementation of
///
/// ```text
/// exp(x) = Σₙ₌₀∞ xⁿ / n!.
/// ```
///
/// Characteristic zero is required because the series contains every natural
/// number as a denominator. In positive characteristic, some `n!` vanishes and
/// the ordinary total field exponential cannot be defined this way.
///
/// The field need not be commutative. For commuting `x` and `y`, the
/// exponential is expected to satisfy
///
/// ```text
/// exp(0)     = 1
/// exp(x + y) = exp(x) exp(y)
/// exp(-x)    = exp(x)⁻¹.
/// ```
///
/// The second identity is asserted only when `x * y == y * x`; it does not hold
/// for arbitrary elements of a noncommutative field.
///
/// # Implementing
///
/// Implementors must provide [`exp`](FieldExp::exp). Types with an appropriate
/// [`Metric`] may delegate to [`exp_by_series`](FieldExp::exp_by_series), while
/// types with a more accurate or efficient implementation—such as a platform
/// real exponential or a closed form—should use that instead.
///
/// The series helper is an implementation convenience, not an additional
/// requirement of `FieldExp`: an exponential may exist without electing the
/// particular metric needed by its scaling-and-squaring algorithm.
pub trait FieldExp: Field<Characteristic = NatZero> {
    /// Computes the elected exponential of this field element.
    fn exp(&self) -> Self;

    /// Approximates the exponential using a scaled Taylor series.
    ///
    /// The input is repeatedly divided by two until its distance from zero is
    /// at most one. The method then evaluates the Taylor polynomial through
    /// degree 20 and reverses the scaling by repeated squaring:
    ///
    /// ```text
    /// exp(x) = exp(x / 2ˢ)^(2ˢ).
    /// ```
    ///
    /// [`Metric`] supplies the compatible positive magnitude used to choose
    /// `s`; it is required by this algorithm rather than by the
    /// [`FieldExp`] abstraction itself.
    ///
    /// This is a general fallback. Implementors which can delegate to a native
    /// exponential or use a suitable closed form will generally obtain better
    /// accuracy and performance by overriding [`exp`](FieldExp::exp).
    fn exp_by_series(&self) -> Self
    where
        Self: Metric,
    {
        let theta = Self::R::one();
        let n = 20;
        let r = self.distance(&Self::zero());

        let div = r.div(theta);
        let s = if !div.exact_lt(Self::R::one()) {
            <i32 as NumCast>::from(div.log2().ceil()).unwrap()
        } else {
            0
        };

        let scaled = self.div((Self::from_nat(2)).powi(s.try_into().unwrap()));

        let (mut result, _, _) = (0..n).fold(
            (Self::one(), Self::one(), Self::one()),
            |(acc, term, n), _| {
                let term = term * scaled.div(n);

                (acc + term, term, n + Self::one())
            },
        );

        for _ in 0..s {
            result = result * result;
        }

        result
    }
}

/// A type-level natural number (Peano encoding).
///
/// Used to carry a [`Field`]'s [`characteristic`](Field::Characteristic) in the
/// type system, where it can be matched on in bounds (`Characteristic = NatZero`)
/// rather than checked at runtime. [`N`](Nat::N) reflects the numeral back to a
/// `usize` for the rare cases that need the value (e.g. the finite characteristic
/// audit in [`check_characteristic_up_to`](Field::check_characteristic_up_to)).
pub trait Nat {
    /// The numeral's value as a `usize`. `NatZero::N == 0`, `Succ<N>::N == N::N + 1`.
    const N: usize;
}

/// The successor `n + 1` at the type level. See [`Nat`].
pub struct Succ<N: Nat>(N);

/// Type-level zero — the base of the [`Nat`] tower.
///
/// Deliberately uninhabited: as a *set* it is the empty set, so its cardinality
/// (`0`) matches the numeral it denotes. A field with `Characteristic = NatZero`
/// is characteristic zero (contains ℚ), which is exactly the precondition the
/// matrix exponential needs to form `1/k!`.
pub enum NatZero {}

impl Nat for NatZero {
    const N: usize = 0;
}

impl<N: Nat> Nat for Succ<N> {
    const N: usize = N::N + 1;
}

/// A field `F` with its involution *trivialised*: `conj = id`, hence
/// `Fixed = Self`.
///
/// This selects the `σ = id` sector of `F` without threading an involution
/// parameter through the hierarchy. On this sector a symmetric bilinear form is
/// the same thing as a Hermitian one, so [`Bilinear`] falls out of
/// [`Sesquilinear`] via the blanket impl. `Symmetrized<Complex<R>>` is ℂ viewed
/// through the identity involution — the home of the ℂ-*bilinear* Killing form,
/// distinct at the type level from `Complex<R>` (whose canonical involution is
/// conjugation, giving a *Hermitian* form).
///
/// It shares every algebraic and analytic fact with the inner `F` except the
/// involution: arithmetic, characteristic, and any forwarded metric all delegate
/// straight through. Note that `Fixed = Self` means its fixed field is *not*
/// `R`, so fields that wish to have analysis done on it should use [`Metric`],
/// which is guaranteed to be a field norm via coherence.
///
/// Since conjugation is anti-multiplicative and `F::Fixed = Self => conj(a) = a`,
/// `ab = conj(ab) = conj(b)conj(a) = ba`. Therefore this construction is only valid
/// for commutative fields hence the trait bound.
///
/// [`Metric`]: crate::traits::Metric
/// [`Bilinear`]: crate::traits::Bilinear
/// [`Sesquilinear`]: crate::traits::Sesquilinear
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Symmetrized<F: CField>(pub F);

impl<F: CField> Sub for Symmetrized<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl<F: CField> Add for Symmetrized<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl<F: CField> Neg for Symmetrized<F> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl<F: CField> Mul for Symmetrized<F> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0.mul(rhs.0))
    }
}

impl<F: CField> One for Symmetrized<F> {
    fn one() -> Self {
        Self(F::one())
    }

    fn is_one(&self) -> bool {
        self.0.is_one()
    }
}

impl<F: CField> Zero for Symmetrized<F> {
    fn zero() -> Self {
        Self(F::zero())
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<F: CField> Inv for NonZero<Symmetrized<F>> {
    type Output = Self;

    fn inv(self) -> Self::Output {
        NonZero::new_unchecked(Symmetrized(
            <F::Mul>::from(NonZero::new_unchecked(self.0.0))
                .inv()
                .into()
                .0,
        ))
    }
}

impl<F: CField> Field for Symmetrized<F> {
    type Fixed = Self;
    type Characteristic = F::Characteristic;

    fn conj(&self) -> Self {
        *self
    }

    fn to_fixed(self) -> Self::Fixed {
        self
    }

    fn from_fixed(x: Self::Fixed) -> Self {
        x
    }
}

impl<F: CField + Interval> Interval for Symmetrized<F> {
    type R = F::R;

    fn interval_squared(&self, other: &Self) -> F::R {
        self.0.interval_squared(&other.0)
    }
}

impl<F: CField<Fixed: Real>> FromReal for Symmetrized<F> {
    fn from_real(r: Self::R) -> Self {
        Self(F::from_fixed(r))
    }
}

impl<F: CField + Metric> Metric for Symmetrized<F> {}
impl<F: CField> CField for Symmetrized<F> {}

impl<R: Real, F: Field<Fixed = R>> Interval for F {
    type R = R;

    fn interval_squared(&self, other: &Self) -> R {
        (*self - *other).norm_squared()
    }
}

/// A *primitive* `N`-th root of unity — one that generates all of `μ_N`.
///
/// [`new`](RootOfUnityPrimitive::new) returns `None` if the given element isn't
/// primitive (some lower power hits `1`), and
/// [`roots_of_unity`](RootOfUnityPrimitive::roots_of_unity) enumerates the full
/// group it generates as [`RootOfUnity`] values.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct RootOfUnityPrimitive<F: Field, const N: usize>(RootOfUnity<F, N>);

impl<F: Field, const N: usize> RootOfUnityPrimitive<F, N> {
    /// Certifies that `x` is a primitive `N`-th root of unity.
    ///
    /// Returns `None` when an earlier positive power of `x` is one or when
    /// `xⁿ != 1`.
    pub fn new(x: F) -> Option<Self> {
        const { assert!(N != 0) }

        (0..N)
            .try_fold(F::one(), |root, n| {
                let power = root * x;
                match n {
                    x if x == N - 1 => power == F::one(),
                    _ => power != F::one(),
                }
                .then_some(power)
            })
            .map(|_| Self(RootOfUnity(x)))
    }

    /// Returns this generator as an ordinary [`RootOfUnity`].
    pub fn inner(&self) -> RootOfUnity<F, N> {
        self.0
    }

    /// Enumerates the cyclic group generated by this primitive root.
    pub fn roots_of_unity(&self) -> impl Iterator<Item = RootOfUnity<F, N>> {
        let mut acc = F::one();

        (0..N).map(move |_| {
            let ret = acc;
            acc = acc * self.0.0;
            RootOfUnity(ret)
        })
    }
}

/// An `N`-th root of unity in `F` — an element of the finite cyclic group `μ_N`.
///
/// A zero-dimensional [`LieGroup`] (its tangent space is trivial, so `exp`/`log`
/// are trivial). Used as the kernel in quotient constructions — e.g. the
/// [`Lorentz`](crate::spacetime::Lorentz) group is `SL(2,ℂ)` quotiented by
/// `RootOfUnity<_, 2> = {±1}`.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct RootOfUnity<F: Field, const N: usize>(F);

impl<F: Field, const N: usize> One for RootOfUnity<F, N> {
    fn one() -> Self {
        const { assert!(N != 0) }
        // One is always a root of unity
        Self(F::one())
    }

    fn is_one(&self) -> bool {
        self.0 == F::one()
    }
}

impl<F: Field, const N: usize> Mul<Self> for RootOfUnity<F, N> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        // Unchecked by definition!
        Self(self.0.mul(rhs.0))
    }
}

impl<F: Field, const N: usize> Inv for RootOfUnity<F, N> {
    type Output = Self;

    fn inv(self) -> Self::Output {
        Self(F::Mul::from(NonZero::new_unchecked(self.0)).inv().into().0)
    }
}

impl_group_via_mul!(RootOfUnity<F, N>, F: Field, const N: usize);

impl<V: Tensor, const N: usize> LieGroup<V> for RootOfUnity<V::F, N> {
    fn identity_exp(_: V) -> Self {
        const { assert!(N != 0) }
        Self::one()
    }

    fn identity_log(p: &Self) -> Option<V> {
        p.is_one().then(|| V::zero())
    }
}

impl<F: Field, const N: usize> RootOfUnity<F, N> {
    /// Certifies that `x` is an `N`-th root of unity.
    pub fn new(x: F) -> Option<Self> {
        const { assert!(N != 0) }
        ((1..N).fold(x, |acc, _| acc * x).is_one()).then_some(Self(x))
    }

    /// Returns the underlying field element.
    pub fn inner(self) -> F {
        self.0
    }
}

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
/// [`Vector`] spaces, [`Z`](crate::discrete::Z), [`S1`](crate::flat::S1))
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
/// [`Vector`]: crate::traits::Vector
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
        Self::identity().compose(self) == *self
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
        Self: PartialEq + core::fmt::Debug,
    {
        // We test it this way to give tolerance relations a scale to measure the
        // relative error from, so that this test passes at all scales.
        // Solves catestrophic cancelling problems.
        (self.inverse()).compose(self).compose(self) == Self::identity().compose(self)
    }

    #[cfg(feature = "testing")]
    fn check_right_inverse(&self) -> bool
    where
        Self: PartialEq,
    {
        self.compose(&self.inverse()).compose(self) == Self::identity().compose(self)
    }
}

/// A Lie group structure on a manifold.
///
/// The space of all values of a type `G: LieGroup<V>` is interpreted as
/// a Lie group — a manifold that is also a group, where the group operations
/// are smooth maps. `V` is the (pseudo) Euclidean space coordinatising the group's
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
pub trait LieGroup<V: Tensor>: Group {
    fn identity_exp(v: V) -> Self;
    fn identity_log(p: &Self) -> Option<V>;
}

// left translation
impl<V: Tensor, L: LieGroup<V>> Smooth<V> for L {
    type Global = Self;

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
pub trait Quotient<G: LieGroup<V>, H: LieGroup<V>, V: Tensor>: Point {
    /// Maps `g` to the `Quotient` value representing its coset `gH`.
    fn new(g: G) -> Self;

    /// Recovers a representative of `self`'s coset, satisfying
    /// `new(self.lift()) == self`.
    ///
    /// This is not merely "some" representative: `lift` must return the
    /// one nearest the identity, in the sense that `identity_log`'s
    /// result (built from `lift`, see `quotient_identity_log`) reports
    /// the same norm as `Metric::distance` from the identity. A `lift`
    /// that satisfies `new(self.lift()) == self` without also being
    /// nearest will still pass every `Quotient`/`Group`/`LieGroup`
    /// axiom test — those are satisfied by any valid representative
    /// choice — but will silently produce a geometrically wrong
    /// `identity_log`, and hence a wrong `Chart`/`ExpMap`/`Metric` for
    /// the whole type. This is exactly what `test_riemannian!`'s
    /// `chart_metric_compatibility` catches: it is the test that
    /// certifies `lift` chose correctly, not a separate property to
    /// verify independently.
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
    ($type:ty, $g:ty, $h:ty, $v:ty, $($generics:tt)*) => {
        impl<$($generics)*> $crate::traits::Group for $type {
            fn identity() -> Self {
                <Self as $crate::traits::Quotient<$g, $h, $v>>::quotient_identity()
            }
            fn compose(&self, rhs: &Self) -> Self {
                <Self as $crate::traits::Quotient<$g, $h, $v>>::quotient_compose(&self, &rhs)
            }
            fn inverse(&self) -> Self {
                <Self as $crate::traits::Quotient<$g, $h, $v>>::quotient_inverse(&self)
            }
        }

        impl<$($generics)*> $crate::traits::LieGroup<$v> for $type {
            fn identity_exp(v: $v) -> Self {
                <Self as $crate::traits::Quotient<$g, $h, $v>>::quotient_identity_exp(v)
            }
            fn identity_log(p: &Self) -> Option<$v> {
                <Self as $crate::traits::Quotient<$g, $h, $v>>::quotient_identity_log(p)
            }
        }
    };
}
