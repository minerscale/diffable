//! A reflected ontology of mathematical structure.
//!
//! This module exists to solve a slightly unusual generic-programming problem.  A
//! mathematical trait is often usable in a context richer than its own canonical
//! one: a jet can act as a field, a vector space carries a scalar field, and a
//! manifold carries tangent structure which in turn carries scalar structure.  The
//! concrete things involved are *not* thereby identified.  What is shared is only
//! the structural role in which a theorem is allowed to use them.
//!
//! The types in this file make that ambient structural context explicit.  A [`Cat`]
//! such as [`𝐅𝐥𝐝`] is a label for a theory/category.  Its associated [`Cat::C`] is a
//! canonical, zero-sized type tree describing the structural context on which that
//! theory depends.  A different context may contain that canonical tree somewhere
//! inside itself; [`Refines`] is the public proposition that this is so.
//!
//! In other words, the ontology records the crate's sanctioned ways for something
//! to "act the same while actually being different".  It does not choose concrete
//! implementations and it carries no runtime data.  The trait impls are the proofs.

use crate::traits::{Nat, NatZero, NatCompare, Succ};
use core::{fmt::Debug, marker::PhantomData};

// -----------------------------------------------------------------------------
// Convenience macros for writing the ontology
// -----------------------------------------------------------------------------

/// Build the child list of a canonical category context.
///
/// The DSL deliberately distinguishes three kinds of entry:
///
/// - `X` inserts the complete canonical context `<X as Cat>::C`.
/// - `~X` inserts only the node `X`, with no descendants.
/// - `X => Y` inserts `X`'s canonical context but cuts every branch at `Y`.
///
/// Shallow and truncated nodes are useful when two proof threads have already
/// accounted for common structure and spelling out the rest again would only
/// duplicate ontology.  They still retain the category label/provenance at the
/// point where the branch was cut.
macro_rules! cat {
    () => {
        Ø
    };

    // Shallow node.
    (~$head:ty, $($tail:tt)*) => {
        Cons<𝒯<$head, Ø>, cat![$($tail)*]>
    };

    (~$head:ty) => {
        Cons<𝒯<$head, Ø>, Ø>
    };

    // Canonical tree, truncated inclusively at every `target`.
    ($head:ty => $target:ty, $($tail:tt)*) => {
        Cons<
            <<$head as Cat>::C as Truncate<$target>>::Output,
            cat![$($tail)*]
        >
    };

    ($head:ty => $target:ty) => {
        Cons<
            <<$head as Cat>::C as Truncate<$target>>::Output,
            Ø
        >
    };

    // Full canonical tree.
    ($head:ty, $($tail:tt)*) => {
        Cons<<$head as Cat>::C, cat![$($tail)*]>
    };

    ($head:ty) => {
        Cons<<$head as Cat>::C, Ø>
    };
}

/// Register the ordinary category labels in one finite constructive universe.
///
/// Besides implementing [`Atom`], this emits an explicit `Different` proof for
/// every off-diagonal pair.  `Compare` therefore never means "Rust failed to
/// prove equality, so assume inequality"; every comparison result used by the
/// recursive machinery is backed by an impl.
///
/// Supplying the same atom twice is intentionally incoherent: the generated
/// off-diagonal `Different` impl would collide with reflexive `Same`.
macro_rules! atoms {
    () => {};

    ($head:ty $(, $tail:ty)* $(,)?) => {
        impl Atom for $head {}

        $(
            impl Compare<$tail> for $head {
                type Relation = Different;
            }

            impl Compare<$head> for $tail {
                type Relation = Different;
            }
        )*

        atoms!($($tail),*);
    };
}

macro_rules! compare_atoms_to_family {
    ($family:ident; $($cat:ident),* $(,)?) => {
        $(
            impl<N: Nat> Compare<$family<N>> for $cat
            where
                $family<N>: Cat,
            {
                type Relation = Different;
            }

            impl<N: Nat> Compare<$cat> for $family<N>
            where
                $family<N>: Cat,
            {
                type Relation = Different;
            }
        )*
    };
}

macro_rules! compare_atoms_to_families {
    ([$($cat:ident),* $(,)?];) => {};

    (
        [$($cat:ident),* $(,)?];
        $family:ident $(, $rest:ident)* $(,)?
    ) => {
        compare_atoms_to_family!($family; $($cat),*);

        compare_atoms_to_families!(
            [$($cat),*];
            $($rest),*
        );
    };
}

/// Declare the ontology and its Nat-indexed higher families.
///
/// Ordinary entries `A => ...;` introduce an [`Atom`] with a fixed canonical
/// context. Family entries `@F<N> => ...;` introduce a tower:
///
/// ```text
/// F<0>      -> the declared base context
/// F<n + 1>  -> the complete canonical context of F<n>
/// ```
///
/// The generated [`𝐂𝐚𝐭<N>`] is the diagonal context of the declaration: it
/// contains the complete canonical tree of every ordinary category and the
/// `N`th member of every indexed family. Thus `𝐂𝐚𝐭<N>` contains exactly the
/// finite ordinary ontology plus the higher structure exposed at depth `N`.
macro_rules! categories {
    (
        $(
            $cat:ident => $children:ty;
        )*

        $(
            @$family:ident<$n:ident> => $base:ty;
        )*
    ) => {
        $(
            #[derive(Copy, Clone, Debug)]
            pub struct $cat;

            impl Cat for $cat {
                type C = 𝒯<$cat, $children>;
            }
        )*

        $(
            #[derive(Copy, Clone, Debug)]
            pub struct $family<$n: Nat>(PhantomData<$n>);

            impl Cat for $family<NatZero> {
                type C = 𝒯<Self, $base>;
            }

            impl<$n: Nat> Cat for $family<Succ<$n>>
            where
                $family<$n>: Cat,
            {
                type C = 𝒯<Self, cat![$family<$n>]>;
            }
        )*

        /// The category of categories at universe/higher-structure level `N`.
        ///
        /// Its canonical context is the diagonal closure of this ontology: all
        /// ordinary categories occur with their full canonical trees, and each
        /// indexed family contributes its `N`th stage (which recursively carries
        /// all preceding stages beneath it).
        #[derive(Copy, Clone, Debug)]
        pub struct 𝐂𝐚𝐭<N: Nat>(PhantomData<N>);

        impl<N: Nat> Cat for 𝐂𝐚𝐭<N>
        where
            $(
                $family<N>: Cat,
            )*
        {
            type C = 𝒯<
                Self,
                cat![
                    $($cat,)*
                    $($family<N>,)*
                ],
            >;
        }

        atoms![$($cat),*];

        compare_atoms_to_families!(
            [$($cat),*];
            $($family),*
        );

        $(
            impl<N, M> Compare<$family<M>> for $family<N>
            where
                N: Nat + NatCompare<M>,
                M: Nat,
                $family<N>: Cat,
                $family<M>: Cat,
            {
                type Relation = <N as NatCompare<M>>::Relation;
            }

            impl<N, M> Compare<𝐂𝐚𝐭<M>> for $family<N>
            where
                N: Nat,
                M: Nat,
                $family<N>: Cat,
                𝐂𝐚𝐭<M>: Cat,
            {
                type Relation = Different;
            }

            impl<N, M> Compare<$family<M>> for 𝐂𝐚𝐭<N>
            where
                N: Nat,
                M: Nat,
                𝐂𝐚𝐭<N>: Cat,
                $family<M>: Cat,
            {
                type Relation = Different;
            }
        )*

        impl<N, M> Compare<𝐂𝐚𝐭<M>> for 𝐂𝐚𝐭<N>
        where
            N: Nat + NatCompare<M>,
            M: Nat,
            𝐂𝐚𝐭<N>: Cat,
            𝐂𝐚𝐭<M>: Cat,
        {
            type Relation = <N as NatCompare<M>>::Relation;
        }
    };
}

// The ontology of the crate.
//
// Read `A => cat![B, C]` as: the canonical context required to talk about `A`
// contains independently addressable `B` and `C` roles.  Descendants recursively
// record the dependencies of those roles.  Repeated labels are not necessarily
// redundant: two branches may reach the same category while referring to different
// associated objects, so their provenance can matter.
//
// The tree is not a taxonomy saying that the root value *is* every descendant.
// It is an ambient dependency/proof context saying which structural roles are
// available, and by which canonical threads.
categories! {
    𝐒𝐞𝐭   => Ø;
    𝐓𝐨𝐩   => cat![𝐒𝐞𝐭];
    𝐌𝐨𝐧   => cat![𝐓𝐨𝐩];
    𝐂𝐌𝐨𝐧  => cat![𝐌𝐨𝐧];
    𝐆𝐫𝐩   => cat![𝐌𝐨𝐧];
    𝐀𝐛    => cat![𝐆𝐫𝐩, ~𝐂𝐌𝐨𝐧];
    𝐑𝐢𝐧𝐠  => cat![𝐂𝐌𝐨𝐧, 𝐆𝐫𝐩 => 𝐌𝐨𝐧];
    𝐂𝐅𝐥𝐝  => cat![𝐑𝐢𝐧𝐠, ~𝐅𝐥𝐝, ~𝐀𝐛];
    𝐅𝐥𝐝   => cat![𝐑𝐢𝐧𝐠, ~𝐆𝐫𝐩, 𝐂𝐅𝐥𝐝];
    𝐓𝐞𝐧𝐬  => cat![𝐂𝐌𝐨𝐧, 𝐅𝐥𝐝];
    𝐕𝐞𝐜𝐭  => cat![𝐓𝐞𝐧𝐬, 𝐆𝐫𝐩 => 𝐌𝐨𝐧];
    𝐌𝐚𝐧   => cat![𝐓𝐨𝐩, 𝐓𝐞𝐧𝐬];
    @𝐇𝐨𝐦<N> => cat![𝐌𝐚𝐧];
}

/// Diffeomorphisms are the first Hom level over smooth manifolds.
pub type 𝐃𝐢𝐟𝐟 = 𝐇𝐨𝐦<NatZero>;

/// The empty type-level list.
///
/// Category contexts are types, not runtime trees; `Ø` and [`Cons`] are merely a
/// type-level encoding of a finite ordered forest of child contexts.
#[derive(Debug, Copy, Clone)]
pub struct Ø;

/// A type-level list node with `Head` followed by `Tail`.
#[derive(Debug, Copy, Clone)]
pub struct Cons<Head, Tail>(PhantomData<(Head, Tail)>);

mod sealed {
    pub trait Category {}
    pub trait CatList {}
}

pub trait CatList: sealed::CatList {}

impl sealed::CatList for Ø {}
impl CatList for Ø {}
impl<C: Category, L: CatList> sealed::CatList for Cons<C, L> {}
impl<C: Category, L: CatList> CatList for Cons<C, L> {}

/// A node in a concrete reflected category context.
///
/// `𝒞` is the category *label* at this node. `L` is the list of structural
/// dependency contexts reachable beneath it.  `𝒯` contains no mathematical value:
/// its sole purpose is to let the type system retain and inspect the ontology.
#[derive(Debug, Copy, Clone)]
pub struct 𝒯<𝒞: Cat, L: CatList>(PhantomData<(𝒞, L)>);

impl<𝒞: Cat, L: CatList> sealed::Category for 𝒯<𝒞, L> {}
impl<𝒞: Cat, L: CatList> Category for 𝒯<𝒞, L> {
    type Children = L;

    fn witness() -> Self {
        Self(PhantomData)
    }
}

/// A concrete type-level structural context.
///
/// Implementations are sealed so that every `Category` has the tree shape understood
/// by the proof machinery below.  A category value is only a witness; every such
/// context has exactly one meaningful inhabitant and carries no runtime information.
pub trait Category: sealed::Category {
    type Children: CatList;

    fn witness() -> Self;
}

/// A label for a mathematical category/theory known to the ontology.
///
/// Do not confuse `𝒞: Cat` with `C: Category`. `𝒞` names the role (for example
/// `𝐅𝐥𝐝`); `𝒞::C` is the distinguished canonical dependency context for that role.
pub trait Cat: Copy + Clone + Debug + Send + Sync + 'static {
    type C: Category;
}

// -----------------------------------------------------------------------------
// Object equivalence and context refinement
// -----------------------------------------------------------------------------

/// Reversible equivalence of concrete objects as observed in category `𝒞`.
///
/// This is value-level mathematics and is conceptually separate from [`Refines`].
/// `Refines` relates zero-sized *contexts*; `Equivalent` supplies actual maps between
/// concrete representations.
pub trait Equivalent<𝒞: Cat, X> {
    fn project(self) -> X;
    fn lift(x: X) -> Self;
}

impl<𝒞: Cat, X> Equivalent<𝒞, X> for X {
    fn project(self) -> X {
        self
    }

    fn lift(x: X) -> Self {
        x
    }
}

/// Proposition that this context can furnish the canonical role `𝒞`.
///
/// `C: Refines<𝒞>` does **not** say that the object whose ambient context is `C`
/// literally is a `𝒞`, nor does it select a concrete implementation.  It says only
/// that `C` contains the canonical context `𝒞::C` somewhere in its ontology.  This
/// is the place where the crate records "acts like `𝒞`, but is actually something
/// richer/different".
///
/// The recursive `Contains` proof searches arbitrarily deeply, so a role exposed
/// through several associated structures is still available to the ambient context.
/// For example, a manifold context may reach a scalar-field context through its
/// tangent structure.  The actual projection of mathematical *values* is supplied by
/// the relevant mathematical traits; projecting these context ZSTs themselves would
/// carry no information.
///
/// Note that the current criterion is *exact canonical subtree inclusion*: a shallow
/// `~𝒞` node or a branch truncated before the full `𝒞::C` tree is present does not,
/// by itself, establish `Refines<𝒞>`.
pub trait Refines<𝒞: Cat>: Category {}
impl<𝒞: Cat, C: Category + Contains<𝒞, Relation = Same>> Refines<𝒞> for C {}

// -----------------------------------------------------------------------------
// Constructive equality / distinctness proofs
// -----------------------------------------------------------------------------

/// Type-level result of a proved comparison.
///
/// These are evidence tags used only to dispatch mutually exclusive recursive impls.
pub struct Same;
pub struct Different;

/// A constructive proof that we know how `Self` relates to `𝒟`.
///
/// `Relation` is either `Same` or `Different`, but only because some
/// corresponding proof has actually been supplied.
pub trait Compare<𝒟: Cat>: Cat {
    type Relation;
}

/// An ordinary member of the finite ontology.
///
/// `Atom` is the enclave in which `atoms!` can enumerate pairwise distinctness.
/// Universe-bearing [`𝐂𝐚𝐭`] is intentionally kept outside it.
pub trait Atom: Cat {}

// Ordinary labels have generic reflexivity. Indexed families and `𝐂𝐚𝐭` use
// their Peano-index comparison impls instead, keeping the coherence domains disjoint.
impl<𝒞: Atom> Compare<𝒞> for 𝒞 {
    type Relation = Same;
}

impl<𝒞: Atom, N: Nat> Compare<𝐂𝐚𝐭<N>> for 𝒞 {
    type Relation = Different;
}

impl<𝒞: Atom, N: Nat> Compare<𝒞> for 𝐂𝐚𝐭<N> {
    type Relation = Different;
}

// -----------------------------------------------------------------------------
// Canonical-tree truncation
// -----------------------------------------------------------------------------

// Truncation is an ontology-writing operation, not a runtime projection.  It walks
// an entire canonical tree and, whenever it encounters the requested label, keeps
// that node but replaces its descendants with `Ø`.  This lets `cat![A => B]` retain
// the fact/provenance that the `B` role is present without duplicating the rest of
// `B`'s dependency tree on that proof thread.

pub trait Truncate<𝒟: Cat>: Category {
    type Output: Category;
}

pub trait TruncateList<𝒟: Cat>: CatList {
    type Output: CatList;
}

impl<𝒟: Cat> TruncateList<𝒟> for Ø {
    type Output = Ø;
}

impl<𝒟: Cat, Head: Category + Truncate<𝒟>, Tail: CatList + TruncateList<𝒟>> TruncateList<𝒟>
    for Cons<Head, Tail>
{
    type Output = Cons<<Head as Truncate<𝒟>>::Output, <Tail as TruncateList<𝒟>>::Output>;
}

pub trait TruncateWith<𝒟: Cat, Relation>: Category {
    type Output: Category;
}

impl<𝒞: Cat, 𝒟: Cat, L: CatList> TruncateWith<𝒟, Same> for 𝒯<𝒞, L> {
    type Output = 𝒯<𝒞, Ø>;
}

impl<𝒞: Cat, 𝒟: Cat, L: CatList + TruncateList<𝒟>> TruncateWith<𝒟, Different> for 𝒯<𝒞, L> {
    type Output = 𝒯<𝒞, <L as TruncateList<𝒟>>::Output>;
}

impl<𝒞: Cat + Compare<𝒟>, 𝒟: Cat, L: CatList> Truncate<𝒟> for 𝒯<𝒞, L>
where
    𝒯<𝒞, L>: TruncateWith<𝒟, <𝒞 as Compare<𝒟>>::Relation>,
{
    type Output = <𝒯<𝒞, L> as TruncateWith<𝒟, <𝒞 as Compare<𝒟>>::Relation>>::Output;
}

// -----------------------------------------------------------------------------
// Structural search: exact tree equality and deep canonical-subtree inclusion
// -----------------------------------------------------------------------------

// This machinery is private because `TreeEq`/`Contains` are implementation details
// of the public proposition `Refines`.  The recursion is deliberately expressed via
// `Same`/`Different` dispatch rather than overlapping positive/negative trait bounds.
//
// `TreeEq` means exact reflected-tree equality: same root category *and* pairwise
// equal ordered child lists.  `Contains<𝒞>` then performs a depth-first existential
// search for an exact occurrence of `𝒞::C`.  If the current root is not that tree,
// it recursively searches each child, however deeply nested.
//
// Consequently, containment is stronger than merely finding the label `𝒞`: shallow
// or truncated occurrences are not matches unless they happen to equal `𝒞::C`.

/// Exact equality of two reflected category trees.
trait TreeEq<Rhs: Category>: Category {
    type Relation;
}

trait TreeEqWith<Rhs: Category, Relation>: Category {
    type Relation;
}

impl<𝒞: Cat + Compare<𝒟, Relation = Different>, 𝒟: Cat, L: CatList, R: CatList>
    TreeEqWith<𝒯<𝒟, R>, Different> for 𝒯<𝒞, L>
{
    type Relation = Different;
}

impl<𝒞: Cat + Compare<𝒟, Relation = Same>, 𝒟: Cat, L: CatList + ListEq<R>, R: CatList>
    TreeEqWith<𝒯<𝒟, R>, Same> for 𝒯<𝒞, L>
{
    type Relation = <L as ListEq<R>>::Relation;
}

impl<𝒞: Cat + Compare<𝒟>, 𝒟: Cat, L: CatList, R: CatList> TreeEq<𝒯<𝒟, R>> for 𝒯<𝒞, L>
where
    𝒯<𝒞, L>: TreeEqWith<𝒯<𝒟, R>, <𝒞 as Compare<𝒟>>::Relation>,
{
    type Relation =
        <𝒯<𝒞, L> as TreeEqWith<𝒯<𝒟, R>, <𝒞 as Compare<𝒟>>::Relation>>::Relation;
}

/// Exact, ordered equality of two child lists.
trait ListEq<Rhs: CatList>: CatList {
    type Relation;
}

impl ListEq<Ø> for Ø {
    type Relation = Same;
}

impl<H: Category, T: CatList> ListEq<Cons<H, T>> for Ø {
    type Relation = Different;
}

impl<H: Category, T: CatList> ListEq<Ø> for Cons<H, T> {
    type Relation = Different;
}

trait ListEqWith<RH: Category, RT: CatList, Relation>: CatList {
    type Relation;
}

impl<H: Category + TreeEq<RH, Relation = Different>, T: CatList, RH: Category, RT: CatList>
    ListEqWith<RH, RT, Different> for Cons<H, T>
{
    type Relation = Different;
}

impl<H: Category + TreeEq<RH, Relation = Same>, T: CatList + ListEq<RT>, RH: Category, RT: CatList>
    ListEqWith<RH, RT, Same> for Cons<H, T>
{
    type Relation = <T as ListEq<RT>>::Relation;
}

impl<H: Category + TreeEq<RH>, T: CatList, RH: Category, RT: CatList> ListEq<Cons<RH, RT>>
    for Cons<H, T>
where
    Cons<H, T>: ListEqWith<RH, RT, <H as TreeEq<RH>>::Relation>,
{
    type Relation = <Cons<H, T> as ListEqWith<RH, RT, <H as TreeEq<RH>>::Relation>>::Relation;
}

/// Whether this tree contains an exact canonical `𝒞::C` subtree at any depth.
trait Contains<𝒞: Cat>: Category {
    type Relation;
}

trait ContainsWith<𝒞: Cat, Relation>: Category {
    type Relation;
}

impl<𝒞: Cat, C: Category + TreeEq<𝒞::C, Relation = Same>> ContainsWith<𝒞, Same> for C {
    type Relation = Same;
}

impl<𝒞: Cat, 𝒟: Cat, L: CatList + ListContains<𝒞>> ContainsWith<𝒞, Different> for 𝒯<𝒟, L> {
    type Relation = <L as ListContains<𝒞>>::Relation;
}

impl<𝒞: Cat, C: Category + TreeEq<𝒞::C> + ContainsWith<𝒞, <C as TreeEq<𝒞::C>>::Relation>>
    Contains<𝒞> for C
{
    type Relation = <C as ContainsWith<𝒞, <C as TreeEq<𝒞::C>>::Relation>>::Relation;
}

/// Existential search for `𝒞::C` across a list of sibling subtrees.
trait ListContains<𝒞: Cat>: CatList {
    type Relation;
}

impl<𝒞: Cat> ListContains<𝒞> for Ø {
    type Relation = Different;
}

trait ListContainsWith<𝒞: Cat, Relation>: CatList {
    type Relation;
}

impl<𝒞: Cat, H: Category + Contains<𝒞, Relation = Same>, T: CatList> ListContainsWith<𝒞, Same>
    for Cons<H, T>
{
    type Relation = Same;
}

impl<𝒞: Cat, H: Category + Contains<𝒞, Relation = Different>, T: CatList + ListContains<𝒞>>
    ListContainsWith<𝒞, Different> for Cons<H, T>
{
    type Relation = <T as ListContains<𝒞>>::Relation;
}

impl<𝒞: Cat, H: Category + Contains<𝒞>, T: CatList> ListContains<𝒞> for Cons<H, T>
where
    Cons<H, T>: ListContainsWith<𝒞, <H as Contains<𝒞>>::Relation>,
{
    type Relation =
        <Cons<H, T> as ListContainsWith<𝒞, <H as Contains<𝒞>>::Relation>>::Relation;
}
