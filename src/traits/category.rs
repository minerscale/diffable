//! A reflected ontology of mathematical structure.
//!
//! Rust traits are not first-class, so this module reifies selected mathematical
//! traits as ordinary zero-sized types implementing [`Cat`].  [`Cat::C`] is the
//! canonical [`Category`] signature of the reflected trait.
//!
//! A category signature has two independent pieces:
//!
//! - labelled structural dependencies, corresponding to associated types;
//! - an unordered set of properties of `Self`, corresponding to inherited or
//!   otherwise available mathematical structure.
//!
//! The category token itself is deliberately *not* stored in its canonical
//! signature.  The signature is structural: if another category has the same
//! required associated dependencies and properties (possibly with additional
//! information), it may [`Refines`] the reflected trait without carrying its
//! nominal label.
//!
//! Associated dependencies are different.  Their labels are observable interface
//! structure: `Vector::F` must be recovered as `F`, not merely as the first
//! field-shaped thing in a list.  Consequently structural dependencies are matched
//! by associated-type name while properties are matched as an unordered set.
//!
//! Everything in this module is zero-sized.  The ontology is a compile-time
//! database consumed by trait resolution; there is no runtime tree.

use crate::{
    coords::Coords,
    traits::{CField, Field, Nat, NatCompare, NatZero, Succ, Tensor, Vector},
};
use core::{fmt::Debug, marker::PhantomData};

// -----------------------------------------------------------------------------
// Generic type-level lists
// -----------------------------------------------------------------------------

/// The empty type-level list.
#[derive(Debug, Copy, Clone)]
pub struct Ø;

/// A type-level list node with `Head` followed by `Tail`.
///
/// This deliberately remains unconstrained: calculus also uses `Cons`/`Ø` for
/// type-level routes unrelated to the category ontology.
#[derive(Debug, Copy, Clone)]
pub struct Cons<Head, Tail>(PhantomData<(Head, Tail)>);

// -----------------------------------------------------------------------------
// Category signatures
// -----------------------------------------------------------------------------

mod sealed {
    pub trait Category {}
    pub trait AssocList {}
    pub trait PropertyList {}
    pub trait PropertyEntry {}
    pub trait EquationList {}
    pub trait AssocEntry {}
}

/// A concrete structural signature of a reflected trait.
///
/// `Structure` is a labelled record of associated-type dependencies.
/// `Properties` is an unordered set of properties of `Self`.
pub trait Category: sealed::Category + Sized {
    type Structure: AssocList;
    type Properties: PropertyList;
    type Equations: EquationList;
}

/// The canonical category/signature pair.
///
/// Unlike the previous `𝒯<Label, Children>`, this type deliberately contains no
/// nominal category label.  The meaning of a [`Cat`] is its associated `C`.
#[derive(Debug, Copy, Clone)]
pub struct 𝒯<Structure: AssocList, Properties: PropertyList, Equations: EquationList = Ø>(
    PhantomData<(Structure, Properties, Equations)>,
);

impl<S: AssocList, P: PropertyList, E: EquationList> sealed::Category for 𝒯<S, P, E> {}

impl<S: AssocList, P: PropertyList, E: EquationList> Category for 𝒯<S, P, E> {
    type Structure = S;
    type Properties = P;
    type Equations = E;
}

/// A first-class name for a reflected Rust trait / mathematical category.
///
/// `Cat::C` is the trait's structural signature.  `Self` is not embedded into
/// that signature: the bold token names the interface, while `C` says what the
/// interface means.
pub trait Cat: Copy + Clone + Debug + Send + Sync + 'static {
    type C: Category;
}

// -----------------------------------------------------------------------------
// Structural dependencies: reflected associated types
// -----------------------------------------------------------------------------

/// Marker implemented by names of reflected associated types.
pub trait AssocName: Copy + Clone + Debug + Send + Sync + 'static {}

/// A canonical associated-type requirement.
///
/// `Name` is literally the associated type's reflected name and `Role` is the
/// reflected trait bound required of that associated type.
#[derive(Debug, Copy, Clone)]
pub struct Requires<Name: AssocName, Role: Cat>(PhantomData<(Name, Role)>);

/// A concrete associated-type binding.
///
/// Reflection of a concrete trait implementation replaces a canonical
/// [`Requires`] entry by `Binds<Name, Role, Value>`.
#[derive(Debug, Copy, Clone)]
pub struct Binds<Name: AssocName, Role: Cat, Value>(PhantomData<(Name, Role, Value)>);

/// Placeholder used as [`AssocEntry::Value`] by a canonical requirement.
#[derive(Debug, Copy, Clone)]
pub struct Unspecified;

pub trait AssocEntry: sealed::AssocEntry {
    type Name: AssocName;
    type Role: Cat;
    type Value;
}

impl<N: AssocName, R: Cat> sealed::AssocEntry for Requires<N, R> {}
impl<N: AssocName, R: Cat> AssocEntry for Requires<N, R> {
    type Name = N;
    type Role = R;
    type Value = Unspecified;
}

impl<N: AssocName, R: Cat, V> sealed::AssocEntry for Binds<N, R, V> {}
impl<N: AssocName, R: Cat, V> AssocEntry for Binds<N, R, V> {
    type Name = N;
    type Role = R;
    type Value = V;
}

pub trait AssocList: sealed::AssocList {}
impl sealed::AssocList for Ø {}
impl AssocList for Ø {}
impl<H: AssocEntry, T: AssocList> sealed::AssocList for Cons<H, T> {}
impl<H: AssocEntry, T: AssocList> AssocList for Cons<H, T> {}

/// Constructive equality of associated-type labels.
pub trait CompareAssoc<Rhs: AssocName>: AssocName {
    type Relation;
}

/// Declare the finite set of reflected associated-type names.
///
/// As with category atoms, inequality is constructive: off-diagonal `Different`
/// impls are generated explicitly rather than inferred from failure.
macro_rules! assoc_names {
    () => {};

    ($head:ident $(, $tail:ident)* $(,)?) => {
        #[derive(Debug, Copy, Clone)]
        pub struct $head;
        impl AssocName for $head {}
        impl CompareAssoc<$head> for $head {
            type Relation = Same;
        }

        $(
            impl CompareAssoc<$tail> for $head {
                type Relation = Different;
            }
            impl CompareAssoc<$head> for $tail {
                type Relation = Different;
            }
        )*

        assoc_names!($($tail),*);
    };
}

// These names intentionally mirror the associated type spellings of the
// reflected traits.
assoc_names!(F, Fixed, Characteristic);

/// Find a structural dependency by associated-type name.
///
/// The list is logically a record, not a tuple: lookup ignores declaration order.
pub trait FindAssoc<Name: AssocName>: AssocList {
    type Found: AssocEntry<Name = Name>;
}

pub trait FindAssocWith<Name: AssocName, Relation>: AssocList {
    type Found: AssocEntry<Name = Name>;
}

impl<Name, Head, Tail> FindAssocWith<Name, Same> for Cons<Head, Tail>
where
    Name: AssocName,
    Head: AssocEntry<Name = Name>,
    Tail: AssocList,
{
    type Found = Head;
}

impl<Name, Head, Tail> FindAssocWith<Name, Different> for Cons<Head, Tail>
where
    Name: AssocName,
    Head: AssocEntry,
    Tail: AssocList + FindAssoc<Name>,
{
    type Found = <Tail as FindAssoc<Name>>::Found;
}

impl<Name, Head, Tail> FindAssoc<Name> for Cons<Head, Tail>
where
    Name: AssocName,
    Head: AssocEntry,
    Head::Name: CompareAssoc<Name>,
    Tail: AssocList,
    Cons<Head, Tail>: FindAssocWith<Name, <Head::Name as CompareAssoc<Name>>::Relation>,
{
    type Found = <Cons<Head, Tail> as FindAssocWith<
        Name,
        <Head::Name as CompareAssoc<Name>>::Relation,
    >>::Found;
}

/// Project a reflected associated dependency by its actual trait name.
pub trait Associated<Name: AssocName>: Category {
    type Role: Cat;
    type Type;
}

impl<Name, S, P, E> Associated<Name> for 𝒯<S, P, E>
where
    Name: AssocName,
    S: AssocList + FindAssoc<Name>,
    P: PropertyList,
    E: EquationList,
{
    type Role = <<S as FindAssoc<Name>>::Found as AssocEntry>::Role;
    type Type = <<S as FindAssoc<Name>>::Found as AssocEntry>::Value;
}

// -----------------------------------------------------------------------------
// Properties: unordered inherited structure
// -----------------------------------------------------------------------------

/// A canonical requirement that `Self` has property/category `𝒞`.
#[derive(Debug, Copy, Clone)]
pub struct Property<𝒞: Cat>(PhantomData<𝒞>);

/// A resolved property edge.
///
/// `Context` is the concrete/refined graph which satisfies `𝒞`.  Unlike a bare
/// [`Property`], this retains the structural information discovered while resolving
/// the property, including associated bindings inherited through it.
#[derive(Debug, Copy, Clone)]
pub struct BindsProperty<𝒞: Cat, Context: Category>(PhantomData<(𝒞, Context)>);

pub trait PropertyEntry: sealed::PropertyEntry {
    type Role: Cat;
}

impl<𝒞: Cat> sealed::PropertyEntry for Property<𝒞> {}
impl<𝒞: Cat> PropertyEntry for Property<𝒞> {
    type Role = 𝒞;
}

impl<𝒞: Cat, C: Category> sealed::PropertyEntry for BindsProperty<𝒞, C> {}
impl<𝒞: Cat, C: Category> PropertyEntry for BindsProperty<𝒞, C> {
    type Role = 𝒞;
}

pub trait PropertyList: sealed::PropertyList {}
impl sealed::PropertyList for Ø {}
impl PropertyList for Ø {}
impl<H: PropertyEntry, T: PropertyList> sealed::PropertyList for Cons<H, T> {}
impl<H: PropertyEntry, T: PropertyList> PropertyList for Cons<H, T> {}

/// Find a property edge by reflected category name.
pub trait FindProperty<𝒞: Cat>: PropertyList {
    type Found: PropertyEntry;
}

pub trait FindPropertyWith<𝒞: Cat, Relation>: PropertyList {
    type Found: PropertyEntry;
}

impl<𝒞, Head, Tail> FindPropertyWith<𝒞, Same> for Cons<Head, Tail>
where
    𝒞: Cat,
    Head: PropertyEntry,
    Head::Role: Compare<𝒞, Relation = Same>,
    Tail: PropertyList,
{
    type Found = Head;
}

impl<𝒞, Head, Tail> FindPropertyWith<𝒞, Different> for Cons<Head, Tail>
where
    𝒞: Cat,
    Head: PropertyEntry,
    Head::Role: Compare<𝒞, Relation = Different>,
    Tail: PropertyList + FindProperty<𝒞>,
{
    type Found = <Tail as FindProperty<𝒞>>::Found;
}

impl<𝒞, Head, Tail> FindProperty<𝒞> for Cons<Head, Tail>
where
    𝒞: Cat,
    Head: PropertyEntry,
    Head::Role: Compare<𝒞>,
    Tail: PropertyList,
    Cons<Head, Tail>: FindPropertyWith<𝒞, <Head::Role as Compare<𝒞>>::Relation>,
{
    type Found = <Cons<Head, Tail> as FindPropertyWith<
        𝒞,
        <Head::Role as Compare<𝒞>>::Relation,
    >>::Found;
}

/// Project the resolved graph which supplies property `𝒞`.
pub trait PropertyRefinement<𝒞: Cat>: Category {
    type Refinement: Category;
}

pub trait ResolvedProperty<𝒞: Cat>: PropertyEntry {
    type Refinement: Category;
}

impl<Actual, Context, Required> ResolvedProperty<Required> for BindsProperty<Actual, Context>
where
    Actual: Cat + Compare<Required, Relation = Same>,
    Context: Category + Refines<Required>,
    Required: Cat,
{
    type Refinement = <Context as Refines<Required>>::Refinement;
}

impl<𝒞, S, P, E> PropertyRefinement<𝒞> for 𝒯<S, P, E>
where
    𝒞: Cat,
    S: AssocList,
    P: PropertyList + FindProperty<𝒞>,
    E: EquationList,
    <P as FindProperty<𝒞>>::Found: ResolvedProperty<𝒞>,
{
    type Refinement = <<P as FindProperty<𝒞>>::Found as ResolvedProperty<𝒞>>::Refinement;
}

// -----------------------------------------------------------------------------
// Graph links: associated paths and equations
// -----------------------------------------------------------------------------

/// The first hop of a labelled path from the current reflected object.
#[derive(Debug, Copy, Clone)]
pub struct At<Name: AssocName>(PhantomData<Name>);

/// One further associated-type projection along an existing path.
#[derive(Debug, Copy, Clone)]
pub struct Follow<Path, Name: AssocName>(PhantomData<(Path, Name)>);

/// Equality of two associated paths.
///
/// Equations turn the otherwise tree-shaped associated dependencies into a finite
/// graph presentation.  Shared descendants and cycles are represented by identifying
/// paths rather than recursively unfolding category signatures.
#[derive(Debug, Copy, Clone)]
pub struct Equal<Left, Right>(PhantomData<(Left, Right)>);

/// An equation list is a list of [`Equal`] constraints.
pub trait EquationList: sealed::EquationList {}

impl sealed::EquationList for Ø {}
impl EquationList for Ø {}

impl<L, R, T: EquationList> sealed::EquationList for Cons<Equal<L, R>, T> {}
impl<L, R, T: EquationList> EquationList for Cons<Equal<L, R>, T> {}

/// Native projection of one reflected associated type from a concrete Rust type.
///
/// This is deliberately tiny: it is the bridge from a graph edge back into rustc's
/// own associated-type projection machinery.
pub trait Project<Name: AssocName> {
    type Output;
}

impl<T: Field> Project<Fixed> for T {
    type Output = T::Fixed;
}

impl<T: Field> Project<Characteristic> for T {
    type Output = T::Characteristic;
}

impl<T: Tensor> Project<F> for T {
    type Output = T::F;
}

/// Resolve a path against a concrete reflected category.
///
/// The first hop is read from the category's labelled `Binds` record.  Further hops
/// are delegated to [`Project`], so path equality ultimately becomes ordinary Rust
/// associated-type equality.
pub trait ResolvePath<Path>: Category {
    type Output;
}

impl<Name, S, P, E> ResolvePath<At<Name>> for 𝒯<S, P, E>
where
    Name: AssocName,
    S: AssocList + FindAssoc<Name>,
    P: PropertyList,
    E: EquationList,
{
    type Output = <Self as Associated<Name>>::Type;
}

impl<Path, Name, S, P, E> ResolvePath<Follow<Path, Name>> for 𝒯<S, P, E>
where
    Name: AssocName,
    S: AssocList,
    P: PropertyList,
    E: EquationList,
    Self: ResolvePath<Path>,
    <Self as ResolvePath<Path>>::Output: Project<Name>,
{
    type Output = <<Self as ResolvePath<Path>>::Output as Project<Name>>::Output;
}

/// Type-equality witness used to hand graph equations back to rustc.
pub trait SameType<Rhs> {}
impl<T> SameType<T> for T {}

trait SatisfiesEquation<Eq>: Category {}

impl<C, Left, Right> SatisfiesEquation<Equal<Left, Right>> for C
where
    C: Category + ResolvePath<Left> + ResolvePath<Right>,
    <C as ResolvePath<Left>>::Output: SameType<<C as ResolvePath<Right>>::Output>,
{
}

trait SatisfiesEquations<Equations: EquationList>: Category {}

impl<C: Category> SatisfiesEquations<Ø> for C {}

impl<C, Left, Right, Tail> SatisfiesEquations<Cons<Equal<Left, Right>, Tail>> for C
where
    C: Category + SatisfiesEquation<Equal<Left, Right>> + SatisfiesEquations<Tail>,
    Tail: EquationList,
{
}

// -----------------------------------------------------------------------------
// Constructive comparison of reflected trait/category names
// -----------------------------------------------------------------------------

/// Type-level result of a proved comparison.
pub struct Same;
pub struct Different;

pub trait Compare<𝒟: Cat>: Cat {
    type Relation;
}

pub trait Atom: Cat {}

impl<𝒞: Atom> Compare<𝒞> for 𝒞 {
    type Relation = Same;
}

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

// -----------------------------------------------------------------------------
// DSL
// -----------------------------------------------------------------------------

macro_rules! assoc_requirements {
    () => {
        Ø
    };

    ($name:ident : $role:ty $(, $rest_name:ident : $rest_role:ty)* $(,)?) => {
        Cons<
            Requires<$name, $role>,
            assoc_requirements!($($rest_name : $rest_role),*)
        >
    };
}

macro_rules! properties {
    () => {
        Ø
    };

    ($head:ty $(, $tail:ty)* $(,)?) => {
        Cons<Property<$head>, properties!($($tail),*)>
    };
}

/// Build one canonical trait/category signature.
///
/// ```text
/// cat![(F: 𝐅𝐥𝐝), {𝐓𝐞𝐧𝐬, 𝐆𝐫𝐩}]
/// ```
///
/// The parenthesised portion is a labelled associated-type record.  The braced
/// portion is an unordered property set.
macro_rules! equations {
    () => {
        Ø
    };

    ($head:ty $(, $tail:ty)* $(,)?) => {
        Cons<$head, equations!($($tail),*)>
    };
}

macro_rules! cat {
    // Full form with equations.
    (
        ($($name:ident : $role:ty),* $(,)?),
        {$($property:ty),* $(,)?},
        {$($equation:ty),* $(,)?}
    ) => {
        𝒯<
            assoc_requirements!($($name : $role),*),
            properties!($($property),*),
            equations!($($equation),*)
        >
    };

    // Full form without equations.
    (
        ($($name:ident : $role:ty),* $(,)?),
        {$($property:ty),* $(,)?}
    ) => {
        𝒯<
            assoc_requirements!($($name : $role),*),
            properties!($($property),*),
            Ø
        >
    };

    // Structure only.
    (
        $($name:ident : $role:ty),* $(,)?
    ) => {
        cat![
            ($($name : $role),*),
            {}
        ]
    };

    // Properties only.
    //
    // Keep this last: `$property:ty` is deliberately permissive and can start
    // consuming the more structured forms before failing partway through them.
    (
        $($property:ty),* $(,)?
    ) => {
        cat![
            (),
            {$($property),*}
        ]
    };
}

/// Declare ordinary reflected traits plus Nat-indexed higher families.
macro_rules! categories {
    (
        $(
            $cat:ident => $context:ty;
        )*

        $(
            @$family:ident<$n:ident> => $base:ty;
        )*
    ) => {
        $(
            #[derive(Copy, Clone, Debug)]
            pub struct $cat;

            impl Cat for $cat {
                type C = $context;
            }
        )*

        $(
            #[derive(Copy, Clone, Debug)]
            pub struct $family<$n: Nat>(PhantomData<$n>);

            impl Cat for $family<NatZero> {
                type C = $base;
            }

            impl<$n: Nat> Cat for $family<Succ<$n>>
            where
                $family<$n>: Cat,
            {
                type C = cat![(), {$family<$n>}];
            }
        )*

        /// The category of categories visible at higher-structure level `N`.
        #[derive(Copy, Clone, Debug)]
        pub struct 𝐂𝐚𝐭<N: Nat>(PhantomData<N>);

        impl<N: Nat> Cat for 𝐂𝐚𝐭<N>
        where
            $(
                $family<N>: Cat,
            )*
        {
            type C = cat![
                (),
                {
                    $($cat,)*
                    $($family<N>,)*
                }
            ];
        }

        atoms![$($cat),*];

        impl<𝒞: Atom, N: Nat> Compare<𝐂𝐚𝐭<N>> for 𝒞
        where
            𝐂𝐚𝐭<N>: Cat,
        {
            type Relation = Different;
        }

        impl<𝒞: Atom, N: Nat> Compare<𝒞> for 𝐂𝐚𝐭<N>
        where
            𝐂𝐚𝐭<N>: Cat,
        {
            type Relation = Different;
        }

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

// -----------------------------------------------------------------------------
// Ontology
// -----------------------------------------------------------------------------

// The bold types are first-class names for the corresponding mathematical
// traits/categories.  Their `C` values are signatures, not nominally rooted trees.
//
// Associated dependencies are declared exactly where the reflected Rust trait
// introduces them.  `Tensor` owns `F`; `Vector` reaches that scalar field through
// its `Tensor` property.  `Field` owns `Fixed` and `Characteristic`.
categories! {
    𝐒𝐞𝐭      => cat!{};
    𝐓𝐨𝐩      => cat!{𝐒𝐞𝐭};
    𝐌𝐨𝐧      => cat!{𝐓𝐨𝐩};
    𝐂𝐌𝐨𝐧     => cat!{𝐌𝐨𝐧};
    𝐆𝐫𝐩      => cat!{𝐌𝐨𝐧};
    𝐀𝐛       => cat!{𝐆𝐫𝐩, 𝐂𝐌𝐨𝐧};
    𝐑𝐢𝐧𝐠     => cat!{𝐂𝐌𝐨𝐧, 𝐆𝐫𝐩};
    𝐍𝐚𝐭      => cat!{};
    𝐂𝐅𝐥𝐝     => cat!{𝐅𝐥𝐝, 𝐀𝐛};
    𝐅𝐥𝐝      => cat![
        (Fixed: 𝐂𝐅𝐥𝐝, Characteristic: 𝐍𝐚𝐭),
        {𝐑𝐢𝐧𝐠, 𝐆𝐫𝐩},
        {Equal<Follow<At<Fixed>, Fixed>, At<Fixed>>}
    ];
    𝐓𝐞𝐧𝐬     => cat![(F: 𝐅𝐥𝐝), {𝐂𝐌𝐨𝐧}];
    𝐕𝐞𝐜𝐭     => cat!{𝐓𝐞𝐧𝐬, 𝐆𝐫𝐩};
    𝐌𝐚𝐧      => cat!{𝐓𝐨𝐩, 𝐓𝐞𝐧𝐬};

    @𝐇𝐨𝐦<N> => cat!{𝐌𝐚𝐧};
}

/// Diffeomorphisms are the first Hom level over smooth manifolds.
pub type 𝐃𝐢𝐟𝐟 = 𝐇𝐨𝐦<NatZero>;

// -----------------------------------------------------------------------------
// Structural refinement
// -----------------------------------------------------------------------------

/// Resolve one property requirement into the graph which actually supplies it.
///
/// A concrete [`BindsProperty`] carries its graph explicitly.  A bare canonical
/// [`Property`] can resolve itself when its canonical signature is already concrete
/// enough to refine that property; this handles ordinary inherited structure whose
/// canonical graph has no unresolved associated bindings.
pub trait RefineProperty<Required: Cat>: PropertyEntry {
    type Refinement: Category;
}

impl<Actual, Context, Required> RefineProperty<Required> for BindsProperty<Actual, Context>
where
    Actual: Cat + Compare<Required, Relation = Same>,
    Context: Category + Refines<Required>,
    Required: Cat,
{
    type Refinement = <Context as Refines<Required>>::Refinement;
}

impl<Actual, Required> RefineProperty<Required> for Property<Actual>
where
    Actual: Cat + Compare<Required, Relation = Same>,
    Required: Cat,
    Actual::C: Refines<Required>,
{
    type Refinement = <Actual::C as Refines<Required>>::Refinement;
}

/// Resolve every required property in `Target`, retaining the graph found for each.
pub trait RefinesProperties<Target: PropertyList>: PropertyList {
    type Refinement: PropertyList;
}

impl<S: PropertyList> RefinesProperties<Ø> for S {
    type Refinement = Ø;
}

impl<S, 𝒞, Tail> RefinesProperties<Cons<Property<𝒞>, Tail>> for S
where
    S: PropertyList + FindProperty<𝒞> + RefinesProperties<Tail>,
    𝒞: Cat,
    Tail: PropertyList,
    <S as FindProperty<𝒞>>::Found: RefineProperty<𝒞>,
{
    type Refinement = Cons<
        BindsProperty<𝒞, <<S as FindProperty<𝒞>>::Found as RefineProperty<𝒞>>::Refinement>,
        <S as RefinesProperties<Tail>>::Refinement,
    >;
}

/// A found associated dependency satisfies a canonical requirement when its
/// reflected role is the same required trait/category and its concrete value can
/// itself be reflected in that role.
trait SatisfiesAssoc<Role: Cat>: AssocEntry {}

impl<Name, ActualRole, Value, RequiredRole> SatisfiesAssoc<RequiredRole>
    for Binds<Name, ActualRole, Value>
where
    Name: AssocName,
    ActualRole: Cat + Compare<RequiredRole, Relation = Same>,
    RequiredRole: Cat,
    Value: Reflect<RequiredRole>,
{
}

/// Resolve every required associated dependency, retaining the actual source binding.
///
/// Matching is by associated-type name, never by declaration position.
pub trait RefinesStructure<Target: AssocList>: AssocList {
    type Refinement: AssocList;
}

impl<S: AssocList> RefinesStructure<Ø> for S {
    type Refinement = Ø;
}

impl<S, Name, Role, Tail> RefinesStructure<Cons<Requires<Name, Role>, Tail>> for S
where
    S: AssocList + FindAssoc<Name> + RefinesStructure<Tail>,
    Name: AssocName,
    Role: Cat,
    Tail: AssocList,
    <S as FindAssoc<Name>>::Found: SatisfiesAssoc<Role>,
{
    type Refinement =
        Cons<<S as FindAssoc<Name>>::Found, <S as RefinesStructure<Tail>>::Refinement>;
}

/// Structural refinement between concrete category signatures.
///
/// The result is the relevant resolved graph: target associated requirements become
/// concrete source bindings, target properties become nested resolved property
/// graphs, and target equations are retained after being proved against the source.
pub trait RefinesCategory<Target: Category>: Category {
    type Refinement: Category;
}

impl<SS, SP, SE, TS, TP, TE> RefinesCategory<𝒯<TS, TP, TE>> for 𝒯<SS, SP, SE>
where
    SS: AssocList + RefinesStructure<TS>,
    SP: PropertyList + RefinesProperties<TP>,
    SE: EquationList,
    TS: AssocList,
    TP: PropertyList,
    TE: EquationList,
    𝒯<SS, SP, SE>: SatisfiesEquations<TE>,
{
    type Refinement = 𝒯<
        <SS as RefinesStructure<TS>>::Refinement,
        <SP as RefinesProperties<TP>>::Refinement,
        TE,
    >;
}

/// Query a concrete reflected graph for its `𝒞`-shaped resolved subgraph.
pub trait Refines<𝒞: Cat>: Category {
    type Refinement: Category;
}

impl<𝒞, C> Refines<𝒞> for C
where
    𝒞: Cat,
    C: Category + RefinesCategory<𝒞::C>,
{
    type Refinement = <C as RefinesCategory<𝒞::C>>::Refinement;
}

/// The public shorthand for reflecting `X` as `𝒞` and returning the resolved graph.
#[allow(type_alias_bounds)]
pub type Refine<𝒞: Cat, X: Reflect<𝒞>> = <<X as Reflect<𝒞>>::C as Refines<𝒞>>::Refinement;

// -----------------------------------------------------------------------------
// Reflection of concrete Rust trait implementations
// -----------------------------------------------------------------------------

/// Reflect a concrete implementation of trait/category `𝒞` into the ontology.
///
/// This is the trust boundary between ordinary Rust trait implementation and the
/// compile-time category database.  The resulting signature must itself refine the
/// canonical signature of `𝒞`.
pub trait Reflect<𝒞: Cat> {
    type C: Category + Refines<𝒞>;
}

impl<N: Nat> Reflect<𝐍𝐚𝐭> for N {
    type C = <𝐍𝐚𝐭 as Cat>::C;
}

impl<T: Field> Reflect<𝐅𝐥𝐝> for T {
    type C = 𝒯<
        Cons<Binds<Fixed, 𝐂𝐅𝐥𝐝, T::Fixed>, Cons<Binds<Characteristic, 𝐍𝐚𝐭, T::Characteristic>, Ø>>,
        properties![𝐑𝐢𝐧𝐠, 𝐆𝐫𝐩],
        Cons<Equal<Follow<At<Fixed>, Fixed>, At<Fixed>>, Ø>,
    >;
}

impl<T: CField> Reflect<𝐂𝐅𝐥𝐝> for T {
    type C = 𝒯<
        Cons<Binds<Fixed, 𝐂𝐅𝐥𝐝, T::Fixed>, Cons<Binds<Characteristic, 𝐍𝐚𝐭, T::Characteristic>, Ø>>,
        Cons<BindsProperty<𝐅𝐥𝐝, <T as Reflect<𝐅𝐥𝐝>>::C>, Cons<Property<𝐀𝐛>, Ø>>,
        Cons<Equal<Follow<At<Fixed>, Fixed>, At<Fixed>>, Ø>,
    >;
}

impl<T: Tensor> Reflect<𝐓𝐞𝐧𝐬> for T {
    type C = 𝒯<Cons<Binds<F, 𝐅𝐥𝐝, T::F>, Ø>, properties![𝐂𝐌𝐨𝐧]>;
}

impl<V: Vector> Reflect<𝐕𝐞𝐜𝐭> for V {
    type C = 𝒯<Ø, Cons<BindsProperty<𝐓𝐞𝐧𝐬, <V as Reflect<𝐓𝐞𝐧𝐬>>::C>, Cons<Property<𝐆𝐫𝐩>, Ø>>>;
}

// -----------------------------------------------------------------------------
// Value-level equivalence remains separate from structural refinement
// -----------------------------------------------------------------------------

/// Reversible equivalence of concrete objects as observed in category `𝒞`.
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

#[allow(unused)]
fn test_this_whole_thing_baby() {
    type V = Refine<𝐕𝐞𝐜𝐭, Coords<f64, 2>>;
    type T = <V as PropertyRefinement<𝐓𝐞𝐧𝐬>>::Refinement;
    type Scalar = <T as Associated<F>>::Type;

    fn assert_same_type<T>(_: T, _: T) {}

    assert_same_type(PhantomData::<Scalar>, PhantomData::<f64>);

    type Fld = Refine<𝐅𝐥𝐝, f64>;
    type Fixed1 = <Fld as Associated<Fixed>>::Type;
    type FixedGraph = Refine<𝐂𝐅𝐥𝐝, Fixed1>;
    // then follow Fixed again and assert equality
}
