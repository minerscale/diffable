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

/// Admit a concrete Rust type into the ontology with its richest structural context.
///
/// This is the object-level trust boundary. An object is admitted once; every
/// weaker category it inhabits is then derived structurally through [`Ob`].
pub trait Object {
    type Context: Category;
}

/// The proposition that `Self` is an object of structural category `C`.
///
/// Objecthood is derived from the single context admitted by [`Object`]. Thus a
/// value admitted with a rich context is automatically an object of every weaker
/// category refined by that context.
pub trait Ob<C: Category>: Object {}

impl<X: Object<Context: RefinesCategory<C>>, C: Category> Ob<C> for X {}

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
pub struct Requires<Name: AssocName, 𝒞: Cat>(PhantomData<(Name, 𝒞)>);

/// A concrete associated-type binding.
///
/// Reflection of a concrete trait implementation replaces a canonical
/// [`Requires`] entry by `Binds<Name, Role, Value>`.
#[derive(Debug, Copy, Clone)]
pub struct Binds<Name: AssocName, 𝒞: Cat, Value>(PhantomData<(Name, 𝒞, Value)>);

/// A concrete binding whose role is witnessed by an explicit structural context.
#[derive(Debug, Copy, Clone)]
pub struct BindsAs<Name: AssocName, 𝒞: Cat, Value, Context: Category>(
    PhantomData<(Name, 𝒞, Value, Context)>,
);

/// The signature of an arrow in the structural category `C`.
///
/// An arrow is typed all at once:
///
/// ```text
/// D -> E
/// ```
///
/// where both `D` and `E` are objects of `C`. Its domain and codomain are
/// therefore not represented as independent associated dependencies; together
/// they form the single signature which inhabits the [`Typing`] role.
///
/// `ArrowSignature<C, D, E>` is the concrete payload bound by
/// [`BindsTyping`]. The [`Signature`] trait exposes `D` and `E` merely as
/// projections of that one typing.
///
/// The `C` parameter records the category in which the signature is
/// interpreted. Objecthood of `D` and `E` in `C` is established at the
/// [`BindsTyping`] boundary rather than by `Signature` itself.
#[derive(Debug, Copy, Clone)]
pub struct ArrowSignature<C: Category, D, E>(PhantomData<fn() -> (C, D, E)>);

/// Project the domain and codomain from an arrow signature.
///
/// `Signature` describes the shape of a typing: it has a domain and a
/// codomain. It does not by itself assert that either projection is an object
/// of any particular category; that proof belongs to the structural binding
/// which admits the signature.
pub trait Signature {
    type Domain;
    type Codomain;
}

impl<C: Category, D, E> Signature for ArrowSignature<C, D, E> {
    type Domain = D;
    type Codomain = E;
}

/// Bind the [`Typing`] of an arrow in `C`.
///
/// This is the admission boundary for an arrow signature. Unlike the
/// projection-only [`Signature`] trait, a `BindsTyping<C, D, E>` certifies that
/// both `D` and `E` are objects of `C`, so an arrow can never acquire a domain
/// independently of its codomain (or vice versa).
///
/// Its associated value is the single [`ArrowSignature<C, D, E>`] from which
/// domain and codomain may subsequently be projected.
#[derive(Debug, Copy, Clone)]
pub struct BindsTyping<C: Category + 'static, D: Ob<C>, E: Ob<C>>(PhantomData<fn() -> (C, D, E)>);

/// Placeholder used as [`AssocEntry::Value`] by a canonical requirement.
#[derive(Debug, Copy, Clone)]
pub struct Unspecified;

pub trait AssocEntry: sealed::AssocEntry {
    type Name: AssocName;
    type Role: Cat;
    type Value;
}

impl<𝒞: Cat, N: AssocName> sealed::AssocEntry for Requires<N, 𝒞> {}
impl<𝒞: Cat, N: AssocName> AssocEntry for Requires<N, 𝒞> {
    type Name = N;
    type Role = 𝒞;
    type Value = Unspecified;
}

impl<𝒞: Cat, N: AssocName, V> sealed::AssocEntry for Binds<N, 𝒞, V> {}
impl<𝒞: Cat, N: AssocName, V> AssocEntry for Binds<N, 𝒞, V> {
    type Name = N;
    type Role = 𝒞;
    type Value = V;
}

impl<𝒞: Cat, N: AssocName, V, C: Category> sealed::AssocEntry for BindsAs<N, 𝒞, V, C> {}
impl<𝒞: Cat, N: AssocName, V, C: Category> AssocEntry for BindsAs<N, 𝒞, V, C> {
    type Name = N;
    type Role = 𝒞;
    type Value = V;
}

impl<C: Category + 'static, D: Ob<C>, E: Ob<C>> sealed::AssocEntry for BindsTyping<C, D, E> {}

impl<C: Category + 'static, D: Ob<C>, E: Ob<C>> AssocEntry for BindsTyping<C, D, E> {
    type Name = Typing;
    type Role = 𝐓𝐲𝐩𝐢𝐧𝐠<C>;
    type Value = ArrowSignature<C, D, E>;
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
assoc_names!(F, Fixed, Characteristic, Tangent, Typing, From, To);

/// Find a structural dependency by associated-type name.
///
/// The list is logically a record, not a tuple: lookup ignores declaration order.
pub trait FindAssoc<Name: AssocName>: AssocList {
    type Found: AssocEntry<Name = Name>;
}

pub trait FindAssocWith<Name: AssocName, Relation>: AssocList {
    type Found: AssocEntry<Name = Name>;
}

impl<Name: AssocName, Head: AssocEntry<Name = Name>, Tail: AssocList> FindAssocWith<Name, Same>
    for Cons<Head, Tail>
{
    type Found = Head;
}

impl<Name: AssocName, Head: AssocEntry, Tail: AssocList + FindAssoc<Name>>
    FindAssocWith<Name, Different> for Cons<Head, Tail>
{
    type Found = <Tail as FindAssoc<Name>>::Found;
}

impl<Name: AssocName, Head: AssocEntry<Name: CompareAssoc<Name>>, Tail: AssocList> FindAssoc<Name>
    for Cons<Head, Tail>
where
    Cons<Head, Tail>: FindAssocWith<Name, <Head::Name as CompareAssoc<Name>>::Relation>,
{
    type Found = <Cons<Head, Tail> as FindAssocWith<
        Name,
        <Head::Name as CompareAssoc<Name>>::Relation,
    >>::Found;
}

/// Project a reflected associated dependency by its actual trait name.
pub trait Ⱶ<Name: AssocName>: Category {
    type 𝒞: Cat;
    type X;
}

impl<Name: AssocName, S: AssocList + FindAssoc<Name>, P: PropertyList, E: EquationList> Ⱶ<Name>
    for 𝒯<S, P, E>
{
    type 𝒞 = <<S as FindAssoc<Name>>::Found as AssocEntry>::Role;
    type X = <<S as FindAssoc<Name>>::Found as AssocEntry>::Value;
}

// -----------------------------------------------------------------------------
// Properties: unordered inherited structure
// -----------------------------------------------------------------------------

/// A resolved property edge.
///
/// `Context` is the concrete/refined graph which satisfies `𝒞`. Unlike a bare
/// category name in a [`PropertyList`], this retains the structural information
/// discovered while resolving the property, including associated bindings inherited
/// through it.
#[derive(Debug, Copy, Clone)]
pub struct BindsProperty<𝒞: Cat, Context: Category>(PhantomData<(𝒞, Context)>);

pub trait PropertyEntry: sealed::PropertyEntry {
    type Role: Cat;
}

impl<𝒞: Cat> sealed::PropertyEntry for 𝒞 {}
impl<𝒞: Cat> PropertyEntry for 𝒞 {
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

/// Append two type-level property lists.
pub trait AppendProperties<Rhs: PropertyList>: PropertyList {
    type Output: PropertyList;
}

impl<Rhs: PropertyList> AppendProperties<Rhs> for Ø {
    type Output = Rhs;
}

impl<Rhs: PropertyList, Head: PropertyEntry, Tail: PropertyList + AppendProperties<Rhs>>
    AppendProperties<Rhs> for Cons<Head, Tail>
{
    type Output = Cons<Head, <Tail as AppendProperties<Rhs>>::Output>;
}

/// The transitive properties supplied by one property edge.
pub trait ExpandProperty: PropertyEntry {
    type Expansion: PropertyList;
}

impl<𝒞: Cat<C: Category<Properties: ExpandProperties>>> ExpandProperty for 𝒞 {
    type Expansion = <<𝒞::C as Category>::Properties as ExpandProperties>::Expansion;
}

impl<𝒞: Cat, Context: Category<Properties: ExpandProperties>> ExpandProperty
    for BindsProperty<𝒞, Context>
{
    type Expansion = <<Context as Category>::Properties as ExpandProperties>::Expansion;
}

/// Flatten the transitive closure of a property graph.
///
/// Every direct property remains present, followed by all properties reachable
/// through its resolved context.
pub trait ExpandProperties: PropertyList {
    type Expansion: PropertyList;
}

impl ExpandProperties for Ø {
    type Expansion = Ø;
}

impl<
    Head: PropertyEntry + ExpandProperty<Expansion: AppendProperties<Tail::Expansion>>,
    Tail: PropertyList + ExpandProperties,
> ExpandProperties for Cons<Head, Tail>
{
    type Expansion = Cons<
        Head,
        <<Head as ExpandProperty>::Expansion as AppendProperties<
            <Tail as ExpandProperties>::Expansion,
        >>::Output,
    >;
}

/// Find a property edge by reflected category name.
pub trait FindProperty<𝒞: Cat>: PropertyList {
    type Found: PropertyEntry;
}

pub trait FindPropertyWith<𝒞: Cat, Relation>: PropertyList {
    type Found: PropertyEntry;
}

impl<𝒞: Cat, Head: PropertyEntry<Role: Compare<𝒞, Relation = Same>>, Tail: PropertyList>
    FindPropertyWith<𝒞, Same> for Cons<Head, Tail>
{
    type Found = Head;
}

impl<
    𝒞: Cat,
    Head: PropertyEntry<Role: Compare<𝒞, Relation = Different>>,
    Tail: PropertyList + FindProperty<𝒞>,
> FindPropertyWith<𝒞, Different> for Cons<Head, Tail>
{
    type Found = <Tail as FindProperty<𝒞>>::Found;
}

impl<𝒞: Cat, Head: PropertyEntry<Role: Compare<𝒞>>, Tail: PropertyList> FindProperty<𝒞>
    for Cons<Head, Tail>
where
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

impl<𝒞: Cat, 𝒟: Cat + Compare<𝒞, Relation = Same>, Context: Category + π<𝒞>> ResolvedProperty<𝒞>
    for BindsProperty<𝒟, Context>
{
    type Refinement = <Context as π<𝒞>>::C;
}

impl<
    𝒞: Cat,
    S: AssocList,
    P: PropertyList + FindProperty<𝒞, Found: ResolvedProperty<𝒞>>,
    E: EquationList,
> PropertyRefinement<𝒞> for 𝒯<S, P, E>
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

impl<Name: AssocName, S: AssocList + FindAssoc<Name>, P: PropertyList, E: EquationList>
    ResolvePath<At<Name>> for 𝒯<S, P, E>
{
    type Output = <Self as Ⱶ<Name>>::X;
}

impl<Path, Name: AssocName, S: AssocList, P: PropertyList, E: EquationList>
    ResolvePath<Follow<Path, Name>> for 𝒯<S, P, E>
where
    Self: ResolvePath<Path, Output: Project<Name>>,
{
    type Output = <<Self as ResolvePath<Path>>::Output as Project<Name>>::Output;
}

/// Type-equality witness used to hand graph equations back to rustc.
pub trait SameType<Rhs> {}
impl<T> SameType<T> for T {}

trait SatisfiesEquation<Eq>: Category {}

impl<
    Left,
    Right,
    C: Category
        + ResolvePath<Right>
        + ResolvePath<Left, Output: SameType<<C as ResolvePath<Right>>::Output>>,
> SatisfiesEquation<Equal<Left, Right>> for C
{
}

trait SatisfiesEquations<Equations: EquationList>: Category {}

impl<C: Category> SatisfiesEquations<Ø> for C {}

impl<
    Left,
    Right,
    Tail: EquationList,
    C: Category + SatisfiesEquation<Equal<Left, Right>> + SatisfiesEquations<Tail>,
> SatisfiesEquations<Cons<Equal<Left, Right>, Tail>> for C
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
        Cons<$head, properties!($($tail),*)>
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

// -----------------------------------------------------------------------------
// Higher-order category constructors
// -----------------------------------------------------------------------------

/// The theory of an object carrying structural category `C`.
pub struct 𝐈𝐝<C: Category + 'static>(PhantomData<fn() -> C>);

impl<C: Category + 'static> Copy for 𝐈𝐝<C> {}
impl<C: Category + 'static> Clone for 𝐈𝐝<C> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<C: Category + 'static> Debug for 𝐈𝐝<C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("𝐈𝐝")
    }
}
impl<C: Category + 'static> Cat for 𝐈𝐝<C> {
    type C = C;
}

/// The theory of arrows in structural category `C`.
pub struct 𝐀𝐫𝐫<C: Category + 'static>(PhantomData<fn() -> C>);

impl<C: Category + 'static> Copy for 𝐀𝐫𝐫<C> {}
impl<C: Category + 'static> Clone for 𝐀𝐫𝐫<C> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<C: Category + 'static> Debug for 𝐀𝐫𝐫<C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("𝐀𝐫𝐫")
    }
}

/// The theory of arrow typings in `C`.
///
/// A typing consists of a domain and codomain which are both objects of `C`.
/// It represents the inseparable judgement `D -> E`, rather than two
/// independently available associated dependencies.
pub struct 𝐓𝐲𝐩𝐢𝐧𝐠<C: Category + 'static>(PhantomData<fn() -> C>);

impl<C: Category + 'static> Copy for 𝐓𝐲𝐩𝐢𝐧𝐠<C> {}
impl<C: Category + 'static> Clone for 𝐓𝐲𝐩𝐢𝐧𝐠<C> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<C: Category + 'static> Debug for 𝐓𝐲𝐩𝐢𝐧𝐠<C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("𝐓𝐲𝐩𝐢𝐧𝐠")
    }
}
impl<C: Category + 'static> Cat for 𝐓𝐲𝐩𝐢𝐧𝐠<C> {
    type C = cat![(From: 𝐈𝐝<C>, To: 𝐈𝐝<C>), {}];
}

impl<C: Category + 'static> Cat for 𝐀𝐫𝐫<C> {
    type C = cat![(Typing: 𝐓𝐲𝐩𝐢𝐧𝐠<C>), {}];
}

/// The theory of homotopies between arrows in `C`.
pub struct 𝐇𝐨𝐦𝐨𝐭𝐨𝐩𝐲<C: Category + 'static>(PhantomData<fn() -> C>);

impl<C: Category + 'static> Copy for 𝐇𝐨𝐦𝐨𝐭𝐨𝐩𝐲<C> {}
impl<C: Category + 'static> Clone for 𝐇𝐨𝐦𝐨𝐭𝐨𝐩𝐲<C> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<C: Category + 'static> Debug for 𝐇𝐨𝐦𝐨𝐭𝐨𝐩𝐲<C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("𝐇𝐨𝐦𝐨𝐭𝐨𝐩𝐲")
    }
}
impl<C: Category + 'static> Cat for 𝐇𝐨𝐦𝐨𝐭𝐨𝐩𝐲<C> {
    type C = cat![(From: 𝐀𝐫𝐫<C>, To: 𝐀𝐫𝐫<C>), {}];
}

pub type C<𝒞> = <𝒞 as Cat>::C;

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
            impl<N: Nat + NatCompare<M>, M: Nat> Compare<$family<M>> for $family<N>
            where
                $family<N>: Cat,
                $family<M>: Cat,
            {
                type Relation = <N as NatCompare<M>>::Relation;
            }

            impl<N: Nat, M: Nat> Compare<𝐂𝐚𝐭<M>> for $family<N>
            where
                $family<N>: Cat,
                𝐂𝐚𝐭<M>: Cat,
            {
                type Relation = Different;
            }

            impl<N: Nat, M: Nat> Compare<$family<M>> for 𝐂𝐚𝐭<N>
            where
                𝐂𝐚𝐭<N>: Cat,
                $family<M>: Cat,
            {
                type Relation = Different;
            }
        )*

        impl<N: Nat + NatCompare<M>, M: Nat> Compare<𝐂𝐚𝐭<M>> for 𝐂𝐚𝐭<N>
        where
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
    𝐌𝐚𝐧      => cat![(Tangent: 𝐓𝐞𝐧𝐬), {𝐓𝐨𝐩}];

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
/// category name can resolve itself when its canonical signature is already concrete
/// enough to refine that property; this handles ordinary inherited structure whose
/// canonical graph has no unresolved associated bindings.
pub trait RefineProperty<𝒞: Cat>: PropertyEntry {
    type Refinement: Category;
}

impl<𝒞: Cat, 𝒟: Cat + Compare<𝒞, Relation = Same>, Context: Category + π<𝒞>> RefineProperty<𝒞>
    for BindsProperty<𝒟, Context>
{
    type Refinement = <Context as π<𝒞>>::C;
}

impl<𝒞: Cat, 𝒟: Cat<C: π<𝒞>> + Compare<𝒞, Relation = Same>> RefineProperty<𝒞> for 𝒟 {
    type Refinement = <𝒟::C as π<𝒞>>::C;
}

/// Resolve every required property in `Target`, retaining the graph found for each.
pub trait RefinesProperties<Target: PropertyList>: PropertyList {
    type Refinement: PropertyList;
}

impl<S: PropertyList> RefinesProperties<Ø> for S {
    type Refinement = Ø;
}

impl<
    𝒞: Cat,
    Tail: PropertyList,
    S: PropertyList
        + ExpandProperties<Expansion: FindProperty<𝒞, Found: RefineProperty<𝒞>>>
        + RefinesProperties<Tail>,
> RefinesProperties<Cons<𝒞, Tail>> for S
{
    type Refinement = Cons<
        BindsProperty<
            𝒞,
            <<<S as ExpandProperties>::Expansion as FindProperty<𝒞>>::Found as RefineProperty<
                𝒞,
            >>::Refinement,
        >,
        <S as RefinesProperties<Tail>>::Refinement,
    >;
}

/// A found associated dependency satisfies a canonical requirement when its
/// reflected role is the same required trait/category and its concrete value can
/// itself be reflected in that role.
trait SatisfiesAssoc<𝒞: Cat>: AssocEntry {}

impl<𝒞: Cat, 𝒟: Cat + Compare<𝒞, Relation = Same>, Name: AssocName, Value: Reflect<𝒞>>
    SatisfiesAssoc<𝒞> for Binds<Name, 𝒟, Value>
{
}

impl<
    Name: AssocName,
    Required: Category + 'static,
    Actual: Category + RefinesCategory<Required> + 'static,
    Context: Category,
    Value: Object<Context = Context> + Ob<Actual>,
> SatisfiesAssoc<𝐈𝐝<Required>> for BindsAs<Name, 𝐈𝐝<Actual>, Value, Context>
{
}

impl<
    Required: Category + 'static,
    Actual: Category + RefinesCategory<Required> + 'static,
    D: Ob<Actual>,
    E: Ob<Actual>,
> SatisfiesAssoc<𝐓𝐲𝐩𝐢𝐧𝐠<Required>> for BindsTyping<Actual, D, E>
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

impl<
    𝒞: Cat,
    Name: AssocName,
    Tail: AssocList,
    S: AssocList + FindAssoc<Name, Found: SatisfiesAssoc<𝒞>> + RefinesStructure<Tail>,
> RefinesStructure<Cons<Requires<Name, 𝒞>, Tail>> for S
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
    type C: Category;
}

impl<
    TS: AssocList,
    TP: PropertyList,
    TE: EquationList,
    SS: AssocList + RefinesStructure<TS>,
    SP: PropertyList + RefinesProperties<TP>,
    SE: EquationList,
> RefinesCategory<𝒯<TS, TP, TE>> for 𝒯<SS, SP, SE>
where
    𝒯<SS, SP, SE>: SatisfiesEquations<TE>,
{
    type C = 𝒯<
        <SS as RefinesStructure<TS>>::Refinement,
        <SP as RefinesProperties<TP>>::Refinement,
        TE,
    >;
}

/// Query a concrete reflected graph for its `𝒞`-shaped resolved subgraph.
#[allow(non_camel_case_types)]
pub trait π<𝒞: Cat>: Category {
    type C: Category;
}

impl<𝒞: Cat, C: Category + RefinesCategory<𝒞::C>> π<𝒞> for C {
    type C = <C as RefinesCategory<𝒞::C>>::C;
}

/// The public shorthand for reflecting `X` as `𝒞` and returning the resolved graph.
#[allow(type_alias_bounds)]
pub type Refine<𝒞: Cat, X: Reflect<𝒞>> = <<X as Reflect<𝒞>>::C as π<𝒞>>::C;

// -----------------------------------------------------------------------------
// Reflection of concrete Rust trait implementations
// -----------------------------------------------------------------------------

/// Reflect a concrete implementation of trait/category `𝒞` into the ontology.
///
/// This is the trust boundary between ordinary Rust trait implementation and the
/// compile-time category database.  The resulting signature must itself refine the
/// canonical signature of `𝒞`.
pub trait Reflect<𝒞: Cat> {
    type C: Category + π<𝒞>;
}

impl<N: Nat> Reflect<𝐍𝐚𝐭> for N {
    type C = C<𝐍𝐚𝐭>;
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
        Cons<BindsProperty<𝐅𝐥𝐝, <T as Reflect<𝐅𝐥𝐝>>::C>, Cons<𝐀𝐛, Ø>>,
        Cons<Equal<Follow<At<Fixed>, Fixed>, At<Fixed>>, Ø>,
    >;
}

impl<T: Tensor> Reflect<𝐓𝐞𝐧𝐬> for T {
    type C = 𝒯<Cons<Binds<F, 𝐅𝐥𝐝, T::F>, Ø>, properties![𝐂𝐌𝐨𝐧]>;
}

impl<V: Vector> Reflect<𝐕𝐞𝐜𝐭> for V {
    type C = 𝒯<Ø, Cons<BindsProperty<𝐓𝐞𝐧𝐬, <V as Reflect<𝐓𝐞𝐧𝐬>>::C>, Cons<𝐆𝐫𝐩, Ø>>>;
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
    type Scalar = <T as Ⱶ<F>>::X;

    fn assert_same_type<T>(_: T, _: T) {}

    assert_same_type(PhantomData::<Scalar>, PhantomData::<f64>);
}
