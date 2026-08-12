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
//! information), it may satisfy the reflected trait through [`Ⱶ`] without
//! carrying its nominal label.
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
    traits::{
        CField, Field, Manifold, Nat, NatCompare, NatZero, Real, Succ, Tensor, Topological, Vector,
    },
};
use core::{
    convert::Infallible,
    fmt::Debug,
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

// -----------------------------------------------------------------------------
// Generic type-level lists
// -----------------------------------------------------------------------------

/// The empty type-level list.
#[derive(Debug, Copy, Clone)]
pub struct Ø;

/// A type-level list node with `Head` followed by `Tail`.
///
/// This deliberately remains unconstrained: calculus also uses `ː`/`Ø` for
/// type-level routes unrelated to the category ontology.
#[derive(Debug, Copy, Clone)]
pub struct ː<Head, Tail>(PhantomData<(Head, Tail)>);

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

/// A concrete structural context rooted at one Rust object and nominal category.
///
/// Canonical [`Cat::C`] values remain unrooted theories. `Rooted` is used when a
/// concrete context has an explicit body rather than the canonical
/// [`Interpretation`] body, recording which object that graph is actually about.
#[derive(Debug, Copy, Clone)]
pub struct Rooted<𝒞: Cat, X, C: Category>(PhantomData<fn() -> (𝒞, X, C)>);

impl<𝒞: Cat, X, C: Category> sealed::Category for Rooted<𝒞, X, C> {}

impl<𝒞: Cat, X, C: Category> Category for Rooted<𝒞, X, C> {
    type Structure = C::Structure;
    type Properties = C::Properties;
    type Equations = C::Equations;
}

/// The canonical concrete context of `X` interpreted in nominal category `𝒞`.
///
/// `Interpretation<𝒞, X>` is the *identity* of the context. [`Reflect::Body`] is
/// merely its elaborated structural body. Keeping those notions separate gives
/// recursive theories such as `Field::Fixed` a finite Rust type while ensuring
/// that a direct interpretation and the same ordinary child reached through
/// [`π`] are literally the same type.
#[derive(Debug, Copy, Clone)]
pub struct Interpretation<𝒞: Cat, X>(PhantomData<fn() -> (𝒞, X)>);

impl<𝒞: Cat, X: Reflect<𝒞>> sealed::Category for Interpretation<𝒞, X> {}

impl<𝒞: Cat, X: Reflect<𝒞>> Category for Interpretation<𝒞, X> {
    type Structure = <<X as Reflect<𝒞>>::Body as Category>::Structure;
    type Properties = <<X as Reflect<𝒞>>::Body as Category>::Properties;
    type Equations = <<X as Reflect<𝒞>>::Body as Category>::Equations;
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

impl<X, C> Ob<C> for X
where
    X: Object,
    C: Category + 'static,
    X::Context: Ⱶ<𝐈𝐝<C>>,
{
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
pub struct Requires<Name: AssocName, 𝒞: Cat>(PhantomData<(Name, 𝒞)>);

/// A concrete associated-type binding.
///
/// `Context` is the *entire child subcontext owned by the parent*. Projection
/// never reconstructs it from `Value`: [`π::C`] simply selects this stored graph.
/// The role `𝒞` records the nominal interface under which the edge is exposed,
/// while `Context` may carry strictly richer information.
#[derive(Debug, Copy, Clone)]
pub struct Binds<Name: AssocName, 𝒞: Cat, Value, Context: Category>(
    PhantomData<(Name, 𝒞, Value, Context)>,
);

/// The ordinary reflected binding, whose child context is the canonical
/// reflection of `Value` in the edge role.
///
/// More informative parents should use [`Binds`] directly and provide the richer
/// child context explicitly.
#[allow(type_alias_bounds)]
pub type BindsReflected<Name: AssocName, 𝒞: Cat, Value: Reflect<𝒞>> =
    Binds<Name, 𝒞, Value, Interpretation<𝒞, Value>>;

/// A concrete binding whose child context is supplied explicitly.
///
/// Unlike [`BindsReflected`], this preserves parent-specific information which
/// is not recoverable from the child's ordinary nominal reflection.
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
    type C: Category;
    type Value;
}

impl<𝒞: Cat, N: AssocName> sealed::AssocEntry for Requires<N, 𝒞> {}
impl<𝒞: Cat, N: AssocName> AssocEntry for Requires<N, 𝒞> {
    type Name = N;
    type Role = 𝒞;
    type C = 𝒞::C;
    type Value = Unspecified;
}

impl<𝒞: Cat, N: AssocName, V, C: Category> sealed::AssocEntry for Binds<N, 𝒞, V, C> {}
impl<𝒞: Cat, N: AssocName, V, C: Category> AssocEntry for Binds<N, 𝒞, V, C> {
    type Name = N;
    type Role = 𝒞;
    type C = C;
    type Value = V;
}

impl<𝒞: Cat, N: AssocName, V, C: Category> sealed::AssocEntry for BindsAs<N, 𝒞, V, C> {}
impl<𝒞: Cat, N: AssocName, V, C: Category> AssocEntry for BindsAs<N, 𝒞, V, C> {
    type Name = N;
    type Role = 𝒞;
    type C = C;
    type Value = V;
}

impl<C: Category + 'static, D: Ob<C>, E: Ob<C>> sealed::AssocEntry for BindsTyping<C, D, E> {}

impl<C: Category + 'static, D: Ob<C>, E: Ob<C>> AssocEntry for BindsTyping<C, D, E> {
    type Name = Typing;
    type Role = 𝐓𝐲𝐩𝐢𝐧𝐠<C>;
    type C = Rooted<
        𝐓𝐲𝐩𝐢𝐧𝐠<C>,
        ArrowSignature<C, D, E>,
        𝒯<
            ː<
                BindsAs<From, 𝐈𝐝<C>, D, <D as Object>::Context>,
                ː<BindsAs<To, 𝐈𝐝<C>, E, <E as Object>::Context>, Ø>,
            >,
            Ø,
        >,
    >;
    type Value = ArrowSignature<C, D, E>;
}

pub trait AssocList: sealed::AssocList {}
impl sealed::AssocList for Ø {}
impl AssocList for Ø {}
impl<H: AssocEntry, T: AssocList> sealed::AssocList for ː<H, T> {}
impl<H: AssocEntry, T: AssocList> AssocList for ː<H, T> {}

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

// Most names mirror reflected Rust associated types; the payload names are
// metadata carried by reflected functor images.
assoc_names!(
    This,
    F,
    Fixed,
    Characteristic,
    Tangent,
    Typing,
    From,
    To,
    JetPayload,
    TensorPayload,
);

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
    for ː<Head, Tail>
{
    type Found = Head;
}

impl<Name: AssocName, Head: AssocEntry, Tail: AssocList + FindAssoc<Name>>
    FindAssocWith<Name, Different> for ː<Head, Tail>
{
    type Found = <Tail as FindAssoc<Name>>::Found;
}

impl<Name: AssocName, Head: AssocEntry<Name: CompareAssoc<Name>>, Tail: AssocList> FindAssoc<Name>
    for ː<Head, Tail>
where
    ː<Head, Tail>: FindAssocWith<Name, <Head::Name as CompareAssoc<Name>>::Relation>,
{
    type Found = <ː<Head, Tail> as FindAssocWith<
        Name,
        <Head::Name as CompareAssoc<Name>>::Relation,
    >>::Found;
}

/// Root metadata carried only by concrete reflected contexts.
///
/// The open canonical theory `C![𝒞]` deliberately has no root object.
/// [`Rooted`] supplies that information for a concrete reflection without
/// recursively inserting the whole category into its own structure list.
pub trait RootContext: Category {
    type 𝒞: Cat;
    type X;
}

impl<𝒞: Cat, X, C: Category> RootContext for Rooted<𝒞, X, C> {
    type 𝒞 = 𝒞;
    type X = X;
}

impl<𝒞: Cat, X: Reflect<𝒞>> RootContext for Interpretation<𝒞, X> {
    type 𝒞 = 𝒞;
    type X = X;
}

/// Internal dispatch for the identity projection (`This`) versus an ordinary
/// named child edge.
pub trait ProjectAssoc<Name: AssocName, Relation>: Category {
    type 𝒞: Cat;
    type C: Category;
    type X;
}

impl<C: RootContext> ProjectAssoc<This, Same> for C {
    type 𝒞 = C::𝒞;
    type C = C;
    type X = C::X;
}

impl<Name: AssocName, C: Category<Structure: FindAssoc<Name>>> ProjectAssoc<Name, Different> for C
where
    <C::Structure as FindAssoc<Name>>::Found: AssocEntry,
{
    type 𝒞 = <<C::Structure as FindAssoc<Name>>::Found as AssocEntry>::Role;
    type C = <<C::Structure as FindAssoc<Name>>::Found as AssocEntry>::C;
    type X = <<C::Structure as FindAssoc<Name>>::Found as AssocEntry>::Value;
}

/// Project a named child of a category context.
///
/// `π::C` is not reconstructed from `π::X`; it is the exact subcontext stored
/// on that edge by the parent. `π<This>` is the identity projection on a rooted
/// concrete context.
#[allow(non_camel_case_types)]
pub trait π<Name: AssocName = This>: Category {
    type 𝒞: Cat;
    type C: Category;
    type X;
}

impl<Name: AssocName + CompareAssoc<This>, C: Category> π<Name> for C
where
    C: ProjectAssoc<Name, <Name as CompareAssoc<This>>::Relation>,
{
    type 𝒞 = <C as ProjectAssoc<Name, <Name as CompareAssoc<This>>::Relation>>::𝒞;
    type C = <C as ProjectAssoc<Name, <Name as CompareAssoc<This>>::Relation>>::C;
    type X = <C as ProjectAssoc<Name, <Name as CompareAssoc<This>>::Relation>>::X;
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
impl<H: PropertyEntry, T: PropertyList> sealed::PropertyList for ː<H, T> {}
impl<H: PropertyEntry, T: PropertyList> PropertyList for ː<H, T> {}

/// Append two type-level property lists.
pub trait AppendProperties<Rhs: PropertyList>: PropertyList {
    type Output: PropertyList;
}

impl<Rhs: PropertyList> AppendProperties<Rhs> for Ø {
    type Output = Rhs;
}

impl<Rhs: PropertyList, Head: PropertyEntry, Tail: PropertyList + AppendProperties<Rhs>>
    AppendProperties<Rhs> for ː<Head, Tail>
{
    type Output = ː<Head, <Tail as AppendProperties<Rhs>>::Output>;
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
> ExpandProperties for ː<Head, Tail>
{
    type Expansion = ː<
        Head,
        <<Head as ExpandProperty>::Expansion as AppendProperties<
            <Tail as ExpandProperties>::Expansion,
        >>::Output,
    >;
}

/// Type-level result of a total property-membership query.
#[derive(Debug, Copy, Clone)]
pub struct Present;
/// Type-level result proving that a property is absent from a closed property graph.
#[derive(Debug, Copy, Clone)]
pub struct Absent;

/// Decide whether an already-expanded property list contains `𝒞`.
///
/// Unlike [`FindProperty`], this query is total: failure to find the property is
/// represented by [`Absent`] rather than by failure of trait resolution.  This
/// makes negative facts available to ordinary stable-Rust coherence.
pub trait PropertyPresence<𝒞: Cat>: PropertyList {
    type Relation;
}

pub trait PropertyPresenceWith<𝒞: Cat, Relation>: PropertyList {
    type Output;
}

impl<𝒞: Cat> PropertyPresence<𝒞> for Ø {
    type Relation = Absent;
}

impl<𝒞: Cat, Head: PropertyEntry<Role: Compare<𝒞, Relation = Same>>, Tail: PropertyList>
    PropertyPresenceWith<𝒞, Same> for ː<Head, Tail>
{
    type Output = Present;
}

impl<
    𝒞: Cat,
    Head: PropertyEntry<Role: Compare<𝒞, Relation = Different>>,
    Tail: PropertyList + PropertyPresence<𝒞>,
> PropertyPresenceWith<𝒞, Different> for ː<Head, Tail>
{
    type Output = <Tail as PropertyPresence<𝒞>>::Relation;
}

impl<𝒞: Cat, Head: PropertyEntry<Role: Compare<𝒞>>, Tail: PropertyList> PropertyPresence<𝒞>
    for ː<Head, Tail>
where
    ː<Head, Tail>: PropertyPresenceWith<𝒞, <Head::Role as Compare<𝒞>>::Relation>,
{
    type Relation = <ː<Head, Tail> as PropertyPresenceWith<
        𝒞,
        <Head::Role as Compare<𝒞>>::Relation,
    >>::Output;
}

/// Decide whether the transitive property graph of a concrete category contains `𝒞`.
///
/// Because [`Category`] graphs are closed compile-time data, `Relation = Absent`
/// is a constructive negative fact rather than merely a failed search.  The same
/// associated projection can therefore be constrained to `Present` or `Absent`
/// in competing impls, giving rustc visibly disjoint coherence regions.
pub trait HasProperty<𝒞: Cat>: Category {
    type Relation;
}

impl<𝒞: Cat, C: Category<Properties: ExpandProperties>> HasProperty<𝒞> for C
where
    <C::Properties as ExpandProperties>::Expansion: PropertyPresence<𝒞>,
{
    type Relation =
        <<C::Properties as ExpandProperties>::Expansion as PropertyPresence<𝒞>>::Relation;
}

/// Find a property edge by reflected category name.
pub trait FindProperty<𝒞: Cat>: PropertyList {
    type Found: PropertyEntry;
}

pub trait FindPropertyWith<𝒞: Cat, Relation>: PropertyList {
    type Found: PropertyEntry;
}

impl<𝒞: Cat, Head: PropertyEntry<Role: Compare<𝒞, Relation = Same>>, Tail: PropertyList>
    FindPropertyWith<𝒞, Same> for ː<Head, Tail>
{
    type Found = Head;
}

impl<
    𝒞: Cat,
    Head: PropertyEntry<Role: Compare<𝒞, Relation = Different>>,
    Tail: PropertyList + FindProperty<𝒞>,
> FindPropertyWith<𝒞, Different> for ː<Head, Tail>
{
    type Found = <Tail as FindProperty<𝒞>>::Found;
}

impl<𝒞: Cat, Head: PropertyEntry<Role: Compare<𝒞>>, Tail: PropertyList> FindProperty<𝒞>
    for ː<Head, Tail>
where
    ː<Head, Tail>: FindPropertyWith<𝒞, <Head::Role as Compare<𝒞>>::Relation>,
{
    type Found =
        <ː<Head, Tail> as FindPropertyWith<𝒞, <Head::Role as Compare<𝒞>>::Relation>>::Found;
}

/// Project the resolved graph which supplies property `𝒞`.
pub trait PropertyRefinement<𝒞: Cat>: Category {
    type Refinement: Category;
}

pub trait ResolvedProperty<𝒞: Cat>: PropertyEntry {
    type Refinement: Category;
}

impl<𝒞: Cat, 𝒟: Cat + Compare<𝒞, Relation = Same>, Context: Ⱶ<𝒞>> ResolvedProperty<𝒞>
    for BindsProperty<𝒟, Context>
{
    // Projection preserves the exact context carried by the property edge.
    // `Ⱶ<𝒞>` proves that this context supplies the requested property; it does
    // not replace the parent's knowledge with a weaker proof-shaped view.
    type Refinement = Context;
}

impl<𝒞: Cat, C: Category<Properties: ExpandProperties>> PropertyRefinement<𝒞> for C
where
    <C::Properties as ExpandProperties>::Expansion: FindProperty<𝒞, Found: ResolvedProperty<𝒞>>,
{
    type Refinement = <<<C::Properties as ExpandProperties>::Expansion as FindProperty<
        𝒞,
    >>::Found as ResolvedProperty<𝒞>>::Refinement;
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

impl<L, R, T: EquationList> sealed::EquationList for ː<Equal<L, R>, T> {}
impl<L, R, T: EquationList> EquationList for ː<Equal<L, R>, T> {}

#[allow(type_alias_bounds)]
pub type Project<T: Reflect<𝒞>, 𝒞: Cat, Name: AssocName> = <Interpretation<𝒞, T> as π<Name>>::X;

/// Resolve a path against a concrete reflected category.
///
/// The first hop is read from the category's labelled [`Binds`] record,
/// yielding the bound Rust type, nominal role, and complete stored child context.
/// Further hops continue directly inside that child context; path resolution never
/// reconstructs or re-reflects an intermediate object.
///
/// Path equality therefore reduces ultimately to ordinary Rust type equality,
/// while every intermediate hop retains exactly the structural context selected
/// from its parent.
pub trait ResolvePath<Path>: Category {
    type 𝒞: Cat;
    type C: Category;
    type Output;
}

impl<Name: AssocName, C: Category + π<Name>> ResolvePath<At<Name>> for C {
    type 𝒞 = <C as π<Name>>::𝒞;
    type C = <C as π<Name>>::C;
    type Output = <C as π<Name>>::X;
}

impl<Path, Name: AssocName, C: Category + ResolvePath<Path>> ResolvePath<Follow<Path, Name>> for C
where
    <C as ResolvePath<Path>>::C: π<Name>,
{
    type 𝒞 = <<C as ResolvePath<Path>>::C as π<Name>>::𝒞;
    type C = <<C as ResolvePath<Path>>::C as π<Name>>::C;
    type Output = <<C as ResolvePath<Path>>::C as π<Name>>::X;
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
> SatisfiesEquations<ː<Equal<Left, Right>, Tail>> for C
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
        ː<
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
        ː<$head, properties!($($tail),*)>
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
        ː<$head, equations!($($tail),*)>
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

// `𝐈𝐝<C>` embeds an arbitrary concrete category into the nominal `Cat` world.
// It is never an inherited property of an ordinary named theory, so ordinary
// atomic properties compare different from it. Exact identity categories compare
// equal only when their concrete category parameter is literally the same type.
impl<C: Category + 'static> Compare<𝐈𝐝<C>> for 𝐈𝐝<C> {
    type Relation = Same;
}

impl<𝒞: Atom, C: Category + 'static> Compare<𝐈𝐝<C>> for 𝒞 {
    type Relation = Different;
}

/// Refine `𝒞` with one associated object carrying functor-specific metadata.
///
/// The original structural signature is retained verbatim; `Name` is simply
/// prepended as one additional associated dependency whose value is itself an
/// object of `𝒞`.  This is useful for functor images whose concrete Rust representation remains
/// available operationally while the ontology records the source object and the
/// complete mathematical structure carried by the image.
pub struct WithPayload<𝒞: Cat, Name: AssocName>(PhantomData<fn() -> (𝒞, Name)>);

impl<𝒞: Cat, Name: AssocName> Copy for WithPayload<𝒞, Name> {}
impl<𝒞: Cat, Name: AssocName> Clone for WithPayload<𝒞, Name> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<𝒞: Cat, Name: AssocName> Debug for WithPayload<𝒞, Name> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("WithPayload")
    }
}
impl<𝒞: Cat, Name: AssocName> Cat for WithPayload<𝒞, Name> {
    type C = 𝒯<
        ː<Requires<Name, 𝒞>, <𝒞::C as Category>::Structure>,
        <𝒞::C as Category>::Properties,
        <𝒞::C as Category>::Equations,
    >;
}

impl<𝒞: Cat, Name: AssocName, C: Category + 'static> Compare<𝐈𝐝<C>> for WithPayload<𝒞, Name> {
    type Relation = Different;
}

// Payload-refined labels are distinct from every ordinary atomic category.
// This keeps closed property-membership queries total when the requested
// category is itself a functor image such as `Jetted<𝒞>`.
impl<𝒞: Atom, Base: Cat, Name: AssocName> Compare<WithPayload<Base, Name>> for 𝒞 {
    type Relation = Different;
}

/// The reflected codomain of jettification on `𝒞`.
///
/// This is not the jet functor itself.  It is the input category with one extra
/// associated dependency recording the source object type carried by the jet
/// representation.
#[allow(type_alias_bounds)]
pub type Jetted<𝒞: Cat> = WithPayload<𝒞, JetPayload>;

/// The reflected codomain of re-presenting an object of `𝒞` over new scalars.
///
/// As with [`Jetted`], the representation itself is not promoted to a category label; only the
/// source object type is retained as additional metadata.
#[allow(type_alias_bounds)]
pub type TensorOf<𝒞: Cat> = WithPayload<𝒞, TensorPayload>;

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

impl<A: Category + 'static, C: Category + 'static> Compare<𝐈𝐝<C>> for 𝐓𝐲𝐩𝐢𝐧𝐠<A> {
    type Relation = Different;
}

impl<C: Category + 'static> Cat for 𝐀𝐫𝐫<C> {
    type C = cat![(Typing: 𝐓𝐲𝐩𝐢𝐧𝐠<C>), {}];
}

impl<A: Category + 'static, C: Category + 'static> Compare<𝐈𝐝<C>> for 𝐀𝐫𝐫<A> {
    type Relation = Different;
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

impl<A: Category + 'static, C: Category + 'static> Compare<𝐈𝐝<C>>
    for 𝐇𝐨𝐦𝐨𝐭𝐨𝐩𝐲<A>
{
    type Relation = Different;
}

#[macro_export]
macro_rules! C {
    [$𝒞:ty] => {
        <$𝒞 as Cat>::C
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

        // Higher-category families and `𝐂𝐚𝐭` are also ordinary named labels.
        // They can never be identical to an embedded concrete category `𝐈𝐝<C>`.
        $(
            impl<N: Nat, C: Category + 'static> Compare<𝐈𝐝<C>> for $family<N>
            where
                $family<N>: Cat,
            {
                type Relation = Different;
            }
        )*

        impl<N: Nat, C: Category + 'static> Compare<𝐈𝐝<C>> for 𝐂𝐚𝐭<N>
        where
            𝐂𝐚𝐭<N>: Cat,
        {
            type Relation = Different;
        }

        // Higher-category families are also ordinary (non-payload) labels, so
        // they compare different from every `WithPayload<_, _>` target.
        $(
            impl<N: Nat, Base: Cat, Name: AssocName> Compare<WithPayload<Base, Name>>
                for $family<N>
            where
                $family<N>: Cat,
            {
                type Relation = Different;
            }
        )*

        impl<N: Nat, Base: Cat, Name: AssocName> Compare<WithPayload<Base, Name>>
            for 𝐂𝐚𝐭<N>
        where
            𝐂𝐚𝐭<N>: Cat,
        {
            type Relation = Different;
        }

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
    𝐑𝐞𝐚𝐥𝐎𝐩𝐬 => cat!{};
    𝐑𝐞𝐚𝐥     => cat!{𝐂𝐅𝐥𝐝, 𝐑𝐞𝐚𝐥𝐎𝐩𝐬};
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
// Reflection of concrete Rust trait implementations
// -----------------------------------------------------------------------------

/// Elaborate the structural body of `X` interpreted as `𝒞`.
///
/// This records the structural claim made by an ordinary Rust trait implementation.
/// Admission through [`Ⱶ`] checks that claim against the requested theory. The
/// canonical *context type* is always
/// [`Interpretation<𝒞, X>`]; `Body` is only the graph exposed by that context.
/// Keeping the identity fixed here prevents equivalent ordinary contexts from
/// acquiring distinct Rust types merely because they were reached by different
/// proof paths.
pub trait Reflect<𝒞: Cat> {
    /// The elaborated structural graph behind this interpretation.
    ///
    /// Reflection is a certificate, not merely metadata: the body must itself
    /// satisfy the nominal theory it claims to interpret.
    type Body: Ⱶ<𝒞>;
}

impl<N: Nat> Reflect<𝐍𝐚𝐭> for N {
    type Body = C![𝐍𝐚𝐭];
}

impl<T: Field> Reflect<𝐅𝐥𝐝> for T {
    type Body = 𝒯<
        ː<
            BindsReflected<Fixed, 𝐂𝐅𝐥𝐝, T::Fixed>,
            ː<BindsReflected<Characteristic, 𝐍𝐚𝐭, T::Characteristic>, Ø>,
        >,
        properties![𝐑𝐢𝐧𝐠, 𝐆𝐫𝐩],
        ː<Equal<Follow<At<Fixed>, Fixed>, At<Fixed>>, Ø>,
    >;
}

impl<T: CField> Reflect<𝐂𝐅𝐥𝐝> for T {
    type Body = 𝒯<
        ː<
            BindsReflected<Fixed, 𝐂𝐅𝐥𝐝, T::Fixed>,
            ː<BindsReflected<Characteristic, 𝐍𝐚𝐭, T::Characteristic>, Ø>,
        >,
        ː<BindsProperty<𝐅𝐥𝐝, Interpretation<𝐅𝐥𝐝, T>>, ː<𝐀𝐛, Ø>>,
        ː<Equal<Follow<At<Fixed>, Fixed>, At<Fixed>>, Ø>,
    >;
}

impl<R: Real> Reflect<𝐑𝐞𝐚𝐥> for R {
    type Body = 𝒯<Ø, ː<BindsProperty<𝐂𝐅𝐥𝐝, Interpretation<𝐂𝐅𝐥𝐝, R>>, ː<𝐑𝐞𝐚𝐥𝐎𝐩𝐬, Ø>>>;
}

impl<T: Tensor> Reflect<𝐓𝐞𝐧𝐬> for T {
    type Body = 𝒯<ː<BindsReflected<F, 𝐅𝐥𝐝, T::F>, Ø>, properties![𝐂𝐌𝐨𝐧]>;
}

impl<V: Vector> Reflect<𝐕𝐞𝐜𝐭> for V {
    type Body = 𝒯<Ø, ː<BindsProperty<𝐓𝐞𝐧𝐬, Interpretation<𝐓𝐞𝐧𝐬, V>>, ː<𝐆𝐫𝐩, Ø>>>;
}

impl<T: Topological> Reflect<𝐓𝐨𝐩> for T {
    type Body = 𝒯<Ø, ː<BindsProperty<𝐒𝐞𝐭, C![𝐒𝐞𝐭]>, Ø>>;
}

impl<M: Manifold> Reflect<𝐌𝐚𝐧> for M {
    type Body = 𝒯<
        ː<BindsReflected<Tangent, 𝐓𝐞𝐧𝐬, M::Tangent>, Ø>,
        ː<BindsProperty<𝐓𝐨𝐩, Interpretation<𝐓𝐨𝐩, M>>, Ø>,
    >;
}

// -----------------------------------------------------------------------------
// Arrow category construction.
// -----------------------------------------------------------------------------

/// The concrete Rust domain of an arrow context.
#[allow(type_alias_bounds)]
pub type DomainOf<C: π<Typing, X: Signature>> = <<C as π<Typing>>::X as Signature>::Domain;

/// The concrete Rust codomain of an arrow context.
#[allow(type_alias_bounds)]
pub type CodomainOf<C: π<Typing, X: Signature>> = <<C as π<Typing>>::X as Signature>::Codomain;

/// Construct the concrete context of an arrow `D -> E` in `C`.
///
/// Both endpoints are bound by one structural association, so a domain cannot
/// exist without its codomain.
#[allow(type_alias_bounds)]
pub type ArrowCategory<C: Category + 'static, D: Ob<C>, E: Ob<C>> =
    𝒯<ː<BindsTyping<C, D, E>, Ø>, Ø>;

/// A concrete callable admitted as a morphism in structural context `C`.
///
/// The `F` field is the complete runtime representation. The context marker is
/// erased, but at compile time it records the exact mathematical interpretation
/// under which the callable was certified. Theorems should retain this richest
/// context and ask it to [`Ⱶ`] whatever weaker arrow theory they require.
///
/// [`Arrow::new`] is deliberately a semantic trust boundary: Rust checks the
/// function signature against the context, but the caller certifies the
/// mathematical claim that the function really is a morphism in `C`.
pub struct Arrow<C: Category, F = Infallible> {
    f: F,
    ctx: PhantomData<fn() -> C>,
}

impl<C: Category, F> Deref for Arrow<C, F> {
    type Target = F;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.f
    }
}

impl<C: Category, F> DerefMut for Arrow<C, F> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.f
    }
}

impl<C: π<Typing, X: Signature>> Arrow<C, Infallible> {
    /// Admit `function` as the morphism described by `C`.
    #[inline]
    pub fn new<F>(f: F) -> Arrow<C, F> {
        Arrow::<C, F> {
            f,
            ctx: PhantomData,
        }
    }
}

impl<C: π<Typing, X: Signature>, F: Fn(&DomainOf<C>) -> CodomainOf<C>> Arrow<C, F> {
    /// Forget the admission and recover the ordinary Rust callable.
    #[inline]
    pub fn into_inner(self) -> F {
        self.f
    }
}

impl<C: Category, F: Clone> Clone for Arrow<C, F> {
    fn clone(&self) -> Self {
        Self {
            f: self.f.clone(),
            ctx: PhantomData,
        }
    }
}

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

impl<𝒞: Cat, 𝒟: Compare<𝒞, Relation = Same>, C: Ⱶ<𝒞>> RefineProperty<𝒞> for BindsProperty<𝒟, C> {
    // A resolved property edge already carries the complete child context.
    // Retain it verbatim so refinement does not encode proof history into the
    // resulting Rust type.
    type Refinement = C;
}

impl<𝒞: Cat, 𝒟: Cat<C: Ⱶ<𝒞>> + Compare<𝒞, Relation = Same>> RefineProperty<𝒞> for 𝒟 {
    type Refinement = <𝒟::C as Ⱶ<𝒞>>::C;
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
> RefinesProperties<ː<𝒞, Tail>> for S
{
    type Refinement = ː<
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
/// stored role is at least as strong as the required role and its stored context
/// is genuinely a context of that value in the stronger role.
///
/// The ordinary interpretation is checked *lazily*: validating the edge does
/// not recursively refine the child's whole graph. This is what keeps cyclic
/// equations such as `Fixed::Fixed = Fixed` finite.
trait SatisfiesAssoc<𝒞: Cat>: AssocEntry {}

/// A context which legitimately describes `X` in nominal role `𝒞`.
trait ChildContext<𝒞: Cat, X>: Category {}

impl<𝒞: Cat, X: Reflect<𝒞>> ChildContext<𝒞, X> for Interpretation<𝒞, X> {}

impl<𝒞: Cat, X, C: Category> ChildContext<𝒞, X> for Rooted<𝒞, X, C> where
    Rooted<𝒞, X, C>: Ⱶ<𝒞>
{
}

/// Dispatch nominal weakening without asking the child's concrete context to
/// recursively prove the weaker theory. `Actual == Required` is immediate; a
/// strictly richer role may satisfy a weaker role through its transitive property
/// graph (for example `Real -> CField -> Field`).
trait RoleSatisfies<Required: Cat, Relation>: Cat {}

impl<Required: Cat, Actual: Cat> RoleSatisfies<Required, Same> for Actual {}

impl<Required: Cat, Actual: Cat<C: HasProperty<Required, Relation = Present>>>
    RoleSatisfies<Required, Different> for Actual
{
}

impl<
    Required: Cat,
    Actual: Cat + Compare<Required>,
    Name: AssocName,
    Value,
    Context: ChildContext<Actual, Value>,
> SatisfiesAssoc<Required> for Binds<Name, Actual, Value, Context>
where
    Actual: RoleSatisfies<Required, <Actual as Compare<Required>>::Relation>,
{
}

/// Explicit structural-category bindings retain their original proof rule.
impl<
    Name: AssocName,
    Required: Category + 'static,
    Actual: Ⱶ<𝐈𝐝<Required>> + 'static,
    Context: Category,
    Value: Object<Context = Context> + Ob<Actual>,
> SatisfiesAssoc<𝐈𝐝<Required>> for BindsAs<Name, 𝐈𝐝<Actual>, Value, Context>
{
}

impl<Required: Category + 'static, Actual: Ⱶ<𝐈𝐝<Required>> + 'static, D: Ob<Actual>, E: Ob<Actual>>
    SatisfiesAssoc<𝐓𝐲𝐩𝐢𝐧𝐠<Required>> for BindsTyping<Actual, D, E>
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
> RefinesStructure<ː<Requires<Name, 𝒞>, Tail>> for S
{
    type Refinement = ː<<S as FindAssoc<Name>>::Found, <S as RefinesStructure<Tail>>::Refinement>;
}

/// Solver kernel for structural refinement between concrete category graphs.
///
/// Public mathematical statements should use `C₁: Ⱶ<𝐈𝐝<C₂>>`, which embeds a
/// concrete `Category` back into the single turnstile relation. This trait merely
/// performs the graph-to-graph comparison underneath that judgement. The result
/// retains the relevant resolved graph: associated requirements become concrete
/// bindings, properties become nested resolved property graphs, and equations are
/// retained after being proved against the source.
#[doc(hidden)]
pub trait StructuralRefinement<Target: Category>: Category {
    type C: Category;
}

impl<
    TS: AssocList,
    TP: PropertyList,
    TE: EquationList,
    SS: AssocList + RefinesStructure<TS>,
    SP: PropertyList + RefinesProperties<TP>,
    SE: EquationList,
> StructuralRefinement<𝒯<TS, TP, TE>> for 𝒯<SS, SP, SE>
where
    𝒯<SS, SP, SE>: SatisfiesEquations<TE>,
{
    type C = 𝒯<
        <SS as RefinesStructure<TS>>::Refinement,
        <SP as RefinesProperties<TP>>::Refinement,
        TE,
    >;
}

/// Root metadata belongs to the full concrete context, not to a weakened proof
/// subgraph. Explicit refinement therefore delegates to the rooted body's graph.
impl<𝒞: Cat, X, C: Category + StructuralRefinement<Target>, Target: Category>
    StructuralRefinement<Target> for Rooted<𝒞, X, C>
{
    type C = <C as StructuralRefinement<Target>>::C;
}

/// A canonical interpretation is structurally transparent to the solver: the
/// low-level graph comparison delegates to its elaborated body without changing
/// the interpretation's Rust type.
impl<𝒞: Cat, X: Reflect<𝒞>, Target: Category> StructuralRefinement<Target> for Interpretation<𝒞, X>
where
    <X as Reflect<𝒞>>::Body: StructuralRefinement<Target>,
{
    type C = <<X as Reflect<𝒞>>::Body as StructuralRefinement<Target>>::C;
}

/// Query a concrete context for its `𝒞`-shaped resolved subgraph.
///
/// A context rooted in `𝒞` refines its own structural body directly. A context
/// rooted in a strictly richer category reaches `𝒞` through the resolved
/// property graph, preserving the rich ambient context while returning the
/// weaker view only as this proof's associated `C`.
pub trait Ⱶ<𝒞: Cat>: Category {
    type C: Category;
}

/// Unrooted structural graphs are already views, so ordinary structural
/// refinement remains the right interpretation for them.
impl<𝒞: Cat, S: AssocList, P: PropertyList, E: EquationList> Ⱶ<𝒞> for 𝒯<S, P, E>
where
    𝒯<S, P, E>: StructuralRefinement<𝒞::C>,
{
    type C = <𝒯<S, P, E> as StructuralRefinement<𝒞::C>>::C;
}

/// Dispatch a rooted judgement according to where the requested category is
/// found in the closed context.
///
/// If `Required` is present in the inherited property graph, use that resolved
/// property subcontext. Otherwise the root body itself must structurally refine
/// `Required`. This distinction is stronger than comparing nominal root labels:
/// `WithPayload<𝐑𝐞𝐚𝐥, _>` is nominally different from `𝐑𝐞𝐚𝐥`, but directly
/// contains the complete Real theory in its own body.
pub trait RootRefinement<𝒞: Cat, Presence>: Category {
    type C: Category;
}

impl<Required: Cat, Actual: Cat, X, Body: Category> RootRefinement<Required, Present>
    for Rooted<Actual, X, Body>
where
    Rooted<Actual, X, Body>: PropertyRefinement<Required>,
{
    type C = <Rooted<Actual, X, Body> as PropertyRefinement<Required>>::Refinement;
}

impl<Required: Cat, Actual: Cat, X, Body: Category + StructuralRefinement<Required::C>>
    RootRefinement<Required, Absent> for Rooted<Actual, X, Body>
{
    type C = <Body as StructuralRefinement<Required::C>>::C;
}

impl<Required: Cat, Actual: Cat, X, Body: Category> Ⱶ<Required> for Rooted<Actual, X, Body>
where
    Rooted<Actual, X, Body>: HasProperty<Required>
        + RootRefinement<Required, <Rooted<Actual, X, Body> as HasProperty<Required>>::Relation>,
{
    type C = <Rooted<Actual, X, Body> as RootRefinement<
        Required,
        <Rooted<Actual, X, Body> as HasProperty<Required>>::Relation,
    >>::C;
}

/// Canonical interpretations delegate admission to a rooted view of their
/// elaborated body. The rooted view exists only while solving the judgement; the
/// canonical context type remains `Interpretation<Actual, X>`.
impl<Required: Cat, Actual: Cat, X: Reflect<Actual>> Ⱶ<Required> for Interpretation<Actual, X>
where
    Rooted<Actual, X, <X as Reflect<Actual>>::Body>: Ⱶ<Required>,
{
    type C = <Rooted<Actual, X, <X as Reflect<Actual>>::Body> as Ⱶ<Required>>::C;
}

/// The public shorthand for reflecting `X` as `𝒞` and returning the resolved graph.
#[allow(type_alias_bounds)]
pub type Refine<𝒞: Cat, X: Reflect<𝒞>> = <Interpretation<𝒞, X> as Ⱶ<𝒞>>::C;

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
    type ReflectedV = Interpretation<𝐕𝐞𝐜𝐭, Coords<f64, 2>>;
    type RootRole = <ReflectedV as π>::𝒞;
    type RootC = <ReflectedV as π>::C;

    type V = Refine<𝐕𝐞𝐜𝐭, Coords<f64, 2>>;
    type T = <V as PropertyRefinement<𝐓𝐞𝐧𝐬>>::Refinement;
    type Scalar = <T as π<F>>::X;
    type ScalarC = <T as π<F>>::C;

    fn assert_same_type<T>(_: T, _: T) {}

    assert_same_type(PhantomData::<RootRole>, PhantomData::<𝐕𝐞𝐜𝐭>);
    assert_same_type(PhantomData::<RootC>, PhantomData::<ReflectedV>);
    assert_same_type(PhantomData::<Scalar>, PhantomData::<f64>);

    fn scalar_context_is_field<C: Ⱶ<𝐅𝐥𝐝>>() {}
    scalar_context_is_field::<ScalarC>();

    // Ordinary projection and direct interpretation have literal Rust type
    // equality: context polymorphism records knowledge, never proof history.
    type DirectFieldC = Interpretation<𝐅𝐥𝐝, f64>;
    assert_same_type(PhantomData::<ScalarC>, PhantomData::<DirectFieldC>);

    // `Ⱶ<𝐈𝐝<C>>` is the public concrete-category refinement judgement.
    fn refines_concrete_field_theory<C: Ⱶ<𝐈𝐝<C![𝐅𝐥𝐝]>>>() {}
    refines_concrete_field_theory::<DirectFieldC>();

    // Self-referential associated structure remains a finite context handle.
    // For a real scalar, Fixed = Self, so following Fixed twice must return
    // literally the same finite child context rather than recursively expanding it.
    type FieldC = Interpretation<𝐅𝐥𝐝, f64>;
    type FixedC = <FieldC as π<Fixed>>::C;
    type FixedFixedC = <FixedC as π<Fixed>>::C;
    assert_same_type(PhantomData::<FixedC>, PhantomData::<FixedFixedC>);

    // A parent may expose a child only as a Field while retaining a strictly
    // richer Real subcontext for it. Projection must select that exact graph.
    type RealC = Interpretation<𝐑𝐞𝐚𝐥, f64>;

    // Property projection obeys the same normalization law as `π`: a direct
    // inherited property returns the exact context stored by its parent.
    type CFieldViaReal = <RealC as PropertyRefinement<𝐂𝐅𝐥𝐝>>::Refinement;
    assert_same_type(
        PhantomData::<CFieldViaReal>,
        PhantomData::<Interpretation<𝐂𝐅𝐥𝐝, f64>>,
    );

    // The law is transitive too. Expanding Real -> CField -> Field must still
    // land on the canonical Field interpretation, not a resolved proof graph.
    type FieldViaReal = <RealC as PropertyRefinement<𝐅𝐥𝐝>>::Refinement;
    assert_same_type(
        PhantomData::<FieldViaReal>,
        PhantomData::<Interpretation<𝐅𝐥𝐝, f64>>,
    );

    type Parent = 𝒯<ː<Binds<F, 𝐑𝐞𝐚𝐥, f64, RealC>, Ø>, Ø>;
    type ChildRole = <Parent as π<F>>::𝒞;
    type ChildC = <Parent as π<F>>::C;
    type ChildRootRole = <ChildC as π>::𝒞;

    assert_same_type(PhantomData::<ChildRole>, PhantomData::<𝐑𝐞𝐚𝐥>);
    assert_same_type(PhantomData::<ChildC>, PhantomData::<RealC>);
    assert_same_type(PhantomData::<ChildRootRole>, PhantomData::<𝐑𝐞𝐚𝐥>);

    type RequiresFieldChild = 𝒯<ː<Requires<F, 𝐅𝐥𝐝>, Ø>, Ø>;
    fn richer_edge_satisfies_weaker_requirement<C: Ⱶ<𝐈𝐝<RequiresFieldChild>>>() {}
    richer_edge_satisfies_weaker_requirement::<Parent>();

    fn child_keeps_real_information<C: Ⱶ<𝐅𝐥𝐝> + HasProperty<𝐑𝐞𝐚𝐥𝐎𝐩𝐬, Relation = Present>>() {
    }
    child_keeps_real_information::<ChildC>();
}
