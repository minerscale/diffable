//! A reflected ontology of mathematical structure.
//!
//! This abstraction is simultaneously:
//!   - A model of category theory.
//!   - A model of model theory.
//!   - A model of dependent types on stable Rust.
//!   - A model of first-class traits in Rust.
//!   - A hack approximating negative trait bounds on stable Rust.
//!   - A place for additional information that can prevent overlapping impls.
//!   - A system admitting interpretations from higher homotopy theory,
//!     in particular as an ∞-groupoid.
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
//! structure: `<V as Tensor>::F` must be recovered as `tensor::F`, not merely as the first
//! field-shaped thing in a list.  Consequently structural dependencies are matched
//! by associated-type name while properties are matched as an unordered set.
//!
//! Everything in this module is zero-sized.  The ontology is a compile-time
//! database consumed by trait resolution; there is no runtime tree.

use crate::{
    coords::Coords,
    traits::{
        CField, Field, Form, Manifold, Nat, NatCompare, NatZero, Real, Succ, Tensor, Topological,
        Vector,
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
/// Canonical [`Cat::C`] values remain unrooted theories. `Rooted` records which
/// object an explicitly supplied concrete body is actually about.
#[derive(Debug, Copy, Clone)]
pub struct Rooted<𝒞: Cat, X, C: Category>(PhantomData<fn() -> (𝒞, X, C)>);

impl<𝒞: Cat, X, C: Category> sealed::Category for Rooted<𝒞, X, C> {}

impl<𝒞: Cat, X, C: Category> Category for Rooted<𝒞, X, C> {
    type Structure = C::Structure;
    type Properties = C::Properties;
    type Equations = C::Equations;
}

/// Finite implementation carrier for a concrete object rooted in nominal theory `𝒞`.
///
/// Public code should normally obtain contexts through [`ι`] and [`Model`]. This
/// carrier exists so [`Reflect::Body`] can refer to recursive child models without
/// recursively expanding their entire structural graphs into the Rust type itself.
#[doc(hidden)]
#[derive(Debug, Copy, Clone)]
pub struct ReflectedContext<𝒞: Cat, X>(PhantomData<fn() -> (𝒞, X)>);

impl<𝒞: Cat, X: Reflect<𝒞>> sealed::Category for ReflectedContext<𝒞, X> {}

impl<𝒞: Cat, X: Reflect<𝒞>> Category for ReflectedContext<𝒞, X> {
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

/// Include a concrete Rust type into the ontology with its canonical structural context.
///
/// This is the canonical inclusion map from an admitted Rust type into the
/// categorical ontology. [`Reflect`] supplies the available interpretations; `ι`
/// selects the distinguished context from which weaker models are derived through
/// [`Ⱶ`].
#[allow(non_camel_case_types)]
pub trait ι: Sized {
    type C: RootContext<X = Self>;
}

/// The canonical model of theory `𝒞` obtained from the included Rust type `X`.
///
/// `Model` is derived notation, not a fourth primitive operation: first [`ι`]
/// includes `X` into its canonical categorical context, then [`Ⱶ`] selects the
/// requested theory view.
#[allow(type_alias_bounds)]
pub type Model<𝒞: Cat, X: ι<C: Ⱶ<𝒞>>> = <<X as ι>::C as Ⱶ<𝒞>>::C;

/// The proposition that `Self` is an object of structural category `C`.
///
/// Objecthood is derived from the single canonical context selected by [`ι`]. Thus a
/// value included with a rich context is automatically an object of every weaker
/// category refined by that context.
pub trait Ob<C: Category + 'static>: ι {
    /// The contextual witness used to admit `Self` as an object of `C`.
    ///
    /// Context-free objecthood chooses the canonical context supplied by [`ι`],
    /// but exposes that witness explicitly so dependent constructions can retain
    /// it instead of reconstructing the endpoint through [`ι`] later.
    type Context: RootContext<X = Self> + Ⱶ<𝐈𝐝<C>>;
}

impl<X, C> Ob<C> for X
where
    X: ι,
    C: Category + 'static,
    X::C: RootContext<X = X> + Ⱶ<𝐈𝐝<C>>,
{
    type Context = X::C;
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
/// The role `𝒞` records the actual nominal interface carried by the edge. A
/// parent's requirement may be weaker; structural refinement checks that relationship
/// without changing either the stored role or `Context`.
#[derive(Debug, Copy, Clone)]
pub struct Binds<Name: AssocName, 𝒞: Cat, Value, Context: Category>(
    PhantomData<(Name, 𝒞, Value, Context)>,
);

/// The ordinary reflected binding, whose child context is the canonical
/// reflection of `Value` in the edge role.
///
/// This is intentionally *role-local*: it records exactly the interpretation
/// justified by the parent's Rust bound. If the parent already owns a richer
/// child context, use [`Binds`] directly; if the child has a distinguished [`ι`]
/// inclusion which should be preserved, use [`BindsIncluded`].
#[allow(type_alias_bounds)]
pub type BindsReflected<Name: AssocName, 𝒞: Cat, Value: Reflect<𝒞>> =
    Binds<Name, 𝒞, Value, ReflectedContext<𝒞, Value>>;

/// Bind an associated object using its distinguished included context.
///
/// Unlike [`BindsReflected`], this does not first choose the minimum role required
/// by the parent and then reflect `Value` in that role. It stores the child
/// selected by [`ι`] verbatim, including that context's actual root role. A parent
/// which merely requires a weaker role can still consume the edge through
/// the nominal weakening machinery.
///
/// This is the canonical dependent-functor operation when the associated object is
/// known to have an [`ι`] inclusion. Generic reflected implementations should keep
/// using [`BindsReflected`] when their Rust bounds do not provide such an inclusion.
#[allow(type_alias_bounds)]
pub type BindsIncluded<Name: AssocName, Value: ι> =
    Binds<Name, <<Value as ι>::C as RootContext>::𝒞, Value, <Value as ι>::C>;

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
/// they form the single signature which inhabits the [`arrow::Typing`] role.
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

/// Bind the [`arrow::Typing`] of an arrow in `C`.
///
/// This is the admission boundary for an arrow signature. Unlike the
/// projection-only [`Signature`] trait, a
/// `BindsTyping<C, D, DContext, E, EContext>` certifies both endpoints together
/// with the exact contexts under which they were admitted as objects of `C`, so
/// an arrow can never acquire a domain independently of its codomain (or vice
/// versa). Keeping those contexts explicit also lets higher arrow constructions
/// retain contextual information instead of re-running [`ι`] on the endpoint
/// carriers.
///
/// Its associated value is the single [`ArrowSignature<C, D, E>`] from which
/// domain and codomain may subsequently be projected.
#[derive(Debug, Copy, Clone)]
pub struct BindsTyping<C: Category + 'static, D, DContext: Category, E, EContext: Category>(
    PhantomData<fn() -> (C, D, DContext, E, EContext)>,
);

/// Placeholder used as [`AssocEntry::Value`] by a canonical requirement.
#[derive(Debug, Copy, Clone)]
pub struct Unspecified;

#[doc(hidden)]
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

impl<C, D, DContext, E, EContext> sealed::AssocEntry for BindsTyping<C, D, DContext, E, EContext>
where
    C: Category + 'static,
    DContext: RootContext<X = D> + Ⱶ<𝐈𝐝<C>>,
    EContext: RootContext<X = E> + Ⱶ<𝐈𝐝<C>>,
{
}

impl<C, D, DContext, E, EContext> AssocEntry for BindsTyping<C, D, DContext, E, EContext>
where
    C: Category + 'static,
    DContext: RootContext<X = D> + Ⱶ<𝐈𝐝<C>>,
    EContext: RootContext<X = E> + Ⱶ<𝐈𝐝<C>>,
{
    type Name = arrow::Typing;
    type Role = 𝐓𝐲𝐩𝐢𝐧𝐠<C>;
    type C = Rooted<
        𝐓𝐲𝐩𝐢𝐧𝐠<C>,
        ArrowSignature<C, D, E>,
        𝒯<
            ː<
                BindsAs<signature::Domain, 𝐈𝐝<C>, D, DContext>,
                ː<BindsAs<signature::Codomain, 𝐈𝐝<C>, E, EContext>, Ø>,
            >,
            Ø,
        >,
    >;
    type Value = ArrowSignature<C, D, E>;
}

#[doc(hidden)]
pub trait AssocList: sealed::AssocList {}
impl sealed::AssocList for Ø {}
impl AssocList for Ø {}
impl<H: AssocEntry, T: AssocList> sealed::AssocList for ː<H, T> {}
impl<H: AssocEntry, T: AssocList> AssocList for ː<H, T> {}

/// Constructive equality of associated-type labels.
#[doc(hidden)]
pub trait CompareAssoc<Rhs: AssocName>: AssocName {
    type Relation;
}

/// Declare a namespace containing reflected associated-type labels.
///
/// Labels are owned by the trait or construction which introduces their semantic
/// role.  This mirrors Rust's own `<X as Trait>::Assoc` qualification and prevents
/// unrelated associated types with the same spelling from becoming equal in the
/// ontology merely because their local names coincide.
macro_rules! assoc_namespace {
    ($namespace:ident { $($name:ident),+ $(,)? }) => {
        pub mod $namespace {
            $(
                #[derive(Debug, Copy, Clone)]
                pub struct $name;
            )+
        }
    };
}

/// Declare constructive equality for the finite universe of associated labels.
///
/// As with category atoms, inequality is explicit: every off-diagonal pair receives
/// a `Different` implementation rather than relying on failed trait resolution.
macro_rules! assoc_relations {
    () => {};

    ($head:path $(, $tail:path)* $(,)?) => {
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

        assoc_relations!($($tail),*);
    };
}

// `This` is the distinguished root projection rather than a reflected Rust
// associated type, so it remains the one unqualified label.  Every ordinary
// associated role is namespaced by its declaring trait/construction.
#[derive(Debug, Copy, Clone)]
pub struct This;

assoc_namespace!(tensor { F });
assoc_namespace!(field {
    Fixed,
    Characteristic
});
assoc_namespace!(manifold { Tangent });
assoc_namespace!(arrow { Typing });
assoc_namespace!(signature { Domain, Codomain });
assoc_namespace!(homotopy { From, To });
assoc_namespace!(jet { Payload });
assoc_namespace!(tensor_of { Payload });

assoc_relations!(
    This,
    tensor::F,
    field::Fixed,
    field::Characteristic,
    manifold::Tangent,
    arrow::Typing,
    signature::Domain,
    signature::Codomain,
    homotopy::From,
    homotopy::To,
    jet::Payload,
    tensor_of::Payload,
);

/// Find a structural dependency by associated-type name.
///
/// The list is logically a record, not a tuple: lookup ignores declaration order.
#[doc(hidden)]
pub trait FindAssoc<Name: AssocName>: AssocList {
    type Found: AssocEntry<Name = Name>;
}

#[doc(hidden)]
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
#[doc(hidden)]
pub trait RootContext: Category {
    type 𝒞: Cat;
    type X;
}

impl<𝒞: Cat, X, C: Category> RootContext for Rooted<𝒞, X, C> {
    type 𝒞 = 𝒞;
    type X = X;
}

impl<𝒞: Cat, X: Reflect<𝒞>> RootContext for ReflectedContext<𝒞, X> {
    type 𝒞 = 𝒞;
    type X = X;
}

/// Internal dispatch for the identity projection (`This`) versus an ordinary
/// named child edge.
#[doc(hidden)]
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

#[doc(hidden)]
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

#[doc(hidden)]
pub trait PropertyList: sealed::PropertyList {}
impl sealed::PropertyList for Ø {}
impl PropertyList for Ø {}
impl<H: PropertyEntry, T: PropertyList> sealed::PropertyList for ː<H, T> {}
impl<H: PropertyEntry, T: PropertyList> PropertyList for ː<H, T> {}

/// Append two type-level property lists.
#[doc(hidden)]
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
#[doc(hidden)]
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
#[doc(hidden)]
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
#[doc(hidden)]
pub trait PropertyPresence<𝒞: Cat>: PropertyList {
    type Relation;
}

#[doc(hidden)]
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

/// Decide whether an expanded source property graph contains every property
/// required by `Target`.
///
/// This is the constructive fragment of negative refinement.  For theories
/// whose canonical signature contains only properties, failure of this query is
/// enough to prove that the source cannot satisfy the theory.
#[doc(hidden)]
pub trait CoversProperties<Target: PropertyList>: PropertyList {
    type Relation;
}

#[doc(hidden)]
pub trait CoversPropertiesWith<Target: PropertyList, Relation>: PropertyList {
    type Output;
}

impl<S: PropertyList> CoversProperties<Ø> for S {
    type Relation = Present;
}

impl<S: PropertyList, Tail: PropertyList> CoversPropertiesWith<Tail, Absent> for S {
    type Output = Absent;
}

impl<Tail: PropertyList, S: PropertyList + CoversProperties<Tail>>
    CoversPropertiesWith<Tail, Present> for S
{
    type Output = <S as CoversProperties<Tail>>::Relation;
}

impl<
    Head: PropertyEntry,
    Tail: PropertyList,
    S: PropertyList + PropertyPresence<<Head as PropertyEntry>::Role>,
> CoversProperties<ː<Head, Tail>> for S
where
    S: CoversPropertiesWith<Tail, <S as PropertyPresence<<Head as PropertyEntry>::Role>>::Relation>,
{
    type Relation = <S as CoversPropertiesWith<
        Tail,
        <S as PropertyPresence<<Head as PropertyEntry>::Role>>::Relation,
    >>::Output;
}

/// Find a property edge by reflected category name.
#[doc(hidden)]
pub trait FindProperty<𝒞: Cat>: PropertyList {
    type Found: PropertyEntry;
}

#[doc(hidden)]
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

/// Solver primitive selecting the exact context carried by property `𝒞`.
#[doc(hidden)]
pub trait SelectProperty<𝒞: Cat>: Category {
    type C: Category;
}

#[doc(hidden)]
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

impl<𝒞: Cat, 𝒟: Cat<C: Ⱶ<𝒞>> + Compare<𝒞, Relation = Same>> ResolvedProperty<𝒞> for 𝒟 {
    // A bare inherited theory carries no concrete child context of its own.
    // Resolve its canonical theory only in this open-theory case.
    type Refinement = <𝒟::C as Ⱶ<𝒞>>::C;
}

impl<𝒞: Cat, C: Category<Properties: ExpandProperties>> SelectProperty<𝒞> for C
where
    <C::Properties as ExpandProperties>::Expansion: FindProperty<𝒞, Found: ResolvedProperty<𝒞>>,
{
    type C = <<<C::Properties as ExpandProperties>::Expansion as FindProperty<
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
#[doc(hidden)]
pub trait EquationList: sealed::EquationList {}

impl sealed::EquationList for Ø {}
impl EquationList for Ø {}

impl<L, R, T: EquationList> sealed::EquationList for ː<Equal<L, R>, T> {}
impl<L, R, T: EquationList> EquationList for ː<Equal<L, R>, T> {}

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
#[doc(hidden)]
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
#[doc(hidden)]
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

#[doc(hidden)]
pub trait Compare<𝒟: Cat>: Cat {
    type Relation;
}

#[doc(hidden)]
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
    ($family:ident; $($cat:ty),* $(,)?) => {
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
    ([$($cat:ty),* $(,)?];) => {};

    (
        [$($cat:ty),* $(,)?];
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

    ($name:path : $role:ty $(, $rest_name:path : $rest_role:ty)* $(,)?) => {
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
/// cat![(tensor::F: 𝐅𝐥𝐝), {𝐓𝐞𝐧𝐬, 𝐆𝐫𝐩}]
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
        ($($name:path : $role:ty),* $(,)?),
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
        ($($name:path : $role:ty),* $(,)?),
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
        $($name:path : $role:ty),* $(,)?
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
pub type Jetted<𝒞: Cat> = WithPayload<𝒞, jet::Payload>;

/// The reflected codomain of re-presenting an object of `𝒞` over new scalars.
///
/// As with [`Jetted`], the representation itself is not promoted to a category label; only the
/// source object type is retained as additional metadata.
#[allow(type_alias_bounds)]
pub type TensorOf<𝒞: Cat> = WithPayload<𝒞, tensor_of::Payload>;

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
    type C = cat![(signature::Domain: 𝐈𝐝<C>, signature::Codomain: 𝐈𝐝<C>), {}];
}

impl<A: Category + 'static, C: Category + 'static> Compare<𝐈𝐝<C>> for 𝐓𝐲𝐩𝐢𝐧𝐠<A> {
    type Relation = Different;
}

impl<C: Category + 'static> Cat for 𝐀𝐫𝐫<C> {
    type C = cat![(arrow::Typing: 𝐓𝐲𝐩𝐢𝐧𝐠<C>), {}];
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
    type C = cat![(homotopy::From: 𝐀𝐫𝐫<C>, homotopy::To: 𝐀𝐫𝐫<C>), {}];
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

/// Elaborate one ordinary ontology declaration into its canonical structural context.
///
/// The declaration syntax names ordinary theories by their bold module name; the
/// actual first-class `Cat` type lives at `𝐗::𝒞`.
macro_rules! category_context {
    (@root {}) => {
        cat!{}
    };

    (@root {$($property:ident),+ $(,)?}) => {
        cat!{$($property::𝒞),+}
    };

    (@root [($($name:path : $role:ident),* $(,)?), {$($property:ident),* $(,)?}, {$($equation:ty),* $(,)?} $(,)?]) => {
        cat![
            ($($name: $role::𝒞),*),
            {$($property::𝒞),*},
            {$($equation),*}
        ]
    };

    (@root [($($name:path : $role:ident),* $(,)?), {$($property:ident),* $(,)?} $(,)?]) => {
        cat![
            ($($name: $role::𝒞),*),
            {$($property::𝒞),*}
        ]
    };

    (@module {}) => {
        cat!{}
    };

    (@module {$($property:ident),+ $(,)?}) => {
        cat!{$(super::$property::𝒞),+}
    };

    (@module [($($name:path : $role:ident),* $(,)?), {$($property:ident),* $(,)?}, {$($equation:ty),* $(,)?} $(,)?]) => {
        cat![
            ($($name: super::$role::𝒞),*),
            {$(super::$property::𝒞),*},
            {$($equation),*}
        ]
    };

    (@module [($($name:path : $role:ident),* $(,)?), {$($property:ident),* $(,)?} $(,)?]) => {
        cat![
            ($($name: super::$role::𝒞),*),
            {$(super::$property::𝒞),*}
        ]
    };
}

macro_rules! category_module_impl {
    ($cat:ident => $context:tt; $($property:ident),* $(,)?) => {
        #[allow(non_snake_case)]
        pub mod $cat {
            use super::*;

            #[derive(Copy, Clone, Debug)]
            pub struct 𝒞;

            impl super::Cat for 𝒞 {
                type C = category_context!(@module $context);
            }

            pub type C = <𝒞 as super::Cat>::C;

            #[allow(non_camel_case_types)]
            pub trait Ⱶ: super::Ⱶ<𝒞> $(+ super::$property::Ⱶ)* {}

            impl<Context> Ⱶ for Context
            where
                Context: super::Ⱶ<𝒞> $(+ super::$property::Ⱶ)*,
            {
            }
        }
    };
}

macro_rules! category_module {
    ($cat:ident => {}) => {
        category_module_impl!($cat => {};);
    };

    ($cat:ident => {$($property:ident),+ $(,)?}) => {
        category_module_impl!($cat => {$($property),+}; $($property),+);
    };

    ($cat:ident => [($($structure:tt)*), {$($property:ident),* $(,)?}, {$($equation:tt)*} $(,)?]) => {
        category_module_impl!(
            $cat => [($($structure)*), {$($property),*}, {$($equation)*}];
            $($property),*
        );
    };

    ($cat:ident => [($($structure:tt)*), {$($property:ident),* $(,)?} $(,)?]) => {
        category_module_impl!(
            $cat => [($($structure)*), {$($property),*}];
            $($property),*
        );
    };
}

/// Declare ordinary reflected theories plus Nat-indexed higher families.
///
/// Every ordinary bold name is a namespace containing the nominal theory `𝒞`,
/// its canonical context `C`, and the compiled judgement `Ⱶ`. Parameterised
/// families remain type constructors because Rust modules cannot be generic.
macro_rules! categories {
    (
        $(
            $cat:ident => cat!$context:tt;
        )*

        $(
            @$family:ident<$n:ident> => cat!$base:tt;
        )*
    ) => {
        $(
            category_module!($cat => $context);
        )*

        $(
            #[derive(Copy, Clone, Debug)]
            pub struct $family<$n: Nat>(PhantomData<$n>);

            impl Cat for $family<NatZero> {
                type C = category_context!(@root $base);
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
                    $($cat::𝒞,)*
                    $($family<N>,)*
                }
            ];
        }

        atoms![$($cat::𝒞),*];

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
            [$($cat::𝒞),*];
            $($family),*
        );

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

// The bold modules are first-class namespaces for the corresponding mathematical
// theories. Each exposes its nominal `𝒞`, canonical context `C`, and judgement `Ⱶ`.
//
// Associated dependencies are declared exactly where the reflected Rust trait
// introduces them.  `Tensor` owns `tensor::F`; `Vector` reaches that scalar field through
// its `Tensor` property.  `Field` owns `field::Fixed` and `field::Characteristic`.
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

    // Order is independent structure on a set. `OrdFld` couples that order to
    // commutative field arithmetic; `Dedekind` adds order completeness.
    𝐎𝐫𝐝      => cat!{𝐒𝐞𝐭};
    𝐎𝐫𝐝𝐅𝐥𝐝   => cat!{𝐂𝐅𝐥𝐝, 𝐎𝐫𝐝};
    𝐃𝐞𝐝𝐞𝐤𝐢𝐧𝐝 => cat!{𝐎𝐫𝐝};
    𝐑𝐞𝐚𝐥     => cat!{𝐎𝐫𝐝𝐅𝐥𝐝, 𝐃𝐞𝐝𝐞𝐤𝐢𝐧𝐝};
    𝐅𝐥𝐝      => cat![
        (field::Fixed: 𝐂𝐅𝐥𝐝, field::Characteristic: 𝐍𝐚𝐭),
        {𝐑𝐢𝐧𝐠, 𝐆𝐫𝐩},
        {Equal<Follow<At<field::Fixed>, field::Fixed>, At<field::Fixed>>}
    ];
    𝐓𝐞𝐧𝐬     => cat![(tensor::F: 𝐅𝐥𝐝), {𝐂𝐌𝐨𝐧}];
    𝐕𝐞𝐜𝐭     => cat!{𝐓𝐞𝐧𝐬, 𝐆𝐫𝐩};
    𝐅𝐨𝐫𝐦     => cat!{𝐓𝐞𝐧𝐬};
    𝐌𝐚𝐧      => cat![(manifold::Tangent: 𝐓𝐞𝐧𝐬), {𝐓𝐨𝐩}];

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
/// Admission through [`Ⱶ`] checks that claim against the requested theory. `Body`
/// is only the elaborated graph behind a finite rooted model; the public entry
/// point from an ordinary Rust type is [`ι`], and [`Model`] composes that inclusion
/// with theory refinement. Keeping the rooted carrier finite prevents equivalent
/// ordinary contexts from acquiring distinct Rust types merely because they were
/// reached by different proof paths.
pub trait Reflect<𝒞: Cat> {
    /// The elaborated structural graph behind this interpretation.
    ///
    /// Reflection is a certificate, not merely metadata: the body must itself
    /// satisfy the nominal theory it claims to interpret.
    type Body: Ⱶ<𝒞>;
}

impl<N: Nat> Reflect<𝐍𝐚𝐭::𝒞> for N {
    type Body = 𝐍𝐚𝐭::C;
}

impl<T: Field> Reflect<𝐅𝐥𝐝::𝒞> for T {
    type Body = 𝒯<
        ː<
            BindsReflected<field::Fixed, 𝐂𝐅𝐥𝐝::𝒞, T::Fixed>,
            ː<BindsReflected<field::Characteristic, 𝐍𝐚𝐭::𝒞, T::Characteristic>, Ø>,
        >,
        properties![𝐑𝐢𝐧𝐠::𝒞, 𝐆𝐫𝐩::𝒞],
        ː<Equal<Follow<At<field::Fixed>, field::Fixed>, At<field::Fixed>>, Ø>,
    >;
}

impl<T: CField> Reflect<𝐂𝐅𝐥𝐝::𝒞> for T {
    type Body = 𝒯<
        ː<
            BindsReflected<field::Fixed, 𝐂𝐅𝐥𝐝::𝒞, T::Fixed>,
            ː<BindsReflected<field::Characteristic, 𝐍𝐚𝐭::𝒞, T::Characteristic>, Ø>,
        >,
        ː<BindsProperty<𝐅𝐥𝐝::𝒞, ReflectedContext<𝐅𝐥𝐝::𝒞, T>>, ː<𝐀𝐛::𝒞, Ø>>,
        ː<Equal<Follow<At<field::Fixed>, field::Fixed>, At<field::Fixed>>, Ø>,
    >;
}

impl<R: Real> Reflect<𝐎𝐫𝐝::𝒞> for R {
    type Body = 𝒯<Ø, ː<BindsProperty<𝐒𝐞𝐭::𝒞, 𝐒𝐞𝐭::C>, Ø>>;
}

impl<R: Real> Reflect<𝐎𝐫𝐝𝐅𝐥𝐝::𝒞> for R {
    type Body = 𝒯<
        Ø,
        ː<
            BindsProperty<𝐂𝐅𝐥𝐝::𝒞, ReflectedContext<𝐂𝐅𝐥𝐝::𝒞, R>>,
            ː<BindsProperty<𝐎𝐫𝐝::𝒞, ReflectedContext<𝐎𝐫𝐝::𝒞, R>>, Ø>,
        >,
    >;
}

impl<R: Real> Reflect<𝐃𝐞𝐝𝐞𝐤𝐢𝐧𝐝::𝒞> for R {
    type Body = 𝒯<Ø, ː<BindsProperty<𝐎𝐫𝐝::𝒞, ReflectedContext<𝐎𝐫𝐝::𝒞, R>>, Ø>>;
}

impl<R: Real> Reflect<𝐑𝐞𝐚𝐥::𝒞> for R {
    type Body = 𝒯<
        Ø,
        ː<
            BindsProperty<𝐎𝐫𝐝𝐅𝐥𝐝::𝒞, ReflectedContext<𝐎𝐫𝐝𝐅𝐥𝐝::𝒞, R>>,
            ː<BindsProperty<𝐃𝐞𝐝𝐞𝐤𝐢𝐧𝐝::𝒞, ReflectedContext<𝐃𝐞𝐝𝐞𝐤𝐢𝐧𝐝::𝒞, R>>, Ø>,
        >,
    >;
}

impl<T: Tensor> Reflect<𝐓𝐞𝐧𝐬::𝒞> for T {
    // `Tensor` guarantees only that `F` is a `Field`; it does not guarantee that
    // `F` has a distinguished `ι` inclusion. Reflection therefore records the
    // role-local Field interpretation here. A construction which already knows
    // a richer scalar context can store it with `Binds`/`BindsIncluded` instead.
    type Body = 𝒯<ː<BindsReflected<tensor::F, 𝐅𝐥𝐝::𝒞, T::F>, Ø>, properties![𝐂𝐌𝐨𝐧::𝒞]>;
}

impl<V: Vector> Reflect<𝐕𝐞𝐜𝐭::𝒞> for V {
    type Body = 𝒯<Ø, ː<BindsProperty<𝐓𝐞𝐧𝐬::𝒞, ReflectedContext<𝐓𝐞𝐧𝐬::𝒞, V>>, ː<𝐆𝐫𝐩::𝒞, Ø>>>;
}

impl<V: Form> Reflect<𝐅𝐨𝐫𝐦::𝒞> for V {
    type Body = 𝒯<Ø, ː<BindsProperty<𝐓𝐞𝐧𝐬::𝒞, ReflectedContext<𝐓𝐞𝐧𝐬::𝒞, V>>, Ø>>;
}

impl<T: Topological> Reflect<𝐓𝐨𝐩::𝒞> for T {
    type Body = 𝒯<Ø, ː<BindsProperty<𝐒𝐞𝐭::𝒞, 𝐒𝐞𝐭::C>, Ø>>;
}

impl<M: Manifold> Reflect<𝐌𝐚𝐧::𝒞> for M {
    type Body = 𝒯<
        ː<BindsReflected<manifold::Tangent, 𝐓𝐞𝐧𝐬::𝒞, M::Tangent>, Ø>,
        ː<BindsProperty<𝐓𝐨𝐩::𝒞, ReflectedContext<𝐓𝐨𝐩::𝒞, M>>, Ø>,
    >;
}

// -----------------------------------------------------------------------------
// Arrow category construction.
// -----------------------------------------------------------------------------

/// The concrete Rust domain of an arrow context.
#[allow(type_alias_bounds)]
pub type DomainOf<C: π<arrow::Typing, X: Signature>> =
    <<C as π<arrow::Typing>>::X as Signature>::Domain;

/// The concrete Rust codomain of an arrow context.
#[allow(type_alias_bounds)]
pub type CodomainOf<C: π<arrow::Typing, X: Signature>> =
    <<C as π<arrow::Typing>>::X as Signature>::Codomain;

/// Construct the concrete context of an arrow `D -> E` in `C`.
///
/// Both endpoints are bound by one structural association, so a domain cannot
/// exist without its codomain.
#[allow(type_alias_bounds)]
pub type ArrowCategory<C: Category + 'static, D: Ob<C>, E: Ob<C>> =
    𝒯<ː<BindsTyping<C, D, <D as Ob<C>>::Context, E, <E as Ob<C>>::Context>, Ø>, Ø>;

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

impl<C: π<arrow::Typing, X: Signature>> Arrow<C, Infallible> {
    /// Admit `function` as the morphism described by `C`.
    #[inline]
    pub fn new<F>(f: F) -> Arrow<C, F> {
        Arrow::<C, F> {
            f,
            ctx: PhantomData,
        }
    }
}

impl<C: π<arrow::Typing, X: Signature>, F: Fn(&DomainOf<C>) -> CodomainOf<C>> Arrow<C, F> {
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
#[doc(hidden)]
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
#[doc(hidden)]
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

impl<𝒞: Cat, X: Reflect<𝒞>> ChildContext<𝒞, X> for ReflectedContext<𝒞, X> {}

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

impl<Required: Cat, Actual: Cat<C: Category<Properties: ExpandProperties>>>
    RoleSatisfies<Required, Different> for Actual
where
    <<Actual::C as Category>::Properties as ExpandProperties>::Expansion:
        PropertyPresence<Required, Relation = Present>,
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
    Context: RootContext<X = Value> + Ⱶ<𝐈𝐝<Actual>>,
    Value,
> SatisfiesAssoc<𝐈𝐝<Required>> for BindsAs<Name, 𝐈𝐝<Actual>, Value, Context>
{
}

impl<Required, Actual, D, DContext, E, EContext> SatisfiesAssoc<𝐓𝐲𝐩𝐢𝐧𝐠<Required>>
    for BindsTyping<Actual, D, DContext, E, EContext>
where
    Required: Category + 'static,
    Actual: Ⱶ<𝐈𝐝<Required>> + 'static,
    DContext: RootContext<X = D> + Ⱶ<𝐈𝐝<Actual>>,
    EContext: RootContext<X = E> + Ⱶ<𝐈𝐝<Actual>>,
{
}

/// Resolve every required associated dependency, retaining the actual source binding.
///
/// Matching is by associated-type name, never by declaration position.
#[doc(hidden)]
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

/// A finite rooted model is structurally transparent to the solver: the
/// low-level graph comparison delegates to its elaborated body without recursively
/// expanding the rooted carrier itself.
impl<𝒞: Cat, X: Reflect<𝒞>, Target: Category> StructuralRefinement<Target>
    for ReflectedContext<𝒞, X>
where
    <X as Reflect<𝒞>>::Body: StructuralRefinement<Target>,
{
    type C = <<X as Reflect<𝒞>>::Body as StructuralRefinement<Target>>::C;
}

/// Judge whether a concrete context satisfies theory `𝒞`.
///
/// The default `Present` judgement is the ordinary refinement relation and
/// returns the exact context which supplies `𝒞`. `Absent` is a constructive
/// negative judgement. It is currently available for property-only theories,
/// where a missing required property proves that refinement is impossible.
///
/// The associated `C` is meaningful for the default positive judgement. In the
/// negative mode it is the unchanged source context and should not be projected
/// as a refinement witness.
pub trait Ⱶ<𝒞: Cat, Relation = Present>: Category {
    type C: Category;
}

/// Select the positive refinement route. If `𝒞` is already carried as an
/// inherited property, preserve that exact stored context; otherwise prove the
/// current graph directly against the canonical theory.
#[doc(hidden)]
pub trait RefinementRoute<𝒞: Cat, Presence>: Category {
    type C: Category;
}

impl<Required: Cat, C: Category + SelectProperty<Required>> RefinementRoute<Required, Present>
    for C
{
    type C = <C as SelectProperty<Required>>::C;
}

impl<Required: Cat, C: Category + StructuralRefinement<Required::C>>
    RefinementRoute<Required, Absent> for C
{
    type C = <C as StructuralRefinement<Required::C>>::C;
}

impl<Required: Cat, C, P> Ⱶ<Required> for C
where
    C: Category<Properties = P>,
    P: ExpandProperties,
    P::Expansion: PropertyPresence<Required>,
    C: RefinementRoute<Required, <P::Expansion as PropertyPresence<Required>>::Relation>,
{
    type C =
        <C as RefinementRoute<Required, <P::Expansion as PropertyPresence<Required>>::Relation>>::C;
}

/// Constructive negative refinement for theories whose canonical signature has
/// no associated requirements or equations.
///
/// This is deliberately a constructive proof, not negation-as-trait-failure:
/// at least one property required by `Required` is known to be absent from the
/// source's closed transitive property graph.
impl<Required: Cat, C, P> Ⱶ<Required, Absent> for C
where
    C: Category<Properties = P>,
    P: ExpandProperties,
    <Required as Cat>::C: Category<Structure = Ø, Equations = Ø>,
    P::Expansion:
        CoversProperties<<<Required as Cat>::C as Category>::Properties, Relation = Absent>,
{
    type C = C;
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
    type ReflectedV = <Coords<f64, 2> as ι>::C;
    type RootRole = <ReflectedV as π>::𝒞;
    type RootC = <ReflectedV as π>::C;

    type V = <Coords<f64, 2> as ι>::C;
    type T = <V as Ⱶ<𝐓𝐞𝐧𝐬::𝒞>>::C;
    type Scalar = <T as π<tensor::F>>::X;
    type ScalarC = <T as π<tensor::F>>::C;

    fn assert_same_type<T>(_: T, _: T) {}
    fn assert_different_names<A, B>()
    where
        A: AssocName + CompareAssoc<B, Relation = Different>,
        B: AssocName,
    {
    }

    // Associated labels are identified by their declaring owner, not merely by
    // spelling. These two `Payload` roles therefore remain constructively distinct.
    assert_different_names::<jet::Payload, tensor_of::Payload>();

    assert_same_type(PhantomData::<RootRole>, PhantomData::<𝐕𝐞𝐜𝐭::𝒞>);
    assert_same_type(PhantomData::<RootC>, PhantomData::<ReflectedV>);
    assert_same_type(PhantomData::<Scalar>, PhantomData::<f64>);

    fn scalar_context_is_field<C: Ⱶ<𝐅𝐥𝐝::𝒞>>() {}
    scalar_context_is_field::<ScalarC>();

    // `Model` is literally the composition of the two public operations:
    // include the Rust type through `ι`, then select a theory through `Ⱶ`.
    type DirectFieldC = Model<𝐅𝐥𝐝::𝒞, f64>;
    assert_same_type(PhantomData::<ScalarC>, PhantomData::<DirectFieldC>);

    // `Ⱶ<𝐈𝐝<C>>` is the public concrete-category refinement judgement.
    fn refines_concrete_field_theory<C: Ⱶ<𝐈𝐝<𝐅𝐥𝐝::C>>>() {}
    refines_concrete_field_theory::<DirectFieldC>();

    // Self-referential associated structure remains a finite context handle.
    // For a real scalar, Fixed = Self, so following Fixed twice must return
    // literally the same finite child context rather than recursively expanding it.
    type FieldC = Model<𝐅𝐥𝐝::𝒞, f64>;
    type FixedC = <FieldC as π<field::Fixed>>::C;
    type FixedFixedC = <FixedC as π<field::Fixed>>::C;
    assert_same_type(PhantomData::<FixedC>, PhantomData::<FixedFixedC>);

    // The canonical inclusion of f64 retains its richest Real context.
    type RealC = <f64 as ι>::C;

    // The new Real decomposition is itself navigable through the turnstile.
    type OrderedFieldViaReal = <RealC as Ⱶ<𝐎𝐫𝐝𝐅𝐥𝐝::𝒞>>::C;
    type DedekindViaReal = <RealC as Ⱶ<𝐃𝐞𝐝𝐞𝐤𝐢𝐧𝐝::𝒞>>::C;
    assert_same_type(
        PhantomData::<OrderedFieldViaReal>,
        PhantomData::<Model<𝐎𝐫𝐝𝐅𝐥𝐝::𝒞, f64>>,
    );
    assert_same_type(
        PhantomData::<DedekindViaReal>,
        PhantomData::<Model<𝐃𝐞𝐝𝐞𝐤𝐢𝐧𝐝::𝒞, f64>>,
    );

    // Theory projection obeys the same normalization law as `π`: an inherited
    // theory returns the exact context stored by its parent.
    type CFieldViaReal = <RealC as Ⱶ<𝐂𝐅𝐥𝐝::𝒞>>::C;
    assert_same_type(
        PhantomData::<CFieldViaReal>,
        PhantomData::<Model<𝐂𝐅𝐥𝐝::𝒞, f64>>,
    );

    // Bare inherited theories use the same frontend even when there is no
    // concrete `BindsProperty` context to preserve.
    fn cfield_is_abelian<C: Ⱶ<𝐀𝐛::𝒞>>() {}
    cfield_is_abelian::<CFieldViaReal>();

    // The law is transitive too. Expanding Real -> CField -> Field must still
    // land on the canonical Field model, not a resolved proof graph.
    type FieldViaReal = <RealC as Ⱶ<𝐅𝐥𝐝::𝒞>>::C;
    assert_same_type(
        PhantomData::<FieldViaReal>,
        PhantomData::<Model<𝐅𝐥𝐝::𝒞, f64>>,
    );

    type Parent = 𝒯<ː<Binds<tensor::F, 𝐑𝐞𝐚𝐥::𝒞, f64, RealC>, Ø>, Ø>;
    type ChildRole = <Parent as π<tensor::F>>::𝒞;
    type ChildC = <Parent as π<tensor::F>>::C;
    type ChildRootRole = <ChildC as π>::𝒞;

    assert_same_type(PhantomData::<ChildRole>, PhantomData::<𝐑𝐞𝐚𝐥::𝒞>);
    assert_same_type(PhantomData::<ChildC>, PhantomData::<RealC>);
    assert_same_type(PhantomData::<ChildRootRole>, PhantomData::<𝐑𝐞𝐚𝐥::𝒞>);

    type RequiresFieldChild = 𝒯<ː<Requires<tensor::F, 𝐅𝐥𝐝::𝒞>, Ø>, Ø>;
    fn richer_edge_satisfies_weaker_requirement<C: Ⱶ<𝐈𝐝<RequiresFieldChild>>>() {}
    richer_edge_satisfies_weaker_requirement::<Parent>();

    fn child_keeps_real_information<C: Ⱶ<𝐅𝐥𝐝::𝒞> + Ⱶ<𝐑𝐞𝐚𝐥::𝒞>>() {}
    child_keeps_real_information::<ChildC>();

    // `BindsIncluded` is the canonical form of the same construction when the
    // child has an `ι` inclusion. The parent can require only Field while the
    // edge retains f64's actual Real role and exact Real context.
    type IncludedParent = 𝒯<ː<BindsIncluded<tensor::F, f64>, Ø>, Ø>;
    type IncludedChildRole = <IncludedParent as π<tensor::F>>::𝒞;
    type IncludedChildC = <IncludedParent as π<tensor::F>>::C;
    assert_same_type(PhantomData::<IncludedParent>, PhantomData::<Parent>);
    assert_same_type(PhantomData::<IncludedChildRole>, PhantomData::<𝐑𝐞𝐚𝐥::𝒞>);
    assert_same_type(PhantomData::<IncludedChildC>, PhantomData::<RealC>);
    richer_edge_satisfies_weaker_requirement::<IncludedParent>();
    child_keeps_real_information::<IncludedChildC>();

    // Negative refinement is a real semantic judgement, not a proxy marker.
    // `Model<CField, f64>` deliberately forgets the order/completeness evidence
    // carried by the canonical Real inclusion of f64.
    fn is_constructively_not_real<C: Ⱶ<𝐑𝐞𝐚𝐥::𝒞, Absent>>() {}
    is_constructively_not_real::<Model<𝐂𝐅𝐥𝐝::𝒞, f64>>();
}
