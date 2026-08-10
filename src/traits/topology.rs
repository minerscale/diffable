//! Context-indexed maps and a tiny allocation-free homotopy theory.
//!
//! This module is a proving ground for the reflected ontology.  The central idea
//! is that the mathematical interpretation belongs to the *trait application*,
//! not to the runtime value:
//!
//! ```text
//! F: Map<C>
//! ```
//!
//! says that the ordinary Rust callable `F` is being used as the arrow described
//! by concrete structural context `C`.  `C` contains the concrete domain and
//! codomain types and may refine any number of named arrow theories.  Consequently
//! a theorem can quantify over "any map living in a context which behaves like a
//! continuous map" without requiring a `ContinuousMap` nominal trait or a wrapper.
//!
//! No trait objects, allocation, or `std` are used.  Arbitrary closures participate
//! directly through the blanket [`Map`] implementation.

use crate::traits::C;

use super::{
    Associated, Binds, BindsAs, BindsProperty, Cat, Category, Codomain, Cons, Domain, Realizes,
    Refines, Reflect, SameType, Tangent, TangentBundle, Tensor, Ø, 𝐌𝐚𝐧, 𝐌𝐚𝐩, 𝐎𝐛, 𝐒𝐞𝐭, 𝐓𝐞𝐧𝐬, 𝐓𝐨𝐩, 𝒯,
};

/// The concrete Rust type found at the `Domain` edge of an arrow context.
#[allow(type_alias_bounds)]
pub type DomainOf<C: Category + Associated<Domain>> = <C as Associated<Domain>>::Type;

/// The concrete Rust type found at the `Codomain` edge of an arrow context.
#[allow(type_alias_bounds)]
pub type CodomainOf<C: Category + Associated<Codomain>> = <C as Associated<Codomain>>::Type;

/// Construct the concrete context of an arrow `Domain -> Codomain` in `C`.
///
/// Notice that `C` is present only in this type-level context.  It never becomes a
/// generic parameter of the runtime function/closure value.
#[allow(type_alias_bounds)]
pub type Arrow<C: Category + 'static, D: Realizes<C>, E: Realizes<C>> = 𝒯<
    Cons<
        BindsAs<Domain, 𝐎𝐛<C>, D, <D as Realizes<C>>::Context>,
        Cons<BindsAs<Codomain, 𝐎𝐛<C>, E, <E as Realizes<C>>::Context>, Ø>,
    >,
    Ø,
>;

/// A callable interpreted through structural arrow context `C`.
///
/// This is deliberately just a refinement of [`Fn`].  Since the input and output
/// types are read from `C`, Rust can blanket-implement it for *every* matching
/// closure without needing to recover a closure's anonymous argument type.
pub trait Map<C>: Fn(&DomainOf<C>) -> CodomainOf<C>
where
    C: Category + Associated<Domain> + Associated<Codomain>,
{
}

impl<C, F> Map<C> for F
where
    C: Category + Associated<Domain> + Associated<Codomain>,
    F: Fn(&DomainOf<C>) -> CodomainOf<C>,
{
}

/// A homotopy between two maps living in the same arrow context.
///
/// Sharing `C` means the endpoint maps have exactly the same domain and codomain.
/// `R` is explicit so arbitrary closures can again receive a blanket impl without
/// allocation or trait objects.  Implementing/using this trait certifies the usual
/// endpoint laws `H(0, x) = from(x)` and `H(1, x) = to(x)`; those semantic laws are
/// not numerically checked here.
pub trait Homotopy<C, From, To, R>: Fn(R, &DomainOf<C>) -> CodomainOf<C>
where
    C: Category + Associated<Domain> + Associated<Codomain>,
    From: Map<C>,
    To: Map<C>,
    R: super::Real,
{
}

impl<C, From, To, R, H> Homotopy<C, From, To, R> for H
where
    C: Category + Associated<Domain> + Associated<Codomain>,
    From: Map<C>,
    To: Map<C>,
    R: super::Real,
    H: Fn(R, &DomainOf<C>) -> CodomainOf<C>,
{
}

/// Two opposite arrows which are certified to be exact inverses.
///
/// `ForwardC` and `BackwardC` remain proof-level contexts; the map values are plain
/// callables.  The `SameType` bounds statically force their endpoints to reverse.
pub trait Isomorphism<ForwardC, BackwardC, Forward, Backward>
where
    ForwardC: Category + Associated<Domain> + Associated<Codomain>,
    BackwardC: Category + Associated<Domain> + Associated<Codomain>,
    DomainOf<ForwardC>: SameType<CodomainOf<BackwardC>>,
    CodomainOf<ForwardC>: SameType<DomainOf<BackwardC>>,
    Forward: Map<ForwardC>,
    Backward: Map<BackwardC>,
{
}

/// A chosen topology on a point type.
pub trait Topological: super::Point {}

/// A smooth manifold compatible with Diffable's existing tangent-bundle API.
pub trait Manifold: Topological + Sized {
    type Tangent: Tensor;
    type Atlas: TangentBundle<Self, Self::Tangent>;
}

// -----------------------------------------------------------------------------
// Object reflection
// -----------------------------------------------------------------------------

impl<T: Topological> Reflect<𝐓𝐨𝐩> for T {
    type C = 𝒯<Ø, Cons<BindsProperty<𝐒𝐞𝐭, C<𝐒𝐞𝐭>>, Ø>>;
}

impl<T: Topological> Realizes<C<𝐓𝐨𝐩>> for T {
    type Context = <T as Reflect<𝐓𝐨𝐩>>::C;
}

impl<M: Manifold> super::Project<Tangent> for M {
    type Output = M::Tangent;
}

impl<M: Manifold> Reflect<𝐌𝐚𝐧> for M {
    type C = 𝒯<
        Cons<Binds<Tangent, 𝐓𝐞𝐧𝐬, M::Tangent>, Ø>,
        Cons<BindsProperty<𝐓𝐨𝐩, <M as Reflect<𝐓𝐨𝐩>>::C>, Ø>,
    >;
}

impl<M: Manifold> Realizes<C<𝐌𝐚𝐧>> for M {
    type Context = <M as Reflect<𝐌𝐚𝐧>>::C;
}

// -----------------------------------------------------------------------------
// The point of the experiment: theorem transport
// -----------------------------------------------------------------------------

/// A theorem which only knows that its arrow context refines arrows in `Top`.
///
/// In particular, `F` need not implement any nominal `ContinuousMap` trait.  A
/// closure over a manifold arrow context is accepted because that context can be
/// structurally projected to `𝐌𝐚𝐩<Top>`.
#[allow(dead_code)]
fn theorem_about_continuous_maps<F, C>(_f: &F)
where
    C: Category
        + Associated<Domain>
        + Associated<Codomain>
        + Refines<𝐌𝐚𝐩<<𝐓𝐨𝐩 as Cat>::C>>,
    F: Map<C>,
{
}

/// The same transport works one level higher for homotopies: the operation is
/// parameterised by the rich arrow context, while the theorem asks only that the
/// context admits the continuous-arrow interpretation.
#[allow(dead_code)]
fn theorem_about_continuous_homotopies<H, F, G, C, R>(_h: &H, _f: &F, _g: &G)
where
    C: Category
        + Associated<Domain>
        + Associated<Codomain>
        + Refines<𝐌𝐚𝐩<<𝐓𝐨𝐩 as Cat>::C>>,
    F: Map<C>,
    G: Map<C>,
    R: super::Real,
    H: Homotopy<C, F, G, R>,
{
}

// This function is intentionally never called.  Its body is a compile-time proof
// that a closure over a *manifold* arrow context is accepted by a theorem stated
// solely for arrows which refine the topological arrow theory.
#[allow(dead_code)]
fn smooth_maps_are_continuous_without_a_nominal_bridge<M, N>(_m: M, n: N)
where
    M: Manifold,
    N: Manifold,
{
    type Smooth<M, N> = Arrow<C<𝐌𝐚𝐧>, M, N>;

    let f = move |_x: &M| n.clone();

    theorem_about_continuous_maps::<_, Smooth<M, N>>(&f);
}
