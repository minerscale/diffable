//! Context-indexed morphisms over the reflected ontology.
//!
//! This module is a proving ground for the reflected ontology.  The central idea
//! is that an ordinary Rust callable becomes a mathematical morphism only when it
//! is explicitly admitted into a concrete structural context:
//!
//! ```text
//! Arrow<C, F>
//! ```
//!
//! `F` remains the concrete computational object. `C` is a compile-time proof
//! context: constructing [`Arrow`] certifies that `F` is the morphism described by
//! `C`. The context may then refine any number of weaker arrow theories, allowing
//! the same admitted arrow to be consumed by theorems about continuous maps, set
//! maps, or any other structurally implied interpretation without manufacturing
//! further wrapper types or trait implementations.
//!
//! No trait objects, allocation, or `std` are used. The callable remains fully
//! monomorphised; only its mathematical interpretation lives in the type system.

use crate::traits::C;
use core::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use super::{
    Associated, Binds, BindsAs, BindsProperty, Cat, Category, Codomain, Cons, Domain, Ob, Object,
    Refines, Reflect, Tangent, TangentBundle, Tensor, Ø, 𝐀𝐫𝐫, 𝐈𝐝, 𝐌𝐚𝐧, 𝐒𝐞𝐭, 𝐓𝐞𝐧𝐬, 𝐓𝐨𝐩, 𝒯,
};

/// The concrete Rust type found at the `Domain` edge of an arrow context.
#[allow(type_alias_bounds)]
pub type DomainOf<C: Category + Associated<Domain>> = <C as Associated<Domain>>::Type;

/// The concrete Rust type found at the `Codomain` edge of an arrow context.
#[allow(type_alias_bounds)]
pub type CodomainOf<C: Category + Associated<Codomain>> = <C as Associated<Codomain>>::Type;

/// Construct the concrete context of an arrow `Domain -> Codomain` in `C`.
///
/// This is the structural description into which an actual callable can later be
/// admitted with [`Arrow::new`].
#[allow(type_alias_bounds)]
pub type ArrowCategory<C: Category + 'static, D: Ob<C>, E: Ob<C>> = 𝒯<
    Cons<
        BindsAs<Domain, 𝐈𝐝<C>, D, <D as Object>::Context>,
        Cons<BindsAs<Codomain, 𝐈𝐝<C>, E, <E as Object>::Context>, Ø>,
    >,
    Ø,
>;

/// A concrete callable admitted as a morphism in structural context `C`.
///
/// The `F` field is the complete runtime representation. The context marker is
/// erased, but at compile time it records the exact mathematical interpretation
/// under which the callable was certified. Theorems should retain this richest
/// context and ask it to [`Refines`] whatever weaker arrow theory they require.
///
/// [`Arrow::new`] is deliberately a semantic trust boundary: Rust checks the
/// function signature against the context, but the caller certifies the
/// mathematical claim that the function really is a morphism in `C`.
pub struct Arrow<C: Category, F> {
    function: F,
    _context: PhantomData<fn() -> C>,
}

impl<C: Category, F> Deref for Arrow<C, F> {
    type Target = F;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.function
    }
}

impl<C: Category, F> DerefMut for Arrow<C, F> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.function
    }
}

impl<C, F> Arrow<C, F>
where
    C: Category + Associated<Domain> + Associated<Codomain>,
    F: Fn(&DomainOf<C>) -> CodomainOf<C>,
{
    /// Admit `function` as the morphism described by `C`.
    #[inline]
    pub fn new(function: F) -> Self {
        Self {
            function,
            _context: PhantomData,
        }
    }

    /// Forget the admission and recover the ordinary Rust callable.
    #[inline]
    pub fn into_inner(self) -> F {
        self.function
    }
}

impl<C: Category, F: Clone> Clone for Arrow<C, F> {
    fn clone(&self) -> Self {
        Self {
            function: self.function.clone(),
            _context: PhantomData,
        }
    }
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

impl<M: Manifold> super::Project<Tangent> for M {
    type Output = M::Tangent;
}

impl<M: Manifold> Reflect<𝐌𝐚𝐧> for M {
    type C = 𝒯<
        Cons<Binds<Tangent, 𝐓𝐞𝐧𝐬, M::Tangent>, Ø>,
        Cons<BindsProperty<𝐓𝐨𝐩, <M as Reflect<𝐓𝐨𝐩>>::C>, Ø>,
    >;
}

// -----------------------------------------------------------------------------
// The point of the experiment: theorem transport
// -----------------------------------------------------------------------------

/// A theorem which only knows that its arrow context refines arrows in `Top`.
///
/// The callable carries its original, possibly much richer context. Nothing is
/// converted to a separate "continuous map" type; the theorem merely asks rustc
/// to project the continuous-arrow interpretation out of that context.
#[allow(dead_code)]
fn theorem_about_continuous_maps<F, C>(_f: &Arrow<C, F>)
where
    C: Category
        + Associated<Domain>
        + Associated<Codomain>
        + Refines<𝐀𝐫𝐫<<𝐓𝐨𝐩 as Cat>::C>>,
    F: Fn(&DomainOf<C>) -> CodomainOf<C>,
{
}

// This function is intentionally never called.  Its body is a compile-time proof
// that an arrow admitted into a *manifold* context is accepted by a theorem stated
// solely for arrows whose original context refines the topological arrow theory.
#[allow(dead_code)]
fn smooth_maps_are_continuous_without_a_nominal_bridge<M, N>(_m: M, n: N)
where
    M: Manifold + Ob<C<𝐌𝐚𝐧>>,
    N: Manifold + Ob<C<𝐌𝐚𝐧>>,
{
    let f: Arrow<ArrowCategory<C<𝐌𝐚𝐧>, _, _>, _> = Arrow::new(move |_x: &M| n.clone());

    theorem_about_continuous_maps(&f);
}
