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
    Binds, BindsProperty, BindsTyping, Cat, Category, Cons, Ob, Reflect, Signature, Tangent,
    TangentBundle, Tensor, Typing, Ø, π, Ⱶ, 𝐀𝐫𝐫, 𝐌𝐚𝐧, 𝐒𝐞𝐭, 𝐓𝐞𝐧𝐬, 𝐓𝐨𝐩, 𝒯,
};

/// The concrete Rust domain of an arrow context.
#[allow(type_alias_bounds)]
pub type DomainOf<C: Ⱶ<Typing, X: Signature>> = <<C as Ⱶ<Typing>>::X as Signature>::Domain;

/// The concrete Rust codomain of an arrow context.
#[allow(type_alias_bounds)]
pub type CodomainOf<C: Ⱶ<Typing, X: Signature>> = <<C as Ⱶ<Typing>>::X as Signature>::Codomain;

/// Construct the concrete context of an arrow `D -> E` in `C`.
///
/// Both endpoints are bound by one structural association, so a domain cannot
/// exist without its codomain.
#[allow(type_alias_bounds)]
pub type ArrowCategory<C: Category + 'static, D: Ob<C>, E: Ob<C>> =
    𝒯<Cons<BindsTyping<C, D, E>, Ø>, Ø>;

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

impl<C: Ⱶ<Typing, X: Signature>, F: Fn(&DomainOf<C>) -> CodomainOf<C>> Arrow<C, F> {
    /// Admit `function` as the morphism described by `C`.
    #[inline]
    pub fn new(f: F) -> Self {
        Self {
            f,
            ctx: PhantomData,
        }
    }

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
    C: π<𝐀𝐫𝐫<<𝐓𝐨𝐩 as Cat>::C>> + Ⱶ<Typing, X: Signature>,
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
