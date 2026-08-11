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

use crate::{
    C,
    traits::{Arrow, ArrowCategory, CodomainOf, DomainOf},
};

use super::{
    Cat, Ob, Signature, TangentBundle, Tensor, Typing, π, Ⱶ, 𝐀𝐫𝐫, 𝐌𝐚𝐧, 𝐓𝐨𝐩
};

/// A chosen topology on a point type.
pub trait Topological: super::Point {}

/// A smooth manifold compatible with Diffable's existing tangent-bundle API.
pub trait Manifold: Topological + Sized {
    type Tangent: Tensor;
    type Atlas: TangentBundle<Self, Self::Tangent>;
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
    C: Ⱶ<𝐀𝐫𝐫<C![𝐓𝐨𝐩]>> + π<Typing, X: Signature>,
    F: Fn(&DomainOf<C>) -> CodomainOf<C>,
{
}

// This function is intentionally never called.  Its body is a compile-time proof
// that an arrow admitted into a *manifold* context is accepted by a theorem stated
// solely for arrows whose original context refines the topological arrow theory.
#[allow(dead_code)]
fn smooth_maps_are_continuous_without_a_nominal_bridge<M, N>(_m: M, n: N)
where
    M: Manifold + Ob<C![𝐌𝐚𝐧]>,
    N: Manifold + Ob<C![𝐌𝐚𝐧]>,
{
    let f = Arrow::<ArrowCategory<C![𝐌𝐚𝐧], M, N>>::new(|_x: &M| n.clone());

    theorem_about_continuous_maps(&f);
}
