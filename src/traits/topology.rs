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

use crate::traits::{Arrow, ArrowCategory, CodomainOf, DomainOf};

use super::{
    Ob, Signature, TangentBundle, Tensor, TensorIn, arrow, manifold, π, Ⱶ, 𝐀𝐫𝐫, 𝐌𝐚𝐧, 𝐓𝐞𝐧𝐬, 𝐓𝐨𝐩
};

/// A chosen topology on a point type.
pub trait Topological<C: 𝐓𝐨𝐩::Ⱶ = 𝐓𝐨𝐩::C<Self>>: super::Point<C> {}

/// A smooth manifold compatible with Diffable's existing tangent-bundle API.
///
/// [`Manifold`] remains the unique carrier-level owner of the tangent and atlas
/// associated types. It is deliberately only a stable Rust frontend shell:
/// the mathematical topological judgement belongs to [`ManifoldIn<C>`], so
/// multiple valid manifold contexts do not create competing `Tangent` owners
/// and the shell does not silently select `Topological`'s default context.
pub trait Manifold: Clone + core::fmt::Debug + Sized {
    type Tangent: Tensor;
    type Atlas: TangentBundle<Self, Self::Tangent>;
}

mod manifold_in_sealed {
    use super::*;

    pub trait Sealed<C: 𝐌𝐚𝐧::Ⱶ>: Manifold + Topological<C> {}

    impl<C, M> Sealed<C> for M
    where
        M: Manifold + Topological<C>,
        C: 𝐌𝐚𝐧::Ⱶ
            + π<X = M>
            + π<manifold::Tangent, X = M::Tangent>,
        <C as π<manifold::Tangent>>::C: 𝐓𝐞𝐧𝐬::Ⱶ,
        M::Tangent: TensorIn<<C as π<manifold::Tangent>>::C>,
    {
    }
}

/// Certifies that the manifold structure of `Self` is valid in ontology context `C`.
#[doc(hidden)]
#[allow(private_bounds)]
pub trait ManifoldIn<C: 𝐌𝐚𝐧::Ⱶ>:
    Manifold + Topological<C> + manifold_in_sealed::Sealed<C>
{
}

impl<C: 𝐌𝐚𝐧::Ⱶ, M> ManifoldIn<C> for M where
    M: Manifold + Topological<C> + manifold_in_sealed::Sealed<C>
{
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
    C: Ⱶ<𝐀𝐫𝐫<𝐓𝐨𝐩::Theory>> + π<arrow::Typing, X: Signature>,
    F: Fn(&DomainOf<C>) -> CodomainOf<C>,
{
}

// This function is intentionally never called.  Its body is a compile-time proof
// that an arrow admitted into a *manifold* context is accepted by a theorem stated
// solely for arrows whose original context refines the topological arrow theory.
#[allow(dead_code)]
fn smooth_maps_are_continuous_without_a_nominal_bridge<M, N>(_m: M, n: N)
where
    M: Manifold + Ob<𝐌𝐚𝐧::Theory>,
    N: Manifold + Ob<𝐌𝐚𝐧::Theory>,
{
    let f = Arrow::<ArrowCategory<𝐌𝐚𝐧::Theory, M, N>>::new(|_x: &M| n.clone());

    theorem_about_continuous_maps(&f);
}
