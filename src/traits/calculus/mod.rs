//! Tensor constructions, tangent lifts, and forward automatic differentiation.
//!
//! The differentiation API is a small typed language. [`d`] introduces a
//! derivative, [`d::along`] contracts one derivative slot with a direction,
//! and [`d::at`] or [`Along::at`] evaluates the completed program. Programs may
//! themselves be differentiated, so `d(d(f))` is the second derivative (the
//! Hessian for a scalar-valued map) and arbitrarily deep nestings use the same
//! machinery.
//!
//! Evaluation is implemented with truncated Taylor jets. [`Jet::new`] and
//! [`Jet::constant`] are the category-restricted constructors of the jet image,
//! while [`TensorOver::new`] performs scalar re-presentation. Their concrete
//! return types remain visible to Rust so all native trait structure is preserved.
//! [`JetVector`] names the tensor presentation needed by extension traits, while
//! [`JetMap`] is the internal interpretation rule for ordinary functions and
//! differential programs. [`ConstantRoute`] records how captured constants
//! must be embedded through nested presentations. [`EvaluableAt`] is the final
//! interpreter boundary and provides the user-facing diagnostic when a program
//! cannot be evaluated.
//!
//! [`Connection`] extends the construction from vector spaces to tangent
//! bundles. [`FormLift`] and [`NondegenerateLift`] state that lowering and
//! raising maps extend coherently when coordinates are replaced by jets.

mod connection;
mod differentiation;
mod jet;
mod tensor;

pub use connection::*;
pub use differentiation::*;
pub use jet::*;
pub use tensor::*;
