//! Tensors, scalar actions, duality, and geometric forms.
//!
//! [`Tensor`] carries the coordinate shape, scalar field, handedness, and
//! available actions needed by the tensor algebra. [`Vector`] selects tensors
//! with an accessible scalar action. [`Dual`] reverses variance, [`Form`] elects
//! a lowering map, and [`Nondegenerate`], [`Sesquilinear`], and [`InnerProduct`]
//! progressively refine its geometry.

use core::ops::{Add, Index, IndexMut, Mul, Neg, Sub};
use num_traits::{Zero, real::Real as _};

#[cfg(feature = "testing")]
use super::Chart;

use super::{Field, LieGroup, Metric, Real, calculus::NondegenerateLift};
use crate::{
    impl_group_via_add, impl_vector_ops,
    traits::{Interval, Point},
};

/// A finite-dimensional Euclidean space.
///
/// The space of all values of a type `E: Euclidean` is interpreted as
/// `R^N` (with `R := E::F` and `N := E::N`) — the canonical flat, *positive-
/// definite* space of dimension `N` over the field `R`. This is the space in
/// which local coordinate charts take their values, and in which tangent
/// vectors live.
///
/// `Euclidean` is the **definite real-valued refinement** of [`Bilinear`]:
/// it is a pseudo-Euclidean space (signature `(N, 0)`) that additionally
/// carries an [`InnerProduct`] — a positive-definite pairing inducing a genuine
/// `norm` and a [`Metric`]. Where the pseudo-Euclidean base has only a signed
/// [`Bilinear`] scalar product, a Euclidean space has all the metric-space
/// structure on top, because definiteness is exactly what makes
/// `sqrt(⟨v,v⟩)` real and the induced distance a metric.
///
/// Beyond the algebraic structure of a vector space (`Add`, `Sub`, `Mul`,
/// `Neg`, `Zero`), it carries that inner product and a canonical tangent
/// bundle ([`TangentBundle`]) whose charts are globally defined with infinite
/// injectivity radius — reflecting the flatness of the space.
///
/// # Flatness
/// Unlike a general Riemannian manifold, a Euclidean space is flat: geodesics
/// are straight lines, parallel transport is path-independent, and the
/// exponential map is a global isomorphism rather than merely a local one.
/// These properties are verified by the `check_*` methods inherited from
/// [`TangentBundle`], [`Vector`], [`Form`], [`Nondegenerate`] and [`Sesquilinear`]
/// together with the definite-only `check_pythagorean` below.
///
/// [`NondegenerateLift`] additionally certifies that the musical isomorphisms
/// extend through jet-valued scalar presentations. This is what permits a
/// function quantified over `V: Euclidean` to be evaluated recursively by
/// `d(d(f))` rather than losing its Euclidean bound at the first jet layer.
///
/// # Implementing
/// Use the `test_euclidean!` macro to verify that your implementation
/// satisfies the Euclidean axioms. (For an indefinite space, implement only
/// [`Sesquilinear`] and use `test_pseudo_euclidean!` instead.)
///
/// [`Bilinear`]: crate::traits::Bilinear
/// [`InnerProduct`]: crate::traits::InnerProduct
/// [`Metric`]: crate::traits::Metric
/// [`TangentBundle`]: crate::traits::TangentBundle
/// [`Form`]: crate::traits::Form
/// [`Nondegenerate`]: crate::traits::Nondegenerate
/// [`NondegenerateLift`]: crate::traits::calculus::NondegenerateLift
/// [`Sesquilinear`]: crate::traits::Sesquilinear
pub trait Euclidean:
    Bilinear<F: Real, Action = BothSided> + InnerProduct + NondegenerateLift + Vector
{
    // Pythagorean theorem: d(a, b)² == |a - b|²
    #[cfg(feature = "testing")]
    fn check_pythagorean(a: &Self, b: &Self) -> bool {
        let dist_sq = a.distance(b);
        let dist_sq = dist_sq * dist_sq;
        let diff = a.clone() - b.clone();
        let norm_sq = diff.norm_squared();
        dist_sq == norm_sq
    }
}

/// The dual space `V*` — the linear functionals on `V`.
///
/// Stored as a `V` internally, because [`pairing`](Tensor::pairing) is fixed to
/// the coordinate dot product, which identifies the dual basis with the primal
/// basis component-wise. A `Dual<V>` is therefore coordinate-identical to the
/// `V` holding its components — the wrapper exists purely so the type system
/// keeps covariant and contravariant tensors apart. That separation is what
/// lets [`Matrix`](crate::matrix::Matrix) enforce index variance and the musical
/// maps [`flat`](Form::flat)/[`sharp`](Nondegenerate::sharp) land in the correct
/// space.
///
/// Dualisation reverses handedness: if `V` is a right module, `Dual<V>` is a
/// left module, and conversely. Its ordinary `Mul<V::F>` implementation uses
/// that opposite action. In particular, constructing a `Dual<V>` from raw
/// coordinates does not apply [`flat`](Form::flat); it merely declares that the
/// supplied coordinates are covector coordinates.
///
/// Obtain a covector with a geometric meaning through [`flat`](Form::flat), not
/// [`from_raw`](Dual::from_raw) — the latter is a bare relabel that ignores the
/// metric.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Dual<V: Tensor>(V);

#[doc(hidden)]
pub enum Undecorated {}
#[doc(hidden)]
pub enum Dualized {}
#[doc(hidden)]
pub enum Sinistered {}
#[doc(hidden)]
pub enum DualSinistered {}

#[doc(hidden)]
pub trait TensorDecoration {
    type ToggleDual: TensorDecoration;
    type ToggleSinister: TensorDecoration;

    type Hand<H: Handedness>: Handedness;
}
impl TensorDecoration for Undecorated {
    type ToggleDual = Dualized;
    type ToggleSinister = Sinistered;
    type Hand<H: Handedness> = H;
}
impl TensorDecoration for Dualized {
    type ToggleDual = Undecorated;
    type ToggleSinister = DualSinistered;
    type Hand<H: Handedness> = H::Opposite;
}
impl TensorDecoration for Sinistered {
    type ToggleDual = DualSinistered;
    type ToggleSinister = Undecorated;
    type Hand<H: Handedness> = H::Opposite;
}
impl TensorDecoration for DualSinistered {
    type ToggleDual = Sinistered;
    type ToggleSinister = Dualized;
    type Hand<H: Handedness> = H;
}

#[doc(hidden)]
pub trait ApplyTensorDecoration<V: Tensor>: TensorDecoration {
    type Output: Tensor<F = V::F, Action = V::Action, Hand = Self::Hand<V::Hand>>;
    fn apply(v: V) -> Self::Output;
}
impl<V: Tensor> ApplyTensorDecoration<V> for Undecorated {
    type Output = V;
    fn apply(v: V) -> V {
        v
    }
}
impl<V: Tensor> ApplyTensorDecoration<V> for Dualized {
    type Output = Dual<V>;
    fn apply(v: V) -> Dual<V> {
        Dual(v)
    }
}
impl<V: Tensor<Action = BothSided>> ApplyTensorDecoration<V> for Sinistered {
    type Output = Sinister<V>;
    fn apply(v: V) -> Sinister<V> {
        Sinister(v)
    }
}
impl<V: Tensor<Action = BothSided>> ApplyTensorDecoration<V> for DualSinistered {
    type Output = Sinister<Dual<V>>;
    fn apply(v: V) -> Sinister<Dual<V>> {
        Sinister(Dual(v))
    }
}

#[doc(hidden)]
pub trait NormalizeWith<D: TensorDecoration>: Tensor {
    type Normalized: Tensor<F = Self::F, Action = Self::Action, Hand = D::Hand<Self::Hand>>;

    fn normalize_with(self) -> Self::Normalized;
}

/// Marks a tensor as a leaf in the tensor-expression normalization tree.
pub struct Atomic;

#[doc(hidden)]
pub struct NormalizeDual;

#[doc(hidden)]
pub struct NormalizeSinister;

impl<T: Tensor> TensorNormalizer<T> for Atomic {
    type Undecorated = T;
    type Dualized = Dual<T>;

    type Sinistered
        = Sinister<T>
    where
        T::Action: Rehandable;

    type DualSinistered
        = Sinister<Dual<T>>
    where
        T::Action: Rehandable;

    fn undecorated(tensor: T) -> Self::Undecorated {
        tensor
    }

    fn dualized(tensor: T) -> Self::Dualized {
        Dual(tensor)
    }

    fn sinistered(tensor: T) -> Self::Sinistered
    where
        T::Action: Rehandable,
    {
        Sinister(tensor)
    }

    fn dual_sinistered(tensor: T) -> Self::DualSinistered
    where
        T::Action: Rehandable,
    {
        Sinister(Dual(tensor))
    }
}

pub trait Normalize: Tensor + NormalizeWith<Undecorated> {
    fn normalize(self) -> <Self as NormalizeWith<Undecorated>>::Normalized {
        <Self as NormalizeWith<Undecorated>>::normalize_with(self)
    }
}

impl<T: Tensor + NormalizeWith<Undecorated>> Normalize for T {}

impl<V: Tensor> Dual<V> {
    /// This is a naive constructor! Do not use this
    /// for geometric computation. It exists only to help
    /// with the implementation of `Form` on types.
    pub fn from_raw(v: V) -> Self {
        Self(v)
    }

    /// This is a naive projection! Do not use this
    /// for geometric computation. It exists only to help
    /// with the implementation of `Form` on types.
    pub fn to_raw(v: Self) -> V {
        v.0
    }
}

impl<V: Tensor> TensorNormalizer<Dual<V>> for NormalizeDual {
    type Undecorated = <V as NormalizeWith<Dualized>>::Normalized;

    type Dualized = <V as NormalizeWith<Undecorated>>::Normalized;

    type Sinistered
        = <V as NormalizeWith<DualSinistered>>::Normalized
    where
        V::Action: Rehandable;

    type DualSinistered
        = <V as NormalizeWith<Sinistered>>::Normalized
    where
        V::Action: Rehandable;

    fn undecorated(tensor: Dual<V>) -> Self::Undecorated {
        <V as NormalizeWith<Dualized>>::normalize_with(tensor.0)
    }

    fn dualized(tensor: Dual<V>) -> Self::Dualized {
        <V as NormalizeWith<Undecorated>>::normalize_with(tensor.0)
    }

    fn sinistered(tensor: Dual<V>) -> Self::Sinistered
    where
        V::Action: Rehandable,
    {
        <V as NormalizeWith<DualSinistered>>::normalize_with(tensor.0)
    }

    fn dual_sinistered(tensor: Dual<V>) -> Self::DualSinistered
    where
        V::Action: Rehandable,
    {
        <V as NormalizeWith<Sinistered>>::normalize_with(tensor.0)
    }
}

impl<V: Tensor> Tensor for Dual<V> {
    type Normalization = NormalizeDual;
    type F = V::F;
    type Hand = <V::Hand as Handedness>::Opposite;
    type Action = V::Action;

    type Array<T: Point> = V::Array<T>;

    fn from_fn(f: impl FnMut(usize) -> Self::F) -> Self {
        Self(V::from_fn(f))
    }
}

impl_vector_ops!(Dual<V>, V: Tensor);

impl<V: Tensor> AsRef<<Dual<V> as Tensor>::Array<V::F>> for Dual<V> {
    fn as_ref(&self) -> &<Dual<V> as Tensor>::Array<V::F> {
        self.0.as_ref()
    }
}

impl<V: Tensor> AsMut<<Dual<V> as Tensor>::Array<V::F>> for Dual<V> {
    fn as_mut(&mut self) -> &mut <Dual<V> as Tensor>::Array<V::F> {
        self.0.as_mut()
    }
}

impl<V: Nondegenerate> Form for Dual<V> {
    fn flat(&self) -> Dual<Self> {
        Dual(Dual(V::sharp(self.clone())))
    }
}

impl<V: Nondegenerate> Nondegenerate for Dual<V> {
    fn sharp(v: Dual<Self>) -> Self {
        v.0.0.flat()
    }
}

impl<V: Nondegenerate + Sesquilinear> Sesquilinear for Dual<V> where Self: Vector {}

impl<V: Nondegenerate + Interval> Interval for Dual<V> {
    type R = V::R;

    fn interval_squared(&self, other: &Self) -> Self::R {
        let a = V::sharp(self.clone());
        let b = V::sharp(other.clone());

        a.interval_squared(&b)
    }
}

impl<V: Nondegenerate + Metric> Metric for Dual<V> {}

impl<V: Euclidean> Euclidean for Dual<V> {}

/// The same two-sided module as `V`, with the opposite preferred hand.
///
/// `Sinister<V>` does not construct a dual space and does not alter the
/// underlying coordinates, scalar field, form, or available scalar actions.
/// It merely changes which of the two existing scalar actions is exposed as
/// ordinary multiplication:
///
/// - if `V` is right-handed, `Sinister<V>` is left-handed;
/// - if `V` is left-handed, `Sinister<V>` is right-handed.
///
/// Consequently, this construction is defined only when
/// `V::Action = BothSided`. Rehanding a genuinely one-sided module would require
/// inventing a scalar action which the original module does not possess.
///
/// This distinction matters over noncommutative fields. For a right-handed
/// `v: V`, `v * a` uses the right action; for the corresponding
/// `Sinister(v)`, `a * v` is selected instead. Over a commutative field these
/// actions agree extensionally, but the wrapper remains useful because the
/// elected hand still controls tensor-product composition.
///
/// # Not a dual
///
/// Although [`Dual<V>`] also reverses handedness, `Sinister<V>` and `Dual<V>`
/// have different meanings. A dual is a space of linear functionals and
/// participates in the canonical evaluation pairing. A sinister vector is
/// still an element of the original module, viewed through its other action.
///
/// Keeping these constructions nominally distinct prevents a rehanded vector
/// from being mistaken for a covector merely because its coordinates and
/// handedness happen to coincide.
///
/// # Geometry
///
/// Any structure which is insensitive to the elected preferred hand may be
/// transported through this wrapper. In particular, forms and their musical
/// isomorphisms can be inherited when their scalar behaviour is compatible
/// with both actions.
///
/// Double rehanding restores the original module up to the canonical
/// coordinate-preserving isomorphism:
///
/// ```text
/// Sinister<Sinister<V>> ≅ V.
/// ```
///
/// Dualisation and rehanding also commute up to a canonical isomorphism:
///
/// ```text
/// Dual<Sinister<V>> ≅ Sinister<Dual<V>>.
/// ```
#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Sinister<V: Tensor<Action: Rehandable>>(pub V);

pub trait ApplicableTensorDecoration<T: Tensor>:
    TensorDecoration + ApplyTensorDecoration<T>
{
}

impl<T: Tensor> ApplicableTensorDecoration<T> for Undecorated {}
impl<T: Tensor> ApplicableTensorDecoration<T> for Dualized {}

impl<T: Tensor<Action = BothSided>> ApplicableTensorDecoration<T> for Sinistered {}

impl<T: Tensor<Action = BothSided>> ApplicableTensorDecoration<T> for DualSinistered {}

#[doc(hidden)]
pub trait TensorNormalizer<T: Tensor> {
    type Undecorated: Tensor<F = T::F, Action = T::Action, Hand = T::Hand>;

    type Dualized: Tensor<F = T::F, Action = T::Action, Hand = <T::Hand as Handedness>::Opposite>;

    type Sinistered: Tensor<F = T::F, Action = T::Action, Hand = <T::Hand as Handedness>::Opposite>
    where
        T::Action: Rehandable;

    type DualSinistered: Tensor<F = T::F, Action = T::Action, Hand = T::Hand>
    where
        T::Action: Rehandable;

    fn undecorated(tensor: T) -> Self::Undecorated;

    fn dualized(tensor: T) -> Self::Dualized;

    fn sinistered(tensor: T) -> Self::Sinistered
    where
        T::Action: Rehandable;

    fn dual_sinistered(tensor: T) -> Self::DualSinistered
    where
        T::Action: Rehandable;
}

impl<V> TensorNormalizer<Sinister<V>> for NormalizeSinister
where
    V: Tensor<Action: Rehandable>,
{
    type Undecorated = <V as NormalizeWith<Sinistered>>::Normalized;

    type Dualized = <V as NormalizeWith<DualSinistered>>::Normalized;

    type Sinistered = <V as NormalizeWith<Undecorated>>::Normalized;

    type DualSinistered = <V as NormalizeWith<Dualized>>::Normalized;

    fn undecorated(tensor: Sinister<V>) -> Self::Undecorated {
        <V as NormalizeWith<Sinistered>>::normalize_with(tensor.0)
    }

    fn dualized(tensor: Sinister<V>) -> Self::Dualized {
        <V as NormalizeWith<DualSinistered>>::normalize_with(tensor.0)
    }

    fn sinistered(tensor: Sinister<V>) -> Self::Sinistered {
        <V as NormalizeWith<Undecorated>>::normalize_with(tensor.0)
    }

    fn dual_sinistered(tensor: Sinister<V>) -> Self::DualSinistered {
        <V as NormalizeWith<Dualized>>::normalize_with(tensor.0)
    }
}

impl<V: Tensor<Action: Rehandable>> Tensor for Sinister<V> {
    type Normalization = NormalizeSinister;
    type F = V::F;
    type Hand = <V::Hand as Handedness>::Opposite;
    type Action = V::Action;

    type Array<T: Point> = V::Array<T>;

    fn from_fn(f: impl FnMut(usize) -> Self::F) -> Self {
        Self(V::from_fn(f))
    }
}

impl<T: Tensor> NormalizeWith<Undecorated> for T {
    type Normalized = <T::Normalization as TensorNormalizer<T>>::Undecorated;

    fn normalize_with(self) -> Self::Normalized {
        <T::Normalization as TensorNormalizer<T>>::undecorated(self)
    }
}

impl<T: Tensor> NormalizeWith<Dualized> for T {
    type Normalized = <T::Normalization as TensorNormalizer<T>>::Dualized;

    fn normalize_with(self) -> Self::Normalized {
        <T::Normalization as TensorNormalizer<T>>::dualized(self)
    }
}

impl<T> NormalizeWith<Sinistered> for T
where
    T: Tensor,
    T::Action: Rehandable,
{
    type Normalized = <T::Normalization as TensorNormalizer<T>>::Sinistered;

    fn normalize_with(self) -> Self::Normalized {
        <T::Normalization as TensorNormalizer<T>>::sinistered(self)
    }
}

impl<T> NormalizeWith<DualSinistered> for T
where
    T: Tensor,
    T::Action: Rehandable,
{
    type Normalized = <T::Normalization as TensorNormalizer<T>>::DualSinistered;

    fn normalize_with(self) -> Self::Normalized {
        <T::Normalization as TensorNormalizer<T>>::dual_sinistered(self)
    }
}

impl_vector_ops!(
    Sinister<V>,
    V: Tensor<Action: Rehandable>
);

impl<V: Tensor<Action: Rehandable>> AsRef<<Sinister<V> as Tensor>::Array<V::F>> for Sinister<V> {
    fn as_ref(&self) -> &<Sinister<V> as Tensor>::Array<V::F> {
        self.0.as_ref()
    }
}

impl<V: Tensor<Action: Rehandable>> AsMut<<Sinister<V> as Tensor>::Array<V::F>> for Sinister<V> {
    fn as_mut(&mut self) -> &mut <Sinister<V> as Tensor>::Array<V::F> {
        self.0.as_mut()
    }
}

impl<V> Form for Sinister<V>
where
    V: Form<Action = BothSided>,
{
    fn flat(&self) -> Dual<Self> {
        let flat: Dual<V> = self.0.flat();

        // Dual<Sinister<V>> and Sinister<Dual<V>>
        // are coordinate-identical.
        Dual(Sinister(flat.0))
    }
}

impl<V: Nondegenerate<Action = BothSided>> Nondegenerate for Sinister<V> {
    fn sharp(v: Dual<Self>) -> Self {
        Sinister(V::sharp(Dual(v.0.0)))
    }
}

// The reversed bilinear form is bilinear. Not true for Sesquilinear forms.
impl<V: Bilinear<Action = BothSided>> Sesquilinear for Sinister<V> {}

impl<V: Tensor<Action = BothSided> + Interval> Interval for Sinister<V> {
    type R = V::R;

    fn interval_squared(&self, other: &Self) -> Self::R {
        self.0.interval_squared(&other.0)
    }
}

impl<V: Tensor<Action = BothSided> + Metric> Metric for Sinister<V> {}

impl<V: Euclidean> Euclidean for Sinister<V> {}

/// The runtime witness for a module's elected scalar-action side.
#[derive(Debug, Copy, Clone)]
pub enum Hand {
    Left,
    Right,
}

/// A type-level choice of left- or right-handed scalar action.
///
/// [`Tensor::Hand`] uses this trait to determine both ordinary scalar
/// multiplication and canonical evaluation order.
pub trait Handedness {
    /// The hand elected by the dual module.
    type Opposite: Handedness<Opposite = Self>;

    /// The runtime value corresponding to this type-level hand.
    const H: Hand;
}

/// Type-level left scalar action.
///
/// This is the opposite of [`Right`] and is selected through
/// [`Tensor::Hand`].
pub enum Left {}

/// Type-level right scalar action.
///
/// This is the conventional hand for concrete coordinate vectors and the
/// opposite of [`Left`].
pub enum Right {}

impl Handedness for Left {
    type Opposite = Right;
    const H: Hand = Hand::Left;
}

impl Handedness for Right {
    type Opposite = Left;
    const H: Hand = Hand::Right;
}

/// The number of externally available scalar actions on a [`Tensor`].
///
/// [`NoSided`] exposes no scalar multiplication, [`OneSided`] exposes only the
/// side elected by [`Tensor::Hand`], and [`BothSided`] exposes compatible
/// actions on both sides. Direct sums take the weaker sidedness through
/// [`Sidedness::Meet`].
pub trait Sidedness: Copy + Clone + core::fmt::Debug {
    /// The lesser of this sidedness and the other sidedness.
    type Meet<T: Sidedness>: Sidedness;
    #[doc(hidden)]
    type MeetOne: Sidedness;

    /// The runtime value corresponding to this type-level side.
    const S: Side;
}

/// Runtime reflection of a type-level [`Sidedness`].
#[derive(Debug, Copy, Clone)]
pub enum Side {
    /// No external scalar action is available.
    None,
    /// Only the side elected by [`Tensor::Hand`] is available.
    Same,
    /// Compatible scalar actions are available on both sides.
    Both,
}

/// A tensor with no externally available scalar action.
#[derive(Debug, Copy, Clone)]
pub enum NoSided {}
/// A tensor with only its preferred scalar action available.
#[derive(Debug, Copy, Clone)]
pub enum OneSided {}
/// A tensor with compatible left and right scalar actions.
#[derive(Debug, Copy, Clone)]
pub enum BothSided {}

/// A [`Sidedness`] which permits participation in another tensor product.
///
/// [`NoSided`] deliberately does not implement this trait. [`Product`](Self::Product)
/// computes how many exterior actions survive after composing two tensors.
pub trait ActionExists: Sidedness {
    /// The sidedness remaining after tensoring with `T`.
    type Product<T: ActionExists>: Sidedness;

    /// Product with `OneSided`.
    #[doc(hidden)]
    type ProductOne: Sidedness;
}

impl ActionExists for OneSided {
    type Product<T: ActionExists> = <T as ActionExists>::ProductOne;

    type ProductOne = NoSided;
}

impl ActionExists for BothSided {
    type Product<T: ActionExists> = T;

    type ProductOne = OneSided;
}

/// A sidedness for which the opposite scalar action exists, so the
/// preferred hand may be reversed without inventing structure.
pub trait Rehandable: ActionExists {}

impl Rehandable for BothSided {}

/// Computes the exterior action and preferred hand of a tensor product.
///
/// The interior actions are consumed by balancing; these associated types
/// record which exterior action remains on the resulting [`TensorProduct`](crate::traits::calculus::TensorProduct).
pub trait TensorProductAction<Rhs: ActionExists>: ActionExists {
    /// The sidedness of the resulting tensor product.
    type Action: Sidedness;
    /// The preferred surviving exterior action.
    type Hand: Handedness;
}

impl<T: ActionExists> TensorProductAction<T> for OneSided {
    type Action = <OneSided as ActionExists>::Product<T>;

    // One ⊗ One is signed zero, defaulting Right.
    // One ⊗ Both retains only the right action.
    type Hand = Right;
}

impl TensorProductAction<OneSided> for BothSided {
    // The left exterior action survives.
    type Action = OneSided;
    type Hand = Left;
}

impl TensorProductAction<BothSided> for BothSided {
    // Both exterior actions survive, defaulting Right.
    type Action = BothSided;
    type Hand = Right;
}

impl Sidedness for OneSided {
    type Meet<T: Sidedness> = T::MeetOne;
    type MeetOne = OneSided;

    const S: Side = Side::Same;
}

impl Sidedness for BothSided {
    type Meet<T: Sidedness> = T;
    type MeetOne = OneSided;

    const S: Side = Side::Both;
}

impl Sidedness for NoSided {
    type Meet<T: Sidedness> = NoSided;
    type MeetOne = NoSided;

    const S: Side = Side::None;
}

/// A fixed-size coordinate container with a canonical flat index order.
///
/// `Array<T>` is the representation layer underlying [`Tensor`]. It describes
/// how a finite collection of coordinates of type `T` is stored, constructed,
/// indexed, and traversed. The container carries no algebraic meaning itself:
/// addition, scalar actions, handedness, forms, and tensor structure belong to
/// the tensor which selects it.
///
/// Unlike a slice, an `Array` has a dimension known as part of its type through
/// [`N`](Array::N). Unlike a built-in `[T; N]`, it need not use contiguous
/// storage. Direct sums and tensor products may use nested representations while
/// still exposing one canonical flat coordinate order.
///
/// All coordinate access must agree on that order:
///
/// - valid indices are `0..Self::N`;
/// - [`Index`] and [`IndexMut`] select the corresponding coordinate;
/// - [`iter`](Array::iter), [`iter_mut`](Array::iter_mut), and consuming
///   iteration visit coordinates in increasing index order;
/// - [`from_fn`](Array::from_fn) places `f(i)` at index `i`.
///
/// This agreement is what allows generic tensor operations to move between
/// indexing, iteration, and reconstruction without knowing the physical
/// representation.
///
/// # Type families
///
/// A tensor selects an array *family*:
///
/// ```text
/// type Array<T: Point>: Array<T>;
/// ```
///
/// The shape and coordinate order therefore remain fixed while the element type
/// changes. This is essential for constructions such as automatic
/// differentiation, where `V::Array<V::F>` may be replaced by
/// `V::Array<Jet<𝐅𝐥𝐝, V::F>>` without changing the logical tensor.
///
/// Implementations must preserve the same dimension and layout for every
/// admissible `T`.
pub trait Array<T: Point>:
    Point + Sized + Index<usize, Output = T> + IndexMut<usize> + IntoIterator<Item = T>
{
    /// The number of elements in this array.
    ///
    /// Exactly the indices `0..Self::N` must be valid.
    const N: usize;

    /// The iterator returned when borrowing this array.
    ///
    /// Items must be yielded in increasing flat-index order.
    type Iter<'a>: Iterator<Item = &'a T>
    where
        Self: 'a,
        T: 'a;

    /// The iterator returned when mutably borrowing this array.
    ///
    /// Items must be yielded in increasing flat-index order.
    type IterMut<'a>: Iterator<Item = &'a mut T>
    where
        Self: 'a,
        T: 'a;

    /// Iterates over shared references in canonical flat-index order.
    fn iter(&self) -> Self::Iter<'_>;

    /// Iterates over mutable references in canonical flat-index order.
    fn iter_mut(&mut self) -> Self::IterMut<'_>;

    /// Constructs an array by evaluating `f` for every coordinate index.
    ///
    /// The value returned by `f(i)` is placed at index `i`. Implementations
    /// should call `f` exactly once for each index in `0..Self::N`, in increasing
    /// order.
    fn from_fn(f: impl FnMut(usize) -> T) -> Self;
}

impl<T: Point, const N: usize> Array<T> for [T; N] {
    const N: usize = N;

    type Iter<'a>
        = core::slice::Iter<'a, T>
    where
        Self: 'a,
        T: 'a;

    type IterMut<'a>
        = core::slice::IterMut<'a, T>
    where
        Self: 'a,
        T: 'a;

    fn iter(&self) -> Self::Iter<'_> {
        self.as_slice().iter()
    }

    fn iter_mut(&mut self) -> Self::IterMut<'_> {
        self.as_mut_slice().iter_mut()
    }

    fn from_fn(f: impl FnMut(usize) -> T) -> Self {
        core::array::from_fn(f)
    }
}

impl<V: Tensor<Action = BothSided>> Sinister<Sinister<V>> {
    /// The enemy of my enemy is my friend.
    /// Dancing with two left feet.
    /// Two wrongs make a right.
    /// You get the point.
    pub fn collapse(self) -> V {
        self.0.0
    }
}

impl<V: Tensor> Dual<Dual<V>> {
    /// The canonical identification `V** ≅ V`, collapsing a twice-dualised
    /// vector back to `V`.
    ///
    /// This is the *evaluation* isomorphism `v ↦ (φ ↦ φ(v))`, which exists for
    /// any finite-dimensional space with no dependence on a metric or
    /// nondegeneracy — every [`Tensor`] qualifies via its fixed dimension `N`.
    /// It is a pure coordinate relabel (strip two [`Dual`] wrappers) precisely
    /// because [`pairing`](Tensor::pairing) is fixed to the coordinate dot
    /// product, which identifies each dual basis with its primal basis
    /// component-wise. Do not confuse this with the *musical* `V** ≅ V` of
    /// [`Nondegenerate`], which routes through the metric and requires an
    /// invertible form.
    pub fn collapse(self) -> V {
        self.0.0
    }
}

impl<V: Tensor<Action = BothSided>> Dual<Sinister<V>> {
    /// The canonical relabelling
    /// `Dual<Sinister<Self>> ≅ Sinister<Dual<Self>>`.
    pub fn dual_sinister(self) -> Sinister<Dual<V>> {
        Sinister(Dual(self.0.0))
    }
}

impl<V: Tensor<Action = BothSided>> Sinister<Dual<V>> {
    /// The inverse canonical relabelling
    /// `Sinister<Dual<Self>> ≅ Dual<Sinister<Self>>`.
    pub fn sinister_dual(self) -> Dual<Sinister<V>> {
        Dual(Sinister(self.0.0))
    }
}

/// A finite-dimensional element of the tensor algebra, represented in a basis.
///
/// A `Tensor` records its scalar field, coordinates, preferred hand, and
/// remaining scalar actions. It need not itself admit scalar multiplication:
/// completed composites may have `Action = NoSided`. [`Vector`] is the
/// refinement which certifies that the elected action exists and exposes it
/// through `Mul<Self::F>`.
///
/// This is the base of the linear hierarchy. A `Tensor` is nothing more than
/// `N` coordinates in `F` — it carries no metric, no notion of length or angle.
/// Those arrive with the refinements: [`Form`] adds a lowering map,
/// [`Nondegenerate`] makes it invertible, [`Sesquilinear`]/[`Bilinear`] fix how
/// it interacts with the field involution, and [`InnerProduct`]/[`Euclidean`]
/// add positive-definiteness.
///
/// Every `Tensor` is its own tangent space: it is an abelian [`LieGroup`] under
/// addition, with `exp` and `log` the identity (`identity_exp(v) = v`). This is
/// what lets a flat coordinate space and a curved manifold share the same chart
/// machinery.
///
/// [`Hand`](Tensor::Hand) elects the side on which the field acts, if that
/// action exists. Concrete coordinate spaces conventionally elect [`Right`];
/// [`Dual<V>`](Dual) always elects the opposite hand.
///
/// The dual space `V*` is [`Dual<Self>`](Dual), and the canonical evaluation
/// pairing between them is [`pairing`](Tensor::pairing). Because that pairing is
/// pinned to the coordinate dot product, `V`, `V*`, and `V**` are
/// coordinate-identical, which is what makes [`collapse`](Dual<Dual<Tensor>>::collapse) a
/// free relabel. Double dualisation restores the original hand.
///
/// [`Add`], [`Sub`], [`Neg`], [`Zero`], [`Index`] and [`IndexMut`] should all be
/// impl'd using the macro [`impl_vector_ops!`](crate::impl_vector_ops)
pub trait Tensor:
    Add<Output = Self>
    + Sub<Output = Self>
    + Neg<Output = Self>
    + Zero
    + Index<usize, Output = Self::F>
    + IndexMut<usize>
    + Point
    + AsRef<Self::Array<Self::F>>
    + AsMut<Self::Array<Self::F>>
{
    /// The scalar field the coordinates live in.
    type F: Field;

    /// The dimension of the space — the number of coordinates.
    const N: usize = Self::Array::<Self::F>::N;

    /// The underlying storage of this tensor; generic over any point.
    type Array<T: Point>: Array<T>;

    /// The side on which `F` acts. [`Dual<Self>`](Dual) elects the opposite
    /// hand, and `Dual<Dual<Self>>` therefore restores this one.
    type Hand: Handedness;

    /// Whether `Self` has a one-sided or both-sided action.
    type Action: Sidedness;

    /// Selects how this constructor participates in tensor normalization.
    /// Ordinary tensor spaces use [`Atomic`]; expression constructors provide
    /// their corresponding structural normalizer.
    type Normalization: TensorNormalizer<Self>;

    /// Builds a vector from a function of coordinate index. The canonical
    /// constructor — most other constructors reduce to this.
    fn from_fn(f: impl FnMut(usize) -> Self::F) -> Self;

    /// Iterates the `N` coordinates in order.
    fn iter(&self) -> <Self::Array<Self::F> as Array<Self::F>>::Iter<'_> {
        self.as_ref().iter()
    }

    /// Applies f to each coordinate of V
    fn map(&self, mut f: impl FnMut(Self::F) -> Self::F) -> Self {
        Self::from_fn(|i| f(self[i]))
    }

    /// The canonical evaluation pairing `(V, V*) -> F`, `⟨v, ω⟩ = ω(v)`.
    ///
    /// Evaluation follows the elected hand:
    ///
    /// - for a right module, `ω(v) = Σ ωᵢvᵢ`;
    /// - for a left module, `ω(v) = Σ vᵢωᵢ`.
    ///
    /// Thus the covector coordinate is placed on the side opposite the vector's
    /// scalar action. The two orders agree over commutative fields but differ
    /// over a noncommutative division ring.
    fn pairing(&self, rhs: &Dual<Self>) -> Self::F {
        self.iter()
            .zip(rhs.iter())
            .fold(Self::F::zero(), |acc, (&vector, &covector)| {
                acc + match <Self::Hand as Handedness>::H {
                    Hand::Left => vector * covector,
                    Hand::Right => covector * vector,
                }
            })
    }

    /// Constructs a tensor from its coordinates in canonical flat-index order.
    ///
    /// The iterator must contain exactly [`Self::N`] elements. Its first element is
    /// placed at coordinate `0`, its second at coordinate `1`, and so on.
    ///
    /// # Panics
    ///
    /// Panics if the iterator yields either fewer or more than [`Self::N`]
    /// coordinates. This exact-length requirement prevents accidental truncation
    /// and prevents omitted coordinates from being silently filled with zero.
    ///
    /// For construction from a function of the coordinate index, use
    /// [`Tensor::from_fn`].
    ///
    /// # Examples
    ///
    /// ```
    /// use diffable::{coords::Coords, traits::Tensor};
    ///
    /// let v = Coords::<f64, 3>::from_iter([1.0, 2.0, 3.0]);
    ///
    /// assert_eq!(v[0], 1.0);
    /// assert_eq!(v[1], 2.0);
    /// assert_eq!(v[2], 3.0);
    /// ```
    fn from_iter(iter: impl IntoIterator<Item = Self::F>) -> Self {
        let mut iter = iter.into_iter();

        let out = Self::from_fn(|_| {
            iter.next()
                .unwrap_or_else(|| panic!("iterator contained fewer than {} elements", Self::N))
        });

        assert!(
            iter.next().is_none(),
            "iterator contained more than {} elements",
            Self::N,
        );

        out
    }

    fn flatten_index<const R: usize>(index: [usize; R]) -> usize {
        let dimension = (1..=Self::N)
            .find(|&i| i.pow(R as u32) == Self::N)
            .expect("tensor component count is not an exact R-th power");

        index.into_iter().fold(0, |flat, i| {
            assert!(
                i < dimension,
                "tensor component index {i} out of bounds for dimension {dimension}"
            );

            flat * dimension + i
        })
    }

    // Flat space has no singularities — to_local is always Some
    #[cfg(feature = "testing")]
    fn check_global_chart(p: &Self, q: &Self) -> bool {
        let chart = Self::chart_at(p);
        chart.to_local(q).is_some()
    }

    // Geodesic scaling holds globally (infinite injectivity radius):
    // to_global(v * t) is parallel to to_global(v) AND scaled by t exactly
    #[cfg(feature = "testing")]
    fn check_global_geodesic_scaling(p: &Self, v: Self, t: <Self::F as Field>::Fixed) -> bool
    where
        Self: Vector + PartialEq,
    {
        let t = Self::F::from_fixed(t);
        let chart = Self::chart_at(p);
        match (
            chart.to_local(&chart.to_global(v.clone() * t)),
            chart.to_local(&chart.to_global(v)),
        ) {
            (Some(tv_local), Some(v_local)) => tv_local == v_local * t,
            _ => false,
        }
    }
}

/// A tensor on which the elected scalar action is available directly.
///
/// [`Tensor`] is the foundational coordinate abstraction. It records a scalar
/// field, dimension, array representation, preferred hand, and the scalar
/// actions possessed by the object. A `Vector` is the refinement which
/// guarantees that at least one such action exists and exposes the preferred
/// one through ordinary scalar multiplication:
///
/// ```text
/// v * a
/// ```
///
/// The actual multiplication order is determined by [`Tensor::Hand`]:
///
/// - for a right-handed vector, this denotes the right action `v a`;
/// - for a left-handed vector, it denotes the left action `a v`.
///
/// The uniform Rust spelling is intentional. Generic vector algorithms can use
/// scalar multiplication without silently assuming that scalars commute, while
/// the type's handedness preserves the mathematical order of the operation.
///
/// # Actions and handedness
///
/// [`Tensor::Action`] describes which scalar actions exist:
///
/// - [`OneSided`] means only the elected action is available;
/// - [`BothSided`] means both left and right actions are available.
///
/// [`ActionExists`] excludes zero-sided tensors. Such tensors may still be valid
/// elements of the tensor algebra, but they cannot be vectors because no scalar
/// action can be exposed as `Mul<Self::F, Output = Self>`.
///
/// Handedness and sidedness are deliberately separate. `Hand` chooses the
/// action used by the ordinary vector interface; `Action` records whether the
/// opposite action is also available. Thus a two-sided module may be viewed
/// through either hand using [`Sinister`], while a one-sided module cannot be
/// rehanded.
///
/// # Tensor algebra
///
/// Every vector is a tensor, but not every tensor is a vector. Tensor products
/// may consume the exposed actions of their factors during balancing and can
/// therefore produce a tensor with no remaining scalar action. The resulting
/// object still supports tensor-algebra operations and contractions, but does
/// not implement `Vector`.
///
/// This separation keeps invalid scalar multiplication unrepresentable without
/// excluding completed tensor composites from the tensor algebra.
pub trait Vector: Tensor<Action: ActionExists> + Mul<Self::F, Output = Self> {}
impl<V: Tensor<Action: ActionExists> + Mul<Self::F, Output = Self>> Vector for V {}

/// Implements the canonical coordinate-wise operations for a tensor type.
///
/// `impl_vector_ops!` supplies every operation whose implementation is uniquely
/// determined by [`Tensor`]'s coordinate representation:
///
/// - [`Add`] and [`Sub`] operate coordinate by coordinate;
/// - [`Neg`] negates each coordinate;
/// - [`Zero`] constructs the all-zero tensor and tests every coordinate;
/// - [`Index`] and [`IndexMut`] delegate to the array exposed by [`AsRef`] and
///   [`AsMut`];
/// - scalar [`Mul`] is provided when [`Tensor::Action`] implements
///   [`ActionExists`].
///
/// Scalar multiplication respects [`Tensor::Hand`]. For a left-handed tensor,
/// the scalar is placed to the left of each coordinate; for a right-handed
/// tensor, it is placed to the right:
///
/// ```text
/// Left:  (a, v) ↦ [a v₀, a v₁, …]
/// Right: (v, a) ↦ [v₀ a, v₁ a, …]
/// ```
///
/// The multiplication implementation is conditional. A tensor with
/// `Action = NoSided` receives no scalar `Mul` implementation, while one-sided
/// and two-sided tensors receive `Mul<Self::F, Output = Self>`. Consequently,
/// invoking this macro does not by itself assert that the target is a
/// [`Vector`]; that follows automatically only when its elected action exists.
///
/// # Implementing a tensor
///
/// The target type must separately implement [`Tensor`], including
/// [`Tensor::from_fn`], and expose its coordinate storage through the required
/// [`AsRef`] and [`AsMut`] implementations. Those are representation-specific
/// choices and therefore cannot be generated by this macro.
///
/// Once those pieces are present, invoke the macro with the target type followed
/// by its generic declarations:
///
/// ```text
/// impl_vector_ops!(
///     MyTensor<F, N>,
///     F: Field,
///     const N: usize
/// );
/// ```
///
/// Bounds may contain associated-type constraints:
///
/// ```text
/// impl_vector_ops!(
///     MyTensor<V>,
///     V: Tensor<Action = BothSided>
/// );
/// ```
///
/// The macro should be invoked exactly once for a given target configuration.
/// Writing any of the generated implementations manually for the same type will
/// produce conflicting trait implementations.
///
/// [`Add`]: core::ops::Add
/// [`Sub`]: core::ops::Sub
/// [`Neg`]: core::ops::Neg
/// [`Mul`]: core::ops::Mul
/// [`Index`]: core::ops::Index
/// [`IndexMut`]: core::ops::IndexMut
/// [`Zero`]: num_traits::Zero
/// [`AsRef`]: core::convert::AsRef
/// [`AsMut`]: core::convert::AsMut
#[macro_export]
macro_rules! impl_vector_ops {
    ($target:ty, $($generics:tt)*) => {
        impl<$($generics)*> core::ops::Add<Self> for $target {
            type Output = $target;

            fn add(self, rhs: Self) -> Self::Output {
                Self::from_fn(|i| self[i] + rhs[i])
            }
        }

        impl<$($generics)*> core::ops::Sub<Self> for $target {
            type Output = $target;

            fn sub(self, rhs: Self) -> Self::Output {
                Self::from_fn(|i| self[i] - rhs[i])
            }
        }

        impl<$($generics)*> core::ops::Neg for $target {
            type Output = $target;

            fn neg(self) -> Self::Output {
                Self::from_fn(|i| -self[i])
            }
        }

        impl<$($generics)*> core::ops::Mul<<$target as $crate::traits::Tensor>::F> for $target
        where
            $target: $crate::traits::Tensor<Action: $crate::traits::ActionExists>,
        {
            type Output = $target;

            fn mul(self, scalar: <$target as $crate::traits::Tensor>::F) -> Self::Output {
                Self::from_fn(|i| match <<$target as $crate::traits::Tensor>::Hand as $crate::traits::Handedness>::H {
                    $crate::traits::Hand::Left => scalar * self[i],
                    $crate::traits::Hand::Right => self[i] * scalar,
                })
            }
        }

        impl<$($generics)*> num_traits::Zero for $target {
            fn zero() -> Self {
                Self::from_fn(|_| <$target as $crate::traits::Tensor>::F::zero())
            }

            fn is_zero(&self) -> bool {
                self.iter().all(num_traits::Zero::is_zero)
            }
        }

        impl<$($generics)*> core::ops::Index<usize> for $target {
            type Output = <$target as $crate::traits::Tensor>::F;

            fn index(&self, index: usize) -> &Self::Output {
                &self.as_ref()[index]
            }
        }

        impl<$($generics)*> core::ops::IndexMut<usize> for $target {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.as_mut()[index]
            }
        }

        impl<const R: usize, $($generics)*> core::ops::Index<[usize; R]> for $target
        where
            $target: $crate::traits::Tensor,
        {
            type Output = <$target as $crate::traits::Tensor>::F;

            fn index(&self, index: [usize; R]) -> &Self::Output {
                &self[<$target as $crate::traits::Tensor>::flatten_index(index)]
            }
        }

        impl<const R: usize, $($generics)*> core::ops::IndexMut<[usize; R]> for $target
        where
            $target: $crate::traits::Tensor,
        {
            fn index_mut(&mut self, index: [usize; R]) -> &mut Self::Output {
                let flat =
                    <$target as $crate::traits::Tensor>::flatten_index(index);

                &mut self[flat]
            }
        }
    };
}

/// A vector space equipped with a *lowering map* `♭: V → V*`.
///
/// This is where geometry enters: [`flat`](Form::flat) turns a vector into the
/// covector `⟨·, b⟩`, and [`dot`](Form::dot) is the induced form
/// `⟨a, b⟩ = pairing(a, b♭)`. No invertibility, definiteness, or symmetry is
/// assumed here — a general (even indefinite or degenerate) form is a `Form`.
/// The refinements add those: [`Nondegenerate`] (invertible), [`Sesquilinear`]
/// (Hermitian), [`Bilinear`] (symmetric), [`InnerProduct`] (positive-definite).
pub trait Form: Tensor {
    fn flat(&self) -> Dual<Self>;

    fn dot(&self, b: &Self) -> Self::F {
        self.pairing(&b.flat())
    }

    fn self_dot(&self) -> Self::F {
        self.dot(self)
    }

    #[cfg(feature = "testing")]
    fn check_dot_agrees_with_pairing(a: &Self, b: &Self) -> bool {
        a.pairing(&b.flat()) == a.dot(b)
    }

    // Translation invariance: Q((a+c) - (b+c)) == Q(a - b),
    // where Q(v) = ⟨v,v⟩ is the form.
    //
    // Stated on norm_squared rather than a distance, since a pseudo-Euclidean
    // space has no metric: the difference is the same vector either way
    // ((a+c) - (b+c) = a - b), so the form agrees exactly.
    #[cfg(feature = "testing")]
    fn check_translation_invariance(a: &Self, b: &Self, c: &Self) -> bool {
        let diff = a.clone() - b.clone();
        let diff_translated = (a.clone() + c.clone()) - (b.clone() + c.clone());
        diff.self_dot() == diff_translated.self_dot()
    }
}

/// A [`Form`] whose lowering map is invertible — a nondegenerate form.
///
/// [`sharp`](Nondegenerate::sharp) is the raising map `♯: V* → V`, inverse to
/// [`flat`](Form::flat). This is the *musical* isomorphism `V ≅ V*` (and, via
/// [`collapse`](Dual<Dual<Tensor>>::collapse), `V ≅ V**`); it depends on the metric, unlike
/// the purely dimensional evaluation iso.
pub trait Nondegenerate: Form {
    fn sharp(v: Dual<Self>) -> Self;

    // check flat/sharp inverse functions
    #[cfg(feature = "testing")]
    fn check_isomorphism(a: &Self) -> bool
    where
        Self: PartialEq<Self>,
    {
        let flat = a.flat();

        Self::sharp(flat.clone()) == *a && Dual::<Self>::sharp(flat.flat()) == flat
    }
}

impl_group_via_add!(V, V: Tensor);

impl<V: Tensor> LieGroup<V> for V {
    fn identity_exp(v: V) -> Self {
        v
    }

    fn identity_log(p: &Self) -> Option<V> {
        Some(p.clone())
    }
}

/// A symmetric bilinear form on a vector space.
///
/// The space of all values of a type `P: Bilinear<R>` is interpreted as a
/// vector space equipped with a symmetric bilinear pairing
/// `⟨·,·⟩: P × P → R`. **No definiteness is assumed**: the induced quadratic
/// form `Q(v) = ⟨v,v⟩` may be positive, negative, or zero for `v ≠ 0`. This is
/// the structure of a pseudo-Euclidean (e.g. Minkowski) space as well as a
/// Euclidean one.
///
/// Because the form may be indefinite, `Bilinear` provides **no norm and no
/// distance**: `⟨v,v⟩` can be negative, so `sqrt(⟨v,v⟩)` need not be real, and
/// the induced "distance" fails the metric-space axioms (null vectors give
/// distinct points at separation zero; the triangle inequality reverses on
/// timelike triples). A norm and a [`Metric`] arise only once definiteness is
/// added — see [`InnerProduct`], which refines this trait with
/// positive-definiteness and is therefore the only branch that induces a
/// metric space.
///
/// `norm_squared` is provided as `⟨v,v⟩` and is **signed** — it is the value
/// of the quadratic form, not the square of a norm. Callers on indefinite
/// spaces should inspect its sign (causal character) rather than take its
/// square root.
///
/// The three certified invariants — symmetry, additivity, and scalar
/// linearity of the pairing — are signature-agnostic and hold in the
/// indefinite case exactly as in the definite one.
pub trait Bilinear: Sesquilinear {}
impl<F: Field<Fixed = F>, V: Sesquilinear<F = F>> Bilinear for V {}

/// A Hermitian (sesquilinear) form on a vector space.
///
/// The space of all values of a type `P: Sesquilinear<F>` is interpreted as a
/// vector space equipped with a Hermitian pairing
/// `⟨·,·⟩: P × P → F`, where `F` is an [`Field`]. The pairing is
/// linear in its first argument and conjugate-linear in its second, satisfying
/// `⟨v,w⟩ = conj(⟨w,v⟩)`.
///
/// Unlike [`Bilinear`], the codomain may be a field with a nontrivial
/// involution, such as the complex numbers. Hermitian forms are the natural
/// analogue of symmetric bilinear forms over such fields.
///
/// No definiteness is assumed. The induced quadratic form
/// `Q(v) = ⟨v,v⟩` is always fixed by the involution (for example, real-valued
/// over `ℂ`), but it may still be positive, negative, or zero for `v ≠ 0`.
/// Consequently, this trait provides no norm or metric. A norm and the
/// associated [`Metric`] arise only once positive-definiteness is imposed
/// (see [`InnerProduct`] or the corresponding positive-definite Hermitian
/// refinement, if provided).
///
/// `self_dot` returns the value `⟨v,v⟩` in the fixed field `F::Fixed`. This is
/// the value of the quadratic form, not the square of a norm, and should not
/// be square-rooted unless positive-definiteness is known.
///
/// The certified invariants are Hermitian symmetry, additivity, and scalar
/// linearity in the first argument. Conjugate-linearity in the second argument
/// follows from these together with Hermitian symmetry.
pub trait Sesquilinear: Form + Vector {
    // Hermitian spaces are exactly the spaces where
    // self.dot(self) lands in the fixed field of F
    fn norm_squared(&self) -> <Self::F as Field>::Fixed {
        self.dot(self).to_fixed()
    }

    // ⟨v,w⟩ = conj(⟨w,v⟩) — Hermitian symmetry, the sesquilinear analogue
    // of Bilinear::check_symmetry. Additivity and conjugate-linearity in
    // the second argument both follow from this plus linearity in the
    // first, and aren't separately checked for the same reason Bilinear
    // doesn't separately check them.
    #[cfg(feature = "testing")]
    fn check_hermitian_symmetry(a: Self, b: Self) -> bool {
        a.dot(&b) == b.dot(&a).conj()
    }

    #[cfg(feature = "testing")]
    fn check_additivity(a: Self, b: Self, c: Self) -> bool {
        (a.clone() + b.clone()).dot(&c) == a.dot(&c) + b.dot(&c)
    }

    #[cfg(feature = "testing")]
    fn check_scalar_linearity(a: Self, c: Self, k: Self::F) -> bool {
        let dot = a.dot(&c);

        (a * k).dot(&c)
            == match <Self as Tensor>::Hand::H {
                Hand::Right => dot * k,
                Hand::Left => k * dot,
            }
    }
}

/// An inner product structure on a vector space.
///
/// Refines [`Bilinear`] with **positive-definiteness**: `⟨v,v⟩ > 0` for all
/// `v ≠ 0`. This is exactly the property that makes the induced quantities
/// well-behaved — `norm(v) = sqrt(⟨v,v⟩)` is real and non-negative, and
/// `d(a,b) = ‖a - b‖` satisfies the metric-space axioms — which is why
/// `InnerProduct` is a refinement of [`Metric`], whereas the bare
/// [`Bilinear`] base is not.
///
/// Not every [`Metric`] is an `InnerProduct` — the sphere's geodesic distance
/// is a metric not arising from any inner product, since the sphere is not a
/// vector space. And not every [`Bilinear`] form is an `InnerProduct` — a
/// Minkowski scalar product is bilinear and symmetric but indefinite, so it
/// induces no metric at all.
pub trait InnerProduct:
    Sesquilinear + Nondegenerate + Metric<R = <Self::F as Field>::Fixed>
where
    <Self::F as Field>::Fixed: Real,
{
    /// The norm `‖v‖ = sqrt(⟨v,v⟩)`. Well-defined and real because the form
    /// is positive-definite. On an indefinite [`Bilinear`] space this would
    /// not be real — which is why it lives here, not on the base.
    fn norm(&self) -> <Self::F as Field>::Fixed {
        self.norm_squared().sqrt()
    }

    #[cfg(feature = "testing")]
    fn check_positive_definite(a: Self) -> bool
    where
        Self: Zero + PartialEq,
    {
        a == Self::zero() || a.norm() > <Self::F as Field>::Fixed::zero()
    }

    #[cfg(feature = "testing")]
    fn check_metric_compatibility(a: Self, b: Self) -> bool {
        a.clone().sub(b.clone()).norm_squared().sqrt() == a.distance(&b)
    }
}

impl<P: Sesquilinear + Nondegenerate + Metric<R = <Self::F as Field>::Fixed>> InnerProduct for P where
    <Self::F as Field>::Fixed: Real
{
}
