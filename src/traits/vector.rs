use num_traits::{Zero, real::Real as _};
use std::{
    convert::Infallible,
    ops::{Add, Index, IndexMut, Mul, Neg, Sub},
};

#[cfg(feature = "testing")]
use super::Chart;

use super::{Field, LieGroup, Metric, Real};
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
/// [`Sesquilinear`]: crate::traits::Sesquilinear
pub trait Euclidean: Bilinear<F: Real, Action = BothSided> + InnerProduct + Vector {
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
/// Stored as a `V` internally, because [`pairing`](Vector::pairing) is fixed to
/// the coordinate dot product, which identifies the dual basis with the primal
/// basis component-wise. A `Dual<V>` is therefore coordinate-identical to the
/// `V` holding its components — the wrapper exists purely so the type system
/// keeps covariant and contravariant vectors apart. That separation is what
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

impl<V: Tensor> Tensor for Dual<V> {
    type F = V::F;
    type Hand = <V::Hand as Handedness>::Opposite;
    type Action = V::Action;

    type Array<T: Point> = V::Array<T>;

    fn from_fn(f: impl FnMut(usize) -> Self::F) -> Self {
        Self(V::from_fn(f))
    }
}

impl_vector_ops!(Dual<V>, V: Tensor);

impl<V: Tensor> Index<usize> for Dual<V> {
    type Output = V::F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<V: Tensor> IndexMut<usize> for Dual<V> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

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

#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Sinister<V: Tensor<Action = BothSided>>(pub V);

impl<V: Tensor<Action = BothSided>> Tensor for Sinister<V> {
    type F = V::F;
    type Hand = <V::Hand as Handedness>::Opposite;
    type Action = V::Action;

    type Array<T: Point> = V::Array<T>;

    fn from_fn(f: impl FnMut(usize) -> Self::F) -> Self {
        Self(V::from_fn(f))
    }
}

impl_vector_ops!(Sinister<V>, V: Tensor<Action = BothSided>);

impl<V: Tensor<Action = BothSided>> Index<usize> for Sinister<V> {
    type Output = V::F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<V: Tensor<Action = BothSided>> IndexMut<usize> for Sinister<V> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<V: Tensor<Action = BothSided>> AsRef<<Sinister<V> as Tensor>::Array<V::F>> for Sinister<V> {
    fn as_ref(&self) -> &<Sinister<V> as Tensor>::Array<V::F> {
        self.0.as_ref()
    }
}

impl<V: Tensor<Action = BothSided>> AsMut<<Sinister<V> as Tensor>::Array<V::F>> for Sinister<V> {
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
/// [`Vector::Hand`] uses this trait to determine both ordinary scalar
/// multiplication and canonical evaluation order.
pub trait Handedness {
    /// The hand elected by the dual module.
    type Opposite: Handedness<Opposite = Self>;

    /// The runtime value corresponding to this type-level hand.
    const H: Hand;
}

/// Type-level left scalar action.
pub enum Left {}

/// Type-level right scalar action.
pub enum Right {}

impl Handedness for Left {
    type Opposite = Right;
    const H: Hand = Hand::Left;
}

impl Handedness for Right {
    type Opposite = Left;
    const H: Hand = Hand::Right;
}

pub trait Sidedness: Copy + Clone + std::fmt::Debug {
    /// The lesser of this sidedness and the other sidedness.
    type Meet<T: Sidedness>: Sidedness;
    #[doc(hidden)]
    type MeetOne: Sidedness;

    /// `T` when a scalar action exists, otherwise an uninhabited type.
    type Exists<T>;

    /// Extracts the value whose existence is certified by this sidedness.
    fn into_existing<T>(value: &Self::Exists<T>) -> &T;

    /// The runtime value corresponding to this type-level side.
    const S: Side;
}

#[derive(Debug, Copy, Clone)]
pub enum Side {
    None,
    Same,
    Both,
}

#[derive(Debug, Copy, Clone)]
pub enum NoSided {}
#[derive(Debug, Copy, Clone)]
pub enum OneSided {}
#[derive(Debug, Copy, Clone)]
pub enum BothSided {}

pub trait ActionExists: Sidedness {
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

pub trait TensorProductAction<Rhs: ActionExists>: ActionExists {
    type Action: Sidedness;
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

    type Exists<T> = T;

    fn into_existing<T>(value: &T) -> &T {
        value
    }

    const S: Side = Side::Same;
}

impl Sidedness for BothSided {
    type Meet<T: Sidedness> = T;
    type MeetOne = OneSided;

    type Exists<T> = T;

    fn into_existing<T>(value: &T) -> &T {
        value
    }

    const S: Side = Side::Both;
}

impl Sidedness for NoSided {
    type Meet<T: Sidedness> = NoSided;
    type MeetOne = NoSided;

    type Exists<T> = Infallible;

    fn into_existing<T>(value: &Infallible) -> &T {
        match *value {}
    }

    const S: Side = Side::None;
}

pub trait Array<T: Point>:
    Point + Sized + Index<usize, Output = T> + IndexMut<usize> + IntoIterator<Item = T>
{
    const N: usize;

    type Iter<'a>: Iterator<Item = &'a T>
    where
        Self: 'a,
        T: 'a;

    type IterMut<'a>: Iterator<Item = &'a mut T>
    where
        Self: 'a,
        T: 'a;

    fn iter(&self) -> Self::Iter<'_>;
    fn iter_mut(&mut self) -> Self::IterMut<'_>;

    fn from_fn(f: impl FnMut(usize) -> T) -> Self;
}

impl<T: Point, const N: usize> Array<T> for [T; N] {
    const N: usize = N;

    type Iter<'a>
        = std::slice::Iter<'a, T>
    where
        Self: 'a,
        T: 'a;

    type IterMut<'a>
        = std::slice::IterMut<'a, T>
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
        std::array::from_fn(f)
    }
}

/// A finite-dimensional left or right module over a [`Field`], equipped with a
/// basis.
///
/// This is the base of the linear hierarchy. A `Vector` is nothing more than
/// `N` coordinates in `F` — it carries no metric, no notion of length or angle.
/// Those arrive with the refinements: [`Form`] adds a lowering map,
/// [`Nondegenerate`] makes it invertible, [`Sesquilinear`]/[`Bilinear`] fix how
/// it interacts with the field involution, and [`InnerProduct`]/[`Euclidean`]
/// add positive-definiteness.
///
/// Every `Vector` is its own tangent space: it is an abelian [`LieGroup`] under
/// addition, with `exp` and `log` the identity (`identity_exp(v) = v`). This is
/// what lets a flat coordinate space and a curved manifold share the same chart
/// machinery.
///
/// [`Hand`](Vector::Hand) elects which side the field acts on. Concrete
/// coordinate spaces conventionally elect [`Right`]; [`Dual<V>`](Dual) always
/// elects the opposite hand. The ordinary `Mul<Self::F>` operation follows that
/// election, so the same vector API represents either kind of module without
/// silently commuting scalars.
///
/// The dual space `V*` is [`Dual<Self>`](Dual), and the canonical evaluation
/// pairing between them is [`pairing`](Vector::pairing). Because that pairing is
/// pinned to the coordinate dot product, `V`, `V*`, and `V**` are
/// coordinate-identical, which is what makes [`collapse`](Vector::collapse) a
/// free relabel. Double dualisation restores the original hand.
pub trait Tensor:
    Add<Output = Self>
    + Sub<Output = Self>
    + Mul<<Self::Action as Sidedness>::Exists<Self::F>, Output = Self>
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

    type Array<T: Point>: Array<T>;

    /// The side on which `F` acts. [`Dual<Self>`](Dual) elects the opposite
    /// hand, and `Dual<Dual<Self>>` therefore restores this one.
    type Hand: Handedness;

    // Whether `Self` has a one-sided or both-sided action.
    type Action: Sidedness;

    /// Builds a vector from a function of coordinate index. The canonical
    /// constructor — most other constructors reduce to this.
    fn from_fn(f: impl FnMut(usize) -> Self::F) -> Self;

    /// Iterates the `N` coordinates in order.
    fn iter(&self) -> <Self::Array<Self::F> as Array<Self::F>>::Iter<'_> {
        self.as_ref().iter()
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

    /// The canonical identification `V** ≅ V`, collapsing a twice-dualised
    /// vector back to `V`.
    ///
    /// This is the *evaluation* isomorphism `v ↦ (φ ↦ φ(v))`, which exists for
    /// any finite-dimensional space with no dependence on a metric or
    /// nondegeneracy — every [`Vector`] qualifies via its fixed dimension `N`.
    /// It is a pure coordinate relabel (strip two [`Dual`] wrappers) precisely
    /// because [`pairing`](Vector::pairing) is fixed to the coordinate dot
    /// product, which identifies each dual basis with its primal basis
    /// component-wise. Do not confuse this with the *musical* `V** ≅ V` of
    /// [`Nondegenerate`], which routes through the metric and requires an
    /// invertible form.
    fn collapse(v: Dual<Dual<Self>>) -> Self {
        v.0.0
    }

    /// The canonical relabelling
    /// `Dual<Sinister<Self>> ≅ Sinister<Dual<Self>>`.
    fn dual_sinister(v: Dual<Sinister<Self>>) -> Sinister<Dual<Self>>
    where
        Self: Tensor<Action = BothSided>,
    {
        Sinister(Dual(v.0.0))
    }

    /// The inverse canonical relabelling
    /// `Sinister<Dual<Self>> ≅ Dual<Sinister<Self>>`.
    fn sinister_dual(v: Sinister<Dual<Self>>) -> Dual<Sinister<Self>>
    where
        Self: Tensor<Action = BothSided>,
    {
        Dual(Sinister(v.0.0))
    }

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

pub trait Vector: Tensor<Action: ActionExists> + Mul<Self::F, Output = Self> {}
impl<V: Tensor<Action: ActionExists> + Mul<Self::F, Output = Self>> Vector for V {}

#[macro_export]
macro_rules! impl_vector_ops {
    ($target:ty, $($generics:tt)*) => {
        impl<$($generics)*> std::ops::Add<Self> for $target {
            type Output = $target;

            fn add(self, rhs: Self) -> Self::Output {
                Self::from_fn(|i| self[i] + rhs[i])
            }
        }

        impl<$($generics)*> std::ops::Sub<Self> for $target {
            type Output = $target;

            fn sub(self, rhs: Self) -> Self::Output {
                Self::from_fn(|i| self[i] - rhs[i])
            }
        }

        impl<$($generics)*> std::ops::Neg for $target {
            type Output = $target;

            fn neg(self) -> Self::Output {
                Self::from_fn(|i| -self[i])
            }
        }

        impl<$($generics)*> std::ops::Mul<<<$target as $crate::traits::Tensor>::Action as $crate::traits::Sidedness>::Exists<<$target as $crate::traits::Tensor>::F>> for $target {
            type Output = $target;

            fn mul(self, scalar: <<$target as $crate::traits::Tensor>::Action as $crate::traits::Sidedness>::Exists<<$target as $crate::traits::Tensor>::F>) -> Self::Output {
                Self::from_fn(|i| match <<$target as Tensor>::Hand as $crate::traits::Handedness>::H {
                    $crate::traits::Hand::Left => *<<$target as $crate::traits::Tensor>::Action as $crate::traits::Sidedness>::into_existing(&scalar) * self[i],
                    $crate::traits::Hand::Right => self[i] * *<<$target as $crate::traits::Tensor>::Action as $crate::traits::Sidedness>::into_existing(&scalar),
                })
            }
        }

        impl<$($generics)*> num_traits::Zero for $target {
            fn zero() -> Self {
                Self::from_fn(|_| <$target as Tensor>::F::zero())
            }

            fn is_zero(&self) -> bool {
                self.iter().all(num_traits::Zero::is_zero)
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
/// [`collapse`](Vector::collapse), `V ≅ V**`); it depends on the metric, unlike
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
