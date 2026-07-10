use crate::{
    discrete::Z,
    impl_lie_group_via_quotient, impl_tangent_bundle_via_bounded,
    traits::{Bounded, BuildNodes, NerveComplexParameters},
};
use std::marker::PhantomData;

use crate::traits::{Chart, Euclidean, Group, LieGroup, Metric, Quotient, Smooth};

use num_traits::{Euclid, NumCast, One, Zero, real::Real};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct S1<V: Euclidean + From<[<V as Euclidean>::F; 1]>>(V);

impl<V: Euclidean + From<[V::F; 1]>> Metric<V::F> for S1<V> {
    fn distance(&self, other: &Self) -> V::F {
        self.to_local(other).unwrap().norm()
    }
}

impl<V: Euclidean + From<[<V as Euclidean>::F; 1]>> Quotient<V, Z<V>, V> for S1<V> {
    fn new(g: V) -> Self {
        let one = V::F::one();
        let mut d = g[0].rem_euclid(&one);
        if d == one {
            // floating point garbage check gotta do it :(
            d = V::F::zero();
        }
        Self([d].into())
    }

    fn lift(&self) -> V {
        // nearest representative to the identity (0), not just the
        // canonical [0,1) one — reduce into (-1/2, 1/2] instead.
        let half = V::F::one() / (V::F::one() + V::F::one());
        let mut d = self.0[0];
        if d > half {
            d = d - V::F::one();
        }
        [d].into()
    }

    fn embed(h: Z<V>) -> V {
        [<V::F as NumCast>::from(h.0).unwrap()].into()
    }
}

impl_lie_group_via_quotient!(S1<V>, V, Z<V>, From<[<V as Euclidean>::F; 1]>);

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Torus<I: Euclidean + From<[I::F; 1]> + From<[V::F; 1]>, V: Euclidean + From<[I::F; 2]>>(
    S1<I>,
    S1<I>,
    PhantomData<V>,
);

impl<I: Euclidean + From<[I::F; 1]> + From<[V::F; 1]>, V: Euclidean + From<[I::F; 2]>> Torus<I, V> {
    pub fn new(a: S1<I>, b: S1<I>) -> Self {
        Self(a, b, PhantomData)
    }
}

impl<I: Euclidean + From<[I::F; 1]> + From<[V::F; 1]>, V: Euclidean + From<[I::F; 2]>> Metric<I::F>
    for Torus<I, V>
where
    I::F: From<V::F>,
{
    fn distance(&self, other: &Self) -> I::F {
        self.to_local(other).unwrap().norm().into()
    }
}

impl<I: Euclidean + From<[I::F; 1]> + From<[V::F; 1]>, V: Euclidean + From<[I::F; 2]>> Group
    for Torus<I, V>
{
    fn identity() -> Self {
        Self::new(S1::identity(), S1::identity())
    }

    fn compose(&self, other: &Self) -> Self {
        Self::new(self.0.compose(&other.0), self.1.compose(&other.1))
    }

    fn inverse(&self) -> Self {
        Self::new(self.0.inverse(), self.1.inverse())
    }
}

impl<I: Euclidean + From<[I::F; 1]> + From<[V::F; 1]>, V: Euclidean + From<[I::F; 2]>> LieGroup<V>
    for Torus<I, V>
{
    fn identity_exp(v: V) -> Self {
        let v0 = [v[0]].into();
        let v1 = [v[1]].into();
        Self::new(S1::identity_exp(v0), S1::identity_exp(v1))
    }

    fn identity_log(p: &Self) -> Option<V> {
        let a = S1::identity_log(&p.0)?;
        let b = S1::identity_log(&p.1)?;

        Some([a[0], b[0]].into())
    }
}

pub trait ICompatible<V: Euclidean<F: Send + Sync> + From<[Self::F; 2]>>:
    Euclidean<F: From<V::F>> + From<[Self::F; 1]> + From<[V::F; 1]> + 'static + Send + Sync
{
}
pub trait VCompatible<I: Euclidean<F: From<Self::F>> + From<[I::F; 1]> + From<[Self::F; 1]>>:
    Euclidean<F: Send + Sync> + From<[I::F; 2]> + 'static + Send + Sync
{
}

impl<
    I: Euclidean<F: From<V::F>> + From<[I::F; 1]> + From<[V::F; 1]> + 'static + Send + Sync,
    V: Euclidean<F: Send + Sync> + From<[I::F; 2]> + 'static + Send + Sync,
> ICompatible<V> for I
{
}

impl<
    I: Euclidean<F: From<V::F>> + From<[I::F; 1]> + From<[V::F; 1]> + 'static + Send + Sync,
    V: Euclidean<F: Send + Sync> + From<[I::F; 2]> + 'static + Send + Sync,
> VCompatible<I> for V
{
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct KleinBottle<I: ICompatible<V>, V: VCompatible<I>>(S1<I>, S1<I>, PhantomData<V>);

impl<I: ICompatible<V>, V: VCompatible<I>> KleinBottle<I, V> {
    pub fn new(a: S1<I>, b: S1<I>) -> Self {
        Self(a, b, PhantomData)
    }
}

impl<I: ICompatible<V>, V: VCompatible<I>> Smooth<V> for KleinBottle<I, V> {
    fn exp(&self, v: V) -> Self {
        let (x, y) = self.coords();
        let vx: I::F = v[0].into();
        let vy: I::F = v[1].into();
        Self::from_cover(x + vx, y + vy)
    }

    fn log(&self, other: &Self) -> Option<V> {
        let one = I::F::one();
        let two = one + one;
        let (sx, sy) = self.coords();
        let (ox, oy) = other.coords();
        let mut best: Option<(I::F, I::F)> = None;
        let mut best_sq = I::F::zero();

        for n in [-one, I::F::zero(), one] {
            let n_odd = n.rem_euclid(&two) != I::F::zero();
            // Reflection formula in the (-1/2,1/2]-centered
            // convention is `-ox`, not `1 - ox` — reflecting about
            // 0 (the domain's center), not about 1/2 (which was
            // only the reflection point under the old [0,1)
            // convention).
            let base_ox = if n_odd { -ox } else { ox };
            for m in [-one, I::F::zero(), one] {
                let cx = base_ox + m;
                let cy = oy + n;
                let dx = cx - sx;
                let dy = cy - sy;
                let sq = dx * dx + dy * dy;
                if best.is_none() || sq < best_sq {
                    best = Some((dx, dy));
                    best_sq = sq;
                }
            }
        }
        best.map(|(dx, dy)| [dx, dy].into())
    }
}

impl<I: ICompatible<V>, V: VCompatible<I>> KleinBottle<I, V> {
    /// Reduce a cover point (x, y) ∈ ℝ² to the fundamental domain via
    /// Γ = ⟨A, B⟩, A: (x,y) ↦ (x+1, y), B: (x,y) ↦ (−x, y+1).
    ///
    /// Uses the SAME (-1/2, 1/2]-centered convention as `S1::lift`
    /// throughout: seam-crossing count is `y.round()` (nearest
    /// integer), not `y.floor()`, since the fundamental domain is
    /// centered at 0 rather than starting at 0. Parity of that
    /// rounded count decides the flip, exactly as before — only the
    /// rounding function and the reflection formula's center point
    /// (0, not 1/2) changed.
    fn from_cover(x: I::F, y: I::F) -> Self {
        let one = I::F::one();
        let two = one + one;
        let ky = y.round(); // nearest-centered seam count
        let y_red = y - ky; // in (-1/2, 1/2]

        let ky_parity_odd = ky.rem_euclid(&two) != I::F::zero();
        let x_oriented = if ky_parity_odd { -x } else { x };

        // S1::new performs the (-1/2,1/2] reduction itself now, so
        // x_oriented can be handed to it directly, unreduced.
        Self(
            S1::new([x_oriented].into()),
            S1::new([y_red].into()),
            PhantomData,
        )
    }

    fn coords(&self) -> (I::F, I::F) {
        (self.0.lift()[0], self.1.lift()[0])
    }
}

impl<I: ICompatible<V>, V: VCompatible<I>> Metric<I::F> for KleinBottle<I, V> {
    fn distance(&self, other: &Self) -> I::F {
        self.to_local(other).unwrap().norm().into()
    }
}

#[derive(Debug)]
pub struct TorusCover<I: ICompatible<V>, V: VCompatible<I>>(Torus<I, V>);

impl<I: ICompatible<V>, V: VCompatible<I>> From<Torus<I, V>> for TorusCover<I, V> {
    fn from(value: Torus<I, V>) -> Self {
        Self(value)
    }
}

impl<I: ICompatible<V>, V: VCompatible<I>> AsRef<Torus<I, V>> for TorusCover<I, V> {
    fn as_ref(&self) -> &Torus<I, V> {
        &self.0
    }
}

const S: usize = 4;

impl<I: ICompatible<V>, V: VCompatible<I>> Bounded<Torus<I, V>, Torus<I, V>, V>
    for TorusCover<I, V>
{
    fn sdf(&self, v: &V) -> <V as Euclidean>::F {
        let to = |x| <V::F as NumCast>::from(x).unwrap();
        v.norm() - (to(2).sqrt() + to(2)) / to(4 * S)
    }
}

impl_tangent_bundle_via_bounded!(
    TorusCover<I, V>,
    Torus<I, V>,
    Torus<I, V>,
    V,
    I: ICompatible<V>,
V: VCompatible<I>
);

impl<I: ICompatible<V>, V: VCompatible<I>> BuildNodes<TorusCover<I, V>> for TorusCover<I, V> {
    fn build_nodes() -> Vec<Self> {
        let to = |x| <I::F as NumCast>::from(x).unwrap();
        let s = to(S);
        let offset = to(1) / (to(2) * s);

        (0..S)
            .flat_map(|y| (0..S).map(move |x| (x, y)))
            .map(|(x, y)| {
                Torus::new(
                    S1([offset + to(x) / s].into()),
                    S1([offset + to(y) / s].into()),
                )
                .into()
            })
            .collect()
    }
}

impl<I: ICompatible<V>, V: VCompatible<I>>
    NerveComplexParameters<Torus<I, V>, V, Torus<I, V>, TorusCover<I, V>> for TorusCover<I, V>
{
}

#[derive(Debug)]
pub struct KleinBottleCover<I: ICompatible<V>, V: VCompatible<I>>(KleinBottle<I, V>);

impl<I: ICompatible<V>, V: VCompatible<I>> From<KleinBottle<I, V>> for KleinBottleCover<I, V> {
    fn from(value: KleinBottle<I, V>) -> Self {
        Self(value)
    }
}

impl<I: ICompatible<V>, V: VCompatible<I>> AsRef<KleinBottle<I, V>> for KleinBottleCover<I, V> {
    fn as_ref(&self) -> &KleinBottle<I, V> {
        &self.0
    }
}

impl<I: ICompatible<V>, V: VCompatible<I>> Bounded<KleinBottle<I, V>, KleinBottle<I, V>, V>
    for KleinBottleCover<I, V>
{
    fn sdf(&self, v: &V) -> <V as Euclidean>::F {
        let to = |x| <V::F as NumCast>::from(x).unwrap();
        v.norm() - (to(2).sqrt() + to(2)) / to(4 * S)
    }
}

impl_tangent_bundle_via_bounded!(
    KleinBottleCover<I, V>,
    KleinBottle<I, V>,
    KleinBottle<I, V>,
    V,
    I: ICompatible<V>, V: VCompatible<I>
);

impl<I: ICompatible<V>, V: VCompatible<I>> BuildNodes<KleinBottleCover<I, V>>
    for KleinBottleCover<I, V>
{
    fn build_nodes() -> Vec<Self> {
        let to = |x| <I::F as NumCast>::from(x).unwrap();
        let s = to(S);
        let offset = to(1) / (to(2) * s);

        (0..S)
            .flat_map(|y| (0..S).map(move |x| (x, y)))
            .map(|(x, y)| {
                KleinBottle::new(
                    S1([offset + to(x) / s].into()),
                    S1([offset + to(y) / s].into()),
                )
                .into()
            })
            .collect()
    }
}

impl<I: ICompatible<V>, V: VCompatible<I>>
    NerveComplexParameters<KleinBottle<I, V>, V, KleinBottle<I, V>, KleinBottleCover<I, V>>
    for KleinBottleCover<I, V>
{
}

#[derive(Debug, Clone)]
pub struct MyopicTorus<I: ICompatible<V>, V: VCompatible<I>>(pub Torus<I, V>);

impl<I: ICompatible<V>, V: VCompatible<I>> MyopicTorus<I, V> {
    pub fn s() -> usize {
        8
    }

    fn radius() -> V::F {
        // 2/s, quite a lot larger than the lattice spacing.
        (V::F::one() + V::F::one()) / <V::F as NumCast>::from(Self::s()).unwrap()
    }
}

impl<I: ICompatible<V>, V: VCompatible<I>> AsRef<Torus<I, V>> for MyopicTorus<I, V> {
    fn as_ref(&self) -> &Torus<I, V> {
        &self.0
    }
}

impl<I: ICompatible<V>, V: VCompatible<I>> From<Torus<I, V>> for MyopicTorus<I, V> {
    fn from(value: Torus<I, V>) -> Self {
        Self(value)
    }
}

impl<I: ICompatible<V>, V: VCompatible<I>> Bounded<Torus<I, V>, Torus<I, V>, V>
    for MyopicTorus<I, V>
{
    fn sdf(&self, v: &V) -> <V as Euclidean>::F {
        v.norm() - Self::radius()
    }
}

impl_tangent_bundle_via_bounded!(
    MyopicTorus<I, V>,
    Torus<I, V>,
    Torus<I, V>,
    V,
    I: ICompatible<V>, V: VCompatible<I>
);

#[derive(Debug, Clone)]
pub struct MyopicTorusCover<I: ICompatible<V>, V: VCompatible<I>>(MyopicTorus<I, V>);

impl<I: ICompatible<V>, V: VCompatible<I>> AsRef<MyopicTorus<I, V>> for MyopicTorusCover<I, V> {
    fn as_ref(&self) -> &MyopicTorus<I, V> {
        &self.0
    }
}

impl<I: ICompatible<V>, V: VCompatible<I>> From<MyopicTorus<I, V>> for MyopicTorusCover<I, V> {
    fn from(value: MyopicTorus<I, V>) -> Self {
        Self(value)
    }
}

impl<I: ICompatible<V>, V: VCompatible<I>> Bounded<MyopicTorus<I, V>, Torus<I, V>, V>
    for MyopicTorusCover<I, V>
{
    fn sdf(&self, v: &V) -> <V as Euclidean>::F {
        let to = |x| <V::F as NumCast>::from(x).unwrap();
        v.norm() - (to(2).sqrt() + to(2)) / to(4 * MyopicTorus::<I, V>::s())
    }
}

impl_tangent_bundle_via_bounded!(
    MyopicTorusCover<I, V>,
    MyopicTorus<I, V>,
    Torus<I, V>,
    V,
    I: ICompatible<V>, V: VCompatible<I>
);

impl<I: ICompatible<V>, V: VCompatible<I>> BuildNodes<MyopicTorusCover<I, V>>
    for MyopicTorusCover<I, V>
{
    fn build_nodes() -> Vec<Self> {
        let to = |x| <I::F as NumCast>::from(x).unwrap();
        let s_usize = MyopicTorus::<I, V>::s();
        let s = to(s_usize);
        let offset = to(1) / (to(2) * s);

        (0..s_usize)
            .flat_map(|y| (0..s_usize).map(move |x| (x, y)))
            .map(|(x, y)| {
                MyopicTorus(Torus::new(
                    S1([offset + to(x) / s].into()),
                    S1([offset + to(y) / s].into()),
                ))
                .into()
            })
            .collect()
    }
}

impl<I: ICompatible<V>, V: VCompatible<I>>
    NerveComplexParameters<Torus<I, V>, V, MyopicTorus<I, V>, MyopicTorusCover<I, V>>
    for MyopicTorusCover<I, V>
{
    fn overestimation_bound() -> Option<(V::F, V::F)> {
        let to = |x| <V::F as NumCast>::from(x).unwrap();
        let s = to(MyopicTorus::<I, V>::s());
        // κ: king-graph worst case at 22.5°, √(4 − 2√2). Scale-free.
        let kappa = (to(4) - to(2) * to(2).sqrt()).sqrt();
        // C = (1+κ)·2δ_s, with δ_s = √2/(2S) the lattice half-diagonal.
        let delta_s = to(2).sqrt() / (to(2) * s);
        Some((kappa, (V::F::one() + kappa) * to(2) * delta_s))
    }
}
