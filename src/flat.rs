use crate::{
    discrete::Z,
    impl_lie_group_via_quotient, impl_tangent_bundle_via_bounded,
    traits::{Bounded, BuildNodes, NerveComplex},
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

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct KleinBottle<
    I: Euclidean<F: From<V::F>> + From<[I::F; 1]> + From<[V::F; 1]>,
    V: Euclidean + From<[I::F; 2]>,
>(S1<I>, S1<I>, PhantomData<V>);

impl<
    I: Euclidean<F: From<V::F>> + From<[I::F; 1]> + From<[V::F; 1]>,
    V: Euclidean + From<[I::F; 2]>,
> KleinBottle<I, V>
{
    pub fn new(a: S1<I>, b: S1<I>) -> Self {
        Self(a, b, PhantomData)
    }
}

impl<
    I: Euclidean<F: From<V::F>> + From<[I::F; 1]> + From<[V::F; 1]>,
    V: Euclidean + From<[I::F; 2]>,
> Smooth<V> for KleinBottle<I, V>
{
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

impl<I, V> KleinBottle<I, V>
where
    I: Euclidean<F: From<V::F>> + From<[I::F; 1]> + From<[V::F; 1]>,
    V: Euclidean + From<[I::F; 2]>,
{
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

impl<I: Euclidean + From<[I::F; 1]> + From<[V::F; 1]>, V: Euclidean + From<[I::F; 2]>> Metric<I::F>
    for KleinBottle<I, V>
where
    I::F: From<V::F>,
{
    fn distance(&self, other: &Self) -> I::F {
        self.to_local(other).unwrap().norm().into()
    }
}

#[derive(Debug)]
pub struct TorusCover<
    I: Euclidean<F: From<V::F>> + From<[I::F; 1]> + From<[V::F; 1]>,
    V: Euclidean + From<[I::F; 2]>,
>(Torus<I, V>);

impl<
    I: Euclidean<F: From<V::F>> + From<[I::F; 1]> + From<[V::F; 1]>,
    V: Euclidean + From<[I::F; 2]>,
> From<Torus<I, V>> for TorusCover<I, V>
{
    fn from(value: Torus<I, V>) -> Self {
        Self(value)
    }
}

impl<
    I: Euclidean<F: From<V::F>> + From<[I::F; 1]> + From<[V::F; 1]>,
    V: Euclidean + From<[I::F; 2]>,
> AsRef<Torus<I, V>> for TorusCover<I, V>
{
    fn as_ref(&self) -> &Torus<I, V> {
        &self.0
    }
}

const S: usize = 4;

impl<
    I: Euclidean<F: From<V::F>> + From<[I::F; 1]> + From<[V::F; 1]>,
    V: Euclidean + From<[I::F; 2]>,
> Bounded<Torus<I, V>, V> for TorusCover<I, V>
{
    fn sdf(&self, v: &V) -> <V as Euclidean>::F {
        let to = |x| <V::F as NumCast>::from(x).unwrap();
        v.norm() - (to(2).sqrt() + to(2)) / to(4 * S)
    }
}

impl_tangent_bundle_via_bounded!(
    TorusCover<I, V>,
    Torus<I, V>,
    V,
    I: Euclidean<F: From<V::F>> + From<[I::F; 1]> + From<[V::F; 1]>, V: Euclidean + From<[I::F; 2]>
);

impl<
    I: Euclidean<F: From<V::F>> + From<[I::F; 1]> + From<[V::F; 1]> + 'static + Send + Sync,
    V: Euclidean + From<[I::F; 2]> + 'static + Send + Sync,
> BuildNodes<TorusCover<I, V>> for TorusCover<I, V>
{
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

impl<
    I: Euclidean<F: From<V::F>> + From<[I::F; 1]> + From<[V::F; 1]> + Send + Sync + 'static,
    V: Euclidean + From<[I::F; 2]> + Send + Sync + 'static,
> NerveComplex<Torus<I, V>, V, Torus<I, V>, TorusCover<I, V>> for TorusCover<I, V>
{
}

#[derive(Debug)]
pub struct KleinBottleCover<
    I: Euclidean<F: From<V::F>> + From<[I::F; 1]> + From<[V::F; 1]>,
    V: Euclidean + From<[I::F; 2]>,
>(KleinBottle<I, V>);

impl<
    I: Euclidean<F: From<V::F>> + From<[I::F; 1]> + From<[V::F; 1]>,
    V: Euclidean + From<[I::F; 2]>,
> From<KleinBottle<I, V>> for KleinBottleCover<I, V>
{
    fn from(value: KleinBottle<I, V>) -> Self {
        Self(value)
    }
}

impl<
    I: Euclidean<F: From<V::F>> + From<[I::F; 1]> + From<[V::F; 1]>,
    V: Euclidean + From<[I::F; 2]>,
> AsRef<KleinBottle<I, V>> for KleinBottleCover<I, V>
{
    fn as_ref(&self) -> &KleinBottle<I, V> {
        &self.0
    }
}

impl<
    I: Euclidean<F: From<V::F>> + From<[I::F; 1]> + From<[V::F; 1]>,
    V: Euclidean + From<[I::F; 2]>,
> Bounded<KleinBottle<I, V>, V> for KleinBottleCover<I, V>
{
    fn sdf(&self, v: &V) -> <V as Euclidean>::F {
        let to = |x| <V::F as NumCast>::from(x).unwrap();
        v.norm() - (to(2).sqrt() + to(2)) / to(4 * S)
    }
}

impl_tangent_bundle_via_bounded!(
    KleinBottleCover<I, V>,
    KleinBottle<I, V>,
    V,
    I: Euclidean<F: From<V::F>> + From<[I::F; 1]> + From<[V::F; 1]>, V: Euclidean + From<[I::F; 2]>
);

impl<
    I: Euclidean<F: From<V::F>> + From<[I::F; 1]> + From<[V::F; 1]> + 'static + Send + Sync,
    V: Euclidean + From<[I::F; 2]> + 'static + Send + Sync,
> BuildNodes<KleinBottleCover<I, V>> for KleinBottleCover<I, V>
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

impl<
    I: Euclidean<F: From<V::F>> + From<[I::F; 1]> + From<[V::F; 1]> + Send + Sync + 'static,
    V: Euclidean + From<[I::F; 2]> + Send + Sync + 'static,
> NerveComplex<KleinBottle<I, V>, V, KleinBottle<I, V>, KleinBottleCover<I, V>>
    for KleinBottleCover<I, V>
{
}
