use std::{
    marker::PhantomData,
    ops::{Add, Neg},
};

use crate::{
    impl_lie_group_via_quotient,
    traits::{Chart, Euclidean, Group, LieGroup, Metric, Quotient, Smooth},
};

use num_traits::{Euclid, NumCast, One, Zero, real::Real};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct S1Quotient<V: Euclidean + From<[<V as Euclidean>::F; 1]>>(V);

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Z<V: Euclidean>(pub isize, PhantomData<V>);

impl<V: Euclidean> Z<V> {
    pub fn new(v: isize) -> Self {
        Self(v, PhantomData)
    }
}

impl<V: Euclidean> Add for Z<V> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.0 + rhs.0)
    }
}

impl<V: Euclidean> Zero for Z<V> {
    fn zero() -> Self {
        Self::new(0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl<V: Euclidean> Neg for Z<V> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0, PhantomData)
    }
}

impl<V: Euclidean> Group for Z<V> {
    fn identity() -> Self {
        Self::zero()
    }

    fn compose(&self, other: &Self) -> Self {
        *self + *other
    }

    fn inverse(&self) -> Self {
        -*self
    }
}

impl<V: Euclidean> LieGroup<V> for Z<V> {
    fn identity_exp(_: V) -> Self {
        Self::zero()
    }

    fn identity_log(p: &Self) -> Option<V> {
        if p.is_zero() { Some(V::zero()) } else { None }
    }
}

impl<V: Euclidean + From<[V::F; 1]>> Metric<V::F> for S1Quotient<V> {
    fn distance(&self, other: &Self) -> V::F {
        self.to_local(other).unwrap().norm()
    }
}

impl<V: Euclidean + From<[<V as Euclidean>::F; 1]>> Quotient<V, Z<V>, V> for S1Quotient<V> {
    fn new(g: V) -> Self {
        Self([g[0].rem_euclid(&V::F::one())].into())
    }

    fn lift(&self) -> V {
        self.0
    }

    fn embed(h: Z<V>) -> V {
        [<V::F as NumCast>::from(h.0).unwrap()].into()
    }
}

impl_lie_group_via_quotient!(S1Quotient<V>, V, Z<V>, From<[<V as Euclidean>::F; 1]>);

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Torus<I: Euclidean + From<[I::F; 1]> + From<[V::F; 1]>, V: Euclidean + From<[I::F; 2]>>(
    S1Quotient<I>,
    S1Quotient<I>,
    PhantomData<V>,
);

impl<I: Euclidean + From<[I::F; 1]> + From<[V::F; 1]>, V: Euclidean + From<[I::F; 2]>> Torus<I, V> {
    pub fn new(a: S1Quotient<I>, b: S1Quotient<I>) -> Self {
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
        Self::new(S1Quotient::identity(), S1Quotient::identity())
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
        Self::new(S1Quotient::identity_exp(v0), S1Quotient::identity_exp(v1))
    }

    fn identity_log(p: &Self) -> Option<V> {
        let a = S1Quotient::identity_log(&p.0)?;
        let b = S1Quotient::identity_log(&p.1)?;

        Some([a[0], b[0]].into())
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct KleinBottle<
    I: Euclidean + From<[I::F; 1]> + From<[V::F; 1]>,
    V: Euclidean + From<[I::F; 2]>,
>(S1Quotient<I>, S1Quotient<I>, PhantomData<V>);

impl<I: Euclidean + From<[I::F; 1]> + From<[V::F; 1]>, V: Euclidean + From<[I::F; 2]>>
    KleinBottle<I, V>
{
    pub fn new(a: S1Quotient<I>, b: S1Quotient<I>) -> Self {
        Self(a, b, PhantomData)
    }
}

impl<I: Euclidean + From<[I::F; 1]> + From<[V::F; 1]>, V: Euclidean + From<[I::F; 2]>> Smooth<V>
    for KleinBottle<I, V>
where
    I::F: From<V::F>,
{
    fn exp(&self, v: V) -> Self {
        let (x, y) = self.coords();
        // v[0] is the fiber (x) tangent, v[1] the base (y) tangent
        let vx: I::F = v[0].into(); // V::F → I::F
        let vy: I::F = v[1].into();
        Self::from_cover(x + vx, y + vy)
    }

    fn log(&self, other: &Self) -> Option<V> {
        let one = I::F::one();
        let (sx, sy) = self.coords();
        let (ox, oy) = other.coords();

        let mut best: Option<(I::F, I::F)> = None;
        let mut best_sq = I::F::zero();

        // search nearby lifts: y-shift n, x-shift m ∈ {−1, 0, 1}
        for n in [-one, I::F::zero(), one] {
            // parity of the y-shift decides whether this lift is flipped
            let two = one + one;
            let n_odd = n.rem_euclid(&two) != I::F::zero();
            // base lifted x depends on flip: even ⇒ ox, odd ⇒ 1 − ox
            let base_ox = if n_odd { one - ox } else { ox };
            for m in [-one, I::F::zero(), one] {
                let cx = base_ox + m; // lifted x of `other`
                let cy = oy + n; // lifted y of `other`
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
    I: Euclidean + From<[I::F; 1]> + From<[V::F; 1]>,
    V: Euclidean + From<[I::F; 2]>,
    I::F: From<V::F>,
{
    /// Reduce a cover point (x, y) ∈ ℝ² to the fundamental domain via the
    /// Klein-bottle group Γ = ⟨A, B⟩:
    ///   A: (x, y) ↦ (x + 1, y)          — fiber translation
    ///   B: (x, y) ↦ (−x, y + 1)         — base translation with fiber flip
    ///
    /// Reducing y mod 1 requires flipping x once per y-unit crossed, so the
    /// flip is keyed on the parity of ⌊y⌋.
    fn from_cover(x: I::F, y: I::F) -> Self {
        let one = I::F::one();

        // reduce y, capturing the integer crossing count for parity
        let ky = y.floor(); // number of y-seams crossed
        let y_red = y - ky; // y mod 1, in [0, 1)

        // odd number of crossings ⇒ fiber is flipped
        let two = one + one;
        let ky_parity_odd = (ky - (ky / two).floor() * two) != I::F::zero();
        // (ky mod 2 != 0); floor-based since I::F is a float, not an int

        let x_oriented = if ky_parity_odd { -x } else { x };

        // reduce x mod 1 (rem_euclid handles the negative-from-flip case)
        let x_red = x_oriented.rem_euclid(&one);

        Self(
            S1Quotient::new([x_red].into()),
            S1Quotient::new([y_red].into()),
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
