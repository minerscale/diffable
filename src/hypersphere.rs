use std::marker::PhantomData;

use crate::{
    impl_lie_group_via_quotient,
    traits::{Chart, Euclidean, LieGroup, Metric, Quotient},
};
use num_traits::{NumCast, One, Zero, real::Real};

#[derive(Debug, PartialEq, Clone)]
pub struct Sphere<const N: usize, V: Euclidean> {
    real: V::F,
    imag: V,
}

#[derive(Clone, Debug)]
pub struct Stereographic<V: Euclidean>(StereographicPole, PhantomData<V>);

impl<V: Euclidean> Stereographic<V> {
    pub const fn south_pole() -> Self {
        Self(StereographicPole::SouthPole, PhantomData)
    }
    pub const fn north_pole() -> Self {
        Self(StereographicPole::NorthPole, PhantomData)
    }
}

#[derive(Clone, Debug)]
enum StereographicPole {
    SouthPole,
    NorthPole,
}

pub const EPSILON: f64 = 1e-10;

impl<const N: usize, V: Euclidean> Chart<Sphere<N, V>, V> for Stereographic<V> {
    fn to_local(&self, point: &Sphere<N, V>) -> Option<V> {
        let first = match self.0 {
            StereographicPole::NorthPole => point.real,
            StereographicPole::SouthPole => -point.real,
        };

        let epsilon = <V::F as NumCast>::from(EPSILON).unwrap();

        let denom = V::F::one() - first;
        if denom.abs() < epsilon {
            return None;
        } // at north pole

        let recip = denom.recip();
        Some(point.imag * recip)
    }

    fn to_global(&self, coord: V) -> Sphere<N, V> {
        let two = V::F::one() + V::F::one();
        let r_sq = coord.norm_squared();
        let denom = V::F::one() + r_sq;
        Sphere::new(
            match self.0 {
                StereographicPole::NorthPole => (r_sq - V::F::one()) / denom,
                StereographicPole::SouthPole => (V::F::one() - r_sq) / denom,
            },
            coord * (two / denom),
        )
    }

    fn chart_at(p: &Sphere<N, V>) -> Self {
        if p.real > V::F::zero() {
            Self::south_pole()
        } else {
            Self::north_pole()
        }
    }
}

impl<const N: usize, V: Euclidean> Sphere<N, V> {
    pub fn real(&self) -> V::F {
        self.real
    }
    pub fn imag(&self) -> V {
        self.imag
    }

    fn normalised(self) -> Self {
        let real = self.real;
        let imag = self.imag;
        let sum = real * real + imag.iter().fold(V::F::zero(), |acc, &v| acc + v * v);

        assert!(sum != V::F::zero());
        let q_rsqrt = V::F::sqrt(sum).recip(); // What the f***?

        Self {
            real: real * q_rsqrt,
            imag: imag * q_rsqrt,
        }
    }

    pub fn new(real: V::F, imag: V) -> Self {
        let sphere = Sphere { real, imag };

        sphere.normalised()
    }
}

impl<V: Euclidean> LieGroup<V> for Sphere<0, V> {
    fn identity() -> Self {
        Self::new(V::F::one(), V::zero())
    }

    fn compose(&self, other: &Self) -> Self {
        Self::new(self.real * other.real, V::zero())
    }

    // in Z/2Z each element is its own inverse.
    fn inverse(&self) -> Self {
        Self::new(self.real, V::zero())
    }

    fn identity_exp(_: V) -> Self {
        Sphere::new(V::F::one(), V::zero())
    }

    fn identity_log(p: &Self) -> Option<V> {
        if p.real > V::F::zero() {
            Some(V::zero())
        } else {
            None
        }
    }
}

impl<V: Euclidean> LieGroup<V> for Sphere<1, V> {
    fn identity() -> Self {
        Sphere::new(V::F::one(), V::zero())
    }

    fn compose(&self, other: &Self) -> Self {
        let (a1, [b1]) = (self.real, self.imag.to_array());
        let (a2, [b2]) = (other.real, other.imag.to_array());

        Sphere::new(a1 * a2 - b1 * b2, V::from_array([a1 * b2 + a2 * b1]))
    }

    fn inverse(&self) -> Self {
        Sphere::new(self.real, -self.imag)
    }

    fn identity_exp(v: V) -> Self {
        let alpha = v[0];

        Sphere::new(alpha.cos(), V::from_array([alpha.sin()]))
    }

    fn identity_log(p: &Self) -> Option<V> {
        Some(V::from_array([V::F::atan2(p.imag[0], p.real)]))
    }
}

impl<V: Euclidean> LieGroup<V> for Sphere<3, V> {
    fn identity() -> Self {
        Sphere::new(V::F::one(), V::zero())
    }

    fn compose(&self, other: &Self) -> Self {
        let (a1, [b1, c1, d1]) = (self.real, self.imag.to_array());
        let (a2, [b2, c2, d2]) = (other.real, other.imag.to_array());
        Sphere::new(
            a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
            V::from_array([
                a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
                a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
            ]),
        )
    }

    fn inverse(&self) -> Self {
        let (a, [b, c, d]) = (self.real, self.imag.to_array());

        Sphere::new(a, V::from_array([-b, -c, -d]))
    }

    fn identity_exp(v: V) -> Self {
        let two = V::F::one() + V::F::one();
        let three = two + V::F::one();
        let six = two * three;
        let alpha = V::F::sqrt(v.iter().fold(V::F::zero(), |acc, &x| acc + x * x));
        let (sin, cos) = alpha.sin_cos();
        let sinc = if alpha < <V::F as NumCast>::from(EPSILON).unwrap() {
            V::F::one() - alpha * alpha / six
        } else {
            sin / alpha
        };
        Sphere::new(cos, v * sinc)
    }

    fn identity_log(p: &Self) -> Option<V> {
        let two = V::F::one() + V::F::one();
        let three = two + V::F::one();
        let six = two * three;
        let epsilon = <V::F as NumCast>::from(EPSILON).unwrap();
        if (p.real + V::F::one()).abs() < epsilon {
            return None; // antipode singularity
        }
        let alpha = V::F::acos(p.real);
        let sin = alpha.sin();
        let sinc_recip = if sin.abs() < epsilon {
            V::F::one() + alpha * alpha / six
        } else {
            alpha / sin
        };
        Some(p.imag * sinc_recip)
    }
}

impl<const N: usize, V: Euclidean> Metric<V::F> for Sphere<N, V> {
    fn distance(&self, other: &Self) -> V::F {
        let dot = self.real * other.real + self.imag.dot(&other.imag);

        V::F::acos(dot.min(V::F::one()).max(-V::F::one()))
    }
}

#[derive(PartialEq, Clone, Debug)]
pub struct So3<V: Euclidean>(Sphere<3, V>);

impl<V: Euclidean> Quotient<Sphere<3, V>, Sphere<0, V>, V> for So3<V> {
    fn new(g: Sphere<3, V>) -> Self {
        // pick the representative with non-negative real component
        if g.real() >= V::F::zero() {
            So3(g)
        } else {
            So3(Sphere::new(-g.real(), -g.imag()))
        }
    }

    fn lift(&self) -> Sphere<3, V> {
        self.0.clone()
    }

    fn embed(h: Sphere<0, V>) -> Sphere<3, V> {
        Sphere::new(h.real(), V::zero())
    }
}

impl_lie_group_via_quotient!(So3<V>, Sphere<3, _>, Sphere<0, _>);
