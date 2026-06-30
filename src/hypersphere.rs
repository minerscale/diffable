use std::marker::PhantomData;

use crate::{
    impl_lie_group_via_quotient,
    traits::{Chart, Euclidean, LieGroup, Metric, Quotient},
};
use num_traits::{NumCast, One, Zero, real::Real};

#[derive(Debug, PartialEq, Clone)]
pub struct Sphere<const N: usize, Rn: Euclidean> {
    real: Rn::Scalar,
    imag: Rn,
}

#[derive(Clone, Debug)]
pub struct Stereographic<Rn: Euclidean>(StereographicPole, PhantomData<Rn>);

impl<Rn: Euclidean> Stereographic<Rn> {
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

impl<const N: usize, Rn: Euclidean> Chart<Sphere<N, Rn>, Rn> for Stereographic<Rn> {
    fn to_local(&self, point: &Sphere<N, Rn>) -> Option<Rn> {
        let first = match self.0 {
            StereographicPole::NorthPole => point.real,
            StereographicPole::SouthPole => -point.real,
        };

        let epsilon = <Rn::Scalar as NumCast>::from(EPSILON).unwrap();

        let denom = Rn::Scalar::one() - first;
        if denom.abs() < epsilon {
            return None;
        } // at north pole

        let recip = denom.recip();
        Some(point.imag * recip)
    }

    fn to_global(&self, coord: Rn) -> Sphere<N, Rn> {
        let two = Rn::Scalar::one() + Rn::Scalar::one();
        let r_sq = coord.norm_squared();
        let denom = Rn::Scalar::one() + r_sq;
        Sphere::new(
            match self.0 {
                StereographicPole::NorthPole => (r_sq - Rn::Scalar::one()) / denom,
                StereographicPole::SouthPole => (Rn::Scalar::one() - r_sq) / denom,
            },
            coord * (two / denom),
        )
    }

    fn chart_at(p: &Sphere<N, Rn>) -> Self {
        if p.real > Rn::Scalar::zero() {
            Self::south_pole()
        } else {
            Self::north_pole()
        }
    }
}

impl<const N: usize, Rn: Euclidean> Sphere<N, Rn> {
    pub fn real(&self) -> Rn::Scalar {
        self.real
    }
    pub fn imag(&self) -> Rn {
        self.imag
    }

    fn normalised(self) -> Self {
        let real = self.real;
        let imag = self.imag;
        let sum = real * real + imag.iter().fold(Rn::Scalar::zero(), |acc, &v| acc + v * v);

        assert!(sum != Rn::Scalar::zero());
        let q_rsqrt = Rn::Scalar::sqrt(sum).recip(); // What the f***?

        Self {
            real: real * q_rsqrt,
            imag: imag * q_rsqrt,
        }
    }

    pub fn new(real: Rn::Scalar, imag: Rn) -> Self {
        let sphere = Sphere { real, imag };

        sphere.normalised()
    }
}

impl<Rn: Euclidean> LieGroup<Rn> for Sphere<0, Rn> {
    fn identity() -> Self {
        Self::new(Rn::Scalar::one(), Rn::zero())
    }

    fn compose(&self, other: &Self) -> Self {
        Self::new(self.real * other.real, Rn::zero())
    }

    // in Z/2Z each element is its own inverse.
    fn inverse(&self) -> Self {
        Self::new(self.real, Rn::zero())
    }

    fn identity_exp(_: Rn) -> Self {
        Sphere::new(Rn::Scalar::one(), Rn::zero())
    }

    fn identity_log(p: &Self) -> Option<Rn> {
        if p.real > Rn::Scalar::zero() {
            Some(Rn::zero())
        } else {
            None
        }
    }
}

impl<Rn: Euclidean> LieGroup<Rn> for Sphere<1, Rn> {
    fn identity() -> Self {
        Sphere::new(Rn::Scalar::one(), Rn::zero())
    }

    fn compose(&self, other: &Self) -> Self {
        let (a1, [b1]) = (self.real, self.imag.to_array());
        let (a2, [b2]) = (other.real, other.imag.to_array());

        Sphere::new(a1 * a2 - b1 * b2, Rn::from_array([a1 * b2 + a2 * b1]))
    }

    fn inverse(&self) -> Self {
        Sphere::new(self.real, -self.imag)
    }

    fn identity_exp(v: Rn) -> Self {
        let alpha = v[0];

        Sphere::new(alpha.cos(), Rn::from_array([alpha.sin()]))
    }

    fn identity_log(p: &Self) -> Option<Rn> {
        Some(Rn::from_array([Rn::Scalar::atan2(p.imag[0], p.real)]))
    }
}

impl<Rn: Euclidean> LieGroup<Rn> for Sphere<3, Rn> {
    fn identity() -> Self {
        Sphere::new(Rn::Scalar::one(), Rn::zero())
    }

    fn compose(&self, other: &Self) -> Self {
        let (a1, [b1, c1, d1]) = (self.real, self.imag.to_array());
        let (a2, [b2, c2, d2]) = (other.real, other.imag.to_array());
        Sphere::new(
            a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
            Rn::from_array([
                a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
                a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
            ]),
        )
    }

    fn inverse(&self) -> Self {
        let (a, [b, c, d]) = (self.real, self.imag.to_array());

        Sphere::new(a, Rn::from_array([-b, -c, -d]))
    }

    fn identity_exp(v: Rn) -> Self {
        let two = Rn::Scalar::one() + Rn::Scalar::one();
        let three = two + Rn::Scalar::one();
        let six = two * three;
        let alpha = Rn::Scalar::sqrt(v.iter().fold(Rn::Scalar::zero(), |acc, &x| acc + x * x));
        let (sin, cos) = alpha.sin_cos();
        let sinc = if alpha < <Rn::Scalar as NumCast>::from(EPSILON).unwrap() {
            Rn::Scalar::one() - alpha * alpha / six
        } else {
            sin / alpha
        };
        Sphere::new(cos, v * sinc)
    }

    fn identity_log(p: &Self) -> Option<Rn> {
        let two = Rn::Scalar::one() + Rn::Scalar::one();
        let three = two + Rn::Scalar::one();
        let six = two * three;
        let epsilon = <Rn::Scalar as NumCast>::from(EPSILON).unwrap();
        if (p.real + Rn::Scalar::one()).abs() < epsilon {
            return None; // antipode singularity
        }
        let alpha = Rn::Scalar::acos(p.real);
        let sin = alpha.sin();
        let sinc_recip = if sin.abs() < epsilon {
            Rn::Scalar::one() + alpha * alpha / six
        } else {
            alpha / sin
        };
        Some(p.imag * sinc_recip)
    }
}

impl<const N: usize, Rn: Euclidean> Metric<Rn::Scalar> for Sphere<N, Rn> {
    fn distance(&self, other: &Self) -> Rn::Scalar {
        let dot = self.real * other.real + self.imag.dot(&other.imag);

        Rn::Scalar::acos(dot.min(Rn::Scalar::one()).max(-Rn::Scalar::one()))
    }
}

#[derive(PartialEq, Clone, Debug)]
pub struct So3<Rn: Euclidean> (Sphere<3, Rn>);

impl<Rn: Euclidean> Quotient<Sphere<3, Rn>, Sphere<0, Rn>, Rn> for So3<Rn> {
    fn new(g: Sphere<3, Rn>) -> Self {
        // pick the representative with non-negative real component
        if g.real() >= Rn::Scalar::zero() {
            So3(g)
        } else {
            So3(Sphere::new(-g.real(), -g.imag()))
        }
    }

    fn lift(&self) -> Sphere<3, Rn> {
        self.0.clone()
    }

    fn embed(h: Sphere<0, Rn>) -> Sphere<3, Rn> {
        Sphere::new(h.real(), Rn::zero())
    }
}

impl_lie_group_via_quotient!(So3<Rn>, Sphere<3, _>, Sphere<0, _>);
