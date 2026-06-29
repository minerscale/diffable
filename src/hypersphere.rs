use std::marker::PhantomData;

use crate::traits::{Chart, Euclidean, ExpMap, LieGroup, Metric};
use num_traits::{NumCast, One, Zero, real::Real};

#[derive(Debug, PartialEq, Clone)]
pub struct Sphere<const N: usize, Rn: Euclidean<N>> {
    real: Rn::Scalar,
    imag: Rn,
}

#[derive(Clone, Debug)]
pub struct Stereographic<const N: usize, Rn: Euclidean<N>>(StereographicPole, PhantomData<Rn>);

impl<const N: usize, Rn: Euclidean<N>> Stereographic<N, Rn> {
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

impl<const N: usize, Rn: Euclidean<N>> Chart<Sphere<N, Rn>, N, Rn> for Stereographic<N, Rn> {
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

impl<const N: usize, Rn: Euclidean<N>> Sphere<N, Rn> {
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

impl<Rn: Euclidean<0>> LieGroup<0> for Sphere<0, Rn> {
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
}

impl<Rn: Euclidean<1>> LieGroup<1> for Sphere<1, Rn> {
    fn identity() -> Self {
        Sphere::new(Rn::Scalar::one(), Rn::zero())
    }

    fn compose(&self, other: &Self) -> Self {
        let (a1, [b1]) = (self.real, self.imag.into());
        let (a2, [b2]) = (other.real, other.imag.into());

        Sphere::new(a1 * a2 - b1 * b2, [a1 * b2 + a2 * b1].into())
    }

    fn inverse(&self) -> Self {
        Sphere::new(self.real, -self.imag)
    }
}

impl<Rn: Euclidean<3>> LieGroup<3> for Sphere<3, Rn> {
    fn identity() -> Self {
        Sphere::new(Rn::Scalar::one(), Rn::zero())
    }

    fn compose(&self, other: &Self) -> Self {
        let (a1, [b1, c1, d1]) = (self.real, self.imag.into());
        let (a2, [b2, c2, d2]) = (other.real, other.imag.into());
        Sphere::new(
            a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
            [
                a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
                a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
            ]
            .into(),
        )
    }

    fn inverse(&self) -> Self {
        let (a, [b, c, d]) = (self.real, self.imag.into());

        Sphere::new(a, [-b, -c, -d].into())
    }
}

#[derive(Clone, Debug)]
pub struct SphereExpMap<const N: usize, Rn: Euclidean<N>>(Stereographic<N, Rn>);

impl<const N: usize, Rn: Euclidean<N>> SphereExpMap<N, Rn> {
    pub fn new(p: Stereographic<N, Rn>) -> Self {
        Self(p)
    }
}

impl<Rn: Euclidean<0>> Chart<Sphere<0, Rn>, 0, Rn> for SphereExpMap<0, Rn> {
    fn to_global(&self, _: Rn) -> Sphere<0, Rn> {
        Sphere::new(
            match self.0.0 {
                StereographicPole::SouthPole => Rn::Scalar::one(),
                StereographicPole::NorthPole => -Rn::Scalar::one(),
            },
            Rn::zero(),
        )
    }

    fn to_local(&self, point: &Sphere<0, Rn>) -> Option<Rn> {
        match (&self.0.0, point.real > Rn::Scalar::zero()) {
            (StereographicPole::SouthPole, true) => Some([].into()),
            (StereographicPole::SouthPole, false) => None,
            (StereographicPole::NorthPole, true) => None,
            (StereographicPole::NorthPole, false) => Some([].into()),
        }
    }

    fn chart_at(p: &Sphere<0, Rn>) -> Self {
        Self(Stereographic::chart_at(p))
    }
}

impl<Rn: Euclidean<1>> Chart<Sphere<1, Rn>, 1, Rn> for SphereExpMap<1, Rn> {
    fn to_global(&self, coord: Rn) -> Sphere<1, Rn> {
        let alpha = coord[0];

        let real = alpha.cos();
        Sphere::new(
            match self.0.0 {
                StereographicPole::SouthPole => real,
                StereographicPole::NorthPole => -real,
            },
            [alpha.sin()].into(),
        )
    }

    fn to_local(&self, point: &Sphere<1, Rn>) -> Option<Rn> {
        let real = match self.0.0 {
            StereographicPole::NorthPole => -point.real,
            StereographicPole::SouthPole => point.real,
        };

        Some([Rn::Scalar::atan2(point.imag[0], real)].into())
    }

    fn chart_at(p: &Sphere<1, Rn>) -> Self {
        Self(Stereographic::chart_at(p))
    }
}

impl<Rn: Euclidean<3>> Chart<Sphere<3, Rn>, 3, Rn> for SphereExpMap<3, Rn> {
    fn to_local(&self, point: &Sphere<3, Rn>) -> Option<Rn> {
        let two = Rn::Scalar::one() + Rn::Scalar::one();
        let three = two + Rn::Scalar::one();
        let six = two * three;

        let real = match self.0.0 {
            StereographicPole::SouthPole => point.real,
            StereographicPole::NorthPole => -point.real,
        };
        let epsilon = <Rn::Scalar as NumCast>::from(EPSILON).unwrap();
        if (real + Rn::Scalar::one()).abs() < epsilon {
            return None; // at the singularity
        }
        let alpha = Rn::Scalar::acos(real);
        let sin = alpha.sin();
        let sinc_recip = if sin.abs() < epsilon {
            Rn::Scalar::one() + alpha * alpha / six
        } else {
            alpha / sin
        };
        Some(point.imag * sinc_recip)
    }

    fn to_global(&self, coord: Rn) -> Sphere<3, Rn> {
        let two = Rn::Scalar::one() + Rn::Scalar::one();
        let three = two + Rn::Scalar::one();
        let six = two * three;

        let alpha = Rn::Scalar::sqrt(coord.iter().fold(Rn::Scalar::zero(), |acc, &x| acc + x * x));
        let (sin, cos) = alpha.sin_cos();
        let real = match self.0.0 {
            StereographicPole::SouthPole => cos,
            StereographicPole::NorthPole => -cos,
        };

        let sinc = if alpha < <Rn::Scalar as NumCast>::from(EPSILON).unwrap() {
            Rn::Scalar::one() - alpha * alpha / six // sinc(0) = 1
        } else {
            sin / alpha
        };

        Sphere::new(real, coord * sinc)
    }

    fn chart_at(p: &Sphere<3, Rn>) -> Self {
        Self(Stereographic::chart_at(p))
    }
}

impl<const N: usize, Rn: Euclidean<N>> ExpMap<Sphere<N, Rn>, N, Rn> for SphereExpMap<N, Rn> where
    SphereExpMap<N, Rn>: Chart<Sphere<N, Rn>, N, Rn>
{
}

impl<const N: usize, Rn: Euclidean<N>> Metric<Rn::Scalar> for Sphere<N, Rn> {
    fn distance(&self, other: &Self) -> Rn::Scalar {
        let dot = self.real * other.real + self.imag.dot(&other.imag);

        Rn::Scalar::acos(dot.min(Rn::Scalar::one()).max(-Rn::Scalar::one()))
    }
}
