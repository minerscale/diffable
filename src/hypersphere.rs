use std::marker::PhantomData;

use crate::{
    coords::Coords,
    impl_lie_group_via_quotient, impl_tangent_bundle_via_bounded,
    traits::{
        Bounded, Chart, Euclidean, ExpMap, Group, InnerProduct, LieGroup, Metric, NerveComplex,
        Quotient, TangentBundle,
    },
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

pub const EPSILON: f64 = 1e-3;

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

impl<V: Euclidean> Group for Sphere<0, V> {
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
}

impl<V: Euclidean> LieGroup<V> for Sphere<0, V> {
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

impl<V: Euclidean> Group for Sphere<1, V> {
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
}

impl<V: Euclidean> LieGroup<V> for Sphere<1, V> {
    fn identity_exp(v: V) -> Self {
        let alpha = v[0];

        Sphere::new(alpha.cos(), V::from_array([alpha.sin()]))
    }

    fn identity_log(p: &Self) -> Option<V> {
        Some(V::from_array([V::F::atan2(p.imag[0], p.real)]))
    }
}

impl<V: Euclidean> Group for Sphere<3, V> {
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
}

impl<V: Euclidean> LieGroup<V> for Sphere<3, V> {
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

        let eps = <V::F as NumCast>::from(EPSILON).unwrap();
        if (p.real + V::F::one()).abs() < eps {
            return None; // antipode singularity
        }
        // use atan2 instead of acos for numerical stability
        let imag_norm = p.imag.norm();
        let alpha = V::F::atan2(imag_norm, p.real);
        let sinc_recip = if imag_norm < eps {
            V::F::one() + alpha * alpha / six
        } else {
            alpha / imag_norm // alpha / sin(alpha) = alpha / ||imag||
        };
        Some(p.imag * sinc_recip)
    }
}

impl<const N: usize, V: Euclidean> Metric<V::F> for Sphere<N, V> {
    fn distance(&self, other: &Self) -> V::F {
        let two = V::F::one() + V::F::one();
        let diff_real = self.real - other.real;
        let diff_imag = self.imag - other.imag;
        let half_chord_sq = (diff_real * diff_real + diff_imag.norm_squared()) / (two * two);
        half_chord_sq.sqrt().asin() * two
    }
}

#[derive(Clone, Debug, PartialEq)]
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

use crate::epsilon_metric::R64;

#[derive(PartialEq, Debug, Clone)]
pub struct S1Cover(Sphere<1, Coords<R64, 1>>);

impl Bounded<Sphere<1, Coords<R64, 1>>, Coords<R64, 1>> for S1Cover {
    fn sdf(&self, v: &Coords<R64, 1>) -> R64 {
        v.norm() - R64(std::f64::consts::PI / 3.0 + 0.1)
    }

    fn new(p: Sphere<1, Coords<R64, 1>>) -> Self {
        Self(p)
    }

    fn inner(&self) -> &Sphere<1, Coords<R64, 1>> {
        &self.0
    }
}

impl_tangent_bundle_via_bounded!(
    S1Cover, Sphere<1, Coords<R64, 1>>, Coords<R64, 1>
);

impl NerveComplex<Sphere<1, Coords<R64, 1>>, Coords<R64, 1>, S1Cover> for S1Cover {
    fn nodes() -> &'static [S1Cover] {
        use std::sync::LazyLock;
        static NODES: LazyLock<Vec<S1Cover>> = LazyLock::new(|| {
            (0..6)
                .map(|i| {
                    let angle: R64 = R64(i.into()) * R64(std::f64::consts::TAU) / R64(6.0);
                    S1Cover(Sphere::new(angle.cos(), [angle.sin()].into()))
                })
                .collect()
        });
        &NODES
    }
}

#[derive(PartialEq, Debug, Clone)]
pub struct So3Cover(So3<Coords<R64, 3>>);

impl Chart<So3<Coords<R64, 3>>, Coords<R64, 3>> for So3Cover {
    fn to_local(&self, point: &So3<Coords<R64, 3>>) -> Option<Coords<R64, 3>> {
        self.0.to_local(point)
    }
    fn to_global(&self, coord: Coords<R64, 3>) -> So3<Coords<R64, 3>> {
        self.0.to_global(coord)
    }
    fn chart_at(p: &So3<Coords<R64, 3>>) -> Self {
        Self(So3::chart_at(p))
    }
}

impl ExpMap<So3<Coords<R64, 3>>, Coords<R64, 3>> for So3Cover {}

impl TangentBundle<So3<Coords<R64, 3>>, Coords<R64, 3>> for So3Cover {}

impl Bounded<So3<Coords<R64, 3>>, Coords<R64, 3>> for So3Cover {
    fn sdf(&self, v: &Coords<R64, 3>) -> R64 {
        // distance from v to self's base point (origin in self's frame)
        let self_dist = v.norm();

        // distance from v to each other node in self's local frame
        let nearest_other = Self::nodes()
            .iter()
            .filter(|node| *node != self)
            .filter_map(|node| {
                // other node's base point in self's local coordinates
                self.to_local(&node.0).map(|local| (*v - local).norm())
            })
            .reduce(|a, b| if a < b { a } else { b })
            .expect("NerveComplex must have at least 2 nodes");

        // negative inside Voronoi cell (self is nearest)
        self_dist - nearest_other
    }

    fn new(p: So3<Coords<R64, 3>>) -> Self {
        Self(p)
    }

    fn inner(&self) -> &So3<Coords<R64, 3>> {
        &self.0
    }
}

impl NerveComplex<So3<Coords<R64, 3>>, Coords<R64, 3>, So3Cover> for So3Cover {
    fn nodes() -> &'static [So3Cover] {
        use std::sync::LazyLock;
        static NODES: LazyLock<Vec<So3Cover>> = LazyLock::new(|| {
            let r = R64(0.8);
            let verts: [[R64; 3]; _] = [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, -1.0],
                [1.0, -1.0, 1.0],
                [-1.0, 1.0, 1.0],
            ]
            .map(|x| x.map(|x| R64(x)));
            verts
                .iter()
                .map(|&[x, y, z]| {
                    let v: Coords<R64, 3> = [x * r, y * r, z * r].into();
                    So3Cover(So3::identity_exp(v))
                })
                .collect()
        });
        &NODES
    }
}
