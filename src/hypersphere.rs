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
        // lexographic ordering on the fields
        match g
            .real()
            .partial_cmp(&V::F::zero())
            .unwrap()
            .then(g.imag().iter().partial_cmp(V::zero().iter()).unwrap())
        {
            std::cmp::Ordering::Less => So3(Sphere::new(-g.real(), -g.imag())),
            std::cmp::Ordering::Equal | std::cmp::Ordering::Greater => So3(g),
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
    // Each node's domain is the open arc of radius ρ = π/6 + 0.05 about its
    // base point. Six such arcs centred at the sixth roots of unity form an
    // open good cover of S¹:
    //   - covering:   arcs of half-length ρ > π/6 centred π/3 apart cover S¹
    //   - goodness:   arcs and their pairwise intersections are arcs (or
    //                 empty), hence contractible
    //   - nerve:      adjacent arcs (d = π/3 < 2ρ ≈ 1.147) overlap;
    //                 next-nearest (d = 2π/3 > 2ρ) do not — the nerve is a
    //                 hexagon, whose π₁ is free on one generator: π₁(S¹) = Z
    fn sdf(&self, v: &Coords<R64, 1>) -> R64 {
        v.norm() - R64(std::f64::consts::PI / 6.0 + 0.05)
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

impl NerveComplex<Sphere<1, Coords<R64, 1>>, Coords<R64, 1>, Sphere<1, Coords<R64, 1>>, S1Cover>
    for S1Cover
{
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

/// Radius of the geodesic-ball domains of [`So3Cover`].
///
/// The 60 nodes are the icosahedral rotation group I ≅ A₅ ⊂ SO(3) — the
/// image of the 120 icosian unit quaternions (the vertices of the 600-cell)
/// under the double cover S³ → SO(3). In the bi-invariant metric
/// `d = |identity_log|` (half the rotation angle; diameter π/2), the
/// pairwise distances realised between nodes are exactly
///
/// ```text
///   π/5 ≈ 0.628,   π/3 ≈ 1.047,   2π/5 ≈ 1.257,   π/2 ≈ 1.571
/// ```
///
/// and the covering radius of the node set is ≈ 0.3857 (the circumradius
/// of a cell of the 600-cell). The radius ρ = 0.42 is chosen so that:
///
/// - **covering**: ρ > 0.3857, so the 60 open balls cover SO(3);
/// - **goodness**: ρ < π/4, the convexity radius of SO(3) ≅ RP³, so every
///   ball is geodesically convex and all intersections of balls are convex,
///   hence contractible or empty — an open *good* cover;
/// - **faithful 1-skeleton**: two equal balls overlap iff their centres are
///   closer than 2ρ = 0.84, which separates π/5 from π/3 with a wide margin
///   on both sides — the nerve's edges are exactly the 600-cell's edges
///   (mod ±1), and the computation is robust to floating-point error;
/// - **faithful 2-skeleton**: every triangle of the overlap graph is an
///   equilateral triangle of side π/5 with spherical circumradius ≈ 0.365
///   < ρ, so all three balls genuinely share a point — mutual pairwise
///   overlap coincides with triple intersection, and the triangles of the
///   nerve are exactly the 600-cell's 2-faces (mod ±1).
///
/// The nerve of this cover is therefore the *hemi-600-cell*: the classical
/// vertex-transitive 60-vertex triangulation of RP³ with f-vector
/// (60, 360, 600, 300), obtained from the boundary complex of the 600-cell
/// by identifying antipodes. By the nerve theorem the nerve is homotopy
/// equivalent to SO(3), and π₁ computed from its 2-skeleton is
/// ⟨x | x²⟩ ≅ Z/2Z.
impl Bounded<So3<Coords<R64, 3>>, Coords<R64, 3>> for So3Cover {
    // Open geodesic ball of radius 0.42 about the base point.
    // In an exponential chart the geodesic distance from the base point is
    // exactly the coordinate norm, so the ball's true signed distance field
    // is radial.
    fn sdf(&self, v: &Coords<R64, 3>) -> R64 {
        v.norm() - R64(0.42)
    }

    fn new(p: So3<Coords<R64, 3>>) -> Self {
        Self(p)
    }

    fn inner(&self) -> &So3<Coords<R64, 3>> {
        &self.0
    }
}

impl NerveComplex<So3<Coords<R64, 3>>, Coords<R64, 3>, So3<Coords<R64, 3>>, So3Cover> for So3Cover {
    fn nodes() -> &'static [So3Cover] {
        use std::sync::LazyLock;
        static NODES: LazyLock<Vec<So3Cover>> = LazyLock::new(|| {
            // The 120 icosians: vertices of the 600-cell on S³.
            let phi = (1.0 + 5f64.sqrt()) / 2.0;
            let mut quats: Vec<[f64; 4]> = Vec::new();

            // 8 unit quaternions: ±1, ±i, ±j, ±k
            for i in 0..4 {
                for s in [-1.0, 1.0] {
                    let mut q = [0.0; 4];
                    q[i] = s;
                    quats.push(q);
                }
            }
            // 16: (±1 ± i ± j ± k)/2
            for a in [-0.5, 0.5] {
                for b in [-0.5, 0.5] {
                    for c in [-0.5, 0.5] {
                        for d in [-0.5, 0.5] {
                            quats.push([a, b, c, d]);
                        }
                    }
                }
            }
            // 96: all even permutations of (±φ, ±1, ±1/φ, 0)/2
            let even_perms: [[usize; 4]; 12] = [
                [0, 1, 2, 3],
                [0, 2, 3, 1],
                [0, 3, 1, 2],
                [1, 0, 3, 2],
                [1, 2, 0, 3],
                [1, 3, 2, 0],
                [2, 0, 1, 3],
                [2, 1, 3, 0],
                [2, 3, 0, 1],
                [3, 0, 2, 1],
                [3, 1, 0, 2],
                [3, 2, 1, 0],
            ];
            let base = [phi / 2.0, 0.5, 1.0 / (2.0 * phi), 0.0];
            for p in even_perms {
                for s0 in [-1.0, 1.0] {
                    for s1 in [-1.0, 1.0] {
                        for s2 in [-1.0, 1.0] {
                            let vals = [s0 * base[0], s1 * base[1], s2 * base[2], base[3]];
                            let mut q = [0.0; 4];
                            for i in 0..4 {
                                q[p[i]] = vals[i];
                            }
                            quats.push(q);
                        }
                    }
                }
            }
            debug_assert_eq!(quats.len(), 120);

            // Quotient by ±1: canonicalise the sign (first non-zero
            // coordinate positive) and deduplicate, leaving one
            // representative per rotation — 60 in total.
            let mut seen = std::collections::HashSet::new();
            let mut nodes = Vec::new();
            for mut q in quats {
                if let Some(c) = q.iter().find(|c| c.abs() > 1e-9)
                    && *c < 0.0
                {
                    q = q.map(|x| -x);
                }
                if seen.insert(q.map(|c| (c * 1e6).round() as i64)) {
                    let [w, x, y, z] = q.map(R64);
                    nodes.push(So3Cover(So3::new(Sphere::new(w, [x, y, z].into()))));
                }
            }
            debug_assert_eq!(nodes.len(), 60);
            nodes
        });
        &NODES
    }
}
