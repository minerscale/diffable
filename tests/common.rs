#![allow(dead_code)]
#![cfg(feature = "testing")]

use diffable::{
    coords::Coords,
    epsilon_metric::R64,
    hypersphere::{S0, S1, S3, So3, Sphere},
    torus::{S1Quotient, Z},
    traits::Quotient,
};

use num_traits::{One, real::Real};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------------
pub fn arb_sphere0() -> impl Strategy<Value = S0<Coords<R64, 0>>> {
    proptest::bool::ANY.prop_map(|positive| {
        if positive {
            S0(Sphere::new(R64::one(), [].into()))
        } else {
            S0(Sphere::new(-R64::one(), [].into()))
        }
    })
}

prop_compose! {
    pub fn arb_sphere1()(angle in -std::f64::consts::PI..std::f64::consts::PI)
        -> S1<Coords<R64, 1>>
    {
        let angle = R64(angle);
        S1(Sphere::new(angle.cos(), [angle.sin()].into()))
    }
}

prop_compose! {
    pub fn arb_sphere3()(
        w in -1.0f64..1.0f64,
        x in -1.0f64..1.0f64,
        y in -1.0f64..1.0f64,
        z in -1.0f64..1.0f64,
    ) -> S3<Coords<R64, 3>> {
        let w = if w.abs() + x.abs() + y.abs() + z.abs() < 1e-10 { 1.0 } else { w };
        S3(Sphere::new(R64(w), [x, y, z].map(|x| R64(x)).into()))
    }
}

prop_compose! {
    pub fn arb_sphere2()(
        w in -1.0f64..1.0f64,
        x in -1.0f64..1.0f64,
        y in -1.0f64..1.0f64,
    ) -> Sphere<2, Coords<R64, 2>> {
        let w = if w.abs() + x.abs() + y.abs() < 1e-10 { 1.0 } else { w };
        Sphere::new(R64(w), [x, y].map(R64).into())
    }
}

prop_compose! {
    pub fn arb_sphere4()(
        w in -1.0f64..1.0f64,
        x in -1.0f64..1.0f64,
        y in -1.0f64..1.0f64,
        z in -1.0f64..1.0f64,
        u in -1.0f64..1.0f64,
    ) -> Sphere<4, Coords<R64, 4>> {
        let w = if w.abs() + x.abs() + y.abs() + z.abs() + u.abs() < 1e-10 { 1.0 } else { w };
        Sphere::new(R64(w), [x, y, z, u].map(R64).into())
    }
}

pub fn arb_vec<const N: usize>() -> impl Strategy<Value = Coords<R64, N>> {
    proptest::array::uniform(-10.0f64..10.0f64).prop_map(|arr| Coords::from(arr.map(|x| R64(x))))
}

pub fn arb_vec0() -> impl Strategy<Value = Coords<R64, 0>> {
    Just([].into())
}

prop_compose! {
    pub fn arb_vec1()(v in -10.0f64..10.0f64) -> Coords<R64, 1> {
        [R64(v)].into()
    }
}

prop_compose! {
    pub fn arb_vec2()(
        x in -10.0f64..10.0f64,
        y in -10.0f64..10.0f64,
    ) -> Coords<R64, 2> {
        [R64(x), R64(y)].into()
    }
}

prop_compose! {
    pub fn arb_vec3()(
        x in -10.0f64..10.0f64,
        y in -10.0f64..10.0f64,
        z in -10.0f64..10.0f64,
    ) -> Coords<R64, 3> {
        [R64(x), R64(y), R64(z)].into()
    }
}

prop_compose! {
    pub fn arb_vec4()(
        x in -10.0f64..10.0f64,
        y in -10.0f64..10.0f64,
        z in -10.0f64..10.0f64,
        w in -10.0f64..10.0f64,
    ) -> Coords<R64, 4> {
        [R64(x), R64(y), R64(z), R64(w)].into()
    }
}

pub fn arb_so3() -> impl Strategy<Value = So3<Coords<R64, 3>>> {
    arb_sphere3().prop_map(|g| So3::new(g))
}

prop_compose! {
    pub fn arb_scalar()(
        x in -10.0f64..10.0f64,
    ) -> R64 {
        R64(x)
    }
}

prop_compose! {
    pub fn arb_z()(
        z in -100isize..100isize,
    ) -> Z<Coords<R64, 1>> {
        Z::new(z)
    }
}

pub fn arb_s1_quotient() -> impl Strategy<Value = S1Quotient<Coords<R64, 1>>> {
    arb_scalar().prop_map(|x| S1Quotient::new([x].into()))
}
