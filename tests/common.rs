#![allow(dead_code)]
#![cfg(feature = "testing")]

use diffable::{
    coords::Coords,
    epsilon_metric::R64,
    hypersphere::{So3, Sphere},
    traits::Quotient,
};

use num_traits::{One, real::Real};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------------
pub fn arb_sphere0() -> impl Strategy<Value = Sphere<0, Coords<R64, 0>>> {
    proptest::bool::ANY.prop_map(|positive| {
        if positive {
            Sphere::new(R64::one(), [].into())
        } else {
            Sphere::new(-R64::one(), [].into())
        }
    })
}

prop_compose! {
    pub fn arb_sphere1()(angle in -std::f64::consts::PI..std::f64::consts::PI)
        -> Sphere<1, Coords<R64, 1>>
    {
        let angle = R64(angle);
        Sphere::new(angle.cos(), [angle.sin()].into())
    }
}

prop_compose! {
    pub fn arb_sphere3()(
        w in -1.0f64..1.0f64,
        x in -1.0f64..1.0f64,
        y in -1.0f64..1.0f64,
        z in -1.0f64..1.0f64,
    ) -> Sphere<3, Coords<R64, 3>> {
        let w = if w.abs() + x.abs() + y.abs() + z.abs() < 1e-10 { 1.0 } else { w };
        Sphere::new(R64(w), [x, y, z].map(|x| R64(x)).into())
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
    pub fn arb_vec3()(
        x in -10.0f64..10.0f64,
        y in -10.0f64..10.0f64,
        z in -10.0f64..10.0f64,
    ) -> Coords<R64, 3> {
        [R64(x), R64(y), R64(z)].into()
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
