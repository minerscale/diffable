#![allow(dead_code)]
#![cfg(feature = "testing")]

pub const EPSILON: f64 = 1e-7;

use diffable::{
    coords::Coords,
    hypersphere::{So3, Sphere},
    traits::Quotient,
};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------------
pub fn arb_sphere0() -> impl Strategy<Value = Sphere<0, Coords<f64, 0>>> {
    proptest::bool::ANY.prop_map(|positive| {
        if positive {
            Sphere::new(1.0, [].into())
        } else {
            Sphere::new(-1.0, [].into())
        }
    })
}

prop_compose! {
    pub fn arb_sphere1()(angle in -std::f64::consts::PI..std::f64::consts::PI)
        -> Sphere<1, Coords<f64, 1>>
    {
        Sphere::new(angle.cos(), [angle.sin()].into())
    }
}

prop_compose! {
    pub fn arb_sphere3()(
        w in -1.0f64..1.0f64,
        x in -1.0f64..1.0f64,
        y in -1.0f64..1.0f64,
        z in -1.0f64..1.0f64,
    ) -> Sphere<3, Coords<f64, 3>> {
        let w = if w.abs() + x.abs() + y.abs() + z.abs() < EPSILON { 1.0 } else { w };
        Sphere::new(w, [x, y, z].into())
    }
}

pub fn arb_vec<const N: usize>() -> impl Strategy<Value = Coords<f64, N>> {
    proptest::array::uniform(-10.0f64..10.0f64).prop_map(|arr| Coords::from(arr))
}

pub fn arb_vec0() -> impl Strategy<Value = Coords<f64, 0>> {
    Just([].into())
}

prop_compose! {
    pub fn arb_vec1()(v in -10.0f64..10.0f64) -> Coords<f64, 1> {
        [v].into()
    }
}

prop_compose! {
    pub fn arb_vec3()(
        x in -10.0f64..10.0f64,
        y in -10.0f64..10.0f64,
        z in -10.0f64..10.0f64,
    ) -> Coords<f64, 3> {
        [x, y, z].into()
    }
}

pub fn arb_so3() -> impl Strategy<Value = So3<Coords<f64, 3>>> {
    arb_sphere3().prop_map(|g| So3::new(g))
}

pub fn arb_scalar() -> impl Strategy<Value = f64> {
    -10.0f64..10.0f64
}
