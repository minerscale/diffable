use num_derive::{Float, FromPrimitive, Num, NumCast, NumOps, One, Signed, ToPrimitive, Zero};
use std::ops::Neg;

// This newtype over floating points identifies point which are close to each other,
// This allows the library to pretend that all numbers it uses are actually real numbers.
// This is only approximate, but the tests pass, so it's pretty good.

macro_rules! define_epsilon_metric {
    ($name:ident, $inner:ty, $epsilon:expr) => {
        #[derive(
            Debug,
            Clone,
            Copy,
            Float,
            Num,
            Signed,
            Zero,
            One,
            NumOps,
            NumCast,
            ToPrimitive,
            FromPrimitive,
        )]
        pub struct $name(pub $inner);

        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                let a = self.0;
                let b = other.0;
                // relative epsilon for large values, absolute epsilon for small values
                let diff = (a - b).abs();
                let magnitude = a.abs().max(b.abs());
                diff < $epsilon || diff < magnitude * $epsilon
            }
        }

        impl PartialOrd for $name {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                if self == other {
                    // uses our epsilon PartialEq
                    Some(std::cmp::Ordering::Equal)
                } else {
                    self.0.partial_cmp(&other.0)
                }
            }
        }

        impl Neg for $name {
            type Output = Self;

            fn neg(self) -> Self::Output {
                Self(-self.0)
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.0)
            }
        }
    };
}

define_epsilon_metric!(R64, f64, 1e-12);
define_epsilon_metric!(R32, f32, 1e-5);
