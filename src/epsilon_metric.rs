use num_derive::{Float, FromPrimitive, Num, NumCast, NumOps, One, Signed, ToPrimitive, Zero};
use num_traits::Euclid;
use std::ops::Neg;

// This newtype over floating points identifies point which are close to each other,
// This allows the library to pretend that all numbers it uses are actually real numbers.
// This is only approximate, but the tests pass, so it's pretty good.

/// Defines a tolerance-comparison scalar newtype over `$inner`, with a
/// `PartialEq` that treats values within `$epsilon` (relative for large
/// magnitudes, absolute for small ones) as equal -- see [`Scalar`] for why
/// this exists and what it means for the resulting equality to be
/// reflexive and symmetric but not transitive.
///
/// Takes a doc string as its first argument (spliced onto the generated
/// struct via `#[doc = $doc]`) because a macro cannot otherwise attach a
/// distinct `///` comment per invocation -- a `///` written inside the
/// macro body would be identical for every instantiation, and a `///`
/// written before the invocation itself is simply discarded.
///
/// [`Scalar`]: crate::traits::Scalar
macro_rules! define_epsilon_metric {
    ($name:ident, $inner:ty, $epsilon:expr, $doc:expr) => {
        #[doc = $doc]
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

        impl Euclid for $name {
            fn div_euclid(&self, v: &Self) -> Self {
                Self(self.0.div_euclid(v.0))
            }

            fn rem_euclid(&self, v: &Self) -> Self {
                Self(self.0.rem_euclid(v.0))
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

define_epsilon_metric!(
    R64,
    f64,
    1e-12,
    "A tolerance-comparison `f64`, treating values within `1e-12` as equal."
);
define_epsilon_metric!(
    R32,
    f32,
    1e-5,
    "A tolerance-comparison `f32`, treating values within `1e-5` as equal."
);
