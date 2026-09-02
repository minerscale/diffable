//! Tolerance-aware floating-point real fields for tests and numerical geometry.
//!
//! [`R32`] and [`R64`] give ordinary floating-point arithmetic an approximate
//! equality suitable for the property-test macros, while
//! [`ExactCmp`](crate::traits::ExactCmp) remains available when an algorithm
//! needs the underlying strict order.

use core::ops::{Add, Div, Mul, Neg, Rem, Sub};
use num_derive::{FromPrimitive, Num, NumCast, One, Signed, ToPrimitive, Zero};
use num_traits::{Euclid, Float, Inv, One, Zero};

use crate::traits::NonZero;

// This newtype over floating points identifies point which are close to each other,
// This allows the library to pretend that all numbers it uses are actually real numbers.
// This is only approximate, but the tests pass, so it's pretty good.

/// Defines a tolerance-comparison scalar newtype over `$inner`, with a
/// `PartialEq` that treats values within `$epsilon` (relative for large
/// magnitudes, absolute for small ones) as equal -- see
/// [`RealNum`](crate::traits::RealNum) for why this exists and what it means
/// for the resulting equality to be
/// reflexive and symmetric but not transitive.
///
/// Takes a doc string as its first argument (spliced onto the generated
/// struct via `#[doc = $doc]`) because a macro cannot otherwise attach a
/// distinct `///` comment per invocation -- a `///` written inside the
/// macro body would be identical for every instantiation, and a `///`
/// written before the invocation itself is simply discarded.
///
macro_rules! define_epsilon_metric {
    ($name:ident, $inner:ty, $epsilon:expr, $doc:expr) => {
        #[doc = $doc]
        #[derive(
            Debug, Clone, Copy, Num, Signed, Zero, One, NumCast, ToPrimitive, FromPrimitive,
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
            fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
                if self == other {
                    // uses our epsilon PartialEq
                    Some(core::cmp::Ordering::Equal)
                } else {
                    self.0.partial_cmp(&other.0)
                }
            }
        }

        impl Euclid for $name {
            fn div_euclid(&self, v: &Self) -> Self {
                Self(<$inner as Euclid>::div_euclid(&self.0, &v.0))
            }

            fn rem_euclid(&self, v: &Self) -> Self {
                Self(<$inner as Euclid>::rem_euclid(&self.0, &v.0))
            }
        }

        impl Neg for $name {
            type Output = Self;

            fn neg(self) -> Self::Output {
                $name::zero() - self
            }
        }

        impl core::fmt::Display for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        impl Inv for NonZero<$name> {
            type Output = Self;

            fn inv(self) -> Self {
                Self::new_unchecked(self.0.recip())
            }
        }

        impl $name {
            #[inline(always)]
            unsafe fn assume_finite(self) -> Self {
                unsafe {
                    core::hint::assert_unchecked(self.is_finite());
                }

                self
            }
        }

        impl Add for $name {
            type Output = Self;

            #[inline]
            fn add(self, rhs: Self) -> Self {
                unsafe {
                    Self(self.assume_finite().0.algebraic_add(rhs.assume_finite().0))
                        .assume_finite()
                }
            }
        }

        impl Sub for $name {
            type Output = Self;

            #[inline]
            fn sub(self, rhs: Self) -> Self {
                unsafe {
                    Self(self.assume_finite().0.algebraic_sub(rhs.assume_finite().0))
                        .assume_finite()
                }
            }
        }

        impl Mul for $name {
            type Output = Self;

            #[inline]
            fn mul(self, rhs: Self) -> Self {
                unsafe {
                    Self(self.assume_finite().0.algebraic_mul(rhs.assume_finite().0))
                        .assume_finite()
                }
            }
        }

        impl Div for $name {
            type Output = Self;

            #[inline]
            fn div(self, rhs: Self) -> Self {
                unsafe {
                    Self(self.assume_finite().0.algebraic_div(rhs.assume_finite().0))
                        .assume_finite()
                }
            }
        }

        impl Rem for $name {
            type Output = Self;

            #[inline]
            fn rem(self, rhs: Self) -> Self {
                unsafe {
                    Self(self.assume_finite().0.algebraic_rem(rhs.assume_finite().0))
                        .assume_finite()
                }
            }
        }

        impl Float for $name {
            #[inline]
            fn nan() -> Self {
                Self(<$inner as Float>::nan())
            }

            #[inline]
            fn infinity() -> Self {
                Self(<$inner as Float>::infinity())
            }

            #[inline]
            fn neg_infinity() -> Self {
                Self(<$inner as Float>::neg_infinity())
            }

            #[inline]
            fn neg_zero() -> Self {
                Self(<$inner as Float>::neg_zero())
            }

            #[inline]
            fn min_value() -> Self {
                Self(<$inner as Float>::min_value())
            }

            #[inline]
            fn min_positive_value() -> Self {
                Self(<$inner as Float>::min_positive_value())
            }

            #[inline]
            fn max_value() -> Self {
                Self(<$inner as Float>::max_value())
            }

            #[inline]
            fn is_nan(self) -> bool {
                <$inner as Float>::is_nan(self.0)
            }

            #[inline]
            fn is_infinite(self) -> bool {
                <$inner as Float>::is_infinite(self.0)
            }

            #[inline]
            fn is_finite(self) -> bool {
                <$inner as Float>::is_finite(self.0)
            }

            #[inline]
            fn is_normal(self) -> bool {
                <$inner as Float>::is_normal(self.0)
            }

            #[inline]
            fn classify(self) -> core::num::FpCategory {
                <$inner as Float>::classify(self.0)
            }

            #[inline]
            fn floor(self) -> Self {
                Self(<$inner as Float>::floor(self.0))
            }

            #[inline]
            fn ceil(self) -> Self {
                Self(<$inner as Float>::ceil(self.0))
            }

            #[inline]
            fn round(self) -> Self {
                Self(<$inner as Float>::round(self.0))
            }

            #[inline]
            fn trunc(self) -> Self {
                Self(<$inner as Float>::trunc(self.0))
            }

            #[inline]
            fn fract(self) -> Self {
                Self(<$inner as Float>::fract(self.0))
            }

            #[inline]
            fn abs(self) -> Self {
                Self(<$inner as Float>::abs(self.0))
            }

            #[inline]
            fn signum(self) -> Self {
                Self(<$inner as Float>::signum(self.0))
            }

            #[inline]
            fn is_sign_positive(self) -> bool {
                <$inner as Float>::is_sign_positive(self.0)
            }

            #[inline]
            fn is_sign_negative(self) -> bool {
                <$inner as Float>::is_sign_negative(self.0)
            }

            #[inline]
            fn mul_add(self, a: Self, b: Self) -> Self {
                // Preserve Float::mul_add's fused, single-rounding semantics.
                Self(<$inner as Float>::mul_add(self.0, a.0, b.0))
            }

            #[inline]
            fn recip(self) -> Self {
                unsafe {
                    Self($name::one().0.algebraic_div(self.assume_finite().0)).assume_finite()
                }
            }

            #[inline]
            fn powi(self, n: i32) -> Self {
                Self(<$inner as Float>::powi(self.0, n))
            }

            #[inline]
            fn powf(self, n: Self) -> Self {
                Self(<$inner as Float>::powf(self.0, n.0))
            }

            #[inline]
            fn sqrt(self) -> Self {
                Self(<$inner as Float>::sqrt(self.0))
            }

            #[inline]
            fn exp(self) -> Self {
                Self(<$inner as Float>::exp(self.0))
            }

            #[inline]
            fn exp2(self) -> Self {
                Self(<$inner as Float>::exp2(self.0))
            }

            #[inline]
            fn ln(self) -> Self {
                Self(<$inner as Float>::ln(self.0))
            }

            #[inline]
            fn log(self, base: Self) -> Self {
                let numerator = <$inner as Float>::ln(self.0);
                let denominator = <$inner as Float>::ln(base.0);

                Self(numerator.algebraic_div(denominator))
            }

            #[inline]
            fn log2(self) -> Self {
                Self(<$inner as Float>::log2(self.0))
            }

            #[inline]
            fn log10(self) -> Self {
                Self(<$inner as Float>::log10(self.0))
            }

            #[inline]
            fn max(self, other: Self) -> Self {
                Self(<$inner as Float>::max(self.0, other.0))
            }

            #[inline]
            fn min(self, other: Self) -> Self {
                Self(<$inner as Float>::min(self.0, other.0))
            }

            #[inline]
            fn abs_sub(self, other: Self) -> Self {
                Self(<$inner as Float>::abs_sub(self.0, other.0))
            }

            #[inline]
            fn cbrt(self) -> Self {
                Self(<$inner as Float>::cbrt(self.0))
            }

            #[inline]
            fn hypot(self, other: Self) -> Self {
                Self(<$inner as Float>::hypot(self.0, other.0))
            }

            #[inline]
            fn sin(self) -> Self {
                Self(<$inner as Float>::sin(self.0))
            }

            #[inline]
            fn cos(self) -> Self {
                Self(<$inner as Float>::cos(self.0))
            }

            #[inline]
            fn tan(self) -> Self {
                Self(<$inner as Float>::tan(self.0))
            }

            #[inline]
            fn asin(self) -> Self {
                Self(<$inner as Float>::asin(self.0))
            }

            #[inline]
            fn acos(self) -> Self {
                Self(<$inner as Float>::acos(self.0))
            }

            #[inline]
            fn atan(self) -> Self {
                Self(<$inner as Float>::atan(self.0))
            }

            #[inline]
            fn atan2(self, other: Self) -> Self {
                Self(<$inner as Float>::atan2(self.0, other.0))
            }

            #[inline]
            fn sin_cos(self) -> (Self, Self) {
                let (sin, cos) = <$inner as Float>::sin_cos(self.0);

                (Self(sin), Self(cos))
            }

            #[inline]
            fn exp_m1(self) -> Self {
                Self(<$inner as Float>::exp_m1(self.0))
            }

            #[inline]
            fn ln_1p(self) -> Self {
                Self(<$inner as Float>::ln_1p(self.0))
            }

            #[inline]
            fn sinh(self) -> Self {
                Self(<$inner as Float>::sinh(self.0))
            }

            #[inline]
            fn cosh(self) -> Self {
                Self(<$inner as Float>::cosh(self.0))
            }

            #[inline]
            fn tanh(self) -> Self {
                Self(<$inner as Float>::tanh(self.0))
            }

            #[inline]
            fn asinh(self) -> Self {
                Self(<$inner as Float>::asinh(self.0))
            }

            #[inline]
            fn acosh(self) -> Self {
                Self(<$inner as Float>::acosh(self.0))
            }

            #[inline]
            fn atanh(self) -> Self {
                Self(<$inner as Float>::atanh(self.0))
            }

            #[inline]
            fn integer_decode(self) -> (u64, i16, i8) {
                <$inner as Float>::integer_decode(self.0)
            }

            #[inline]
            fn epsilon() -> Self {
                Self(<$inner as Float>::epsilon())
            }

            #[inline]
            fn to_degrees(self) -> Self {
                Self(<$inner as Float>::to_degrees(self.0))
            }

            #[inline]
            fn to_radians(self) -> Self {
                Self(<$inner as Float>::to_radians(self.0))
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
