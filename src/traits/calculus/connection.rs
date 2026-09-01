use core::marker::PhantomData;

use num_traits::{NumCast, One, Zero, real::Real as _};

use crate::{
    matrix::endomorphism_exp,
    traits::{
        Absent, Array, Atomic, BothSided, Cat, Category, Chart, Dual, ExactCmp, ExpMap, Field,
        Form, FromReal, Interval, Left, Nondegenerate, OptionallyOption, Point, Right, Sinister,
        TangentBundle, Tensor, Vector,
        calculus::{
            Contract, Here, Jet, JetVector, JetVectorIn, OnLeft, OnRight, Reassociate,
            ReassociateKernel, TensorOver, TensorProduct, ThroughSinister, d,
            evaluate_derivative_at,
        },
        Ø, ː, ι, Ⱶ, 𝐃𝐢𝐟𝐟, 𝐅𝐥𝐝, 𝐅𝐨𝐫𝐦, 𝐌𝐞𝐭, 𝐑𝐞𝐚𝐥, 𝐓𝐞𝐧𝐬,
    },
};

pub type Christoffel<V> = TensorProduct<TensorProduct<V, Dual<V>>, Dual<V>>;

/// Extends a tangent-bundle chart to jet-valued tangent coordinates.
///
/// Implementing this trait is the admission point for differentiating through
/// a manifold. It states how a tangent element is expressed in the local chart
/// at another tangent element, and how that local jet is returned to the global
/// bundle. [`Tangent`] is the first lifted element; [`TM`] and [`LiftedTM`]
/// describe its iterated tangent bundles. Vector spaces receive the canonical
/// translation-based implementation.
pub trait Connection<P: Point, V: Tensor>: TangentBundle<P, V> {
    /// Expresses `local` in the lifted chart centred at `base`.
    fn tangent_to_local<const N: usize>(
        base: Tangent<P, V, N>,
        local: Tangent<P, V, N>,
    ) -> Option<JetVector<V, N>>;
    /// Reconstructs a global point and tangent jet from a lifted coordinate.
    fn tangent_to_global<const N: usize>(
        base: Tangent<P, V, N>,
        coordinate: JetVector<V, N>,
    ) -> <Self::Global as OptionallyOption<P>>::Mapped<(P, JetVector<V, N>)>;

    fn christoffel_symbols(&self, p: P) -> Option<Christoffel<V>>
    where
        V: Vector<Hand = Right, Action = BothSided>,
    {
        let origin = TangentElement::new(
            LiftedTM::<P, V, Self, 1>::new(p, Zero::zero()),
            Zero::zero(),
        );
        let observer = TangentElement::new(
            LiftedTM::<P, V, Self, 1>::new(self.base_point(), Zero::zero()),
            Zero::zero(),
        );

        let success = core::cell::Cell::new(true);

        let transition = |v: JetVector<V, 1, Jet<𝐅𝐥𝐝::𝒞, <V as Tensor>::F, 1>>| -> _ {
            match Prolongation::<P, V, Self, 1>::tangent_to_global::<1>(
                origin.clone(),
                TensorOver(v.0, PhantomData),
            )
            .into_option()
            {
                Some((point, tangent)) => {
                    match Prolongation::<P, V, Self, 1>::tangent_to_local::<1>(
                        observer.clone(),
                        TangentElement::new(point, tangent),
                    ) {
                        Some(local) => TensorOver(local.0, PhantomData),

                        None => {
                            success.set(false);
                            Zero::zero()
                        }
                    }
                }
                None => {
                    success.set(false);
                    Zero::zero()
                }
            }
        };

        let christoffel = -evaluate_derivative_at(&d(d(transition)), V::zero()).0;

        success.get().then_some(christoffel)
    }

    /// Returns the coordinate acceleration at `p` of the geodesic with
    /// initial tangent `v`, expressed in the fixed chart `self`.
    ///
    /// If
    /// ```text
    /// γ_v(t) = exp_p(t v)
    /// ```
    ///
    /// and `x_v(t)` is that geodesic expressed in the coordinates of
    /// `self`, this returns
    ///
    /// ```text
    ///     x_v''(0).
    /// ```
    ///
    /// The observing chart must contain `p`.
    #[cfg(feature = "testing")]
    fn geodesic_acceleration(&self, p: P, v: V) -> Option<V>
    where
        V: Vector,
    {
        // Observe everything in the fixed chart `self`.
        //
        // By the ExpMap law,
        //
        // Self::chart_at(&self.base_point()) == self,
        //
        // so the zero tangent based at `self.base_point()` selects exactly
        // this lifted chart.
        let observer =
            Tangent::<P, V, 2>::new(self.base_point(), JetVectorIn::<𝐅𝐥𝐝::𝒞, V, 2>::zero());

        // The exponential chart centred at p.
        let origin = Tangent::<P, V, 2>::new(p, JetVectorIn::<𝐅𝐥𝐝::𝒞, V, 2>::zero());

        // The 2-jet of t ↦ t v:
        //
        //     0 + v t + 0 t².
        //
        // Pushing this through the lifted exponential chart therefore
        // constructs the 2-jet of γ_v(t) = exp_p(t v).
        let radial = JetVectorIn::<𝐅𝐥𝐝::𝒞, V, 2>::from_fn(|i| {
            Jet::from_parts(V::F::zero(), [v[i], V::F::zero()])
        });

        let (point, tangent) = Self::tangent_to_global::<2>(origin, radial).into_option()?;

        let geodesic = Tangent::<P, V, 2>::new(point, tangent);

        // Re-express the geodesic in ONE FIXED chart. Using the chart
        // centred at p here would make its coordinate acceleration vanish
        // tautologically.
        let local = Self::tangent_to_local::<2>(observer, geodesic)?;

        // Jets store Taylor coefficients:
        //
        //     jet[2] = x''(0) / 2!
        //
        // so recover the actual acceleration.
        let two = V::F::from_nat(2);

        Some(V::from_fn(|i| local[i][2] * two))
    }

    /// Certifies that the geodesic spray is quadratic in tangent velocity.
    ///
    /// For a fixed chart and point p, define
    ///
    /// ```text
    /// A_p(v) = d²/dt² |₀ chart(exp_p(t v)).
    /// ```
    ///
    /// `Connection` requires:
    ///
    /// ```text
    /// A_p(u + v) + A_p(u - v) = 2 A_p(u) + 2 A_p(v)
    /// ```
    ///
    /// and
    ///
    /// ```text
    ///     A_p(a v) = a² A_p(v).
    /// ```
    ///
    /// Thus A_p is quadratic. Polarization therefore determines a unique
    /// symmetric bilinear Christoffel operation
    ///
    /// ```text
    ///     Γ_p(u, v)
    ///       = -½ (A_p(u + v) - A_p(u) - A_p(v)),
    /// ```
    ///
    /// so the lifted geodesic structure determines a torsion-free affine
    /// connection.
    #[cfg(feature = "testing")]
    fn check_quadratic_geodesic_acceleration(&self, p: P, u: V, v: V, a: V::F) -> bool
    where
        Self: Sized,
        V: Vector + PartialEq,
    {
        // The assertion is local to the observing chart. If p is not in
        // this chart, there is no coordinate acceleration here to test.
        if self.to_local(&p).is_none() {
            return true;
        }

        let Some(a_u) = self.geodesic_acceleration(p.clone(), u.clone()) else {
            return false;
        };

        let Some(a_v) = self.geodesic_acceleration(p.clone(), v.clone()) else {
            return false;
        };

        let Some(a_u_plus_v) = self.geodesic_acceleration(p.clone(), u.clone() + v.clone()) else {
            return false;
        };

        let Some(a_u_minus_v) = self.geodesic_acceleration(p.clone(), u - v.clone()) else {
            return false;
        };

        let two = V::F::from_nat(2);

        // Quadratic parallelogram identity.
        if a_u_plus_v + a_u_minus_v != (a_u + a_v.clone()) * two {
            return false;
        }

        let Some(a_av) = self.geodesic_acceleration(p, v * a) else {
            return false;
        };

        // Degree-two homogeneity.
        a_av == a_v * (a * a)
    }

    #[cfg(feature = "testing")]
    fn check_tangent_to_local_agrees_with_chart(base: P, point: P) -> bool
    where
        V: PartialEq,
        JetVector<V>: PartialEq,
    {
        let chart = Self::chart_at(&base);

        let Some(local) = chart.to_local(&point) else {
            return true;
        };

        let lifted_base = Tangent::new(base, JetVectorIn::<𝐅𝐥𝐝::𝒞, V>::zero());

        let lifted_point = Tangent::new(point, JetVectorIn::<𝐅𝐥𝐝::𝒞, V>::zero());

        let expected = constant_jet_vector(local);

        Self::tangent_to_local(lifted_base, lifted_point).is_some_and(|actual| actual == expected)
    }

    #[cfg(feature = "testing")]
    fn check_tangent_to_global_agrees_with_chart(base: P, local: V) -> bool
    where
        V: PartialEq,
        JetVector<V>: PartialEq,
    {
        use crate::traits::OptionallyOption;

        let chart = Self::chart_at(&base);

        let expected = match chart.to_global(local.clone()).into_option() {
            Some(point) => point,
            None => return true,
        };

        let lifted_base = Tangent::new(base, JetVectorIn::<𝐅𝐥𝐝::𝒞, V>::zero());

        let coordinate = constant_jet_vector(local);

        let (actual, tangent) = match Self::tangent_to_global(lifted_base, coordinate).into_option()
        {
            Some(x) => x,
            None => return true,
        };

        tangent == JetVectorIn::zero() && chart.to_local(&actual) == chart.to_local(&expected)
    }

    /// Certifies that the lifted tangent charts form a coherent tower under
    /// truncation.
    ///
    /// Given `M <= N`, truncating an order-`N` tangent coordinate before
    /// applying `tangent_to_global` must agree exactly with applying the
    /// order-`N` map first and then truncating its jet component.
    ///
    /// Coherence of `tangent_to_local` follows from `check_tangent_isomorphism`.
    #[cfg(feature = "testing")]
    fn check_truncation_coherence<const M: usize, const N: usize>(
        base: Tangent<P, V, N>,
        coordinate: JetVector<V, N>,
    ) -> bool
    where
        P: PartialEq,
        JetVector<V, M>: PartialEq,
    {
        const { assert!(M <= N) };

        let (point_n, tangent_n) =
            match Self::tangent_to_global::<N>(base.clone(), coordinate.clone()).into_option() {
                Some(x) => x,
                None => return true,
            };

        let base_m = Tangent::new(base.0, base.1.truncate::<M>());

        let coordinate_m = coordinate.truncate::<M>();

        let (point_m, tangent_m) =
            match Self::tangent_to_global::<M>(base_m, coordinate_m).into_option() {
                Some(x) => x,
                None => return true,
            };

        point_n == point_m && tangent_n.truncate::<M>() == tangent_m
    }

    /// Certifies that the lifted local and global tangent charts are mutual
    /// inverses at jet order `N`.
    ///
    /// On the domain of `tangent_to_local`,
    ///
    /// ```text
    /// tangent_to_global(base, tangent_to_local(base, point))
    ///     == point,
    /// ```
    ///
    /// while every lifted local coordinate must round-trip as
    ///
    /// ```text
    /// tangent_to_local(base, tangent_to_global(base, coordinate))
    ///     == Some(coordinate).
    /// ```
    ///
    /// Thus `tangent_to_local::<N>` and `tangent_to_global::<N>` describe a
    /// genuine lifted chart isomorphism rather than independent choices of
    /// higher-order tangent data.
    #[cfg(feature = "testing")]
    fn check_tangent_isomorphism<const N: usize>(
        base: Tangent<P, V, N>,
        local: Tangent<P, V, N>,
        coordinate: JetVector<V, N>,
    ) -> bool
    where
        V: PartialEq,
        JetVector<V, N>: PartialEq,
    {
        // First check:
        //
        //     local -> coordinate -> global == local
        //
        // `tangent_to_local` is partial, so points outside this lifted chart
        // impose no inversehood obligation.
        if let Some(local_coordinate) = Self::tangent_to_local::<N>(base.clone(), local.clone()) {
            let (point, tangent) =
                match Self::tangent_to_global::<N>(base.clone(), local_coordinate).into_option() {
                    Some(x) => x,
                    None => return true,
                };

            // Since `P` need not implement PartialEq, compare the reconstructed
            // point through the ordinary chart in which `local` was expressible.
            let chart = Self::chart_at(&base.0);

            if tangent != local.1 || chart.to_local(&point) != chart.to_local(&local.0) {
                return false;
            }
        }

        // Then check:
        //
        //     coordinate -> global -> local == coordinate
        //
        // This obligation applies when coefficient zero belongs to the branch
        // selected by the ordinary chart. `tangent_to_global` may be total even
        // when the exponential chart is not globally injective, so arbitrary
        // vectors need not be canonical local coordinates of their image.
        let coordinate_primal = V::from_fn(|index| coordinate[index][0]);

        let (point, tangent) =
            match Self::tangent_to_global::<N>(base.clone(), coordinate.clone()).into_option() {
                Some(value) => value,
                None => return true,
            };

        let chart = Self::chart_at(&base.0);

        // The generated coordinate lies outside the branch represented by this
        // chart. There is no inversehood obligation for this presentation.
        if chart.to_local(&point) != Some(coordinate_primal) {
            return true;
        }

        let reconstructed = Tangent::new(point, tangent);

        Self::tangent_to_local::<N>(base, reconstructed).is_some_and(|actual| actual == coordinate)
    }
}

/// A connection with an explicitly supplied metric tensor field.
///
/// `MetricTensor` refines [`Connection`]: it is an implementation strategy for a
/// differential structure in which the metric itself is available pointwise,
/// rather than reconstructed from parallel transport of the model-space form.
///
/// For a right-handed tangent space `V`, the covariant rank-two metric is
/// represented as `Sinister<V*> ⊗ V*`. The musical maps are consequences of
/// this tensor: lowering contracts `g_p` with a vector, while raising contracts
/// the inverse tensor with a covector.
pub trait MetricTensor<P: Point, V: Tensor<Hand = Right, Action = BothSided>>:
    Connection<P, V>
{
    /// Evaluate the supplied metric tensor in the tangent space selected by
    /// `target`, expressed in this connection's local tangent coordinates.
    fn g(&self, target: V) -> TensorProduct<Sinister<Dual<V>>, Dual<V>>;
}

/// A point together with a jet-valued tangent coordinate and its tower tag.
///
/// `Tower` distinguishes iterated tangent constructions that have identical
/// runtime representations. Use [`TangentElement::new`] rather than spelling
/// the marker explicitly.
#[derive(Debug, Clone)]
pub struct TangentElement<P: Point, V: Tensor, Tower, const N: usize = 1>(
    pub P,
    pub JetVector<V, N>,
    PhantomData<Tower>,
);

impl<P: Point + PartialEq, V: Tensor, Tower, const N: usize> PartialEq
    for TangentElement<P, V, Tower, N>
{
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}

impl<P: Point, V: Tensor, Tower, const N: usize> TangentElement<P, V, Tower, N> {
    /// Constructs a tangent element from its base point and local jet.
    pub fn new(p: P, v: JetVector<V, N>) -> Self {
        Self(p, v, PhantomData)
    }

    /// Returns a clone of the base point.
    pub fn base_point(&self) -> P {
        self.0.clone()
    }

    /// Borrows the jet-valued tangent coordinate.
    pub fn jet(&self) -> &JetVector<V, N> {
        &self.1
    }
}

impl<P: Point, V: Tensor, const N: usize> Tangent<P, V, N> {
    pub fn into_jet(self, point_to_coordinate: impl FnOnce(P) -> V) -> JetVector<V, N> {
        let coordinate = point_to_coordinate(self.0);

        JetVector::from_fn(|i| {
            let mut jet = self.1[i];
            jet[0] = coordinate[i];
            jet
        })
    }
}

type Prolongation<P, V, T, const N: usize = 1> = TangentElement<P, V, ː<T, Ø>, N>;

/// A first [`TangentElement`] at a point of `P`, expressed in `V` coordinates.
pub type Tangent<P, V, const N: usize = 1> = TangentElement<P, V, Ø, N>;
/// An iterated [`TangentElement`] with explicit [`TangentBundle`] witnesses.
pub type TM<P, V, T, U, const N: usize = 1> = TangentElement<P, V, ː<T, ː<U, Ø>>, N>;
/// The tangent bundle of `T`, represented by the canonical jet prolongation.
///
/// This is the concrete iterated-tangent representation constructed by
/// [`Connection`].
pub type LiftedTM<P, V, T, const N: usize = 1> = TM<P, V, T, Prolongation<P, V, T, N>, N>;

impl<
    P: Point,
    V: Tensor,
    T: TangentBundle<P, V>,
    U: TangentBundle<Self, JetVector<V, N>>,
    const N: usize,
> Chart<P, V> for TM<P, V, T, U, N>
{
    type Global = T::Global;

    fn to_local(&self, point: &P) -> Option<V> {
        T::chart_at(&self.0).to_local(point)
    }

    fn to_global(&self, coord: V) -> Self::Global {
        T::chart_at(&self.0).to_global(coord)
    }

    fn chart_at(p: &P) -> Self {
        Self(p.clone(), JetVectorIn::zero(), PhantomData)
    }
}

impl<
    P: Point,
    V: Tensor,
    T: TangentBundle<P, V>,
    U: TangentBundle<Self, JetVector<V, N>>,
    const N: usize,
> ExpMap<P, V> for TM<P, V, T, U, N>
{
}
impl<
    P: Point,
    V: Tensor,
    T: TangentBundle<P, V>,
    U: TangentBundle<Self, JetVector<V, N>>,
    const N: usize,
> TangentBundle<P, V> for TM<P, V, T, U, N>
{
}

impl<
    P: Point,
    V: Tensor,
    T: TangentBundle<P, V>,
    U: TangentBundle<Self, JetVector<V, N>>,
    const N: usize,
> Chart<Self, JetVector<V, N>> for TM<P, V, T, U, N>
{
    type Global = U::Global;

    fn to_local(&self, point: &Self) -> Option<JetVector<V, N>> {
        U::chart_at(self).to_local(point)
    }

    fn to_global(&self, coord: JetVector<V, N>) -> Self::Global {
        U::chart_at(self).to_global(coord)
    }

    fn chart_at(p: &Self) -> Self {
        p.clone()
    }
}
impl<
    P: Point,
    V: Tensor,
    T: TangentBundle<P, V>,
    U: TangentBundle<Self, JetVector<V, N>>,
    const N: usize,
> ExpMap<Self, JetVector<V, N>> for TM<P, V, T, U, N>
{
}

impl<
    P: Point,
    V: Tensor,
    T: TangentBundle<P, V>,
    U: TangentBundle<Self, JetVector<V, N>>,
    const N: usize,
> TangentBundle<Self, JetVector<V, N>> for TM<P, V, T, U, N>
{
}

impl<P: Point, V: Tensor, T: Connection<P, V>, U: TangentBundle<Self, JetVector<V>>>
    Connection<P, V> for TM<P, V, T, U>
{
    fn tangent_to_local<const M: usize>(
        base: Tangent<P, V, M>,
        local: Tangent<P, V, M>,
    ) -> Option<JetVector<V, M>> {
        T::tangent_to_local(base, local)
    }

    fn tangent_to_global<const N: usize>(
        base: Tangent<P, V, N>,
        coordinate: JetVector<V, N>,
    ) -> <Self::Global as OptionallyOption<P>>::Mapped<(P, JetVector<V, N>)> {
        T::tangent_to_global(base, coordinate)
    }
}

impl<P: Point, V: Tensor, T: Connection<P, V>, const N: usize>
    Connection<LiftedTM<P, V, T, N>, JetVector<V, N>> for Prolongation<P, V, T, N>
{
    fn tangent_to_local<const M: usize>(
        base: Tangent<LiftedTM<P, V, T, N>, JetVector<V, N>, M>,
        local: Tangent<LiftedTM<P, V, T, N>, JetVector<V, N>, M>,
    ) -> Option<JetVector<JetVector<V, N>, M>> {
        let point = T::tangent_to_local::<N>(
            TangentElement::new(base.0.0.clone(), base.0.1.clone()),
            TangentElement::new(local.0.0.clone(), local.0.1.clone()),
        )?;

        Some(JetVectorIn::from_fn(|i| {
            local.1[i] - base.1[i] + Jet::from_parts(point[i], [Jet::<𝐅𝐥𝐝::𝒞, V::F, N>::zero(); M])
        }))
    }

    fn tangent_to_global<const M: usize>(
        base: Tangent<LiftedTM<P, V, T, N>, JetVector<V, N>, M>,
        coordinate: JetVector<JetVector<V, N>, M>,
    ) -> <Self::Global as OptionallyOption<LiftedTM<P, V, T, N>>>::Mapped<(
        LiftedTM<P, V, T, N>,
        JetVector<JetVector<V, N>, M>,
    )> {
        let combined =
            JetVectorIn::<𝐅𝐥𝐝::𝒞, JetVector<V, N>, M>::from_fn(
                |i| base.1[i] + coordinate[i],
            );

        // The outer constant coefficient is an ordinary coordinate in the
        // Prolongation chart on LiftedTM.
        let point_coordinate = JetVectorIn::<𝐅𝐥𝐝::𝒞, V, N>::from_fn(|i| combined[i][0]);

        T::tangent_to_global::<N>(
            TangentElement::new(base.0.0.clone(), base.0.1.clone()),
            point_coordinate,
        )
        .cast_option(|(point, jet)| {
            let point = TangentElement::new(point, jet);

            // Everything above outer order zero is the tangent part.
            let tangent = JetVectorIn::from_fn(|i| {
                let mut value = combined[i];
                value[0] = Jet::<𝐅𝐥𝐝::𝒞, V::F, N>::zero();
                value
            });

            (point, tangent)
        })
    }
}

fn tangent_lerp<
    P: Point,
    V: Vector<Hand = Right, Action = BothSided, F: FromReal>,
    T: Connection<P, V>,
    const N: usize,
>(
    connection: &T,
    target: V,
    t: Jet<𝐑𝐞𝐚𝐥::𝒞, <V::F as Interval>::R, N>,
) -> Tangent<P, V, N> {
    let t = Jet::<𝐅𝐥𝐝::𝒞, V::F, N>::from_parts(
        V::F::from_real(t[0]),
        core::array::from_fn(|i| V::F::from_real(t[i + 1])),
    );

    let radial = JetVectorIn::<𝐅𝐥𝐝::𝒞, V, N>::from_fn(|i| {
        Jet::<𝐅𝐥𝐝::𝒞, V::F, N>::from_parts(target[i], [Zero::zero(); N]) * t
    });

    let base = Tangent::<P, V, N>::new(connection.base_point(), JetVectorIn::zero());

    let (point, tangent) = T::tangent_to_global::<N>(base, radial)
        .into_option()
        .unwrap();

    TangentElement::new(point, tangent)
}

pub const TRANSPORT_ORDER: usize = 6;

pub struct Ordered<
    'a,
    𝒞: Cat,
    𝒟: Cat,
    P: Point,
    V: Vector<Hand = Right, Action = BothSided, F: FromReal> + ι<C: TransportRegion<𝒞>>,
    T: ParallelTransport<𝒞, 𝒟, P, V>,
    const N: usize,
> {
    connection: &'a T,
    _phantom: PhantomData<fn() -> (𝒞, 𝒟, P, V)>,
}

impl<
    'a,
    𝒞: Cat,
    𝒟: Cat,
    P: Point,
    V: Vector<Hand = Right, Action = BothSided, F: FromReal> + ι<C: TransportRegion<𝒞>>,
    T: ParallelTransport<𝒞, 𝒟, P, V>,
    const N: usize,
> Ordered<'a, 𝒞, 𝒟, P, V, T, N>
{
    pub fn transport(
        &self,
        curve: impl Fn(Jet<𝐑𝐞𝐚𝐥::𝒞, <V::F as Interval>::R, N>) -> Tangent<P, V, N>,
        from: <V::F as Interval>::R,
        to: <V::F as Interval>::R,
    ) -> TensorProduct<V, Dual<V>> {
        self.connection.transport_with::<N>(curve, from, to)
    }

    pub fn lower(&self, target: V, v: V) -> Dual<V>
    where
        V: Form,
        <T as ι>::C: MusicalRegion<𝒟, 𝒞, P, V, T>,
    {
        self.connection.lower_with::<N>(target, v)
    }

    pub fn raise(&self, target: V, v: Dual<V>) -> V
    where
        V: Nondegenerate,
        <T as ι>::C: MusicalRegion<𝒟, 𝒞, P, V, T>,
    {
        self.connection.raise_with::<N>(target, v)
    }
}

pub trait ParallelTransport<
    𝒞: Cat,
    𝒟: Cat,
    P: Point,
    V: Vector<Hand = Right, Action = BothSided, F: FromReal> + ι<C: TransportRegion<𝒞>>,
>: Connection<P, V> + ι
{
    fn transport_with<const N: usize>(
        &self,
        curve: impl Fn(Jet<𝐑𝐞𝐚𝐥::𝒞, <V::F as Interval>::R, N>) -> Tangent<P, V, N>,
        from: <V::F as Interval>::R,
        to: <V::F as Interval>::R,
    ) -> TensorProduct<V, Dual<V>>;

    fn order<'a, const N: usize>(&'a self) -> Ordered<'a, 𝒞, 𝒟, P, V, Self, N> {
        Ordered {
            connection: self,
            _phantom: PhantomData,
        }
    }

    fn transport(
        &self,
        curve: impl Fn(
            Jet<𝐑𝐞𝐚𝐥::𝒞, <V::F as Interval>::R, TRANSPORT_ORDER>,
        ) -> Tangent<P, V, TRANSPORT_ORDER>,
        from: <V::F as Interval>::R,
        to: <V::F as Interval>::R,
    ) -> TensorProduct<V, Dual<V>> {
        self.transport_with::<TRANSPORT_ORDER>(curve, from, to)
    }

    fn lower_with<const N: usize>(&self, target: V, v: V) -> Dual<V>
    where
        V: Form,
        <Self as ι>::C: MusicalRegion<𝒟, 𝒞, P, V, Self>,
    {
        <<Self as ι>::C as MusicalRegion<𝒟, 𝒞, P, V, Self>>::lower::<N>(self, target, v)
    }

    fn lower(&self, target: V, v: V) -> Dual<V>
    where
        V: Form,
        <Self as ι>::C: MusicalRegion<𝒟, 𝒞, P, V, Self>,
    {
        self.lower_with::<TRANSPORT_ORDER>(target, v)
    }

    fn raise_with<const N: usize>(&self, target: V, v: Dual<V>) -> V
    where
        V: Nondegenerate,
        <Self as ι>::C: MusicalRegion<𝒟, 𝒞, P, V, Self>,
    {
        <<Self as ι>::C as MusicalRegion<𝒟, 𝒞, P, V, Self>>::raise::<N>(self, target, v)
    }

    fn raise(&self, target: V, v: Dual<V>) -> V
    where
        V: Nondegenerate,
        <Self as ι>::C: MusicalRegion<𝒟, 𝒞, P, V, Self>,
    {
        self.raise_with::<TRANSPORT_ORDER>(target, v)
    }

    /// Certifies that parallel transport around every sufficiently small closed
    /// curve based at `target` preserves the form at that fibre.
    ///
    /// `closed_curve` is assumed by the test harness to satisfy
    ///
    /// ```text
    /// closed_curve(0) = closed_curve(1) = target
    /// ```
    ///
    /// in the corresponding tangent-bundle sense.
    ///
    /// For arbitrary `u, v ∈ T_target M`, this checks
    ///
    /// ```text
    /// g(Pγ u, Pγ v) = g(u, v),
    /// ```
    ///
    /// where `Pγ` is parallel transport around the closed curve `γ`.
    ///
    /// Since the manifold form is derived from the model-space form by parallel
    /// transport, this is precisely the path-independence condition required for
    /// that transported form to be well-defined.
    #[cfg(feature = "testing")]
    fn check_holonomy_preserves_form<const N: usize>(
        &self,
        target: V,
        closed_curve: impl Fn(Jet<𝐑𝐞𝐚𝐥::𝒞, <V::F as Interval>::R, N>) -> Tangent<P, V, N>,
        u: V,
        v: V,
    ) -> bool
    where
        V: Form + PartialEq,
        <Self as ι>::C: MusicalRegion<𝒟, 𝒞, P, V, Self>,
    {
        use num_traits::One;

        let base = self.order::<N>();
        let before = u.pairing(&base.lower(target.clone(), v.clone()));

        let transport = base.transport(&closed_curve, Zero::zero(), One::one());
        let u = transport.mul_v(&u);
        let v = transport.mul_v(&v);

        let after = u.pairing(&base.lower(target, v));

        before == after
    }
}

/// Closed structural region selecting the musical implementation for a connection.
///
/// The dispatch theory is inferred exactly like [`TransportRegion`]: a context
/// constructively outside `Met` selects the connection-derived implementation,
/// while a context refining `Met` selects the supplied metric tensor.
#[doc(hidden)]
pub trait MusicalRegion<
    𝒟: Cat,
    𝒞: Cat,
    P: Point,
    V: Vector<Hand = Right, Action = BothSided, F: FromReal> + ι<C: TransportRegion<𝒞>>,
    T: Connection<P, V>,
>: Category
{
    fn lower<const N: usize>(connection: &T, target: V, v: V) -> Dual<V>
    where
        V: Form;

    fn raise<const N: usize>(connection: &T, target: V, v: Dual<V>) -> V
    where
        V: Nondegenerate;
}

// Generic connection region: there is constructively no supplied metric tensor,
// so reconstruct the musical maps from parallel transport of the model-space form.
impl<
    C: Category,
    𝒞: Cat,
    P: Point,
    V: Vector<Hand = Right, Action = BothSided, F: FromReal> + ι<C: TransportRegion<𝒞>>,
    T: Connection<P, V>,
> MusicalRegion<𝐃𝐢𝐟𝐟::𝒞, 𝒞, P, V, T> for C
where
    C: Ⱶ<𝐌𝐞𝐭::𝒞, Absent>,
{
    fn lower<const N: usize>(connection: &T, target: V, v: V) -> Dual<V>
    where
        V: Form,
    {
        const {
            assert!(N > 0, "lowering requires a positive Taylor order");
        }

        let zero = <V::F as Interval>::R::zero();
        let one = <V::F as Interval>::R::one();
        let curve =
            |t: Jet<𝐑𝐞𝐚𝐥::𝒞, <V::F as Interval>::R, N>| tangent_lerp(connection, target.clone(), t);
        let transport = <V as ι>::C::parallel_transport(connection, &curve, one, zero);
        let v = transport.mul_v(&v);

        Dual::<V>::from_fn(|i| {
            let basis = V::from_fn(|j| if i == j { V::F::one() } else { V::F::zero() });
            transport.mul_v(&basis).dot(&v)
        })
    }

    fn raise<const N: usize>(connection: &T, target: V, v: Dual<V>) -> V
    where
        V: Nondegenerate,
    {
        const {
            assert!(N > 0, "raising requires a positive Taylor order");
        }

        let zero = <V::F as Interval>::R::zero();
        let one = <V::F as Interval>::R::one();
        let curve =
            |t: Jet<𝐑𝐞𝐚𝐥::𝒞, <V::F as Interval>::R, N>| tangent_lerp(connection, target.clone(), t);
        let transport = <V as ι>::C::parallel_transport(connection, &curve, zero, one);

        let v = Dual::<V>::from_fn(|i| {
            let basis = V::from_fn(|j| if i == j { V::F::one() } else { V::F::zero() });

            transport.mul_v(&basis).pairing(&v)
        });

        transport.mul_v(&V::sharp(v))
    }
}

// Metric region: use the supplied metric tensor directly in the tangent
// coordinate space selected by `target`.
impl<
    C: Category,
    𝒞: Cat,
    P: Point,
    V: Vector<Hand = Right, Action = BothSided, F: FromReal, Normalization = Atomic>
        + ι<C: TransportRegion<𝒞>>,
    T: Connection<P, V> + MetricTensor<P, V>,
> MusicalRegion<𝐌𝐞𝐭::𝒞, 𝒞, P, V, T> for C
where
    C: Ⱶ<𝐌𝐞𝐭::𝒞>,
{
    fn lower<const N: usize>(connection: &T, target: V, v: V) -> Dual<V>
    where
        V: Form,
    {
        let product = TensorProduct::pure(connection.g(target), Sinister(v));
        let reassociated = ReassociateKernel::<Right>::reassociate_kernel(product);
        let lowered: Sinister<Dual<V>> = reassociated.contract::<OnRight<ThroughSinister<Here>>>();

        Sinister(lowered).collapse()
    }

    fn raise<const N: usize>(connection: &T, target: V, v: Dual<V>) -> V
    where
        V: Nondegenerate,
    {
        let inverse: TensorProduct<V, Sinister<V>> = connection.g(target).inverse();
        let product = TensorProduct::pure(Sinister(v), Sinister(inverse));
        let reassociated = ReassociateKernel::<Left>::reassociate_kernel(product);

        reassociated.contract::<OnLeft<Here>>()
    }
}

impl<𝒞, 𝒟, P, V, T> ParallelTransport<𝒞, 𝒟, P, V> for T
where
    𝒞: Cat,
    𝒟: Cat,
    P: Point,
    V: Vector<Hand = Right, Action = BothSided, F: FromReal> + ι<C: TransportRegion<𝒞>>,
    T: Connection<P, V> + ι,
    <T as ι>::C: MusicalRegion<𝒟, 𝒞, P, V, T>,
{
    fn transport_with<const N: usize>(
        &self,
        curve: impl Fn(Jet<𝐑𝐞𝐚𝐥::𝒞, <V::F as Interval>::R, N>) -> Tangent<P, V, N>,
        from: <V::F as Interval>::R,
        to: <V::F as Interval>::R,
    ) -> TensorProduct<V, Dual<V>> {
        <V as ι>::C::parallel_transport(self, curve, from, to)
    }
}

#[doc(hidden)]
pub trait TransportRegion<𝒞: Cat>: Category {
    fn parallel_transport<
        P: Point,
        V: Vector<Hand = Right, Action = BothSided, F: FromReal>,
        T: Connection<P, V>,
        F: Fn(Jet<𝐑𝐞𝐚𝐥::𝒞, <V::F as Interval>::R, N>) -> Tangent<P, V, N>,
        const N: usize,
    >(
        connection: &T,
        curve: F,
        from: <V::F as Interval>::R,
        to: <V::F as Interval>::R,
    ) -> TensorProduct<V, Dual<V>>;
}

fn transport_accurate<V, const N: usize>(
    full: &TensorProduct<V, Dual<V>>,
    half: &TensorProduct<V, Dual<V>>,
) -> bool
where
    V: Vector<Hand = Right, Action = BothSided, F: FromReal>,
{
    let epsilon = <V::F as Interval>::R::epsilon();
    let epsilon_squared = epsilon * epsilon;

    let richardson = <V::F as Interval>::R::from_nat((1usize << N) - 1);
    let richardson_squared = richardson * richardson;

    (0..V::N).all(|i| {
        (0..V::N).all(|j| {
            let error = full[(i, j)].interval_squared(&half[(i, j)]).abs();
            let magnitude = half[(i, j)].interval_squared(&V::F::zero()).abs();
            let one = <V::F as Interval>::R::one();
            let scale = if magnitude.exact_lt(one) {
                one
            } else {
                magnitude
            };
            let estimated_error_squared = error / richardson_squared;

            estimated_error_squared.exact_le(epsilon_squared * scale)
        })
    })
}

fn adaptive_parallel_transport<
    V: Vector<Hand = Right, Action = BothSided, F: FromReal>,
    F: Fn(<V::F as Interval>::R, <V::F as Interval>::R) -> Option<TensorProduct<V, Dual<V>>>,
    const N: usize,
>(
    step: F,
    from: <V::F as Interval>::R,
    to: <V::F as Interval>::R,
) -> TensorProduct<V, Dual<V>> {
    let mut t = from;
    let mut transport = TensorProduct::<V, Dual<V>>::identity();
    let mut h = to - from;

    if from.exact_eq(to) {
        return transport;
    }

    let two = <V::F as Interval>::R::one() + <V::F as Interval>::R::one();

    loop {
        let half_h = h / two;
        let midpoint = t + half_h;
        let next = midpoint + half_h;
        let full_h = next - t;

        let full = step(t, full_h);
        let half = step(t, half_h)
            .and_then(|first| step(midpoint, half_h).map(|second| second.compose(&first)));

        let Some(half) = half else {
            h = half_h;
            continue;
        };

        let Some(full) = full else {
            h = half_h;
            continue;
        };

        if !transport_accurate::<V, N>(&full, &half) {
            h = half_h;
            continue;
        }

        // The two-half-step operator is the better local approximation.  A
        // later transport acts on the left, so accumulate in path order.
        transport = half.compose(&transport);

        let progresses = (to - next).abs().exact_lt((to - t).abs());

        if !progresses {
            return transport;
        }

        t = next;

        if t.exact_eq(to) {
            return transport;
        }

        let doubled = h * two;
        let remaining = to - t;

        h = if h.is_sign_negative() {
            if remaining.exact_lt(doubled) {
                doubled
            } else {
                remaining
            }
        } else if doubled.exact_lt(remaining) {
            doubled
        } else {
            remaining
        };
    }
}

fn parallel_transport_taylor<
    P: Point,
    V: Vector<Hand = Right, Action = BothSided, F: FromReal>,
    T: Connection<P, V>,
    F: Fn(Jet<𝐑𝐞𝐚𝐥::𝒞, <V::F as Interval>::R, N>) -> Tangent<P, V, N>,
    const N: usize,
>(
    connection: &T,
    curve: F,
    from: <V::F as Interval>::R,
    to: <V::F as Interval>::R,
) -> TensorProduct<V, Dual<V>> {
    const {
        assert!(N > 0, "parallel transport requires a positive Taylor order");
    }

    let step = |t: <V::F as Interval>::R,
                h: <V::F as Interval>::R|
     -> Option<TensorProduct<V, Dual<V>>> {
        let time = Jet::<𝐑𝐞𝐚𝐥::𝒞, <V::F as Interval>::R, N>::new(
            t,
            core::array::from_fn(|i| {
                if i == 0 {
                    <V::F as Interval>::R::one()
                } else {
                    <V::F as Interval>::R::zero()
                }
            }),
        );

        let path = curve(time);
        let point = LiftedTM::<P, V, T, N>::new(path.0.clone(), path.1.clone());
        let connection = Prolongation::<P, V, T, N>::new(
            connection.base_point(),
            JetVectorIn::<𝐅𝐥𝐝::𝒞, V, N>::zero(),
        );
        let velocity = TensorOver(V::Array::from_fn(|i| path.1[i].derivative()), PhantomData);
        let christoffel = connection.christoffel_symbols(point)?;

        let a = -TensorProduct::pure(christoffel, Sinister(velocity))
            .reassociate::<Right>()
            .contract::<OnRight<ThroughSinister<Here>>>();

        let compose =
            |lhs: &TensorProduct<JetVector<V, N>, Dual<JetVector<V, N>>>,
             rhs: &TensorProduct<JetVector<V, N>, Dual<JetVector<V, N>>>| {
                TensorProduct::<JetVector<V, N>, Dual<JetVector<V, N>>>::from_fn_ij(|i, j| {
                    (0..V::N).fold(Jet::<𝐅𝐥𝐝::𝒞, V::F, N>::zero(), |sum, k| {
                        sum + lhs[(i, k)].clone() * rhs[(k, j)].clone()
                    })
                })
            };

        // Solve the fundamental equation X' = A X, X(0) = I.  This computes
        // the transport itself rather than re-solving the same linear ODE for
        // each vector to which it is later applied.
        let mut x = TensorProduct::<JetVector<V, N>, Dual<JetVector<V, N>>>::from_fn_ij(|i, j| {
            Jet::from_parts(
                if i == j { V::F::one() } else { V::F::zero() },
                core::array::from_fn(|_| V::F::zero()),
            )
        });

        for _ in 0..N {
            let derivative = compose(&a, &x);

            x = TensorProduct::<JetVector<V, N>, Dual<JetVector<V, N>>>::from_fn_ij(|i, j| {
                Jet::integrate_from(
                    if i == j { V::F::one() } else { V::F::zero() },
                    derivative[(i, j)].clone(),
                )
            });
        }

        let h = V::F::from_real(h);

        Some(TensorProduct::<V, Dual<V>>::from_fn_ij(|i, j| {
            let coefficient = &x[(i, j)];
            let mut value = coefficient[N];

            for n in (0..N).rev() {
                value = value * h + coefficient[n];
            }

            value
        }))
    };

    adaptive_parallel_transport::<V, _, N>(step, from, to)
}

impl<C> TransportRegion<𝐓𝐞𝐧𝐬::𝒞> for C
where
    C: Ⱶ<𝐓𝐞𝐧𝐬::𝒞> + Ⱶ<𝐅𝐨𝐫𝐦::𝒞, Absent>,
{
    fn parallel_transport<
        P: Point,
        V: Vector<Hand = Right, Action = BothSided, F: FromReal>,
        T: Connection<P, V>,
        F: Fn(Jet<𝐑𝐞𝐚𝐥::𝒞, <V::F as Interval>::R, N>) -> Tangent<P, V, N>,
        const N: usize,
    >(
        connection: &T,
        curve: F,
        from: <V::F as Interval>::R,
        to: <V::F as Interval>::R,
    ) -> TensorProduct<V, Dual<V>> {
        parallel_transport_taylor(connection, curve, from, to)
    }
}

// Form-bearing region: solve the Magnus equation for Ω and return exp(Ω), the
// transport operator itself.  Vector transport is only an application of this
// result and therefore never needs to repeat the connection solve.
impl<C> TransportRegion<𝐅𝐨𝐫𝐦::𝒞> for C
where
    C: Ⱶ<𝐅𝐨𝐫𝐦::𝒞>,
{
    fn parallel_transport<
        P: Point,
        V: Vector<Hand = Right, Action = BothSided, F: FromReal>,
        T: Connection<P, V>,
        F: Fn(Jet<𝐑𝐞𝐚𝐥::𝒞, <V::F as Interval>::R, N>) -> Tangent<P, V, N>,
        const N: usize,
    >(
        connection: &T,
        curve: F,
        from: <V::F as Interval>::R,
        to: <V::F as Interval>::R,
    ) -> TensorProduct<V, Dual<V>> {
        const {
            assert!(N > 0, "parallel transport requires a positive Taylor order");
        }

        let step = |t: <V::F as Interval>::R,
                    h: <V::F as Interval>::R|
         -> Option<TensorProduct<V, Dual<V>>> {
            let time = Jet::<𝐑𝐞𝐚𝐥::𝒞, <V::F as Interval>::R, N>::new(
                t,
                core::array::from_fn(|i| {
                    if i == 0 {
                        <V::F as Interval>::R::one()
                    } else {
                        <V::F as Interval>::R::zero()
                    }
                }),
            );

            let path = curve(time);
            let point = LiftedTM::<P, V, T, N>::new(path.0.clone(), path.1.clone());
            let connection = Prolongation::<P, V, T, N>::new(
                connection.base_point(),
                JetVectorIn::<𝐅𝐥𝐝::𝒞, V, N>::zero(),
            );
            let velocity = TensorOver(V::Array::from_fn(|i| path.1[i].derivative()), PhantomData);
            let christoffel = connection.christoffel_symbols(point)?;

            let a = -TensorProduct::pure(christoffel, Sinister(velocity))
                .reassociate::<Right>()
                .contract::<OnRight<ThroughSinister<Here>>>();

            let compose =
                |lhs: &TensorProduct<JetVector<V, N>, Dual<JetVector<V, N>>>,
                 rhs: &TensorProduct<JetVector<V, N>, Dual<JetVector<V, N>>>| {
                    TensorProduct::<JetVector<V, N>, Dual<JetVector<V, N>>>::from_fn_ij(|i, j| {
                        (0..V::N).fold(Jet::<𝐅𝐥𝐝::𝒞, V::F, N>::zero(), |sum, k| {
                            sum + lhs[(i, k)].clone() * rhs[(k, j)].clone()
                        })
                    })
                };

            let mut bernoulli = [<V::F as Interval>::R::zero(); N];
            bernoulli[0] = <V::F as Interval>::R::one();

            for n in 1..N {
                let mut sum = <V::F as Interval>::R::zero();
                let mut factorial = <V::F as Interval>::R::one();

                for k in 1..=n {
                    factorial =
                        factorial * <<V::F as Interval>::R as NumCast>::from(k + 1).unwrap();
                    sum = sum + bernoulli[n - k] / factorial;
                }

                bernoulli[n] = -sum;
            }

            let mut omega =
                TensorProduct::<JetVector<V, N>, Dual<JetVector<V, N>>>::from_fn_ij(|_, _| {
                    Jet::from_parts(V::F::zero(), core::array::from_fn(|_| V::F::zero()))
                });

            for _ in 0..N {
                let mut rhs = a.clone();
                let mut ad = a.clone();

                for k in 1..N {
                    ad = compose(&omega, &ad) - compose(&ad, &omega);

                    let coefficient = Jet::<𝐅𝐥𝐝::𝒞, V::F, N>::from_parts(
                        V::F::from_real(bernoulli[k]),
                        core::array::from_fn(|_| V::F::zero()),
                    );

                    rhs = rhs + ad.clone() * coefficient;
                }

                omega =
                    TensorProduct::<JetVector<V, N>, Dual<JetVector<V, N>>>::from_fn_ij(|i, j| {
                        Jet::integrate_from(V::F::zero(), rhs[(i, j)].clone())
                    });
            }

            let h = V::F::from_real(h);
            let omega = TensorProduct::<V, Dual<V>>::from_fn_ij(|i, j| {
                let coefficient = &omega[(i, j)];
                let mut value = coefficient[N];

                for n in (0..N).rev() {
                    value = value * h + coefficient[n];
                }

                value
            });

            Some(endomorphism_exp(omega))
        };

        adaptive_parallel_transport::<V, _, N>(step, from, to)
    }
}

#[cfg(feature = "testing")]
fn constant_jet_vector<V: Tensor, const N: usize>(v: V) -> JetVector<V, N> {
    JetVectorIn::from_fn(|i| Jet::from_parts(v[i], [V::F::zero(); N]))
}

impl<P: Point, V: Tensor, T: Connection<P, V>, const N: usize>
    Chart<LiftedTM<P, V, T, N>, JetVector<V, N>> for Prolongation<P, V, T, N>
{
    type Global = <T::Global as OptionallyOption<P>>::Mapped<LiftedTM<P, V, T, N>>;

    fn to_local(&self, point: &LiftedTM<P, V, T, N>) -> Option<JetVector<V, N>> {
        T::tangent_to_local(
            TangentElement::new(self.0.clone(), self.1.clone()),
            TangentElement::new(point.0.clone(), point.1.clone()),
        )
    }

    fn to_global(&self, coordinate: JetVector<V, N>) -> Self::Global {
        T::tangent_to_global(
            TangentElement::new(self.0.clone(), self.1.clone()),
            coordinate,
        )
        .cast_option(|(base, jet)| TangentElement::new(base, jet))
    }

    fn chart_at(point: &LiftedTM<P, V, T, N>) -> Self {
        TangentElement::new(point.0.clone(), point.1.clone())
    }
}

impl<P: Point, V: Tensor, T: Connection<P, V>, const N: usize>
    ExpMap<LiftedTM<P, V, T, N>, JetVector<V, N>> for Prolongation<P, V, T, N>
{
}
impl<P: Point, V: Tensor, T: Connection<P, V>, const N: usize>
    TangentBundle<LiftedTM<P, V, T, N>, JetVector<V, N>> for Prolongation<P, V, T, N>
{
}
