use core::marker::PhantomData;

use crate::traits::{
    Absent, ActionExists, Array, Cat, Dual, Field, Point, Right, Tensor, TensorProductAction,
    Vector,
    calculus::{
        Connection, ConstantRoute, HomOf, Jet, JetLayer, JetRegion, JetVector, JetVectorIn,
        Tangent, TangentMap, TensorOver, TensorProduct, TensorProductArray,
    },
    Ø, ː, ι, Ⱶ, 𝐅𝐥𝐝, 𝐓𝐞𝐧𝐬,
};
use num_traits::{One, Zero};

/// A composable differential program for `F`.
///
/// Construct it as `d(f)`. Calling [`d::at`] evaluates the full derivative,
/// while [`d::along`] contracts the next derivative slot with a direction.
/// Since `d<F>` itself implements [`JetMap`], differential programs can be
/// nested: `d(d(f))` and `d(d(d(f)))` use the same machinery as `d(f)`.
#[allow(non_camel_case_types)]
pub struct d<F>(pub F);

/// A differential program with its next input slot contracted with `direction`.
///
/// This represents the function `p ↦ Dfₚ(direction)`. It remains a
/// differentiable program until [`Along::at`] evaluates it.
pub struct Along<F, V> {
    f: F,
    direction: V,
}

impl<F> d<F> {
    pub fn at<𝒞: Cat, Point: DifferentialRegion<𝒞, F, Output, Route>, Output, Route>(
        &self,
        point: Point,
    ) -> Output
    where
        Self: EvaluableAt<𝒞, Point, Output, Route>,
    {
        <Self as EvaluableAt<𝒞, Point, Output, Route>>::evaluate_at(self, point)
    }

    pub fn along<V>(self, direction: V) -> Along<F, V> {
        Along {
            f: self.0,
            direction,
        }
    }
}

impl<F, V> Along<F, V> {
    /// Evaluates the directional derivative at `point`.
    pub fn at<𝒞: Cat, Point, Output, Route>(&self, point: Point) -> Output
    where
        Self: EvaluableAt<𝒞, Point, Output, Route>,
    {
        <Self as EvaluableAt<𝒞, Point, Output, Route>>::evaluate_at(self, point)
    }
}

impl<𝒞, F, P, V, Q, W, JP, JQ> EvaluableAt<𝒞, P, W, ManifoldRoute<Q, JP, JQ>> for Along<F, V>
where
    𝒞: Cat,
    P: Connection<P, V> + ι,
    P::C: Ⱶ<𝐓𝐞𝐧𝐬::𝒞, Absent>,
    V: Tensor<F: ι<C: JetRegion<𝒞>>, Hand = Right, Action: ActionExists>,
    Q: Connection<Q, W>,
    W: Tensor<F = V::F, Hand = Right, Action: TensorProductAction<V::Action>>,
    JP: Point,
    JQ: Point,
    F: ManifoldJetMap<P, V, Q, W, 1, ManifoldRoute<Q, JP, JQ>>,
    Jet<𝒞, V::F>: Field,
{
    fn evaluate_at(&self, point: P) -> W {
        let tangent = JetVector::from_fn(|coordinate| {
            Jet::from_parts(V::F::zero(), [self.direction[coordinate]])
        });

        let output = <F as ManifoldJetMap<P, V, Q, W, 1, ManifoldRoute<Q, JP, JQ>>>::jet_at(
            &self.f,
            Tangent::new(point, tangent),
        );

        W::from_fn(|coordinate| output.1[coordinate][1])
    }
}

#[diagnostic::on_unimplemented(
    message = "this differential program cannot be evaluated at `{Point}`",
    label = "the composed differential operations are not defined for this point type",
    note = "the function may not accept the required jet presentation",
    note = "the input and output tensors may have incompatible fields, handedness, or actions",
    note = "a required form or musical isomorphism may not lift through nested jets"
)]
#[doc(hidden)]
/// The diagnostic evaluation boundary used by [`d::at`] and [`Along::at`].
///
/// Keeping their large proof obligations behind this trait replaces a wall of
/// nested associated-type failures with one explanation of why a differential
/// program is not evaluable at a particular point type.
pub trait EvaluableAt<𝒞: Cat, Point, Output, Route = Ø> {
    fn evaluate_at(&self, point: Point) -> Output;
}

pub(crate) fn evaluate_derivative_at<𝒞, F, BT, FT>(
    derivative: &d<F>,
    point: BT,
) -> TangentMap<BT, FT, FT, FT>
where
    𝒞: Cat,
    F: JetMap<𝒞, BT, FT, 1, BT::F>,
    BT: Tensor<Hand = Right, Action: ActionExists>,
    FT: Tensor<F = BT::F, Hand = Right, Action: TensorProductAction<BT::Action>>,
    Jet<𝒞, BT::F>: Field,
{
    let columns: BT::Array<FT> = BT::Array::from_fn(|input_coordinate| {
        let input = JetVectorIn::<𝒞, BT>::from_fn(|coordinate| {
            Jet::from_parts(
                point[coordinate],
                [if input_coordinate == coordinate {
                    BT::F::one()
                } else {
                    BT::F::zero()
                }],
            )
        });

        let output = <F as JetMap<𝒞, BT, FT, 1, BT::F, Ø>>::jet_at(&derivative.0, input);

        FT::from_fn(|output_coordinate| output[output_coordinate][1])
    });

    let rows: FT::Array<<Dual<BT> as Tensor>::Array<BT::F>> =
        FT::Array::from_fn(|output_coordinate| {
            <Dual<BT> as Tensor>::Array::from_fn(|input_coordinate| {
                columns[input_coordinate][output_coordinate]
            })
        });

    TangentMap::new(TensorProduct(TensorProductArray(rows, PhantomData)))
}

impl<
    𝒞: Cat,
    F: JetMap<𝒞, BT, FT, 1, BT::F>,
    BT: Tensor<F: ι<C: JetRegion<𝒞>>, Hand = Right, Action: ActionExists>,
    FT: Tensor<F = BT::F, Hand = Right, Action: TensorProductAction<BT::Action>>,
> EvaluableAt<𝒞, BT, TangentMap<BT, FT, FT, FT>> for d<F>
where
    Jet<𝒞, BT::F>: Field,
{
    fn evaluate_at(&self, point: BT) -> TangentMap<BT, FT, FT, FT> {
        evaluate_derivative_at::<𝒞, _, _, _>(self, point)
    }
}

impl<
    𝒞: Cat,
    F: JetMap<𝒞, BT, FT, 1, BT::F>,
    BT: Vector<F: ι<C: JetRegion<𝒞>>>,
    FT: Tensor<F = BT::F, Hand = Right, Action: TensorProductAction<BT::Action>>,
> EvaluableAt<𝒞, BT, FT> for Along<F, BT>
where
    Jet<𝒞, BT::F>: Field,
{
    fn evaluate_at(&self, point: BT) -> FT {
        let input = JetVectorIn::<𝒞, BT, 1, BT::F>::from_fn(|coordinate| {
            Jet::new(point[coordinate], [self.direction[coordinate]])
        });

        let output: JetVectorIn<𝒞, FT, 1, BT::F> =
            <F as JetMap<𝒞, BT, FT, 1, BT::F, Ø>>::jet_at(&self.f, input);

        FT::from_fn(|coordinate| output[coordinate][1])
    }
}

/// A map that can be evaluated through a selected categorical jet presentation.
///
/// Ordinary generic Rust functions implement this trait through the blanket
/// `Fn(JetVector<𝒞, BT, ..>)` implementation. Differential programs implement it
/// recursively, adding jet layers while `Route` remembers how to inject
/// captured base-field constants into the current scalar type.
pub trait JetMap<𝒞: Cat, BT: Tensor, FT: Tensor<F = BT::F>, const N: usize, S: Field, Route = Ø> {
    /// Evaluates the map without discarding any jet coefficients.
    fn jet_at(&self, input: JetVectorIn<𝒞, BT, N, S>) -> JetVectorIn<𝒞, FT, N, S>;
}

impl<
    𝒞: Cat,
    F: Fn(JetVectorIn<𝒞, BT, N, S>) -> JetVectorIn<𝒞, FT, N, S>,
    BT: Tensor,
    FT: Tensor<F = BT::F>,
    const N: usize,
    S: Field,
    Route,
> JetMap<𝒞, BT, FT, N, S, Route> for F
{
    fn jet_at(&self, input: JetVectorIn<𝒞, BT, N, S>) -> JetVectorIn<𝒞, FT, N, S> {
        self(input)
    }
}

fn evaluate_manifold_derivative_at<F, P, V, Q, W, Route>(
    derivative: &d<F>,
    point: P,
) -> TangentMap<V, Q, W, Q>
where
    P: Connection<P, V>,
    V: Tensor<Hand = Right, Action: ActionExists>,
    Q: Connection<Q, W>,
    W: Tensor<F = V::F, Hand = Right, Action: TensorProductAction<V::Action>>,
    F: ManifoldJetMap<P, V, Q, W, 1, Route>,
{
    let columns: V::Array<W> = V::Array::from_fn(|input_coordinate| {
        let tangent = JetVector::from_fn(|coordinate| {
            Jet::from_parts(
                V::F::zero(),
                [if input_coordinate == coordinate {
                    V::F::one()
                } else {
                    V::F::zero()
                }],
            )
        });

        let output = <F as ManifoldJetMap<P, V, Q, W, 1, Route>>::jet_at(
            &derivative.0,
            Tangent::new(point.clone(), tangent),
        );

        W::from_fn(|output_coordinate| output.1[output_coordinate][1])
    });

    let rows: W::Array<<Dual<V> as Tensor>::Array<V::F>> = W::Array::from_fn(|output_coordinate| {
        <Dual<V> as Tensor>::Array::from_fn(|input_coordinate| {
            columns[input_coordinate][output_coordinate]
        })
    });

    TangentMap::new(TensorProduct(TensorProductArray(rows, PhantomData)))
}

/// A manifold-valued map evaluated by commuting intrinsic tangent jets through
/// the source and target Rust representations.
pub trait ManifoldJetMap<P: Point, V: Tensor, Q: Point, W: Tensor<F = V::F>, const N: usize, Route>
{
    fn jet_at(&self, input: Tangent<P, V, N>) -> Tangent<Q, W, N>;
}

/// Selects the differential evaluator from the canonical category of the
/// point supplied to `d::at`.
#[doc(hidden)]
pub trait DifferentialRegion<𝒞: Cat, F, Output, Route>: Point {}

impl<𝒞, F, BT, FT> DifferentialRegion<𝒞, F, TangentMap<BT, FT, FT, FT>, Ø> for BT
where
    𝒞: Cat,
    F: JetMap<𝒞, BT, FT, 1, BT::F>,
    BT: Tensor<F: ι<C: JetRegion<𝒞>>, Hand = Right, Action: ActionExists>,
    FT: Tensor<F = BT::F, Hand = Right, Action: TensorProductAction<BT::Action>>,
    Jet<𝒞, BT::F>: Field,
{
}

impl<𝒞, F, P, V, Q, W, JP, JQ>
    DifferentialRegion<𝒞, F, TangentMap<V, Q, W, Q>, ManifoldRoute<Q, JP, JQ>> for P
where
    𝒞: Cat,
    P: Point + ι + Connection<P, V>,
    P::C: Ⱶ<𝐓𝐞𝐧𝐬::𝒞, Absent>,
    V: Tensor<F: ι<C: JetRegion<𝒞>>, Hand = Right, Action: ActionExists>,
    Q: Connection<Q, W>,
    W: Tensor<F = V::F, Hand = Right, Action: TensorProductAction<V::Action>>,
    JP: Point,
    JQ: Point,
    F: ManifoldJetMap<P, V, Q, W, 1, ManifoldRoute<Q, JP, JQ>>,
{
}

#[doc(hidden)]
pub struct ManifoldRoute<Q: Point, JP: Point, JQ: Point>(PhantomData<fn(JP) -> (Q, JQ)>);

#[doc(hidden)]
pub struct DifferentiatedManifoldRoute<
    𝒞: Cat,
    JP: Point,
    JV: Tensor,
    JQ: Point,
    JW: Tensor,
    InnerRoute,
>(PhantomData<fn(𝒞, JP, JV, InnerRoute) -> (JQ, JW)>);

impl<𝒞, F, P, V, Q, W, JP, JV, JQ, JW, InnerRoute, const N: usize>
    ManifoldJetMap<
        P,
        V,
        TangentMap<V, Q, W>,
        TangentMap<V, Q, W>,
        N,
        DifferentiatedManifoldRoute<𝒞, JP, JV, JQ, JW, InnerRoute>,
    > for d<F>
where
    𝒞: Cat,
    P: Connection<P, V>,
    V: Tensor<Hand = Right, Action: ActionExists>,
    Q: Connection<Q, W>,
    W: Tensor<F = V::F, Hand = Right, Action: TensorProductAction<V::Action>>,
    JP: Point + Connection<JP, JV> + CommutesJet<P, V, N>,
    JV: Tensor<F = Jet<𝒞, V::F, N>, Hand = Right, Action: ActionExists>,
    JQ: Point + Connection<JQ, JW> + CommutesJet<Q, W, N>,
    JW: Tensor<F = Jet<𝒞, V::F, N>, Hand = Right, Action: TensorProductAction<JV::Action>>,
    F: ManifoldJetMap<JP, JV, JQ, JW, 1, InnerRoute>,
    Jet<𝒞, V::F, N>: Field,
    Jet<𝐅𝐥𝐝::𝒞, V::F, N>: Field,
    TangentMap<V, Q, W>: Tensor<F = V::F, Hand = Right>,
{
    fn jet_at(
        &self,
        input: Tangent<P, V, N>,
    ) -> Tangent<TangentMap<V, Q, W>, TangentMap<V, Q, W>, N> {
        const {
            assert!(JV::N == V::N);
            assert!(JW::N == W::N);
        }

        let outer_point = <JP as CommutesJet<P, V, N>>::commute_jet(input);

        let columns: V::Array<JetVectorIn<𝒞, W, N>> = V::Array::from_fn(|input_coordinate| {
            let inner_tangent: JetVector<JV, 1> = TensorOver::from_fn(|coordinate| {
                Jet::<𝐅𝐥𝐝::𝒞, Jet<𝒞, V::F, N>, 1>::from_parts(
                    Jet::<𝒞, V::F, N>::zero(),
                    [if coordinate == input_coordinate {
                        Jet::<𝒞, V::F, N>::one()
                    } else {
                        Jet::<𝒞, V::F, N>::zero()
                    }],
                )
            });

            let inner_input: Tangent<JP, JV, 1> =
                Tangent::<JP, JV, 1>::new(outer_point.clone(), inner_tangent);

            let output =
                <F as ManifoldJetMap<JP, JV, JQ, JW, 1, InnerRoute>>::jet_at(&self.0, inner_input);

            JetVectorIn::<𝒞, W, N>::from_fn(|output_coordinate| {
                output.1[output_coordinate][1].clone()
            })
        });

        let derivative = JetVectorIn::<𝒞, TangentMap<V, Q, W>, N>::from_fn(|index| {
            let output_coordinate = index / V::N;
            let input_coordinate = index % V::N;

            columns[input_coordinate][output_coordinate].clone()
        });

        derivative.retag::<𝐅𝐥𝐝::𝒞>().into_tangent(|value| value)
    }
}

impl<𝒞, F, P, V, Q, W, JP, JV, JQ, JW, InnerRoute>
    DifferentialRegion<
        𝒞,
        d<F>,
        TangentMap<V, TangentMap<V, Q, W>, TangentMap<V, Q, W>>,
        DifferentiatedManifoldRoute<𝒞, JP, JV, JQ, JW, InnerRoute>,
    > for P
where
    𝒞: Cat,
    P: Point + ι + Connection<P, V>,
    P::C: Ⱶ<𝐓𝐞𝐧𝐬::𝒞, Absent>,
    V: Tensor<Hand = Right, Action: ActionExists>,
    Q: Connection<Q, W>,
    W: Tensor<F = V::F, Hand = Right, Action: TensorProductAction<V::Action>>,
    JP: Point + Connection<JP, JV> + CommutesJet<P, V, 1>,
    JV: Tensor<F = Jet<𝒞, V::F, 1>, Hand = Right, Action: ActionExists>,
    JQ: Point + Connection<JQ, JW> + CommutesJet<Q, W, 1>,
    JW: Tensor<F = Jet<𝒞, V::F, 1>, Hand = Right, Action: TensorProductAction<JV::Action>>,
    F: ManifoldJetMap<JP, JV, JQ, JW, 1, InnerRoute>,
    Jet<𝒞, V::F, 1>: Field,
    Jet<𝐅𝐥𝐝::𝒞, V::F, 1>: Field,
    TangentMap<V, Q, W>: Tensor<F = V::F, Hand = Right, Action: TensorProductAction<V::Action>>,
    d<F>: ManifoldJetMap<
            P,
            V,
            TangentMap<V, Q, W>,
            TangentMap<V, Q, W>,
            1,
            DifferentiatedManifoldRoute<𝒞, JP, JV, JQ, JW, InnerRoute>,
        >,
{
}

impl<𝒞, F, P, V, Q, W, JP, JV, JQ, JW, InnerRoute>
    EvaluableAt<
        𝒞,
        P,
        TangentMap<V, TangentMap<V, Q, W>, TangentMap<V, Q, W>>,
        DifferentiatedManifoldRoute<𝒞, JP, JV, JQ, JW, InnerRoute>,
    > for d<d<F>>
where
    𝒞: Cat,
    P: Connection<P, V>,
    V: Tensor<Hand = Right, Action: ActionExists>,
    Q: Connection<Q, W>,
    W: Tensor<F = V::F, Hand = Right, Action: TensorProductAction<V::Action>>,
    JP: Point + Connection<JP, JV> + CommutesJet<P, V, 1>,
    JV: Tensor<F = Jet<𝒞, V::F, 1>, Hand = Right, Action: ActionExists>,
    JQ: Point + Connection<JQ, JW> + CommutesJet<Q, W, 1>,
    JW: Tensor<F = Jet<𝒞, V::F, 1>, Hand = Right, Action: TensorProductAction<JV::Action>>,
    F: ManifoldJetMap<JP, JV, JQ, JW, 1, InnerRoute>,
    Jet<𝒞, V::F, 1>: Field,
    Jet<𝐅𝐥𝐝::𝒞, V::F, 1>: Field,
    TangentMap<V, Q, W>: Tensor<F = V::F, Hand = Right, Action: TensorProductAction<V::Action>>
        + Connection<TangentMap<V, Q, W>, TangentMap<V, Q, W>>,
    d<F>: ManifoldJetMap<
            P,
            V,
            TangentMap<V, Q, W>,
            TangentMap<V, Q, W>,
            1,
            DifferentiatedManifoldRoute<𝒞, JP, JV, JQ, JW, InnerRoute>,
        >,
{
    fn evaluate_at(&self, point: P) -> TangentMap<V, TangentMap<V, Q, W>, TangentMap<V, Q, W>> {
        evaluate_manifold_derivative_at::<
            d<F>,
            P,
            V,
            TangentMap<V, Q, W>,
            TangentMap<V, Q, W>,
            DifferentiatedManifoldRoute<𝒞, JP, JV, JQ, JW, InnerRoute>,
        >(self, point)
    }
}

impl<𝒞, F, P, V, Q, W, JP, JQ> EvaluableAt<𝒞, P, TangentMap<V, Q, W, Q>, ManifoldRoute<Q, JP, JQ>>
    for d<F>
where
    𝒞: Cat,
    P: Connection<P, V>,
    V: Tensor<F: ι<C: JetRegion<𝒞>>, Hand = Right, Action: ActionExists>,
    Q: Connection<Q, W>,
    W: Tensor<F = V::F, Hand = Right, Action: TensorProductAction<V::Action>>,
    JP: Point,
    JQ: Point,
    F: ManifoldJetMap<P, V, Q, W, 1, ManifoldRoute<Q, JP, JQ>>,
{
    fn evaluate_at(&self, point: P) -> TangentMap<V, Q, W, Q> {
        evaluate_manifold_derivative_at::<F, P, V, Q, W, ManifoldRoute<Q, JP, JQ>>(self, point)
    }
}

impl<P, V, Q, W, F, const N: usize, JP, JQ> ManifoldJetMap<P, V, Q, W, N, ManifoldRoute<Q, JP, JQ>>
    for F
where
    P: Connection<P, V>,
    V: Tensor,
    Q: Connection<Q, W>,
    W: Tensor<F = V::F>,
    JP: CommutesJet<P, V, N>,
    JQ: CommutesJet<Q, W, N>,
    F: Fn(JP) -> JQ,
{
    fn jet_at(&self, input: Tangent<P, V, N>) -> Tangent<Q, W, N> {
        JQ::uncommute_jet(self(JP::commute_jet(input)))
    }
}

/// An isomorphism between a connection's intrinsic split jet and one concrete
/// Rust presentation obtained by commuting jettification through a nominal
/// type constructor.
pub trait CommutesJet<P: Point, V: Tensor, const N: usize>: Point
where
    P: Connection<P, V>,
{
    fn commute_jet(value: Tangent<P, V, N>) -> Self;

    fn uncommute_jet(value: Self) -> Tangent<P, V, N>;
}

impl<P, V, const N: usize> CommutesJet<P, V, N> for Tangent<P, V, N>
where
    P: Connection<P, V>,
    V: Tensor,
{
    fn commute_jet(value: Tangent<P, V, N>) -> Self {
        value
    }

    fn uncommute_jet(value: Self) -> Tangent<P, V, N> {
        value
    }
}

impl<
    𝒞: Cat,
    F: JetMap<𝒞, BT, FT, 1, Jet<𝒞, S, N>, ː<JetLayer<𝒞, N>, Route>>,
    BT: Vector<F = FT::F, Hand = Right>,
    FT: Vector<Hand = Right, Action: TensorProductAction<BT::Action>>,
    const N: usize,
    S: Field,
    Route: ConstantRoute<BT::F, Output = S>,
> JetMap<𝒞, BT, HomOf<BT, FT>, N, S, Route> for d<F>
where
    // The outer presentation.
    JetVectorIn<𝒞, FT, N, S>: Vector<F = Jet<𝒞, S, N>>,
    JetVectorIn<𝒞, BT, N, S>: Tensor<F = Jet<𝒞, S, N>>,
    // One additional derivative layer over the existing outer scalar.
    JetVectorIn<𝒞, BT, 1, Jet<𝒞, S, N>>: Tensor<F = Jet<𝒞, Jet<𝒞, S, N>>>,
    JetVectorIn<𝒞, FT, 1, Jet<𝒞, S, N>>: Tensor<F = Jet<𝒞, Jet<𝒞, S, N>>>,
    Jet<𝒞, S, N>: Field,
{
    fn jet_at(&self, input: JetVectorIn<𝒞, BT, N, S>) -> JetVectorIn<𝒞, HomOf<BT, FT>, N, S> {
        #[allow(type_alias_bounds)]
        type OuterScalar<𝒞: Cat, S, const N: usize> = Jet<𝒞, S, N>;

        let columns: BT::Array<JetVectorIn<𝒞, FT, N, S>> = BT::Array::from_fn(|input_coordinate| {
            let nested_input =
                JetVectorIn::<𝒞, BT, 1, OuterScalar<𝒞, S, N>>::from_fn(|coordinate| {
                    Jet::from_parts(
                        input[coordinate],
                        [if input_coordinate == coordinate {
                            OuterScalar::<𝒞, S, N>::one()
                        } else {
                            OuterScalar::<𝒞, S, N>::zero()
                        }],
                    )
                });

            let nested_output: JetVectorIn<𝒞, FT, 1, OuterScalar<𝒞, S, N>> = <F as JetMap<
                𝒞,
                BT,
                FT,
                1,
                OuterScalar<𝒞, S, N>,
                ː<JetLayer<𝒞, N>, Route>,
            >>::jet_at(
                &self.0,
                nested_input,
            );

            JetVectorIn::<𝒞, FT, N, S>::from_fn(|output_coordinate| {
                nested_output[output_coordinate][1]
            })
        });

        let rows: FT::Array<<Dual<BT> as Tensor>::Array<OuterScalar<𝒞, S, N>>> =
            FT::Array::from_fn(|output_coordinate| {
                <Dual<BT> as Tensor>::Array::from_fn(|input_coordinate| {
                    columns[input_coordinate][output_coordinate]
                })
            });

        TensorOver::<HomOf<BT, FT>, Jet<𝒞, S, N>>(
            TensorProductArray(rows, PhantomData),
            PhantomData,
        )
    }
}

impl<𝒞: Cat, F, BT, FT, const N: usize, S, Route> JetMap<𝒞, BT, FT, N, S, Route> for Along<F, BT>
where
    BT: Vector<F = FT::F>,
    FT: Vector,
    S: Field,
    Route: ConstantRoute<BT::F, Output = S>,
    Jet<𝒞, S, N>: Field,
    JetVectorIn<𝒞, FT, N, S>: Tensor<F = Jet<𝒞, S, N>>,
    JetVectorIn<𝒞, BT, N, S>: Tensor<F = Jet<𝒞, S, N>>,
    JetVectorIn<𝒞, BT, 1, Jet<𝒞, S, N>>: Tensor<F = Jet<𝒞, Jet<𝒞, S, N>>>,
    JetVectorIn<𝒞, FT, 1, Jet<𝒞, S, N>>: Tensor<F = Jet<𝒞, Jet<𝒞, S, N>>>,
    F: JetMap<𝒞, BT, FT, 1, Jet<𝒞, S, N>, ː<JetLayer<𝒞, N>, Route>>,
{
    fn jet_at(&self, input: JetVectorIn<𝒞, BT, N, S>) -> JetVectorIn<𝒞, FT, N, S> {
        #[allow(type_alias_bounds)]
        type OuterScalar<𝒞: Cat, S, const N: usize> = Jet<𝒞, S, N>;

        let nested_input =
            JetVectorIn::<𝒞, BT, 1, OuterScalar<𝒞, S, N>>::from_fn(|coordinate| {
                Jet::from_parts(
                    input[coordinate],
                    [Jet::from_parts(
                        Route::constant(self.direction[coordinate]),
                        [S::zero(); N],
                    )],
                )
            });

        let nested_output: JetVectorIn<𝒞, FT, 1, OuterScalar<𝒞, S, N>> = <F as JetMap<
            𝒞,
            BT,
            FT,
            1,
            OuterScalar<𝒞, S, N>,
            ː<JetLayer<𝒞, N>, Route>,
        >>::jet_at(
            &self.f, nested_input
        );

        JetVectorIn::<𝒞, FT, N, S>::from_fn(|coordinate| nested_output[coordinate][1])
    }
}
