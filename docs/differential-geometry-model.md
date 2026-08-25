# Diffable's model of differential geometry

Diffable models differential geometry as a hierarchy of progressively stronger
structures. Each trait adds a specific piece of geometric data to the structure
inherited from the trait below it. Supplying operations with the required Rust
types is not, by itself, sufficient to establish that structure: the operations
are also required to satisfy the mathematical axioms attached to the trait. Those
axioms are expressed as `check_*` methods, whose universal validity certifies that
the supplied operations have the relationships claimed by the abstraction.

The central hierarchy is:

```text
Point
  ↓
Chart<P, V>
  ↓
ExpMap<P, V>
  ↓
TangentBundle<P, V>
  ↓
Connection<P, V>
```

`Point` supplies the underlying space. `Chart<P, V>` equips that space with local
coordinates in a model vector space `V`. `ExpMap<P, V>` strengthens those charts by
identifying the coordinate origin with a base point and radial lines with geodesics.
`TangentBundle<P, V>` requires such a centred exponential chart at every point, so
`V` serves uniformly as the model for every tangent fibre. `Connection<P, V>` then
lifts this tangent-chart structure to finite jets, making it possible to transport
and compare derivative information between tangent fibres.

Metric structure enters through a nondegenerate form on the common model vector
space. Parallel transport induced by the connection carries that form from one
fibre to another. If every sufficiently small closed-loop holonomy preserves the
form, the transported form is independent of path and therefore defines a metric
tensor field on the manifold. The quadratic geodesic spray determines the unique
torsion-free affine connection compatible with those geodesics; because the metric
is parallel by construction, this connection is the Levi-Civita connection of the
derived metric.

---

## 1. Traits, data, and axioms

The trait hierarchy separates the **data** of a geometric structure from the
**axioms** that make that data an instance of the intended mathematical object.
Rust can require an implementation to provide operations with particular types,
but the type system does not by itself prove laws such as invertibility of chart
maps, centring of exponential charts, or coherence of lifted jet coordinates.

The ordinary trait methods therefore provide the operations from which the
structure is built, while the corresponding `check_*` methods state the laws those
operations must satisfy. `Chart`, `ExpMap`, `TangentBundle`, and `Connection` all
follow this pattern.

Each check is intended to certify one axiom under the assumptions already supplied
by the surrounding hierarchy. A refinement may therefore use the established laws
of its parent traits as hypotheses. The result is an axiomatic tower: once the laws
of one layer hold, the next layer may treat that structure as given and introduce
only the additional data and axioms needed for its own refinement.

---

## 2. `Point`: the underlying manifold

In Diffable, a type represents a manifold and values of that type are points on the
manifold. `P: Point` therefore introduces the manifold `P` without yet requiring any
additional geometric structure on it. Local coordinates, tangent structure,
geodesics, connections, and metric structure are introduced by later refinements.

A chart on `P` will take values in another manifold `V`. The additional bound
`V: Vector` equips that manifold with the vector-space structure required of a local
model. Once tangent structure is introduced, the same `V` is used to model each
tangent fibre `TₚM`.

---

## 3. `Chart`: local coordinates and chart coverage

`Chart<P, V>` introduces local coordinates on `P`. Its essential operations are

```text
to_local  : P ⇀ V
to_global : V ⇀ P
chart_at  : P → C
```

where `C: Chart<P, V>` is itself the chart type.

The important point is that a value of `C` represents one chart, while the space of
all values of `C` represents the atlas. The operation `chart_at` is therefore not a
convenience for searching an externally stored atlas: it is part of the structure of
the atlas itself. For every point `p ∈ P`, `chart_at(p)` must return a chart whose
domain contains `p`.

Consequently, the covering condition for an atlas is built directly into the trait:

```text
∀ p ∈ P,  p ∈ dom(chart_at(p)).
```

There is no separate collection of charts which must later be proved to cover the
manifold. A `Chart<P, V>` implementation supplies a chart at every point, so the
manifold is covered by charts by construction. The axiom checks certify that the
objects returned by `chart_at` really have the local-coordinate behaviour claimed by
the trait.

For a particular chart `c: C`, `to_local` maps manifold points into the model space
and `to_global` maps coordinates back to the manifold. `to_local` may fail outside
the domain of `c`. `Chart::Global` records whether `to_global` is known to be total;
otherwise its coordinate domain may also be a proper subset of `V`.

At this stage a coordinate `v ∈ V` is simply a local description of a manifold
point. The origin of `V` has no distinguished geometric meaning yet, and straight
lines in `V` have not yet been declared to represent geodesics.

### 3.1 The atlas and inverse axiom

`check_local_inverse` tests both the covering obligation of `chart_at` and the local
inverse law. Given an arbitrary `p ∈ P`, it first obtains

```text
c = chart_at(p).
```

The check fails if `c.to_local(p)` is undefined. Thus universal quantification of
`check_local_inverse(p)` establishes that every point is contained in the chart
selected for it. If `v = c.to_local(p)`, the same check then requires

```text
c.to_global(v) = p.
```

or equivalently

```text
to_global(to_local(p)) = p
```

for the chart chosen at `p`.

The result is an atlas whose covering map is explicit and whose selected local
coordinates round-trip to the point they describe. Later refinements may therefore
assume, for every `p`, the existence of a valid local chart containing `p` without
introducing a separate atlas object or a separate covering proof.

---

## 4. `ExpMap`: charts centred on geodesics

`ExpMap<P, V>` refines a chart by giving its origin and radial directions geometric
meaning.

An exponential chart has a base point `p`, and the coordinate origin represents
that point:

```text
expₚ(0) = p.
```

A vector `v ∈ V` is then interpreted as an initial tangent direction at `p`, and the
radial curve

```text
γₚ,ᵥ(t) = expₚ(tv)
```

is the geodesic through `p` with initial velocity `v`.

This is the first point at which the model vector space is used as tangent data.
The construction proceeds from exponential/geodesic structure toward a connection;
it does not begin by taking Christoffel symbols as primitive input.

### 4.1 Exponential-map axioms

The `ExpMap` checks make the distinguished origin, base point, and inherited chart
structure agree.

`check_base_point_is_origin` verifies that the base point has local coordinate zero.
`check_preservation_of_origin` verifies the corresponding round trip through the
chart. `check_chart_at_base_point` verifies that asking the atlas for a chart at the
base point yields one centred there.

After these laws hold, later code may use the chart origin, the chart base point,
and the exponential image of zero as equivalent descriptions of the same geometric
point.

---

## 5. `TangentBundle`: a tangent chart at every point

`TangentBundle<P, V>` makes the exponential-chart construction available at every
point of the manifold.

For every `p ∈ P`, `chart_at(p)` is required to be centred at `p`:

```text
chart_at(p).to_global(0) = p.
```

The model space of that centred exponential chart is therefore identified with the
tangent fibre at `p`:

```text
V ≅ TₚM.
```

Thus every point has a tangent chart, and every `v ∈ TₚM` determines the geodesic

```text
t ↦ expₚ(tv).
```

### 5.1 `check_universal_centring`

`check_universal_centring(p)` certifies that the exponential chart returned by
`chart_at(p)` is actually centred at the requested point. Together with the
inherited `Chart` and `ExpMap` laws, this establishes the tangent-space structure
used by every later differential construction.

At this point the manifold hierarchy has established:

- local coordinates on the manifold;
- exponential charts with distinguished origins;
- a centred exponential chart at every point;
- a model tangent space `TₚM ≅ V` at every point; and
- geodesics given by radial lines in those tangent charts.

The traits above are generic over the model type `V`, but their meaning depends on
`V` actually carrying the vector-space structure used by the charts and tangent
fibres. Now that `TangentBundle` has fixed the geometric role of `V`, the algebraic
and flat-geometric laws required of that model space can be stated without using
notions that have not yet been introduced.

## 6. The model vector space

Every manifold chart in this hierarchy takes values in a common model space `V`.
`V` is not an informal coordinate container: it is itself an axiomatically specified
vector space, and in Diffable it is also a `TangentBundle` with its canonical flat
geometry. These two structures must agree.

### 6.1 Scalars and coordinate algebra

A tensor has an associated scalar field `V::F`. The field hierarchy supplies the
addition, multiplication, additive and multiplicative identities, inverses, and
distributive laws used by all coordinate calculations. These laws are themselves
axiomatically tested by the corresponding algebraic `test_*` suites.

`Tensor` then gives `V` a finite coordinate representation over that field. It
specifies the number of coordinates, the underlying array representation, the
preferred scalar-action hand, and the constructor `from_fn`. It also fixes the
canonical evaluation pairing between a tensor and its dual.

The ordinary additive operations on tensor types are not arbitrary implementation
choices. `impl_vector_ops!` defines addition, subtraction, negation, zero, and the
available scalar action coordinate by coordinate. For example, addition is

```text
(u + v)ᵢ = uᵢ + vᵢ,
```

and, for a right-handed vector,

```text
(v a)ᵢ = vᵢ a.
```

For a left-handed vector the scalar order is reversed. Because these operations are
fixed coordinatewise, the ordinary vector-space or module laws reduce directly to
the corresponding laws of the scalar field. Associativity and commutativity of
vector addition, existence of zero and additive inverses, distributivity of scalar
multiplication, and compatibility of successive scalar actions are therefore not
independent geometric assumptions.

`Vector` is the refinement of `Tensor` for which the elected scalar action actually
exists. This distinction matters over noncommutative scalar systems and inside the
tensor algebra: a tensor product may remain a valid tensor after balancing has
consumed every exposed scalar action, in which case it is not a `Vector`.

### 6.2 The vector-space test suite

`test_vector!` composes the inherited algebraic and geometric obligations rather
than attempting to restate the whole definition of a vector space in one test.
It includes the additive group tests, which universally check identity,
associativity, and inverses, and it includes the `TangentBundle` tests established above.

Two additional checks certify the special global geometry of a vector space.
`check_global_chart` requires that the canonical chart contain every pair of points:
there are no chart singularities or finite injectivity radius. In other words, the
model vector space is globally coordinatised by itself.

Because `Vector` is itself a `TangentBundle`, its flat linear structure must agree
with the tangent-bundle laws established above. `check_global_geodesic_scaling`
certifies the resulting theorem that radial geodesics in the canonical vector-space
chart scale exactly with their parameter. If `v` is a displacement from `p`, then
for every scalar `t`,

```text
to_local(to_global(v t)) = to_local(to_global(v)) t.
```

This is not an independent definition of `TangentBundle`; it is the specialization
of that structure to a vector space whose exponential map is ordinary translation.
Straight lines are therefore globally defined and their parameter agrees exactly
with scalar multiplication in `V`.

This makes `V` a particularly rigid model for tangent fibres: its algebraic laws
come from the certified field and coordinatewise vector operations, while its chart
and geodesic laws identify that algebra with the flat differential geometry used as
the local model of the manifold.

## 7. `Form`, `Sesquilinear`, and `Nondegenerate`

The differential-geometric construction also assumes a form on the model vector
space. `Form` supplies the lowering map

```text
♭ : V → V*
```

and defines the associated pairing by

```text
⟨u, v⟩ = pairing(u, v♭).
```

The form hierarchy again separates supplied data from certified laws.
`check_dot_agrees_with_pairing` verifies that the public form operation agrees with
the canonical tensor/dual pairing, while `check_translation_invariance` verifies
that the induced quadratic form depends only on displacement in the flat model
space.

`Sesquilinear` adds the algebraic laws needed for a Hermitian or symmetric pairing:
Hermitian symmetry, additivity, and scalar linearity in the first argument are
checked directly. Conjugate-linearity in the second argument then follows from
those laws and Hermitian symmetry. Over a field fixed by its involution this
specialises to the symmetric bilinear case used by pseudo-Euclidean spaces.

`Nondegenerate` requires the lowering map to be invertible. Its `check_isomorphism`
verifies that `flat` and `sharp` are mutual inverses, giving the musical
isomorphism

```text
V ≅ V*.
```

For the pseudo-Riemannian theorem proved below, the relevant model-space form is
symmetric, bilinear, and nondegenerate. It may be positive-definite or indefinite.
No positive-definiteness assumption is needed for the construction of the
Levi-Civita connection.

The important structural point is that this is one fixed form on the common model
space `V`. A separate metric tensor is not chosen independently at every manifold
point. The manifold metric will be obtained by transporting this model-space form
between tangent fibres using the connection.

The hierarchy has now established both sides needed for that construction: a
centred tangent chart modelled on `V` at every manifold point, and a certified
nondegenerate form on `V`. What remains is to lift the tangent charts so that they
act coherently on derivative information.

---

## 8. Jets and higher tangent information

Diffable represents finite derivative data by truncated Taylor jets. An order-`N`
scalar jet is conceptually an element of

F[ε] / (εᴺ⁺¹),

and an order-`N` vector jet is the corresponding tensor-valued object.

A jet therefore contains a value together with finite derivative information. In
Diffable's storage convention the higher slots are Taylor coefficients, so for a
curve `x(t)` the order-two coefficient is `x''(0) / 2!`, and similarly at higher
orders.

The type

```text
Tangent<P, V, N>
```

pairs a manifold point with an order-`N` tangent jet. It is the first lifted object
on which a `Connection` acts.

Higher-order jets are useful for much more than computing Christoffel symbols. They
allow ordinary differential programs to be differentiated through manifold
operations, and they provide the data required by Diffable's prolongation and
parallel-transport machinery.

The crucial semantic requirement is that the order-`N` versions are not unrelated
implementations. They must form one coherent tower. This is enforced explicitly by
the truncation axiom below.

---


## 9. `Connection`: the lifted tangent charts

A `Connection<P, V>` is a `TangentBundle<P, V>` together with two primitive
operations:

```rust
fn tangent_to_local<const N: usize>(
    base: Tangent<P, V, N>,
    local: Tangent<P, V, N>,
) -> Option<JetVector<..., V, N>>;

fn tangent_to_global<const N: usize>(
    base: Tangent<P, V, N>,
    coordinate: JetVector<..., V, N>,
) -> (P, JetVector<..., V, N>);
```

These are intended to be the jet-valued analogues of `Chart::to_local` and
`Chart::to_global`.

The trait intentionally does **not** take Christoffel symbols as primitive data.
The primitive choice is how tangent charts act on derivative information. From
that lifted chart behaviour Diffable can derive:

- geodesic acceleration in a fixed observing chart;
- the symmetric Christoffel operation;
- prolongations of tangent bundles;
- parallel transport; and
- ultimately the metric field induced from the model-space form.

Because arbitrary Rust functions can be written for these two methods, their type
signatures alone do not guarantee that they describe a mathematically meaningful
connection. The six checks below provide the missing laws.

---

## 10. The six `Connection` axioms

The `Connection` abstraction requires six check methods representing five
conceptual kinds of law. The two chart-agreement checks are the two directions of
one order-zero anchoring condition.

### 10.1 `check_tangent_to_local_agrees_with_chart`

At order zero, lifting a manifold point into the tangent machinery and converting
it to local coordinates must agree with the underlying ordinary chart.

If

x = to_localₚ(q),

then the constant lifted point at `q` must map to the constant jet with value `x`.

This prevents `tangent_to_local` from describing an unrelated zeroth-order
geometry.

### 10.2 `check_tangent_to_global_agrees_with_chart`

Conversely, an order-zero local tangent coordinate must reconstruct the same point
as `Chart::to_global`, with zero higher tangent information.

Together the two agreement checks say:

> the lifted chart covers the actual tangent chart already supplied by the
> `TangentBundle` hierarchy.

They anchor the higher construction to the manifold rather than allowing an
independent order-zero map to hide inside `Connection`.

---

### 10.3 `check_tangent_isomorphism`

Order-zero agreement is not enough. At order three and above one could otherwise
modify one lifted direction by an arbitrary higher-order transformation while
leaving all zeroth-, first-, and second-order geometry unchanged.

Therefore, at every finite order `N`, `tangent_to_local::<N>` and
`tangent_to_global::<N>` must be mutual inverses on the domain of the local chart.

Writing

Lᴺ = tangent_to_localᴺ,

Gᴺ = tangent_to_globalᴺ,

the axiom is

Lᴺ ∘ Gᴺ = id

on all local coordinates, and

Gᴺ ∘ Lᴺ = id

wherever `Lᴺ` is defined.

Without this law, converting

```text
global jet → local jet → global jet
```

could silently alter higher derivative information. The two methods would then not
constitute a lifted chart at all.

The axiom still permits a simultaneous change of higher-jet coordinates.
If `Hᴺ` is an invertible higher-order coordinate transformation, replacing

Lᴺ ↦ HᴺLᴺ,

Gᴺ ↦ GᴺHᴺ⁻¹

preserves inversehood. Such a transformation is a change of presentation of the
higher jet fibre, not a change of the underlying affine connection.

---

### 10.4 `check_truncation_coherence`

The generic `const N` interface must describe one higher-order object, not an
unrelated implementation for each value of `N`.

Let

τ(N→M) : Jᴺ → Jᴹ,    M ≤ N,

be jet truncation. The global lifted chart is required to commute with truncation:

(idₚ × τ(N→M)) ∘ Gᴺ = Gᴹ ∘ τ(N→M).

In words:

> applying an order-`N` lifted chart and then forgetting derivatives above order
> `M` must give exactly the same result as forgetting those derivatives first and
> using the order-`M` chart.

Because `Lᴺ` and `Gᴺ` are already certified to be inverse isomorphisms, it is
sufficient to test truncation on one direction. Truncation coherence of the inverse
follows algebraically.

### Proof

Assume

Lᴺ = Gᴺ⁻¹,    Lᴹ = Gᴹ⁻¹

and

τ Gᴺ = Gᴹτ.

For an element `y` in the domain of `Lᴺ`, write `x = Lᴺ(y)`. Then `y = Gᴺ(x)`, so

τ y
= τ Gᴺ(x)
= Gᴹ(τx).

Applying `Lᴹ` gives

Lᴹ(τ y)
=τ x
= τ Lᴺ(y).

Hence

Lᴹτ = τ Lᴺ.

Thus one truncation check plus lifted isomorphism certifies the whole tower.

This is a coherence law, not a geometric choice. It prevents `::<3>`, `::<4>`, and
so on from independently redefining the lower derivative slots.

---

### 10.5 `check_quadratic_geodesic_acceleration`

The inherited `ExpMap` structure supplies geodesics, but an arbitrary geodesic
spray need not come from an affine connection.

Fix an observing chart and a point `p`. For `v ∈ TₚM`, let

γᵥ(t) = expₚ(tv)

and let `xᵥ(t)` be this geodesic expressed in the fixed observing chart. Define

Aₚ(v) = xᵥ″(0).

A general spray may depend nonlinearly on velocity, so its coordinate acceleration need not come from an affine connection. An affine connection requires `A_p` to be a genuine quadratic
map.

The check therefore verifies both

Aₚ(u + v) + Aₚ(u − v) = 2Aₚ(u) + 2Aₚ(v)

and

Aₚ(av) = a²Aₚ(v).

The first is the quadratic parallelogram identity; the second is degree-two
homogeneity. Together they certify quadraticity rather than merely a scaling law.

### From quadratic spray to Christoffel operation

Over the scalar fields used here, where `2` is invertible, polarization defines

Γₚ(u, v) = −½(Aₚ(u + v) − Aₚ(u) − Aₚ(v)).

Because `A_p` is quadratic, `Γₚ` is bilinear and symmetric, and

Aₚ(v) = −Γₚ(v, v).

Consequently the geodesic equation in coordinates is

ẍᵏ + Γᵏᵢⱼ(x) ẋⁱ ẋʲ = 0.

This is precisely the geodesic equation of an affine connection.

The symmetry

Γₚ(u, v) = Γₚ(v, u)

means that the connection represented by the spray is the torsion-free
representative. Torsion is not additional information encoded by a geodesic spray:
the antisymmetric part of the connection coefficients disappears when contracted
with `vⁱvʲ`. Diffable therefore takes the symmetric connection determined by
polarization.

This establishes:

quadratic geodesic spray  ⇒  torsion-free affine connection.

### Why `christoffel_symbols` cannot replace this axiom

`Connection::christoffel_symbols` differentiates a transition and extracts a
bilinear Hessian tensor. The result is bilinear by construction. It therefore
already projects the lifted behaviour onto its quadratic component.

A malformed spray could contain, for example,

A(v) = −Γ(v, v) + Q₄(v),

where `Q₄` is quartic. The Christoffel tensor would still recover `Γ` and would
not detect the quartic residue. The quadratic-spray check is therefore needed: it proves that the Christoffel tensor is a complete description of the
spray rather than merely its quadratic part.

---

### 10.6 `check_holonomy_preserves_form`

The final geometric axiom connects the affine connection to the model-space form.
For the Levi-Civita theorem below, this form is understood to satisfy the
`Bilinear + Nondegenerate` refinements; the check itself can be stated more
generally for `Form`.

For every finite jet order `N`, every base point/fibre under test, every sufficiently
small closed curve `γ` based at that target, and arbitrary vectors `u` and `v`
in the corresponding tangent space, parallel transport must preserve the form:

⟨Pγ(u), Pγ(v)⟩ = ⟨u, v⟩.

The test harness supplies a closed curve based at
the target. Conceptually its precondition is

γ(0) = γ(1) = p.

The check then compares the form before and after transport around the loop.

The check is quantified over every relevant `N`, so the holonomy law applies to the entire coherent jet tower rather than to one fixed derivative order.

This is the path-independence condition required to transport the model-space form consistently onto the manifold.

---

## 11. Deriving the metric tensor from parallel transport

The metric tensor is not independent input.

Let `η` denote the nondegenerate form on the model tangent space at a reference
fibre. Let

Pα : TₒM → TₚM

be parallel transport along a path `α` from a reference point `o` to `p`.
Define

g(p)(u, v) = η(Pα⁻¹(u), Pα⁻¹(v)).


For this to define a genuine tensor field, it must be independent of the chosen
path `α`.

### 11.1 Path independence from holonomy

Let `α` and `β` be two admissible paths from `o` to `p`. Then

ℓ = β⁻¹ ∘ α

is a closed loop based at `o`, and

Pℓ = Pβ⁻¹ Pα.

Holonomy preservation gives

η(Pℓ(x), Pℓ(y)) = η(x, y)

for arbitrary `x,y`.

Taking `x = Pα⁻¹(u)` and `y = Pα⁻¹(v)`, or equivalently rearranging the
isometry identity, yields

η(Pα⁻¹(u), Pα⁻¹(v)) = η(Pβ⁻¹(u), Pβ⁻¹(v)).

Therefore the value of `g(p)(u, v)` is independent of the path.

Hence the connection and the model-space form determine a well-defined field of
nondegenerate bilinear forms on the manifold.

If `η` is positive-definite, `g` is Riemannian. If `η` has indefinite signature,
`g` is pseudo-Riemannian with the transported signature.

The argument is local if the holonomy axiom is only asserted for sufficiently
small loops. It extends across any connected region in which admissible paths can
be composed from such local pieces. A global statement requires the corresponding
global holonomy/path assumptions.

---

## 12. The derived metric is parallel

Let `γ` be a path from `p` to `q`. Because the metric at each point was defined
by parallel transport of the same model form, parallel transport along `γ` is
an isometry between the derived tangent-space metrics:

g(q)(Pγ(u), Pγ(v)) = g(p)(u, v).


This is not an additional axiom; it follows from the definition of `g` and the
path-independence just proved.

To obtain the usual infinitesimal statement, let `U(t)` and `V(t)` be parallel
vector fields along a curve `γ(t)`. The transport-isometry identity says that

g(γ(t))(U(t), V(t))

is constant. Differentiating gives

0 = d/dt [g(U, V)] = (∇γ̇g)(U, V) + g(∇γ̇ U, V) + g(U, ∇γ̇ V).

The last two terms vanish because `U` and `V` are parallel. Therefore

(∇γ̇g)(U, V) = 0.

Since the curve velocity and the initial vectors are arbitrary,

∇g = 0.

Thus the connection is metric-compatible with the metric which it induces.

Notice the direction of dependence:

```text
model-space form + connection
              |
              | parallel transport
              v
       manifold metric g
```

not

```text
arbitrary manifold metric g + arbitrary connection
              |
              v
       compatibility check
```

This direction of dependence is part of the `Connection` contract.

---

## 13. The Levi-Civita theorem in Diffable

We can now state the main result.

### Theorem

Let `P` be a manifold carrier with model tangent space `V`. Assume:

1. the inherited `Chart` axioms hold;
2. the inherited `ExpMap` axioms hold, so radial tangent lines are geodesics;
3. the inherited `TangentBundle` axioms hold, so every point has a centred
   exponential tangent chart;
4. `tangent_to_local` and `tangent_to_global` agree with those charts at order
   zero;
5. the lifted local/global maps are mutual inverses at every finite order;
6. the lifted maps are coherent under truncation;
7. the geodesic acceleration is quadratic in tangent velocity; and
8. the model tangent space carries the required nondegenerate symmetric bilinear
   form; and
9. parallel transport around every admissible sufficiently small closed loop
   preserves that form.

Then, on every connected region covered by these assumptions:

- the lifted geodesic spray determines a torsion-free affine connection `∇`;
- parallel transport of the model-space form defines a well-defined
  pseudo-Riemannian metric tensor `g`;
- `∇ g = 0`; and therefore
- `∇` is the Levi-Civita connection of `g`.

### Proof

By assumptions 1–3, every point `p` has an exponential tangent chart and radial
lines `t ↦ tv` define the geodesics through `p`.

Assumption 7 says the coordinate acceleration map of these geodesics is quadratic.
Polarization therefore yields a unique symmetric bilinear Christoffel operation
`Γₚ`, and the geodesic equation is the geodesic equation of the affine
connection determined by `Γ`. Symmetry of `Γ` gives zero torsion.

Assumptions 4–6 ensure that the arbitrary-order lifted maps describe a
coherent jet-valued lifting of the same tangent-chart structure: they cover the
ordinary charts, are inverse descriptions at each finite order, and agree under
forgetting higher derivatives. Thus the connection machinery is attached to the
manifold supplied by the weaker hierarchy rather than to an unrelated or
order-dependent lifted structure.

By assumptions 8–9, the model form is nondegenerate and symmetric bilinear, and
closed-loop parallel transport preserves it.
Therefore transporting that form from a reference fibre to any other fibre is
independent of the chosen admissible path. This defines a nondegenerate metric
tensor field `g`.

By construction, parallel transport is an isometry for `g`. Differentiating this
identity along arbitrary curves shows `∇ g = 0`.

The connection is therefore both torsion-free and metric-compatible with a
nondegenerate symmetric bilinear metric tensor. By uniqueness
of the Levi-Civita connection,

∇ = ∇ᴸᶜ(g).

QED.

---

## 14. Higher-jet presentation versus geometric information

The axioms intentionally do not require one unique syntactic representation of
all higher jet coefficients.

Suppose `Hᴺ` is a coherent family of invertible transformations of the higher jet
fibres which is identity through the orders carrying the affine connection data.
Then the simultaneous change

Lᴺ ↦ HᴺLᴺ,

Gᴺ ↦ GᴺHᴺ⁻¹

preserves lifted inversehood. If the `Hᴺ` commute with truncation, it also
preserves tower coherence.

Such a family is analogous to changing coordinates on the higher-order jet fibre.
It need not represent new differential geometry.

Diffable therefore does not need an axiom saying that every coefficient is written
in one privileged Taylor-coordinate presentation merely for its own sake. What is
required is that:

- the presentation covers the correct underlying chart;
- its two directions are inverse;
- different finite orders are coherent; and
- the second-order spray and induced transport satisfy the geometric laws.


---

## 15. `MetricTensor` is an optimisation, not new geometry

`MetricTensor<P, V>` refines `Connection<P, V>` by supplying

```rust
fn g(&self, target: V) -> TensorProduct<Sinister<Dual<V>>, Dual<V>>;
```

This should not be interpreted as granting the implementation a second independent
choice of metric.

The metric tensor is already determined by the model-space form and connection:

gₚ=Pα \*η.

A valid `MetricTensor` implementation supplies a more efficient pointwise
calculation of that same tensor.

This matters because reconstructing the metric by parallel transport every time a
vector is lowered or a covector is raised may be unnecessarily expensive. If a
closed-form metric tensor is known, lowering can instead contract directly with
`gₚ`, and raising can contract with its inverse.

Thus the two execution paths are semantically equivalent:

```text
Connection only
    model form
        |
        | parallel transport
        v
      gₚ
        |
        +--> lower / raise

Connection + MetricTensor
    supplied fast evaluation of the already determined gₚ
        |
        +--> lower / raise
```

The category/context machinery uses this distinction for compile-time
specialisation. A connection constructively known to carry the metric feature can
select the explicit tensor implementation; an ordinary connection uses the
transport-derived implementation. The result is an optimisation choice with no
additional geometric degree of freedom.

Consequently, any testing law for `MetricTensor` should be understood as an
**agreement** law with the connection-derived metric, not as a second compatibility
axiom defining another geometry.

---

## 16. Parallel transport as a linear operator

Diffable represents parallel transport along a curve by an endomorphism

Pγ : TₚM → T_qM,

rather than by an operation which transports only one vector at a time.

This is the natural object mathematically and computationally. Once the transport
operator has been computed, it can be applied to any number of vectors or
covectors, composed with transport along another segment, and used directly in
holonomy calculations.

For a closed loop at `p`,

Pγ : TₚM → TₚM

is the holonomy operator tested by `check_holonomy_preserves_form`.

The induced metric construction depends only on this operator-level fact: path
independence is precisely the statement that changing the transport path changes
the identification of a fibre only by a form-preserving holonomy transformation.

---

## 17. What torsion means in this model

A geodesic spray cannot see the antisymmetric part of arbitrary affine connection
coefficients. In coordinates,

Γᵏᵢⱼ ẋⁱẋʲ = Γᵏ₍ᵢⱼ₎ ẋⁱẋʲ,

because `ẋⁱẋʲ` is symmetric.

Therefore infinitely many affine connections differing only by torsion can share
the same parametrised geodesics.

Diffable's `Connection` abstraction does not attempt to encode that extra torsion
choice. Once a quadratic spray has been certified, polarization returns the unique
symmetric bilinear Christoffel operation. The represented connection is therefore
torsion-free by construction.

This is not a hidden additional theorem supplied by holonomy. It is the convention
which turns spray data into an affine connection.


---

## 18. Local and global scope

Most of the construction is inherently local.

Charts are local, `to_local` may fail outside an injectivity domain, and the
holonomy check is stated for sufficiently small loops. The theorem should therefore
be read first as a theorem on each connected region over which the required charts
and transport paths are admissible.

Global conclusions require the corresponding global hypotheses:

- enough admissible paths to connect the region;
- composition and reversal of transport paths;
- holonomy preservation for the loops needed to compare competing paths; and
- whatever completeness assumptions are required by the particular manifold.

Nothing in the local Levi-Civita argument requires geodesic completeness.
Diffable's `Chart::Global` separately records when a coordinate/global operation is
known to be total.

---

## 19. Related geometric traits

`Interval`, `PseudoRiemannian`, and `Smooth` describe nearby parts or alternative
presentations of the same geometric picture, but they are not prerequisites of the
`Connection` theorem proved above.

`Interval` supplies a signed scalar separation between manifold points.
`PseudoRiemannian<V>` relates that operation to the quadratic form in exponential
coordinates: where the logarithm is defined, the signed squared interval agrees
with the tangent-space form. This connects an existing point-separation operation
to the metric geometry derived from the tangent structure.

`Smooth<V>` provides a more intrinsic presentation of smooth manifold structure
from which the explicit chart and tangent hierarchy can be obtained. `Connection`
is stated against `TangentBundle<P, V>` because its primitive operations act on a
specific point carrier and model tangent type.

These traits describe related structure without changing the assumptions used to
construct the affine connection and its transported metric.

---

## 20. Relationship to the category/context layer

The differential-geometric theorem above is expressed operationally by ordinary
Rust traits. Diffable's category machinery serves a different purpose: it records
which mathematical theory a concrete implementation has been admitted into and
retains dependent structural information needed for theorem selection.

In this construction, the important example is the distinction between an
ordinary `Connection` and a connection carrying the `MetricTensor` optimisation.
The contextual ontology can constructively distinguish those two cases and select
the appropriate `MusicalRegion` implementation without unstable Rust
specialisation or runtime feature checks.

The category layer therefore does **not** replace the geometric axioms. Instead it
lets the compiler transport evidence about which implementation strategy is valid
once the mathematical structure has been supplied.

The roles are therefore distinct:

```text
ordinary traits
    executable mathematical operations

check_* axioms
    certification that the operations satisfy their theory

category contexts
    retained compile-time evidence used to select and transport theorems
```

---

## 21. Summary of assumptions and consequences

The hierarchy can be read as the following ladder.

### `Point`

Provides the carrier `P` of manifold points.

### `Chart<P, V>`

Provides local coordinate maps between `P` and the model coordinate object `V`.

### `ExpMap<P, V>`

Interprets radial lines through the local origin as geodesics and identifies the
origin with a chart base point.

### `TangentBundle<P, V>`

Provides such a centred exponential chart at every point, allowing `V` to represent
`TₚM` for arbitrary `p`.

### `Connection<P, V>`

Lifts those tangent charts to arbitrary finite jets.

Its axioms certify:

1. **order-zero local agreement** — the lifted local map covers the ordinary
   chart;
2. **order-zero global agreement** — the lifted global map covers the ordinary
   chart;
3. **lifted isomorphism** — local/global lifted coordinates are inverse at every
   order;
4. **truncation coherence** — all finite orders are one coherent tower;
5. **quadratic geodesic acceleration** — the spray is the spray of a
   torsion-free affine connection; and
6. **form-preserving holonomy** — the model form descends consistently under
   parallel transport.

### `Form` / `Nondegenerate`

Supplies the bilinear or sesquilinear geometry on the model tangent space. A
nondegenerate symmetric/bilinear form gives the usual pseudo-Riemannian setting;
positive definiteness further specialises this to the Riemannian case.

### Derived metric tensor

Parallel transport of the model form defines `gₚ` independently of path because
closed-loop holonomy preserves the form.

### Derived Levi-Civita connection

Quadraticity gives a torsion-free affine connection, while the transported metric
is parallel by construction. Therefore the connection is the Levi-Civita
connection of `g`.

### `MetricTensor<P, V>`

Adds no new geometry. It supplies an efficient direct evaluation of the metric
which the preceding construction has already determined.

The core result may therefore be compressed to:


centred exponential tangent charts
+ coherent invertible jet lift
+ quadratic geodesic spray
+ nondegenerate model form
+ form-preserving holonomy

⇒

pseudo-Riemannian metric tensor g
+ its Levi-Civita connection ∇

This is the mathematical contract implemented by Diffable's differential
geometry hierarchy.
