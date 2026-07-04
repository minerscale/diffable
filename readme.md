# Diffable

A differential geometry framework for Rust. Implementing a trait certifies
that a type carries the corresponding mathematical structure — a group
operation, a metric, a smooth atlas — and the trait hierarchy encodes the
dependency order between those structures, so blanket implementations derive
the consequences automatically.

## Structure

The library is organised around a hierarchy of traits that mirror the mathematical structure of differential geometry:

### Foundation — sets, distances, inner products

| Trait | Meaning |
|---|---|
| `Point` | An element of a carrier set — no topology, no smoothness |
| `Scalar` | A real-number type for coordinates (see its doc for the stance on approximate equality) |
| `Metric` | A distance function `d: M × M → R` |
| `InnerProduct` | A bilinear inner product, inducing a norm and metric |

### Charts — local coordinate structure

| Trait | Meaning |
|---|---|
| `Chart<P, V>` | A coordinate chart mapping points of `P` to a Euclidean space `V` |
| `ExpMap<P, V>` | A chart whose coordinate lines are geodesics |
| `TangentBundle<P, V>` | A family of `ExpMap` charts, one centred at each point — the tangent bundle `TM` |

### Smooth structure — self-charting manifolds

| Trait | Meaning |
|---|---|
| `Smooth<V>` | A manifold that charts itself via `exp`/`log`; automatically generates `Chart`, `ExpMap`, and `TangentBundle` |

### Algebra — groups and Lie groups

| Trait | Meaning |
|---|---|
| `Group` | Associative composition with identity and inverses |
| `LieGroup<V>` | A group with a smooth exponential map at the identity; automatically derives `Smooth` via left translation |
| `Quotient<G, H, V>` | A quotient `G/H` by a central subgroup, inheriting Lie group structure |

### Euclidean — flat space

| Trait | Meaning |
|---|---|
| `Euclidean` | Flat space `Rⁿ` — simultaneously an inner-product space, its own tangent bundle, and an additive Lie group |

### Global topology

| Trait | Meaning |
|---|---|
| `Bounded<P, V>` | A `TangentBundle` chart with a bounded domain (signed distance field) |
| `NerveComplex` | A finite cover by `Bounded` charts forming a simplicial complex; computes `π₁(M)` |
| `GroupPresentation` | A group by generators and relations — the output of `fundamental_group` |

### Blanket chains

Implement the left-hand trait, receive the right-hand traits for free:

```text
Smooth<V>              →  Chart<Self, V>, ExpMap<Self, V>, TangentBundle<Self, V>
LieGroup<V>            →  Smooth<V>  →  Chart, ExpMap, TangentBundle
Euclidean              →  Group, LieGroup<Self>  →  Smooth<Self>  →  Chart, ExpMap, TangentBundle
Quotient<G, H, V>      (via macro)  →  Group, LieGroup<V>  →  Smooth  →  ...
```

## Implementations

| Type | Description |
|---|---|
| `Coords<R, N>` | The canonical Euclidean space `Rⁿ` |
| `Sphere<N, V>` | The unit hypersphere `Sⁿ` as a smooth manifold, any dimension |
| `S0<V>`, `S1<V>`, `S3<V>` | Lie group structures on S⁰, S¹, S³ — newtypes of `Sphere` adding group operations |
| `So3<V>` | The rotation group SO(3), the quotient S³/{±1} — a newtype of `S3` |
| `Stereographic<V>` | Stereographic projection charts for spheres |

The newtype layering reflects the mathematical structure: `Sphere` is the
bare manifold (geometry only), `S3` adds the quaternion group operation,
and `So3` adds the antipodal identification. Each wrapper is zero-cost and
peelable — `.0` is the forgetful functor dropping one layer of algebraic
structure.

## Testing

Enable the `testing` feature to access the `test_*` macros, which verify
that your implementations satisfy the mathematical invariants certified by
each trait. The scalar types `R64` and `R32` provide tolerance-based
equality suitable for property testing with floating point.

```toml
[dev-dependencies]
diffable = { version = "...", features = ["testing"] }
```

## License

MIT OR Apache-2.0
