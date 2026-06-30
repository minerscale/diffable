# Diffable

A differential geometry framework for Rust, providing abstractions for smooth manifolds, charts, Lie groups, and tangent bundles.

## Structure

The library is organised around a hierarchy of traits that mirror the mathematical structure of differential geometry:

| Trait | Description |
|-------|-------------|
| `traits::Point` | A point on a smooth manifold |
| `traits::Chart` | A coordinate chart; the space of all charts is an atlas |
| `traits::ExpMap` | A chart with geodesic structure; straight lines map to geodesics |
| `traits::TangentBundle` | A family of exp charts, one per point; the tangent bundle `TM` |
| `traits::LieGroup` | A manifold with a smooth group structure, carrying an exponential map at the identity that automatically generates a full tangent bundle via left translation |
| `traits::Quotient` | A quotient `G/H` of a Lie group by a central subgroup, automatically inheriting Lie group and tangent bundle structure from the parent group |
| `traits::Euclidean` | Flat Euclidean space Rⁿ; the canonical model for local coordinates, carrying a canonical inner product, norm, metric, and tangent bundle |
| `traits::InnerProduct` | An inner product space, inducing a norm and metric |
| `traits::Metric` | A notion of distance on a manifold |

## Implementations

- `coords::Coords` — the canonical Euclidean space Rⁿ
- `hypersphere` — hyperspheres S⁰, S¹, S³ as Lie groups with geodesic structure
- `hypersphere::So3` — the rotation group SO(3), as the quotient S³/{±1}

## Testing

Enable the `testing` feature to access the `test_*` macros, which verify that your implementations satisfy the mathematical invariants certified by each trait:

```toml
[dev-dependencies]
diffable = { version = "...", features = ["testing"] }
```

## Optional Features

| Feature | Description |
|---------|-------------|
| `nalgebra` | Interop with nalgebra's `SVector` and `UnitQuaternion` |
| `testing` | Property-testing macros for verifying trait implementations |
| `all` | Enables all features |
