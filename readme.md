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
| `traits::LieGroup` | A manifold with a smooth group structure |
| `traits::Euclidean` | Flat Euclidean space R^N; the canonical model for local coordinates |
| `traits::Metric` | A notion of distance on a manifold |

## Implementations

- `coords::Coords` — the canonical Euclidean space R^N
- `hypersphere` — hyperspheres S⁰, S¹, S³ as Lie groups with geodesic structure

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
