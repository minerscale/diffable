# Contexts, inclusion, and dependent projections

The category submodule models a finite nominal fragment of mathematical structure at the type level. Its purpose is not to replace the ordinary Rust trait hierarchy. Ordinary traits remain the operational interface: `Field`, `Tensor`, `Vector`, `Manifold`, and so on describe what values can do. Contexts describe a particular mathematical interpretation of those values and retain the evidence needed to navigate dependent structure.

The core operations are deliberately small:

- `ι` includes a Rust carrier into its distinguished, richest known context.
- `Ⱶ` asks an existing context for a weaker theory without reconstructing the object from scratch.
- `π<Name>` follows a named dependent edge and returns its stored role, child carrier, and exact child context.
- `Model<𝒞, X>` is derived notation for "include `X`, then view that included context as theory `𝒞`."

A concrete `Binds<Name, Role, Value, Context>` is therefore more than an associated-type equality. It is a finite dependent edge. The parent stores both the child value and the context in which that child is known, and `π` returns the stored context verbatim.

## The full-threading experiment

An experimental branch threaded a context parameter through essentially the entire mathematical trait hierarchy. Instead of only having ordinary judgements such as

```text
T: Tensor
F: Field
V: Vector
```

many traits became context-indexed judgements, either directly (`Trait<C>`) or through a parallel proof trait (`TraitIn<C>`). The intended benefit was very strong: the same rich context would accompany every inheritance edge, allowing arbitrary context-sensitive specialization anywhere in the hierarchy.

The experiment worked. It demonstrated that Rust can encode that design and that a rich context can be preserved through long chains of weakening. It also exposed the cost of doing so.

The implementation became substantially larger and more indirect. Traits which own ergonomic methods or associated types often had to remain context-free so Rust could resolve ordinary expressions unambiguously, while a second `*In<C>` trait carried the contextual judgement. More importantly, compiler performance deteriorated dramatically. On the development machine, a clean build reached roughly 2.5 minutes with the ordinary trait solver and roughly 30 seconds with the next trait solver. The latter was a major improvement, but still slow enough to make interactive LSP use impractical.

That cost would be defensible if full hierarchy threading were necessary for the dependent behaviour we care about. The experiment showed that it is not.

## Where the dependency actually lives

Consider a tensor whose scalar field is `f64`. The tensor theory only requires its scalar associated object to be a `Field`, but `f64` has a richer canonical interpretation as `Real`.

The important fact is not that every trait between `Real`, `Field`, `Tensor`, and `Vector` carries one generic `C`. The important fact is that the *dependent edge* from the tensor to its scalar stores the richer child context:

```text
parent context
    |
    `-- tensor::F
          role    = Real
          value   = f64
          context = Real context of f64
```

A consumer which only requires `tensor::F: Field` may still accept this edge because the stored `Real` role satisfies the weaker `Field` requirement. Projection nevertheless returns the exact stored Real context. Role may weaken; stored context does not.

This capability already exists in the category machinery through `Binds`, `RoleSatisfies`, `ChildContext`, `π`, and `Ⱶ`. It does not require `Tensor<C>`, `Field<C>`, or a parallel `TensorIn<C>` hierarchy.

This suggests a cleaner interpretation: **contexts live primarily on contextual objects and the dependent functors/projections between them, not on every ordinary Rust trait judgement.**

The ordinary trait hierarchy remains the computational and mathematical API. The category graph becomes a separate language for saying that a particular object has been admitted under some context and that a named dependent projection carries a particular child context.

## Reflection versus inclusion

There are two legitimate ways to construct a dependent edge, and they should not be conflated.

`BindsReflected<Name, Role, Value>` is role-local. It means: interpret `Value` under exactly `Role`, using the ordinary `Reflect<Role>` implementation. This is the correct fallback for blanket reflection. For example, `Tensor` guarantees only `T::F: Field`; it does not guarantee that an arbitrary user-defined field has a distinguished `ι` inclusion from which a richer role can be inferred. Consequently the blanket `Reflect<Tensor>` implementation must remain able to use the ordinary Field reflection.

`BindsIncluded<Name, Value>` is stronger. It means: the child already has a distinguished `ι` inclusion, so store that entire included context and its actual root role. If the parent requires a weaker role, nominal weakening proves that the richer edge is admissible without erasing the richer child evidence.

A third case remains explicit `Binds`: a parent may possess a context that is richer or more specific than anything recoverable from the child carrier alone. In that case the parent should store exactly the context it owns.

This gives a simple rule:

> Never reconstruct a dependent child context when the construction already owns the context that should travel along the edge.

Reflection is for obtaining an interpretation from a Rust bound. Inclusion is for choosing the distinguished richest interpretation of an ordinary carrier. Projection is for retaining an already-owned dependent context.

## Inclusion is a boundary operation

`ι` is still important. It is the canonical bridge from the context-free Rust world to the contextual ontology. This is especially visible in APIs such as automatic differentiation, where the public call intentionally carries no explicit context. A canonical inclusion is needed to choose the relevant contextual interpretation.

Once code is already operating inside a context tree, however, dependent constructions should normally project or preserve contexts rather than call `ι` again. Re-inclusion can throw away information which exists only because of the path by which the object was reached.

The same principle applies to arrows. Ordinary `Ob<C>` admission begins with the endpoint's canonical `ι` context, but the resulting proof exposes that chosen context. `BindsTyping` then stores the endpoint contexts explicitly. Higher arrow constructions can therefore retain the contexts of lower-dimensional endpoints rather than re-canonicalizing their Rust carrier types.

## What remains intentionally context-free

The failed experiment is not an argument against contexts. It is an argument against making context an index of every theorem in the ordinary hierarchy.

Traits such as `Monoid`, `Group`, `Ring`, `Field`, `Tensor`, and `Vector` should remain ordinary Rust interfaces unless a concrete use case demonstrates that their *operations themselves* must vary with context. Context-sensitive reasoning is expressed separately by quantifying over a context and following its dependent edges.

Likewise, abstractions with useful non-semantic roles remain useful. `EvaluableAt`, for example, deliberately centralizes the large proof obligation behind differentiation so compiler failures point at a comprehensible diagnostic boundary. `Model` remains useful mathematical notation for the composition `ι` followed by `Ⱶ`.

The result is less ambitious than a fully context-indexed trait hierarchy, but it is also simpler, faster, and closer to the dependency actually present in the mathematics:

```text
ordinary Rust object
       |
       |  ι  (only when entering contextual semantics)
       v
contextual object
       |
       |  π<Name>  (dependent projection)
       v
contextual child object
```

That is the design boundary this module should preserve.
