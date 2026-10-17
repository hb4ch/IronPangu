# Reusable token-graph modules

`ascend_token_v1` authoring supports file-local modules, compile-time folds,
compile-time conditionals and scoped checkpoint bindings. These elaborate into
ordinary primitive graph nodes before the existing type/effect checks and
initialization planner. They do not add a runtime interpreter or native model
forward function. Other graph targets and cross-file imports are not yet supported.

```inferfabric
module TPLinear(x, w, width) {
    let local = linear(x, weight = w, width = div(width, 8), placement = "tp_rows");
    let output = gather(local);
    yield output;
}
```

Parameters accept graph values, compile-time scalar values, or weight scopes.
They also accept named records of graph handles, described below.
Arguments can be positional or named; positional arguments must precede named
arguments. Missing, unknown and repeated arguments fail. Each body has immutable
`let` bindings followed by one `yield`. Module arguments form its entire lexical
environment: modules cannot implicitly capture a model's local variables.
Types and shapes are inferred and checked after instantiation. A module is a DSL
body, never a native model selector. No recursive module calls are permitted.

```inferfabric
let w = weights("model");
let hidden = fold i in range(0, 46) carry(x = streams) {
    let layer_weights = child(w.layers, i);
    let window = choose(eq(mod(i, 3), 0), 2048, 512);
    let next = DecoderBlock(x, layer_weights, i, window);
    yield next;
};
```

`range(begin, end)` is end-exclusive. The index is a compile-time nonnegative
integer; the carry is a graph value. Each iteration receives the previous
iteration's result. A zero-iteration fold returns its initial value. The body
has its own immutable locals; it cannot leak bindings into the containing scope.
Loops are unrolled, with at most 4,096 iterations per loop, 10,000 emitted nodes,
100,000 elaboration work steps and depth 64. The work bound also covers loops
that emit no nodes. Recursive declarations, including indirect cycles, fail.

## Multiple results and loop carries

`record(name = handle, ...)` groups named graph handles at compile time. Modules
can return records, other modules can receive them, and a fold can carry them.
Dotted access selects fields; nested records are allowed. For example:

```inferfabric
module Advance(state, w) {
    let hidden = mome(state.hidden, weight = w, taps = 3);
    let next = record(hidden = hidden, shared = state.shared);
    yield next;
}
let initial = record(hidden = embedded, shared = embedded);
let state = fold i in range(0, 2) carry(s = initial) {
    let next = Advance(s, child(w.layers, i, "conv"));
    yield next;
};
let combined = add(state.hidden, state.shared);
```

Here `state.shared` still refers to the original embedding node: there is no
record operation, allocation, cast or copy in the expanded execution graph.
This supports a module returning both its updated hidden state and a shared
auxiliary graph result, such as sparse indices, once those primitive operations
are implemented. It does not itself implement an indexer or prefix snapshot.

Each record has 1–64 unique named fields containing graph handles or records;
scalars and weight scopes are rejected. Field names cannot contain dots. Missing
fields and using a record directly as a primitive input fail. Fold iterations
must preserve the complete field structure; graph tensor types and widths remain
subject to normal downstream type checking. A zero-iteration fold returns the
initial record. Record structures are bounded to 65,536 bytes of structural
description, including nested fields, preventing exponential alias expansion.
The model's final yield must still select a single output graph handle.

`choose(condition, then, else)` evaluates only the selected branch. Conditions
must be compile-time booleans; tensor-dependent control flow is rejected.
`eq`, `lt`, `mod`, and `div` take nonnegative integers. `div` requires exact
integer division; division/modulo by zero fail. This is deliberately a small
constant evaluator, with no host-language evaluation or filesystem access.

`weights("model")` constructs a typed weight scope. Dotted selection such as
`w.self_attn.q_a_proj` extends it. `child(w.layers, i)` adds a constant string
or integer component. Only valid dot-separated components are accepted; path
traversal is rejected. `child(w.experts, "{expert}")` preserves the existing
expert-owner binding placeholder for the expert kernel contract. Passing a tensor
where a weight scope or static attribute is needed fails. Scopes resolve to
concrete parameter names before initialization planning.

## Identity and inspection

Each call and iteration has a hygienic instance path, for example
`hidden.i12.next.attention.query.local`. Primitive graph node names use these
paths. Two calls sharing a weight scope still have different state identities;
MoME, KV and mHC state remains keyed by graph-node identity in the native backend.
Weight sharing and state sharing are separate concepts.

The expanded `execution.json` retains all primitive operations, concrete weight
names, input IDs, and attributes. Each node also has a `source` object containing
its declaration's UTF-8 byte range and a `frames` array with the module/fold call
sites and instance paths. Ranges refer to the original DSL input. `execution.dot`
uses the same expanded node names and edges. No source/module lookup is needed
at execution time; `execution.bin` still contains only primitive instructions.
Compiled-language authors can supply the same graph IR; source metadata is
optional and does not affect code generation.

The initial 92B migration was checked against the saved qualified expanded graph:
operation order, input IDs, output, numerical attributes and concrete bindings
must match exactly. Hygienic node names and source metadata may differ. Weight
initialization binaries must remain byte-identical, followed by full NPU output
comparison. This is stronger than checking only graph size or a few operators.

The migration passed the [full 92B NPU qualification](validation/2026-10-16-reusable-dsl/README.md),
with all 33 output tokens matching and complete recorded results byte-identical
to the flat-DSL baseline. The report includes the expanded JSON and Graphviz graph.

Subsequent placement optimization uses `PlacedLinear` to choose replicated shared
experts and rank-local MLA query projection. The current graph has 1,931 nodes;
its changed weight placement is independently qualified against the reference,
then graph replay is compared exactly with the new eager results. The historical
migration fixture remains intact; placement changes do not claim byte-identical
initialization or logits against that fixture.
