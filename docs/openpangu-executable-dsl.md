# OpenPangu 92B executable DSL

The model is defined in [`examples/openpangu/92b-tp8-ep8.inferfabric`](../examples/openpangu/92b-tp8-ep8.inferfabric).
It defines the complete 46-layer model using six reusable DSL
modules: `TPLinear`, `PlacedLinear`, `DenseFFN`, `RoutedMoE`, `MLAAttention`, and `DecoderBlock`.
A compile-time fold instantiates the blocks, choosing dense/MoE branches and
attention windows. Scoped weights resolve checkpoint bindings without repeated
full paths. See [module syntax and expansion rules](dsl-modules.md).

The compiler expands this source into 1,699 primitive operations. The initial
module migration preserved all 2,109 operations of the qualified flat graph;
the current placement leaves one gather and 92 explicit all-reduces per token. Shared experts are replicated
as in the matched OmniInfer configuration, while MLA query projection emits only
the rank's six heads. `mla_query(input_layout="rank_local")` consumes that local
projection; the original replicated layout remains validated and supported.
Attention output MoME channels remain local, and output/down projections use
`tp_columns` plus explicit reductions. The generic initialization executor packs
strided checkpoint columns; no model-specific loader or runtime weight-name rule
selects the placement.
These choices are explicit in the expanded graph and viewer. There is no contract selector that invokes a C++ model
forward function. C++ supplies registered operator implementations; the DSL
supplies operation order, edges, parameters, collectives, routing, residual
connections, state instances and output. Expanded SSA is generated output.

```inferfabric
let hidden = fold i in range(0, 46) carry(x = streams) {
    let window = choose(eq(mod(i, 3), 0), 2048, 512);
    let block = DecoderBlock(x, child(w.layers, i), i, window);
    yield block;
};
```

For each MoE block the graph separately describes routing, owner-local paired gate/up
projections, fused quantized SwiGLU/down, weighted expert reduction, FP32
all-reduce, the shared expert's three projections and activation, and BF16 add.
`{expert}` in a binding is expanded over the router's selected experts at runtime
and over the local rank's parameter ownership during initialization planning.
`expert_linear(paired_weight=...)` binds both projections explicitly and produces
packed expert rows with one input quantization and one grouped matmul. The two-input
`expert_swiglu_down` consumes those rows directly. The original separate-projection
form remains validated and executable. The fused activation/down kernel preserves the qualified W8A8 rounding boundary.

The explicit `cache_norm_rope_append(latent, rotary, sinks, weight=..., eps=1e-5,
theta=6400000)` operation accepts 512 latent values, 64 unrotated key values and
128 normalized sinks. It fuses RMS normalization, half-rotary and the paged cache
write, returning a single-consumer cache handle. Its gamma binding is loaded by
the generic initialization planner. The physical plan records BF16 rotary
coefficients and the device position control; this differs numerically from the
separate FP32 rotary contract. The separate operations remain supported.

## Compilation and initialization

Both the DSL and compiled Rust `ModelDefinition::Execution` produce the same
validated `execution::Graph`. No compiled author gets to bypass validation.
The compiler rejects unknown operators/attributes, shape/type mismatches,
forward references, dead state effects, mismatched expert-route provenance,
unpaired mHC/cache handles and unsupported backend specializations.

`inferfabric-compiler::token::initialization_plans` visits graph operators and
builds their parameter requirements. It reads the safetensors inventory directly;
it does **not** run the old OpenPangu layer binder or infer TP placement from
names such as `q_b_proj`. Projection `placement` explicitly selects replicated
weights, contiguous output-row TP slices or strided input-column TP slices. MLA absorption/expansion contracts
specify head sharding. Expert projections bind only the owning rank's experts.
Only referenced tensors are loaded; unused indexer/MTP tensors are omitted.
Shapes, dtype, conversions and source ranges are validated before publication.

The result contains:

| Artifact | Meaning |
|---|---|
| `execution.json` | Complete dataflow with instance paths, input IDs, attributes, source spans and module call stacks |
| `execution.dot` | Graphviz graph of every operation and dependency |
| `execution.html` | Offline hierarchical viewer: model overview, expandable layers/modules, validated types, bindings and dependency navigation |
| `physical.json` | Verified conservative physical inspection plan: implementation choices, rank bindings, serial dependencies, host lifetimes and explicit unresolved requirements; not a native executable |
| `execution.bin` | Versioned `IFTOKN01` program consumed by the native executor |
| `rank-N.init.json` | Readable permanent initialization IR |
| `rank-N.bin` | Executable `IFINIT01` initialization program |
| `plan.json` | Bundle metadata and topology |

The native executable requires `execution.bin`; it has no fallback model loop.
Before ACL initialization it checks every kernel contract and all eight ranks'
parameter bindings. During execution it follows the program, verifies result
shapes/finite tensor values, and releases intermediate host values at last use.
State is keyed by graph-node identity, so repeated weight bindings do not alias
MoME, mHC or KV state accidentally. Kernel scratch and device state still use
the existing backend allocations; this is not a new device arena optimizer.

## Build and run

Cross-compile Rust locally in WSL using [the build guide](build.md):

```sh
bash scripts/cross-compiler.sh
```

Transfer `.deploy/inferfabric-compiler` and the DSL source into the remote Docker workspace.
Inside Docker:

```sh
.deploy/inferfabric-compiler compile-token \
  examples/openpangu/92b-tp8-ep8.inferfabric /models/92B dsl-plan --allow-stale-index-size
bash scripts/qualify-openpangu-decoder.sh /models/92B dsl-plan decoder-dsl-qualified
```

Use the stale-index option only for the already-audited checkpoint whose index
has stale aggregate size metadata. It does not suppress shape or shard checks.
`compile-token` also accepts a compiled model artifact from the same authoring IR. It publishes a complete new bundle through a sibling-directory rename and refuses to overwrite an existing bundle. The older `export_openpangu_load_plan` example delegates to this same compiler API for token graphs. `plan-openpangu` remains an audit command and rejects executable graphs rather than dropping their computation.
C++/CANN builds and NPU execution remain inside Docker.

## Current backend boundary

Collective precision can be declared explicitly:
`allreduce(x, dtype = "bf16")` or `allreduce(x, dtype = "f32")`.
Omitting `dtype` preserves FP32 reduction. Input conversion, collective payload,
output storage and physical numerical contract follow that declaration; other
dtypes are rejected by both authoring and native validation. The canonical 92B
DSL retains FP32: the BF16/AIV experiment did not improve on the qualified HOST
path. See [collective qualification](validation/2026-10-16-collective-contracts/README.md).

`ascend_token_v1` is the qualified TP8/EP8, BF16/W8A8 correctness backend:
2560 hidden, 48 MLA heads, 512 latent, 128+64 query dimensions, 128 learned sinks,
4 mHC streams, 3-tap MoME, 256 experts/top-8, and vocabulary 151552. These are
explicit operator attributes/contracts, validated against the selected kernels.
Unsupported dimensions fail rather than silently selecting different math.

The graph declares a **512-token context**, one sequence, one token per invocation.
At this bound all causal keys fit both the SWA window (512) and sparse candidate
budget (2048); no learned-indexer selection is needed. It does not claim general
long-context sparse attention. The diagnostic runner uses greedy decoding.
Full-model ACL graph capture/replay is implemented and qualified with device-resident
activations. Continuous batching, Rust serving integration for this program format,
long-context indexing and production performance remain
separate milestones. Existing kernel plugin ABI support remains available.

## Qualification

Explore the [hierarchical plan viewer](validation/2026-10-16-reusable-dsl/execution.html)
or see [generation and navigation instructions](plan-viewer.md).

[Initial reusable-module NPU qualification](validation/2026-10-16-reusable-dsl/README.md): the original 107-line source expanded to the same 2,109 operations; all eight initialization programs and the complete three-prompt results were byte-identical to the flat-DSL baseline. The report includes expanded JSON with source mappings and a Graphviz diagram.

[Full NPU evidence and plan-control experiment](validation/2026-10-15-dsl-execution/README.md): all 33 tokens across three prompts match the pinned offline reference. Recorded results, including logits/top-five values, are identical to the preceding native decoder.
