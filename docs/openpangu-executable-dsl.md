# OpenPangu 92B executable DSL

The model is defined in [`examples/openpangu/92b-tp8-ep8.inferfabric`](../examples/openpangu/92b-tp8-ep8.inferfabric).
It contains the complete 46-layer, 2,109-node token dataflow. There is no contract
selector that invokes a C++ model forward function. C++ supplies registered
operator implementations; the DSL supplies operation order, edges, parameters,
TP collectives, EP routing, residual connections, state instances and output.
The file is expanded SSA so every layer and binding can be inspected directly.
Reusable modules/loops remain a language ergonomics task; no generator is needed
to compile or execute this checked-in source.

An excerpt from the first attention block:

```inferfabric
let l0_qa = linear(l0_an, weight = "model.layers.0.self_attn.q_a_proj", width = 1024, placement = "replicated");
let l0_qc = mome(l0_qa, weight = "model.layers.0.self_attn.qa_conv", taps = 3);
let l0_qn = norm(l0_qc, weight = "model.layers.0.self_attn.q_a_layernorm", eps = 1e-5);
let l0_q_local = linear(l0_qn, weight = "model.layers.0.self_attn.q_b_proj", width = 1152, placement = "tp_rows");
let l0_q = gather(l0_q_local);
```

For each MoE block the graph separately describes routing, owner-local gate/up
projections, fused quantized SwiGLU/down, weighted expert reduction, FP32
all-reduce, the shared expert's three projections and activation, and BF16 add.
`{expert}` in a binding is expanded over the router's selected experts at runtime
and over the local rank's parameter ownership during initialization planning.
The fused activation/down kernel preserves the qualified W8A8 rounding boundary.

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
weights or contiguous output-row TP slices. MLA absorption/expansion contracts
specify head sharding. Expert projections bind only the owning rank's experts.
Only referenced tensors are loaded; unused indexer/MTP tensors are omitted.
Shapes, dtype, conversions and source ranges are validated before publication.

The result contains:

| Artifact | Meaning |
|---|---|
| `execution.json` | Complete typed-checkable dataflow with names, input IDs and attributes |
| `execution.dot` | Graphviz graph of every operation and dependency |
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
bash scripts/cross-openpangu-probe.sh
```

Transfer the resulting exporter and sources into the remote Docker workspace.
Inside Docker:

```sh
.deploy/export_openpangu_load_plan /models/92B \
  examples/openpangu/92b-tp8-ep8.inferfabric dsl-plan --allow-stale-index-size
bash scripts/qualify-openpangu-decoder.sh /models/92B dsl-plan decoder-dsl-qualified
```

Use the stale-index option only for the already-audited checkpoint whose index
has stale aggregate size metadata. It does not suppress shape or shard checks.
The exporter also accepts a compiled model artifact from the same authoring IR.
C++/CANN builds and NPU execution remain inside Docker.

## Current backend boundary

`ascend_token_v1` is the qualified TP8/EP8, BF16/W8A8 correctness backend:
2560 hidden, 48 MLA heads, 512 latent, 128+64 query dimensions, 128 learned sinks,
4 mHC streams, 3-tap MoME, 256 experts/top-8, and vocabulary 151552. These are
explicit operator attributes/contracts, validated against the selected kernels.
Unsupported dimensions fail rather than silently selecting different math.

The graph declares a **512-token context**, one sequence, one token per invocation.
At this bound all causal keys fit both the SWA window (512) and sparse candidate
budget (2048); no learned-indexer selection is needed. It does not claim general
long-context sparse attention. The diagnostic runner uses greedy decoding.
Continuous batching, Rust serving integration for this program format, full-model
ACL graph capture, long-context indexing and production performance remain
separate milestones. Existing kernel plugin ABI support remains available.
