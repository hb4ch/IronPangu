# Model authoring and compiled implementations

There are two equal authoring paths. A versioned DSL parses to a span-bearing AST and elaborates to `inferfabric_authoring::ModelDefinition`. A compiled Rust package implements `ModelAuthor` and returns that same definition. Both undergo the same semantic checks and use the same checkpoint binding, initialization lowering and physical planner. A compiled author can use normal loops/functions; the parser never evaluates host code. Other compiled languages can emit the documented `inferfabric.model` version-1 JSON envelope, but Rust is the provided SDK.

The legacy unversioned `key = value` parsers are removed, including the separate OpenPangu parser. No automatic fallback exists. Checked-in Qwen, OpenPangu and frontend fixtures use `language inferfabric 1;` with separate model and plan declarations. Existing compiled runtime formats are not invalidated just because source syntax changed.

## Supported language subset

See [graph example](../examples/authoring/residual.inferfabric) and [92B definition](../examples/openpangu/92b-tp8-ep8.inferfabric). The CPU graph target supports static f32 inputs, constants, persistent state, add/mul/relu/matmul, named outputs and simultaneous state updates. The CPU planner validates shapes/effects and emits source-independent executable plans. The 92B example now uses `ascend_token_v1`: its complete 2,109-operation computation, state and weight placement are defined in the DSL. See [the executable 92B DSL](openpangu-executable-dsl.md). The older Qwen/OpenPangu contract adapters remain metadata/placement interfaces; the 92B executor requires an explicit execution program.

File-local reusable modules, compile-time folds/conditionals, and scoped weights now elaborate into the token graph with hygienic state identities and source provenance. The 92B source is 107 lines; its expanded plan still contains 2,109 operations. See [module authoring](dsl-modules.md).

The lexer understands quoted strings, comments, lists and calls; declarations and members carry byte spans. Unknown fields, duplicate declarations, invalid versions, unsupported operations and excessive input/nesting are rejected. This is an implemented subset of the larger language design: cross-file module imports, lossless CST/formatting, symbolic shapes, arbitrary foreign operations and general NPU graph lowering remain unsupported. Unsupported constructs fail rather than reverting to the retired parser.

## Compiled-language route

[The compiled model example](../crates/inferfabric-authoring/examples/model.rs) has no DSL dependency. It emits either the same residual graph as the DSL example or an OpenPangu TP8/EP8 placement-only definition:

```bash
cargo run -p inferfabric-authoring --example model -- residual residual.ifmodel.json
cargo run -p inferfabric-authoring --example model -- openpangu92 openpangu92.ifmodel.json
cargo run -p inferfabric-cli -- model-source examples/authoring/residual.inferfabric dsl.ifmodel.json
cargo run -p inferfabric-cli -- plan residual.ifmodel.json residual.ifplan --dump-ir ir
```

The compiled OpenPangu artifact can replace the DSL argument to `export_openpangu_load_plan` and `plan-openpangu`. The Qwen compiler/frontend source readers likewise accept the versioned compiled model artifact. Serialized model definitions are authoring interchange, not executable foreign code. The `openpangu92` compiled example remains placement-only and cannot run the new decoder by itself. A compiled author can instead construct `ModelDefinition::Execution { parallel, graph }`, using the same typed operator IR as the full DSL. Its encoded artifact generates the same initialization/execution programs without parsing DSL.

## Debugging and tuning plans

`plan-debug` disables dead-node removal and constant folding while retaining validation. `plan --dump-ir` records logical, typed, optimized and physical stages. `explain --html` visualizes the actual physical plan. `tune-plan` accepts a map from step name to registered kernel variant, verifies that operation semantics are unchanged, revalidates the whole physical plan and emits a new binary:

```bash
inferfabric plan-debug examples/authoring/residual.inferfabric debug.ifplan --dump-ir debug-ir
# overrides.json: {"projected":"matmul_blocked_f32_v1"}
inferfabric tune-plan debug.ifplan overrides.json tuned.ifplan
inferfabric explain tuned.ifplan --html tuned.html
```

Compiled callers use `compile_with_options`, `override_kernels` and `Executor::bind_implementation`. `CompiledKernel` supplies actual compiled Rust math for a named existing operation; the executor retains input/output shape and finiteness checks and commits state only after successful execution. These callbacks are explicitly installed by trusted compiled code, never loaded implicitly by a plan. Numerical equivalence remains a test obligation. This CPU interface copies inputs and is a debugging/reference seam, not an Ascend performance API.

## Native Ascend C boundary

Model definition must remain independent of implementation language. An Ascend C implementation may replace an operation or a fused subgraph only through the native implementation contract, preserving tensor layouts, effects, workspace, stream ordering and numerical acceptance criteria. The implementation must retain editable source and build provenance, rather than requiring edits to generated code that are lost on recompile. Plan overrides select a separately built implementation; they must not disguise changed model semantics as a backend optimization.

The 92B decoder now binds its dense/shared SwiGLU through the native [kernel ABI](../native/include/inferfabric_kernel_v1.h) and reusable [compiled-kernel loader](../native/include/compiled_kernel.h). The default [C++ implementation](../native/kernels/swiglu_cann.cpp) calls CANN; a separately compiled Ascend C host wrapper can export the same function table. This is an actual model call site, not just an extension mock. The initial hook covers this one operation, not arbitrary replacements throughout the 92B model.

Select an implementation explicitly at process startup:

```bash
INFERFABRIC_SWIGLU_LIBRARY=/absolute/path/to/custom-swiglu.so \
  bash scripts/qualify-openpangu-decoder.sh /models/92B dsl-plan custom-kernel-results
```

The qualifier snapshots and hashes the selected shared object alongside the executable. Parsing a model never loads a shared object. An invalid ABI, contract, target, descriptor, tensor alias or workspace is rejected. The default implementation requires one contiguous BF16 `[T, 2H]` input and `[T, H]` output, with the first half as the SiLU gate and the second half as the multiplier. It does not change the routed-expert fused INT8 activation path.

A custom implementation exports `inferfabric_kernel_v1`, returning the versioned POD function table for `inferfabric.swiglu.bf16.v1` on target `ascend`. `prepare` validates specialization and reports caller-owned workspace; `enqueue` launches on the supplied stream; `release` frees metadata only after the caller fences completion. Prepared instances are single-enqueue. The loader retains the shared object until prepared instances are destroyed. No C++ objects or exceptions cross the ABI. Library code is trusted native code: interface checks do not prove its math or enforce its no-allocation/no-hidden-effects obligations.

This hook enables independent source-level optimization without editing the model definition. Keep its source, build flags/toolchain and output-library hash with test results. A custom implementation still requires independent numerical and capture qualification. General plan-selected native subgraph replacements, hermetic Ascend C/PTO/TileLang/Triton build adapters and arbitrary NPU graph lowering remain work; an environment-selected debug override is not a production implementation-selection manifest. The CPU `tune-plan` command does not select this native library.

## Validation

Current token-graph validation and NPU results are documented in [the executable DSL guide](openpangu-executable-dsl.md). The following records the earlier dual-authoring milestone, before the topology-only 92B example was replaced by an executable graph.

DSL and compiled Rust definitions produce identical model artifacts and executable CPU plans. Tests cover stateful execution, debug planning, implementation overrides, compiled execution callbacks, malformed definitions and explicit legacy rejection. All six Rust frontend tests pass after fixture migration. On the pinned 92B checkpoint, the DSL, compiled author and pre-migration definitions produce byte-identical IFINIT01 programs on all eight ranks. The full 46-layer TP8/EP8 NPU decoder still produces the identical complete three-prompt results JSON (33 tokens including EOS).
