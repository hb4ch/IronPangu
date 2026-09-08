# Ascend qualification of physical planning — 2026-09-08

All hardware runs took place inside `ironpangu-npu` on `root@7.156.99.58`, under `/data/p00603624/ironpangu/plan-npu-20260908`. The image ID was `sha256:19cc2ebef117fc6d026d10994ea5c1225459e69318adac058348417ca3747629`. Rust was built/cross-compiled locally in WSL; C++ was compiled in that container with GCC 10.3.1. See [environment.txt](environment.txt) for the exact CANN/driver/device versions and [sha256.txt](sha256.txt) for source, binary and fixture hashes. No model payload is included in this record.

## Scope

The proposed v1 text parser and general Ascend physical-plan backend are still unimplemented. This milestone qualifies the implemented JSON logical/typed/physical planning subset using an explicit native test adapter. It is **not** proof that the new v1 grammar compiles Qwen into an Ascend executable bundle.

The local `export_npu_fixture` Rust example decodes and verifies an `IFPLAN01` CPU plan, preserves its operation order, shapes, arena offsets and state updates, and emits test inputs plus CPU execution references. The C++ `physical_plan_probe` maps add/multiply/ReLU/matmul semantics to ACLNN. Both CPU matmul implementations map to ACLNN Matmul: this is an explicit qualification translation, not execution of a CPU kernel ID on Ascend. The probe accepts trusted exporter fixtures, not arbitrary unverified runtime programs. It does not read DSL source or invoke a planner on device.

All device allocation, tensor creation, ACLNN preparation and workspace sizing happen before warmup/capture. One stable activation arena uses the emitted offsets. The serial stream respects schedule order and reuse dependencies. State commits use separate temporary snapshots for simultaneous updates. Workspace and state snapshots are additional allocations, not silently counted as part of the planned activation arena. No claim is made about unmeasured allocator activity inside vendor libraries.

## Results

Each fixture passed on devices 0 and 1 with three changed/continued invocations in eager mode, then three graph replay passes, resetting state between passes.

| Fixture | Planned calls | Planned activation arena | Checked values per device | Maximum absolute error |
|---|---:|---:|---:|---:|
| Residual + persistent history | 6 | 192 bytes | 96 | 0 |
| 17×19 @ 19×23, ReLU, accumulated history | 3 | 3200 bytes | 9384 | 0 |
| Simultaneous swap of two state arrays | 0 | 0 bytes | 84 | 0 |

The residual case retains constant folding, dead-node removal and arena reuse. Its first output is `[10,13,2,2]`, independently covered by the planner's hand-calculated test. The rectangular case exercises tail dimensions and the planner's blocked-matmul choice. The swap case detects sequential instead of simultaneous updates and qualifies a capture containing only state-copy commands. ACLNN reported zero workspace for these particular shapes. Total: **19,128 output/state values checked**, all exact matches. See the six fixture/device JSON reports.

The existing ACL graph-copy probe also passed on both devices, including changed inputs and invalid-handle/bounds/thread rejection followed by successful graph reuse.

Separately, the current checkpoint DSL and canonical Qwen native block path were run with `/data/p00603624/models/qwen35` on both devices. All 24 layers and 320 weights were covered for token IDs `[1,2,3]`. Eager and graph logits and persistent state were **bitwise equal** at every position. The scalar Rust CPU reference agreed on top-1 tokens `[5328,220,16]`; relative L2 logits error was approximately 1.42%, 1.13%, 1.46%, below the existing 5% gate. See [qwen-device0.json](qwen-device0.json) and [qwen-device1.json](qwen-device1.json), including weight hashes and program identities. This checks the existing optimized Qwen route separately; it does not establish new-DSL Qwen execution or serving readiness.

Local workspace tests and Clippy (`--all-targets`, warnings denied) passed. C++ built with `-Wall -Wextra -Werror`. The final formatted C++ source was rebuilt and all six fixture/device cases and both graph-copy probes rerun.

## Reproduce

From the repository in local WSL, regenerate a fixture from its verified binary and invocation inputs:

```sh
cargo run -p inferfabric-cli --example export_npu_fixture -- \
  docs/validation/2026-09-08-npu-planning/fixtures/residual.ifplan \
  docs/validation/2026-09-08-npu-planning/fixtures/residual-inputs.json \
  .deploy/residual.txt
bash scripts/cross-npu-probe.sh
```

`fixtures/` preserves all three binaries, exported fixtures, input arrays, and the additional tail/swap logical graphs. The residual source is `examples/physical-plan/residual-state.logical.json`. Do not hand-edit fixture offsets or schedules: the C++ reader is a bounded test protocol reader, not the Rust plan verifier. Export rejects outputs that directly expose a state value being updated, since this adapter observes expected values after commit.

Copy the local cross-compiled Rust binary and native sources to a dedicated directory under the remote working directory. Inside Docker only:

```sh
source /usr/local/Ascend/cann/set_env.sh
cmake -S native -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
./build/physical_plan_probe fixtures/residual.txt 0
./build/physical_plan_probe fixtures/tail.txt 0
./build/physical_plan_probe fixtures/swap.txt 0
./build/acl_graph_probe 0
./inferfabric-npu npu-model-probe examples/qwen35-2b-checkpoint.inferfabric \
  /data/p00603624/models/qwen35 build/libinferfabric_acl.so 0 tokens.json qwen.json
```

Repeat for device 1. `tokens.json` is included here. The probe is separate from the production library and vLLM frontend; no serving process was deployed or changed.

## Remaining qualification gates

Implement the v1 parser/elaborator and a versioned Ascend physical bundle with target kernel descriptors before claiming end-to-end new-DSL execution. Native fusion/provenance, per-rank planning, workspace bounds, graph buckets and the general runtime loader still need integration. Triton/TileLang/PTO/custom-op adapters, TP/PD, mixed serving workloads and full-model new-plan execution are not qualified by these tests.
