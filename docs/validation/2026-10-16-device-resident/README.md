# Device-resident 92B execution — 2026-10-16

**Passed:** complete 46-layer OpenPangu V2 92B generation on eight Ascend NPUs
(TP8/EP8), with no executor-owned activation/routing/KV readback. All **33 tokens**
across three complete answers match the pinned reference, including EOS.

Implementation: `7f43818` (resident execution and hard planner constraint),
`58d5df2` (build-enforced transfer guards and native ownership/negative tests).

| Prompt | Complete answer before EOS | Tokens |
|---|---|---:|
| `1+1=?` | `1 + 1 = 2` | 8 |
| `What is the capital of France?` | `The capital of France is Paris.` | 8 |
| `请用一句话介绍北京。` | `北京是中国的首都，拥有悠久的历史、丰富的文化遗产和现代化的城市风貌。` | 17 |

## Evidence and numerical scope

- [Summary](summary.json), [raw outputs](results.json), [artifact hashes](artifacts.sha256).
- [Pinned reference comparison](reference-comparison.json): exact sequences.
- [Previous native-path comparison](native-comparison.json): exact sequences,
  **not bitwise logits**. Maximum selected-logit absolute differences are 9.5,
  5.0 and 9.5; maximum selected-log-probability differences are 0.131915,
  0.01245247 and 0.336834. Device arithmetic replaces host arithmetic and output
  log-softmax now runs in FP32 on NPU. This run does not isolate the contribution
  of each numerical change; broader golden-logit qualification remains open.
- [All 24 case/rank transfer audits](transfer-audits.json): zero unclassified
  transfers, zero activation transfers, zero control-input readbacks. Total
  request-output readback is **19,536 bytes** across all ranks and generations.
- [Current plan viewer](../2026-10-16-reusable-dsl/execution.html): physical mode
  exposes device locations, transfer roles, fixed-slot expert selection and
  unresolved memory/cost details. The viewer is deterministically compiler-derived.

The transfer guard is forced into the execution translation unit before operator
headers, with a compile-time check. Direct `DeviceTensor::upload/read` and async
host copies fail without an explicit boundary role. The NPU ownership probe
checks these failures, shared-buffer lifetime, slice/concat device copies and
BF16 ties-to-even conversion. Host-only negative guard tests also pass.
The ledger covers executor-owned transfers, not vendor-internal control traffic
or initial checkpoint ingestion by the permanent initialization executor.

## Implementation and remaining work

Generic device values replace host vectors; device operators implement slices,
concatenation, repeats, arithmetic, rotary application, collectives, router output
handling, expert selection/reduction and greedy output selection. There is no
activation offload/fallback option. The planner rejects host/opaque candidates and
unclassified host transfers before comparing costs. Activation tracing to host
is explicitly rejected.

The resident EP baseline packs 32 owner-local experts per rank and selects weights
on device for eight fixed route slots. Foreign-owner slots compute and are masked
on device before route-order FP32 reduction. This costs extra bank memory and
redundant compute; sparse grouped dispatch and direct initialization-time bank
packing are needed for production performance.

Immutable checkpoint-parameter transformations still use one-time host C++
preparation. Position/index metadata and position-derived rotary coefficients
are control uploads. Outputs contain only selected token IDs and top-five
request diagnostics; full vocabulary logits stay on device.

This qualifies one sequence, greedy generation, serial token prefill and the
existing context-512 specialization. It does not qualify long-context indexing,
continuous batching, full-model ACLgraph, frozen physical-plan execution, memory
reuse, Rust serving integration or production throughput for this program format.
No existing graph-capture implementation was changed.

## Reproduction

Rust tests, Clippy with warnings denied, and the AArch64 compiler build ran locally
in WSL. C++ compilation and all device execution ran inside Docker
`inferfabric-92b` on `7.156.99.58`. No Torch/Python inference engine participated;
Python is only the offline tokenizer/test driver.

```sh
# Local WSL, using the configured Rust cross-toolchain:
bash scripts/cross-compiler.sh

# Inside the remote Docker workspace after transferring the compiler/source:
.deploy/inferfabric-compiler compile-token \
  examples/openpangu/92b-tp8-ep8.inferfabric /models/92B resident-dsl-plan \
  --allow-stale-index-size
bash scripts/qualify-openpangu-decoder.sh \
  /models/92B resident-dsl-plan NEW_OUTPUT_DIR
python3 scripts/compare-openpangu-decoder.py \
  NEW_OUTPUT_DIR/results.json \
  docs/validation/2026-10-15-openpangu-decoder/reference/results.json comparison.json
# With the same CANN environment:
cmake --build native-build --target resident_transfer_test resident_tensor_probe -j4
native-build/resident_transfer_test
native-build/resident_tensor_probe
```

Choose a fresh output directory. The audited checkpoint still requires the
explicit stale-index-size option. Remote evidence resides at
`/data/p00603624/inferfabric/openpangu92/decoder-resident-qualified`.
The fresh `resident-dsl-plan` and the qualified `reusable-dsl-plan` have identical
execution binaries. The runtime binary hash is
`3e7e99d4f0a41953d32a27f181a842a7af3cc9cabee9ed5bfc242f0e883754dc`.
