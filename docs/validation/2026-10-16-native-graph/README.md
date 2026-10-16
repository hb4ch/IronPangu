# Full-model graph milestone and matched OmniInfer reference — 2026-10-16

Same checkpoint `/models/92B`, devices 0–7 on `7.156.99.58`, TP8/EP8, batch one,
context 512, max output 32, greedy generation and top-five diagnostics. Each
measurement contains the same three prompts repeated twice, 66 output tokens and
60 decode intervals. All native output records match the pinned resident results
exactly; all reference token sequences match. This milestone **does not yet match
OmniInfer's optimized graph performance**.

| Execution | Mean TPOT | Tokens/s |
|---|---:|---:|
| Previous native prepared eager | 1,281.69 ms | 0.780 |
| Native full token graph | **76.02 ms** | **13.15** |
| OmniInfer eager | 147.75 ms | 6.77 |
| OmniInfer full decode graph | **18.12 ms** | **55.20** |

Native median 75.89 ms, empirical p95 77.23 ms. The remaining gap is 4.20× versus
the optimized reference. The comparison uses same-machine sequential runs rather
than a randomized paired experiment. Native timestamps are flushed output receipt
times; reference timestamps are offline engine.step output receipt times. Both
exclude first-token loading/prefill/preparation and include output delivery, but
native starts a fresh process per request while reference uses a warmed persistent
engine with prefix caching disabled. Native graph capture finishes before the
first output for these prompts. This is not an HTTP, concurrent or long-context test.

## Implementation and verification

A generic prepared-context capture/replay owner retains graphs and rejects device
pool growth during capture. One stream-local preparation completion precedes
capture; replay introduces no device-global fence or per-operation host wait.
Normal completion remains at the explicitly permitted request-output boundary.

Token ID, position, sequence length, causal indices and rotary constants use
stable device buffers, refreshed outside replay. Embedding selection and KV-write
indexing now consume device indices, so replay does not retain an old token or
cache offset. Immutable weights, recurrent state, intermediate buffers, descriptors,
workspace and foreign compiled-kernel metadata outlive the graph. The expanded
DSL graph still determines all 2,109 operations.

The final six runs have 48 passing case/rank audits: zero activation host transfers,
zero allocations/completions inside generated-token steps, bounded output bytes,
and graph replay recorded on every rank. Before capture qualification, the changed
dynamic-input path also passed a complete 17-token eager output check. The first
full-model graph check passed another 17-token exact-record comparison. All 42
compiler tests pass locally. C++ and NPU execution are Docker-only.

`INFERFABRIC_DECODE_GRAPH=1` enables the native graph path; absence retains the
prepared eager oracle. `benchmark-native-tpot.py --decode-graph --resident-audit`
enables and verifies graph execution. Capture failure is explicit, with no hidden
host execution fallback. Generic static physical arena bounds and broad graph
shape/context qualification remain open.

## Reference identity and reproduction

OmniInfer commit `0c7e75a7b206550229210ebf16399f08a88625bc`; vLLM 0.25.1 commit
`752a3a504485790a2e8491cacbb35c137339ad34`. Both custom-operator environments must
be sourced; omitting them caused an initial missing-symbol startup failure, which
is excluded from measurements. The successful optimized reference log confirms
FULL decode capture completed in approximately 40 seconds. Configuration is
preserved in `scripts/benchmark-omni-reference-config.json`; TP8, EP enabled,
max_num_seqs=1, max_num_batched_tokens=512, memory utilization=.5 and disabled
prefix caching are explicit in the reference runner.

Inside the configured Docker workspace, run the reference sequentially with the
native benchmark on the same cards:

```sh
bash scripts/benchmark-omni-reference.sh --checkpoint /models/92B \
  --cases decoder-resident-qualified/results.json --output NEW_OMNI_GRAPH
# Add --eager for the eager reference.
python3 scripts/benchmark-native-tpot.py --binary native-build/openpangu_decode \
  --plan resident-dsl-plan --checkpoint /models/92B \
  --cases decoder-resident-qualified/results.json --output NEW_NATIVE_GRAPH \
  --resident-audit --decode-graph --repeats 2
```

Native commands require the CANN and both omni operator environment scripts plus
`HCCL_OP_EXPANSION_MODE=HOST`. Reference runs are offline benchmarks only; no
Torch/OmniInfer dependency is introduced into native serving.

Raw output intervals/sequences and summaries are in `native-fullgraph-tpot/`,
`omni-eager-tpot-v3/` and `omni-graph-tpot/`. Native per-rank audits accompany every
run. Native binary SHA-256:
`433dc69561f19c69065b4dc87740ae28d7728276bca1b9281bee2870ceedd670`.
Execution program remains
`46fa67de15080a6ccf370ea852f2d7a6744b532603bdf73ded7f59c34a082513`.

Next priorities are fewer materialized copies/casts and less redundant expert
work inside the captured device graph. The graph removes host submission cost;
it does not make its existing device work free.
