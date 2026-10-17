# Full 92B decoder across SWA wraps

Date: 2026-10-17. **All 72 generated tokens matched the independent OmniInfer C8
reference**, for 529-, 1,024- and 2,040-token prompts, eight generated tokens each,
repeated three times. This is full 46-layer TP8/EP8 execution, including packed
prefill, repeated SWA wraps and resident-state reset between requests.

The prompts put access code `7319` at the start, followed by maintenance entries
and a request to return the code. Both implementations return `7319` then EOS.
The test deliberately generates eight tokens, ignoring EOS, and also matches
the forced continuation exactly. [Comparison](comparison.json) keeps both
through-EOS and full-token results; [reference](reference.json) retains inputs,
outputs and reference top-five log probabilities. This small retrieval suite does
not replace diverse long-context output or the earlier failing boundary cases.

## Implementation and admission

Authoring and physical plan bounds now permit contexts through 65,536, with paged
BF16 windows or learned-index packed C8. The 512-candidate causal shortcut and
row-quantized cache paths reject larger contexts. Compiler manifests explicitly
mark larger contexts unqualified. Shipped model DSL files remain at context 512.

The native engine **rejects contexts above 512 by default**, before loading model
or device state. Qualification requires the exact environment value
`INFERFABRIC_EXPERIMENTAL_LONG_CONTEXT=1`. The engine qualification harness clears
an inherited value unless `--experimental-long-context` is supplied. Absent, `0`
and `true` values were checked and rejected. This opt-in is an experiment switch,
not evidence of production readiness or permission to advertise 64K support.

Local compiler tests cover 513, 2,048, 2,049 and 65,536 plan capacities, rejection
above the upper bound, and rejection of the unsupported causal-C8 long path.
Fourteen compiler token tests and four DSL graph tests passed. Native validation
accepted all eight rank bindings and rejected 36 corrupted contracts.

## Measurements

Warm loaded model, serial requests, four-token packed prefill, eight output tokens,
three repetitions per prompt. TTFT includes the full native prefill and first
sample; it excludes loading and HTTP/queue overhead. TPOT includes each following
sample, including the forced post-EOS continuation.

| Input tokens | Mean TTFT | Mean TPOT | Median TPOT |
|---:|---:|---:|---:|
| 529 | 4,988.950 ms | 19.781 ms | 19.782 ms |
| 1,024 | 9,628.534 ms | 19.897 ms | 19.896 ms |
| 2,040 | 19,304.666 ms | 19.971 ms | 19.955 ms |

TTFT remains too high for production long-context serving. These results do not
claim parity with OmniInfer's production prefill configuration; the reference
used a four-token budget to align arithmetic shapes, rather than optimize throughput.

A live `npu-smi` sample reported **32,074 MB process memory per rank**; this is
a sample, not a measured peak. [Raw sample](npu-smi.txt). Frozen arena:
12,489,607,680 bytes/rank; shared workspace: 130,916,352 bytes/rank. All eight ranks
reported zero activation host transfers, workspace growth, retained-step device
allocations and internal completion boundaries. [Long-run audit](long-audit.json).
The long run used noncontiguous page bindings and exact ACLGraph replay. The 99
standard short-fixture tokens also passed discovery, preparation and frozen
execution under the same 2K compiled plan. [Short results](short-results.json).

## Reproduction and provenance

The [model specification](model.inferfabric) is the selected learned-index C8 DSL
with only model context and plan max-model-length raised to 2048. The
[case generator](make-cases.py) uses `/models/92B/tokenizer.json`. Reference:
`scripts/reference-openpangu-decoder.sh --cases swa-wrap-reference-input.json
--output swa-wrap-reference-c8-chunk4.json --max-model-len 2048
--max-num-batched-tokens 4 --kv-cache-dtype int8_ds_mla`.

Native qualification uses `scripts/qualify-program-engine.py` with the frozen
`swa-wrap-bound` plan, the generated reference JSON as cases, `--permuted-pages
--packed-prefill --experimental-long-context --output-diagnostics`. Run the usual
discovery, maximum workspace planning, preparation and arena binding first; their
short fixtures must pass. Rust was cross-compiled locally in WSL. Native/reference
runs were inside Docker `inferfabric-92b` on `7.156.99.58`, working directory
`/data/p00603624/inferfabric/openpangu92`. Native used devices 0–7; offline reference
used 8–15 and had finished before the measured long native run.

Artifacts use `swa-wrap-` prefix (`source`, `discovery`, `workspace`, `preparation`,
`bound`, `qualified`, `long-qualified`, `reference-c8-chunk4.json`). Sampler
diagnostics are retained in `swa-wrap-long-qualified/output-diagnostics.json`.

| Artifact | SHA256 |
|---|---|
| Native library | 0cc6b9278955d321b85cde4fe123ff1d46fd7f1911b5270ff16cbacc75b81c0b |
| execution.bin | 6781c083d05d761289b70a4d4c51315b6f82c23863e252e9fded0c7498138605 |
| Reference fixture | 38de8f2fe5ca143e36af08eac5607b49738b29bf8f418c0f03b99ec1b6d95b6f |

Remaining gates include real selection beyond 2,048 candidates, diverse/reference
accuracy, larger contexts through 64K, concurrent long HTTP workloads, memory
pressure, faster prefill and hybrid prefix checkpoints. Default admission stays
512 pending that broader qualification; existing short-boundary divergences are
not erased by this successful retrieval experiment.
