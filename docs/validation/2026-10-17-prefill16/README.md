# Sixteen-token prefill with a matched reference budget

Date: 2026-10-17. **Experimental; full output qualification fails.** A sixteen-token
packed prefill variant reduces measured 4,088-token TTFT from 38.05 to 25.68 seconds
while retaining device residency and a frozen workspace. All seven tested cases
match the independent reference through EOS. The 529- and 3,072-token cases differ
in forced continuation after EOS. No default model, admission limit or production
qualification is changed by this experiment.

## Correctness and reference sensitivity

The same full 46-layer TP8/EP8 learned-index C8 model runs with two resident slots
and sixteen-token chunks. The offline OmniInfer reference uses token budget 16,
one sequence, eager execution, C8 and no prefix caching. The
[reference](reference-budget16.json) contains all seven exact inputs, outputs and
top-five log probabilities. It does not replace the previous budget-four oracle.

Discovery and workspace preparation each execute the three standard cases plus
the 529-token retrieval case, repeated three times (123 outputs). All 99 standard
tokens match. The 529-token case emits `7319` and EOS exactly, but its eighth forced
token is 148900 rather than reference 1114 in every repetition. Both qualification
commands therefore return failure. [Discovery](discovery-comparison.json),
[preparation](preparation-comparison.json).

The independently run frozen long experiment produces 72 outputs across prompts
of 2,049, 3,072 and 4,088 tokens. All repetitions are stable and **all 72 native
tokens are unchanged from the four-token implementation**. Against the matched
budget-16 reference, 2,049 and 4,088 match all eight outputs; 3,072 differs at its
eighth forced token (native 1114, reference EOS 148902). The full comparison fails.
[Long comparison](long-comparison.json).

Changing only the reference prefill budget from four to sixteen changes forced
continuation at 3,072 and 4,088, after their first EOS. This explains why the
location of the reported long-run mismatch changes despite identical native
outputs. It does not prove all remaining numerical differences harmless.
[Both reference outputs and native comparison](chunk-comparison.json). The earlier
short-boundary mismatches before EOS remain an independent accuracy debt.

## Isolated native measurements

Warm loaded model, serial requests, eight forced outputs, three repetitions each.
TTFT includes full native prefill and first sample, excluding model loading, HTTP
and queueing. The reference had finished and the HTTP test had not started.

| Prompt tokens | Chunk-4 mean TTFT | Chunk-16 mean TTFT | Reduction | Chunk-16 mean TPOT |
|---:|---:|---:|---:|---:|
| 2,049 | 19,064.159 ms | 12,808.125 ms | 32.82% | 20.081 ms |
| 3,072 | 28,549.600 ms | 19,192.844 ms | 32.77% | 20.103 ms |
| 4,088 | 38,050.014 ms | 25,684.915 ms | 32.50% | 20.092 ms |

These are separate measured runs, not confidence intervals. Prefill remains too
slow for production 64K. Attention and recurrent work still use per-query calls;
larger matrix batches alone do not remove that work.

Frozen arena is 12,611,207,168 bytes/rank (74,332,672 bytes above the chunk-four
variant); maximum shared workspace remains 130,916,352 bytes/rank. The live process
sample is 32,216 MB/rank, not a peak. All eight [long-run audits](long-audit.json)
report zero activation host transfers, workspace growth, retained-step allocations
and internal completion boundaries. The [preparation audits](preparation-audit.json)
are explicitly unfrozen preparation, not a passing frozen qualification.

## Concurrent 4K HTTP experiment

The vLLM Rust 0.25.1 frontend passes full fixture text, SSE agreement, six
concurrent short requests, queue isolation and disconnect recovery with
`max_num_seqs=2` and `max_num_batched_tokens=32`. A separate mixed test extends the
4,088-token prompt while cancelling another request and completing a short one.
The long case's full eight-token output matches its independent budget-16 oracle;
the short request finishes first. Context overflow is rejected, SIGTERM drains
workers and shared page occupancy returns to zero after 17 admissions.
[HTTP results](http-results.json), [rank audits](http-audit.json).

The longest request uses all 4,096 total tokens, with HTTP TTFT 26.753 seconds.
The concurrently served short request has TTFT 1.508 seconds and streamed text
chunk intervals around 121 ms. These intervals are not necessarily individual
token latencies. Short requests below sixteen tokens also fall back to serial
prefill in this single-chunk-size bundle. Thus the larger chunk improves isolated
long TTFT but does **not** establish acceptable mixed-workload latency.

The scheduler executed 13 mixed ticks, consumed 4,446 input tokens and replayed
the prefill graph 256 times per rank. Peak occupied shared-page payload was
547,266,560 bytes (distinct from total HBM); final occupancy was zero. All eight
HTTP device-residency and frozen-arena audits pass. This qualifies these specific
HTTP output/lifecycle cases only; the failed forced-output cases above remain.

This evidence calls for both faster batched attention and a finite family of
compiled prefill sizes selected using measured decode-latency budgets. Discover
the maximum shared workspace and arena across that family before binding it;
do not introduce live reallocations or activation offload to change variants.

## Reproduction details

Rust compiler/frontend were built locally in WSL. Native execution and the offline
reference ran in Docker `inferfabric-92b` on 7.156.99.58, under
`/data/p00603624/inferfabric/openpangu92`. Use the prior experiment's
[4K model DSL](../2026-10-17-long-index-selection/model.inferfabric) with
`compile-token ... --resident-sequences 2 --prefill-tokens 16 --shared-pages`.

Generate the same seven token-ID cases and run
`scripts/reference-openpangu-decoder.sh --cases batched16-reference-input.json
--output batched16-reference-c8.json --max-model-len 4096
--max-num-batched-tokens 16 --kv-cache-dtype int8_ds_mla`.
The first four reference cases are `batched16-preparation-reference.json`;
the last three are `batched16-long-reference.json`.

Remote artifacts use `batched16-{source,discovery,workspace,preparation,bound}` and
`batched16-long-experiment`. The discovery/preparation commands fail on output
comparison; retain that failure. Workspace planning and arena binding were then
invoked separately against their device audits **solely for this unqualified
experiment**, not as promotion. The long run uses `--permuted-pages
--packed-prefill --experimental-long-context --output-diagnostics` and again exits
nonzero. `scripts/compare-program-reference.py` preserves all mismatch and timing
evidence. No expected native token was substituted into either reference.

| Artifact | SHA256 |
|---|---|
| Native library | 0d9c48e55251441ea1c32b3f8b6332b6e3c594279feba83b7580faf593c75c2a |
| execution.bin | fa56a54d2c5623044a52e97592e9a098ce8af5aaad5d1d29e27f4ebb2622fa53 |
| Rust frontend | 077d9c060b0838a53d68724385092bc15a6d09540bd39179a491e18396c3ddfd |

Next: batch attention with an independent causal oracle, validate full decoder
output, and extend diverse/reference testing through 8K/16K/32K/64K. General
accuracy, long workload pressure, multi-request batched decode and hybrid prefix
checkpoint/restore remain open. Keep default context admission at 512.
