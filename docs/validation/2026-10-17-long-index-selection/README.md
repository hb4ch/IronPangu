# Full 92B selection beyond 2,048 candidates

Date: 2026-10-17. **Experimental, not qualified 64K serving.** Full 46-layer
TP8/EP8 C8 execution recovered access code `7319` and matched the independent
reference through EOS for 2,049-, 3,072- and 4,088-token prompts in all three
repetitions. The forced eight-token continuation matched at the first two
lengths but diverged after EOS at 4,088. The qualification command therefore
returned failure; this record does not relabel it as a pass.

[Comparison](comparison.json) retains all repetitions and separate full-token
and through-EOS results. [Reference](reference.json) includes the exact inputs.
At 4,088 tokens the native seventh output was 148900 versus reference 1114;
both emitted EOS 148902 at the fifth output. All native repetitions were stable.
Earlier short-boundary numerical divergences remain unresolved.

## Measurements and residency

Warm loaded model, serial requests, four-token packed prefill, eight forced
outputs, three repetitions. These timings exclude loading, HTTP and queueing.
The reference on devices 8–15 finished before the measured native run on 0–7.

| Prompt tokens | Mean TTFT | Mean TPOT |
|---:|---:|---:|
| 2,049 | 19,349.869 ms | 20.141 ms |
| 3,072 | 29,033.372 ms | 20.098 ms |
| 4,088 | 38,660.339 ms | 20.134 ms |

Prefill remains too slow for production long contexts. A live process-memory
sample reported 32,120 MB per rank, not a peak. Frozen arena is 12,536,874,496
bytes/rank and maximum shared workspace 130,916,352 bytes/rank. All eight
[rank audits](long-audit.json) report zero activation host transfers, workspace
growth, retained-step allocations and internal completion boundaries. Page
bindings were permuted and decode used ACLGraph. The standard 99-token fixtures
passed discovery, preparation and [frozen execution](short-results.json).

## Larger chunk experiment

A separate 16-token prefill discovery run passed all 99 standard tokens, but
the 529-token retrieval case differed on its eighth forced output after EOS in
all three repetitions. [Evidence](unpromoted-chunk16-comparison.json). It matched
through EOS but did not pass the full comparison. Its interleaved timings also
overlapped reference work and cannot establish a speedup. This variant was not
promoted or used for the table above.

The independent `quant_sparse_indexer_probe` now exercises 2, 4 and 16 query
rows, alternating query directions, causal lengths and noncontiguous pages
through 65,536 keys. All three passed the CPU top-k cutoff, uniqueness and padding
oracle, including 24 ACLGraph replays each. Checked indices were respectively
107,508 / 214,959 / 858,405; workspace was 70,279,680 bytes each. Tied scores may
select different valid indices: this checks semantics, not identical index order.
These operator results do not qualify full-model 64K output.

## Reproduction

[Model](model.inferfabric) changes the selected learned-index DSL's context to
4096. [Generator](make-cases.py) uses `/models/92B/tokenizer.json` and reserves
eight output tokens. Run `scripts/reference-openpangu-decoder.sh` with generated
cases, `--max-model-len 4096 --max-num-batched-tokens 4 --kv-cache-dtype int8_ds_mla`.
Compile locally, then perform discovery, maximum-workspace planning, preparation
and frozen arena binding for two residents and four prefill tokens. Run
`scripts/qualify-program-engine.py` against the reference with `--permuted-pages
--packed-prefill --experimental-long-context --output-diagnostics`.

Remote artifacts use prefix `index-select-`, including `bound`, `qualified`,
`long-qualified` and `reference-c8-chunk4.json`, under
`/data/p00603624/inferfabric/openpangu92` in Docker `inferfabric-92b` on
7.156.99.58. The 16-token experiment uses `index-select16-discovery` and
`chunk16-preparation-cases.json`. Native builds and execution stayed in Docker.

The generic `scripts/compare-program-reference.py` writes mismatch evidence
before returning nonzero. It checks every repetition and rejects missing,
duplicate or out-of-order steps. Native `elapsed_us` values are individual step
durations, not cumulative timestamps; its tests protect that interpretation.

| Artifact | SHA256 |
|---|---|
| Native library | 0cc6b9278955d321b85cde4fe123ff1d46fd7f1911b5270ff16cbacc75b81c0b |
| Four-token execution.bin | 3b0c4789a6743ffdbc5b22e9d4b7dd67651e97b0840f0559945ba178169d1084 |
| Unpromoted 16-token execution.bin | fa56a54d2c5623044a52e97592e9a098ce8af5aaad5d1d29e27f4ebb2622fa53 |

Next: batch sparse-index queries per prefill chunk and update the physical
recipe, invocation count and workspace before binding. Preserve explicit casts,
device residency and graph pointers; compare full-model output before promotion.
Then qualify diverse prompts at 8K, 16K, 32K and 64K, concurrent HTTP workloads,
memory pressure, cancellation/reuse and hybrid prefix checkpoint correctness.
The default native context admission remains 512.
