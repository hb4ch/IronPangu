# Packed prefill: first native execution variant

Date: 2026-10-17. Native TP8/EP8, OpenPangu V2 92B, C8 DSA cache and
BF16 SWA cache. This qualifies the four-token chunk path within the existing
512-token envelope; it does not qualify 64K or HTTP packed scheduling.

## Execution contract

`compile-token --prefill-tokens 4 --resident-sequences 2` emits an `IFTOKN04`
program containing the resident capacity and exact chunk width. The compiler
accepts widths 1, 2, 4, 8 and 16; only width 4 has the NPU evidence here.
Width 1 retains the existing executable format. Unsupported row-quantized cache
variants fail compilation rather than falling back. The serving manifest records
a distinct numerical class for packed prefill; it does not grant prefix reuse
across classes or hardware qualification by itself.

Each chunk traverses the decoder once. Embeddings, projections, normalization,
mHC, routing, grouped expert matmuls and SwiGLU process token matrices. The first
variant executes attention, query transforms and recurrent convolution in
ordered per-query calls. Per-query causal lengths exclude future chunk rows;
convolution updates the selected resident history in order. The final prompt
row alone is sampled. Short tails use the explicit single-token variant.

The physical recipes and viewer expose the chunk width, shapes, per-query API
repetition and row-assembly device copies. AllGather has an explicit rank-major
to token-major rearrangement. These copies stay on device. This is a bounded
first implementation, not a claim of fused multi-query attention.

Both decode and prefill graphs are warmed and captured before readiness. A
single maximum workspace and frozen arena cover both variants. The new C ABI
validates exact chunk width, vocabulary and continuation position before dispatch
to the rank group; resident selection and resets preserve other requests.

## Correctness and memory evidence

- [Short fixtures](interleaved-results.json): 99 exact expected output tokens,
  nine alternating requests, two resident slots, shuffled per-rank page mappings.
- [Boundary oracle](boundaries-results.json): 336 exact output tokens, 42 requests,
  prompt lengths 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 255, 480 and 504.
  The oracle is the previously qualified serial implementation, not an independent
  long-context reference. Chunk tails, page crossings and resident reuse are covered.
- [Short audits](interleaved-audit.json) and [boundary audits](boundaries-audit.json):
  all eight ranks have zero activation host transfers, rejected transfers,
  workspace growth, retained-step device allocations and internal fences.
- Arena: **12,450,806,784 bytes/rank**, 19,312,064 bytes above the two-resident
  serial arena. Maximum shared workspace: **130,916,352 bytes/rank**.
  HBM snapshots are device-wide and include runtime overhead.

The harness checks the exact packed-graph replay count against the number of
whole chunks in every input, plus one warmup replay. A serial-only run cannot
satisfy the packed qualification flag. Full output matching is mandatory.

## Same-plan timing comparison

Three sequential requests, each 504 prompt tokens and 8 output tokens, on the
same frozen two-resident bundle and native library. These engine timings include
reset/prefill and sampling but exclude model loading, tokenization and HTTP.
All 24 output tokens in each mode match the prior serial oracle.

| Mode | TTFT range | Median TTFT | TPOT range |
|---|---:|---:|---:|
| [Four-token chunks](long-packed-results.json) | 4.278–4.294 s | 4.282 s | 18.60–18.72 ms |
| [Serial prefill](long-serial-results.json) | 9.175–9.214 s | 9.177 s | 18.61–18.75 ms |

Packed prefill reduces median TTFT by about **53.3% (2.14×)** in this case.
This is not an OmniInfer comparison, a 64K result or a throughput saturation test.
The [packed](long-packed-audit.json) and [serial](long-serial-audit.json) audits
pass the same eight-rank frozen-memory and device-residency gates.

## Artifact identity

| Artifact | SHA-256 |
|---|---|
| Execution binary | `9e3cd4108712b0e8c47f1e26e489e71bd0e06351c4911961366a3958e76890db` |
| Native library | `9056cc3bdca21c36b3213cc69c3e753caa6e748af0851eef371062c66a92adce` |
| ABI probe | `7b11de8c272d3709c9687fdae30127b64ebe5a2a7624171c304c6d5321b17683` |

Reproduce with workspace discovery, `plan-shared-workspace.py`, fresh preparation,
and `bind-prepared-arena.py`; run `qualify-program-engine.py` with
`--packed-prefill --interleave --permuted-pages`. The remote bundle is
`packed4-v2-bound`; raw audits are retained beside its qualification directories.
Local compiler tests pass, including binary chunk bounds and materialization
metadata. Strict compiler Clippy remains red on existing `emit` argument-count,
test-module placement and nested-if warnings; it is not claimed green.

## Remaining release work

The Rust HTTP scheduler still uses single-token invocations and rejects the new
multi-variant manifest. Next integrate the chunk API with its token budget and
fairness rules, qualify HTTP cancellation and mixed requests, then increase chunk
efficiency. Dynamic page pools/SWA reclamation, learned DSA, hybrid prefix caching,
stochastic sampling and the 64K correctness/performance/robustness gates remain
required. Neither this evidence nor a larger allocator justifies raising context.
