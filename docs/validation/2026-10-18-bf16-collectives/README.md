# Explicit BF16 collectives and authored serving budgets — 2026-10-18

The BF16 collective candidate passes discovery, fixed-workspace and frozen-arena
qualification on the short/8K six-request suite: **39/39 output tokens at each
stage**. It also preserves all 128 reference tokens for the longer 8K answer.
**The 64K extended-answer exact-output gate still fails at step 7.** Both 128-token
native answers are identical to those from the previous FP32 collective plan.
This candidate is not a solution to the remaining numerical disagreement, and
has not replaced the prior qualified HTTP bundle.

## Changes and configuration

`examples/openpangu/92b-tp8-ep8-c8-serving.inferfabric` explicitly sets BF16 on all
92 AllReduce nodes: 46 attention output reductions, 2 dense FFN reductions and
44 routed MoE reductions. The reference implementation passes BF16 projection
outputs into attention/dense reduction, and casts its quantized MoE result to
BF16 before EP reduction. Native previously widened these values for FP32 sums.
The physical program now contains the necessary explicit casts and BF16
collectives; the executor adds no implicit conversion or host fallback.

Ignoring source-location metadata, the two 1,765-node logical graphs differ
**only** by those 92 `dtype=bf16` attributes. `logical-changes.json` records them.
The physical viewer exposes the `ascend_token_v1/allreduce/bf16` numerical
contract and BF16 collective payloads. The native library/probe were unchanged.

The DSL declares three resident slots, 4096 batched tokens and context 65536.
An obsolete single-sequence/token check was removed from shared authoring, DSL
validation and initialization-plan construction. Both the CLI and the compiled
language `write_bundle` API now use authored capacities by default; explicit
CLI resource flags remain overrides. Unsupported physical capacities still fail
validation. This run compiled without `--resident-sequences` or `--prefill-tokens`
and emitted three residents plus eager 4096-token prefill correctly.

## Hardware results and limits

All runs execute in `inferfabric-92b` on 7.156.99.58. Compiler build/crosscompile
runs locally in WSL. The candidate's frozen arena is **37,968,015,872 bytes per
rank**, 41,922,560 bytes (about 40 MiB) below the previous bundle's arena. Shared
workspace remains **159,221,248 bytes per rank**. These are planned allocation
sizes, not a new measurement of total HBM including weights and HCCL resources.

All eight frozen-run audits report zero activation host transfers, rejected
transfers, execution-time device allocations, internal completion boundaries,
workspace growth, transient invocations or retained transient descriptors.
Decode graphs replay and prefill remains uncaptured. Native processes exit zero
for the completed short/8K and extended runs; the extended qualification wrapper
exits one because the exact-output comparison fails. No HTTP extended-output
run was started after that failure.

The extended fixture contains 8,064/65,408-token prompts and two arithmetic peers.
All 272 requested records arrive. The 8K request and peers match; the 64K request
selects space (220) instead of `2` (17) at output index 7, exactly as before.
Do not replace the independent reference or normalize its whitespace to claim
this gate passed. Timing is diagnostic only: 111 size-two pure decode batches
have mean 26.885 ms, and 16 trailing size-one calls have mean 22.517 ms. This is
not a matched performance win, HTTP TPOT, or a qualified production baseline.

The first discovery attempt stopped after warmup with `unregistered named cache
state`: it incorrectly reused bindings from the old physical plan. It produced
no qualified request output. The corrected run regenerated bindings from the
new plan. `qualify-sequence-serving.py` now rejects stale cache names before
loading weights, avoiding that initialization cost on subsequent mistakes.
The failed attempt remains remote as `bf16-collectives8k-discovery`.

## Reproduction and evidence

Remote workspace `/data/p00603624/inferfabric/openpangu92`:

```sh
INFERFABRIC_EXPERIMENTAL_OUTPUT_ROWS=1 INFERFABRIC_EXPERIMENTAL_BULK_COLLECTIVES=1 \
../inferfabric-compiler-authored-budgets compile-token 92b-tp8-ep8-c8-serving.inferfabric \
  /models/92B bf16-collectives-source --allow-stale-index-size --shared-pages
```

Then run the retained corrected-binding script (HCCL 47400), followed by the
extended script (47600). Existing directories intentionally refuse overwrite.
The resulting bundle is `bf16-collectives8k-bound-frozen-plan`.
`hashes.json` pins the compiler, library, probe, execution binary and arena.
Each stage retains raw output records, native process status, exact comparison,
batch traces and compact all-rank resource observations. Full audits remain in
the corresponding remote directories, pinned by SHA-256 in the observations.
The extended batch summary was computed from the failed run afterward, without
repeating NPU execution or changing its failed result.

Local validation: 26 compiler token tests, 20 DSL/authoring tests, 6 CLI tests,
and 3 sequence-comparison/preflight tests pass; the compiler crossbuild passes.
New tests cover authored-budget bundle publication, both authoring routes,
explicit BF16 reduction operands and rejecting stale binding names. These
checks do not close the failed 64K exact-output gate.
