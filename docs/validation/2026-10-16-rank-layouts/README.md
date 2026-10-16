# Rank-local queries and replicated shared experts — 2026-10-16

**29.9477 ms/token**, 33.3916 tokens/s, versus the previous 32.3856 ms/token
and the matched OmniInfer full-graph reference of 18.1171 ms/token. This removes
7.5% of latency; a 1.65× gap remains. Median is 29.8626 ms and empirical p95 is
30.6347 ms across 60 intervals, six runs and three prompts.

## Explicit DSL placement

The reusable `PlacedLinear` module selects replicated or TP-row linear operations
at compile time. The two dense FFNs retain TP8; the 44 shared experts are
replicated, matching OmniInfer's default `enable_share_expert_tp=false`.
This removes three gathers per MoE layer. The MLA query projection emits its
six local heads directly; `mla_query(input_layout="rank_local")` removes the
gather-and-slice pair in all 46 layers. The replicated query contract remains
supported and tested. Unknown layouts and inconsistent input widths fail before
device initialization.

The expanded graph has 1,931 nodes, 99 gathers and 44 all-reduces, compared with
2,109 nodes, 277 gathers and 44 all-reduces before this change. Weight placement,
shapes, scope hierarchy and dependencies are visible in the deterministically
generated [plan viewer](execution.html) and [expanded IR](execution.json).
No host activation fallback or device-global fence was introduced.

## Qualification

All 33 greedy tokens including EOS match the pinned OmniInfer reference.
Selected-token log-probability maximum absolute differences are 0.030587,
0.004586 and 0.210205 for arithmetic, geography and Chinese. Changed matrix
shapes can select different tiling; this does not claim bitwise logits against
the previous placement or broad numerical equivalence.

All six graph runs match their independently reference-qualified eager records
exactly: 66 complete output records and 48 rank residency/replay/allocation/
completion audits. Activations stay on device, generated steps allocate no new
device buffers, and completion is limited to the existing output boundary.
All 42 compiler and 16 DSL tests pass locally. Rust is cross-compiled locally;
native build and full model runs execute inside Docker. Native contract tests
accept both the new 1,931-node and previous 2,109-node plans on all eight ranks,
including nine negative contract checks.

Same checkpoint `/models/92B`, devices 0–7, TP8/EP8, batch one, context 512,
max batched tokens 512, greedy decoding, logprobs five and the reference prompts.
Loading/prefill and first output are excluded. Native uses fresh processes;
OmniInfer uses a warmed persistent engine. This is a short-context offline
comparison, not concurrent-serving or long-context performance. Raw runs and
binary/plan/qualification hashes are recorded in `summary.json` and `runs.json`.

## Matched device traces and remaining work

`device-profile.json` records the native graph's two decode iterations on all
eight devices. `omni-device-profile.json` summarizes two decode iterations from
the same TP8/EP8 reference (184 mHC pre and 92 total attention calls per rank).
The reference trace uses the arithmetic prompt; native uses Chinese. Both are
short decode contexts, but timings are diagnostic, not paired TPOT samples.
Reference profiling is enabled with `benchmark-omni-reference.py --profile`;
its summary marks latency as non-comparable. Export each rank's trace with
`msprof --export=on`, then run `summarize-cann-kernels.py --root TRACE
--iterations 2 --output SUMMARY.json`.

Observed differences guide subsequent milestones:

- Batch MLA heads: native runs six query matmuls, six value matmuls and six query
  rotary calls per layer; OmniInfer uses one per head group.
- Fuse expert gate/up projections: native runs 132 grouped matmuls per token;
  OmniInfer runs 88, with fewer quantization and packing operations.
- Fuse device routing and weighted reduction: OmniInfer uses MoeInitRoutingV3
  and MoeFinalizeRoutingV2. Native primitive composition adds TopK, gathers,
  repeats, FP32 casts and a costly small-row ReduceSum.
- Fuse KV norm/rotary/cache update and select attention by its explicit contract.
  OmniInfer uses a fused KV operator and a mix of sink and sparse attention;
  native currently uses separate operations and sparse attention throughout.
- Plan column-sharded output projections and reductions. Native still has 99
  logical gathers plus 44 all-reduces; OmniInfer's trace has 93 all-reduces and
  one gather per token. This needs generic initialization-plan column slicing,
  local tensor contracts and explicit reduction semantics, not name-based loading.

Profiler durations include overlap, rank waiting and collection overhead; they
must not be added to predict TPOT. Unprofiled full-output qualification remains
the acceptance gate for every optimization.
