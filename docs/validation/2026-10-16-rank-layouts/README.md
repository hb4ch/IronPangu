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
