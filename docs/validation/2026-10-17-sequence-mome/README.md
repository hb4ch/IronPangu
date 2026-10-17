# Sequence MoME: full-model regression

The native 92B TP8/EP8 engine completed cleanly and matched all 123 reference
output tokens across 12 requests. The launcher left `ACLNN_CACHE_LIMIT` unset,
exercising the engine's default of zero before ACL initialization. The engine
rejects an explicit nonzero value for the uncaptured envelope.

Uncaptured prefill now invokes causal convolution once per layer and sequence,
using `[0, actual_query_tokens]` and the absolute chunk start. Decode keeps its
existing single-query graph path. Physical recipes expose this batching and its
control input. Owned contiguous cache pages are coalesced during reset without
clearing unowned pages.

See [raw summary, hashes and per-rank audits](summary.json). Each rank recorded:

- MoME preparations: 2,346, down from 1,366,890 in the row-wise run.
- Only `token_decode` graph, with 112 replays; zero activation host transfers.
- 14 uncaptured forwards / 9,902 query tokens, including two 4K warmups.
- Arena discovery: 38,253,633,536 bytes; maximum workspace: 159,221,248 bytes.

The 529-token fixture's TTFT was 26.26, 22.82 and 7.03 seconds; TPOT was
23.98, 23.65 and 23.68 ms. Total run time was 499.85 seconds. The preparation-count
reduction is established; the noisy TTFT samples do not establish a proportional
speedup. Memory records are runtime HBM snapshots, not continuous peak sampling.

Scope remains limited: prompt lengths 12/15/14/529, each repeated three times,
plus zero-input 4K warmups. This is not a frozen-arena or real 4K/64K qualification.
The next run uses planned shared workspace and actual 2049/3072/4088-token
independent-reference prompts. Production context remains gated at 512.

Local validation: all 21 compiler token-plan tests and four shared-workspace
tests passed. The complete native candidate built inside the remote Docker;
page-reset tests cover byte equivalence, sparse ownership and 64K coalescing.
