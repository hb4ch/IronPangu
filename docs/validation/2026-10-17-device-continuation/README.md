# Complete device continuation copies — 2026-10-17

The native TP8/EP8 92B backend copies complete continuation state between private
resident allocations without host tensor transfers. All ranks preflight before
any copy launches; destination progress is published only after every owning
stream completes. Copies include MoME history, retained BF16 windows, packed C8
payloads and separate learned-index keys/scales. They do not copy logits.

Fresh discovery, preparation and frozen baseline execution each pass all 123
reference tokens. The previous backend's workspace plan was rejected before
inference, as required by its fingerprint contract; that failure is retained.
Fresh workspace is 131,020,288 bytes/rank and arena is 12,620,067,840 bytes/rank.

The copy test uses a 1,024-token boundary, fragmented per-rank bindings and three
long prompts (2,049/3,072/4,088 tokens), each repeated three times. It advances the
source fully before continuing the copied destination. **All 72 destination output
tokens equal the source tokens.** Invalid same-slot, out-of-range, wrong-boundary
and nonempty-destination calls are rejected before worker dispatch, with valid
continuation succeeding afterward.

The 2,049- and 4,088-token cases match the full independent reference. The 3,072
case retains its known post-EOS divergence; all three match through EOS and are
stable. Native exit is 0 after 209.106 seconds; the wrapper/reference comparison
returns 1 for the retained full-reference mismatch. No full-reference pass is
claimed for that case. The first-output durations in `comparison.json` measure
the **restored suffix only**, excluding prefix computation and copy latency.

All eight final audits pass frozen arena, planned workspace, zero growth, zero
activation host/rejected transfers, zero per-step allocations/internal fences.
Each rank copies 31,037,440 bytes per checkpoint, 279,336,960 bytes across nine
copies. Actual operation counts equal the rank's planned coalesced ranges times
nine; fragmentation causes counts to differ across ranks. Exact graph counts
are decode=133, T2=1, T4=1, T8=7, T16=2875, including warmup, source and destination.
`copy-plan-rank-0.json` exposes logical node, field, byte ranges and D2D primitive;
all-rank counters and original prepared-report hashes are retained.

The first copy client mistakenly fell through into an additional ordinary serial
run after its successful copy assertions. A corrected client returned immediately
and produced this clean 72-token run. A subsequent rebuild initially omitted the
CANN environment and failed to link; that launch did not execute on NPU. Neither
attempt is used as passing qualification evidence.

Rust adapter ownership/history guards are locally tested and cross-compiled.
End-to-end HTTP prefix hits, eviction, salt isolation, concurrent request progress
and shutdown reclamation remain separate qualification gates. This milestone
does not claim production prefix caching or 64K serving readiness.
