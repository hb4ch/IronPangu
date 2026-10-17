# Updated frontend: 64K prefix restoration — 2026-10-18

The completion-order frontend now passes the full 64K prefix-enabled HTTP gate
on the unchanged v9 native plan/library. It preserves first-EOS reference output,
SSE/collected agreement, queued isolation, cancellation recovery, short-peer
progress, context overflow rejection, prefix salt isolation and eviction.
The process exits zero after graceful shutdown.

Two prefix hits each restore 61,440 tokens of a 65,520-token prompt, leaving a
4,080-token suffix. Observed warm TTFT is **1.261 / 1.246 seconds**. Cold salt A,
cold salt B and evicted salt A measure **9.595 / 10.338 / 10.333 seconds**.
The three mixed long requests have TTFT **10.977–11.033 seconds**, and short
peers **738–772 ms**. These are run-specific samples, not a controlled speedup
attributed to completion ordering or sustained production percentiles.

The scheduler accounts for 405,921 input tokens across 38 admissions with two
active requests and a 4096-token budget. Prefix shutdown records two hits,
122,880 restored tokens, six published snapshots and zero remaining entries.
Shared-page occupancy drains to zero. All eight rank audits preserve decode
replay, frozen arena/workspace, zero execution-time allocations or internal
completion boundaries, zero activation-host transfers and drained transient
metadata. No prefill graph replays occur. Prefix snapshot work stays D2D:
1600 operations / 6,952,009,728 bytes per rank across the complete run.

## Reproduction

Use the same frontend, plan and library as
`../2026-10-18-completion-order/README.md`, with normal CANN/vendor environment,
HOST expansion and HCCL base port 45000. Run `qualify-program-http.py` with
`batched16-preparation-reference.json`, `retrieval-64k-reference.json`,
`device-zero-apc-prefix64-case.json`, two active sequences, 4096 token budget,
packed prefill, first-EOS and arithmetic-history checks, HTTP port 18085 and
300-second request inactivity timeout. Output: `release-order64k-prefix-http`.

The terminal report and raw per-rank/request audits are preserved in that
subdirectory; `summary.json` records headline observations and frontend identity.
This completes the updated frontend's prefix regression, not the remaining
long-duration soak, full-model slow-socket, backend-failure/restart, stochastic
sampling or matched OmniInfer performance gates.
