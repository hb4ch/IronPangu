# Communication environment A/B — 2026-10-18

Same native library, frozen plan, 4088-token reference prompt and five outputs
through EOS as the preceding profiling run. Both runs disabled profiling.
Only `HCCL_OP_EXPANSION_MODE` changed (AIV versus HOST), with independent HCCL
ports 62200 and 62400. All tokens, residency and frozen-resource gates passed.

First full prefill ABI: AIV 7.543737 seconds; HOST 7.149745 seconds.
These single samples do not establish a stable performance ratio. They show
that setting AIV alone did not resolve the observed multi-second prefill.
The environment request is not proof of which HCCL implementation was selected.
No default communication mode change is justified by this test.

The preceding HOST profiler's `timeline.json` records rank 0 task span 6.805 s,
notify-wait union 5.724 s and inclusive AllReduce durations around 5.247 s.
Wait durations include dependency/rank readiness; they do not prove a link
bandwidth bottleneck. Host/API nesting and inter-stream overlap prevent adding
these numbers into a latency decomposition.

Next inspect host-side work between launches. Source inspection found eager
creation of top-k row descriptors for packed expert output, even when the fused
consumer uses only packed storage. That optimization is a separate pending
change and is not included in these measurements.
