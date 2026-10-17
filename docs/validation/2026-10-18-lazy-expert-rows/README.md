# Lazy expert row descriptors — 2026-10-18

Packed expert outputs now retain row width instead of eagerly creating a tensor
view for every routed row. Packed grouped-matmul and reduction consumers never
needed those row descriptors. Consumers that explicitly pack individual rows
still create bounded views on demand and retain the same copy semantics.

The isolated `native-build-lazy-row` library completed discovery, fixed workspace
and frozen-arena qualification against the 4088-token reference prompt. Each
stage matched all five output tokens through EOS. The new frozen bundle is
`lazy-row4k-frozen-plan`; do not pair the new library with the old frozen bundle.

Rank 0 retained descriptors fell from 75,772 in the old HOST frozen run to
10,180 in the new frozen run. Arena bytes remain 39,572,690,960 per rank;
workspace remains 159,221,248. All eight ranks passed zero activation-host
transfer, zero in-execution allocation and zero internal completion-boundary
checks. Decode graph replays remain enabled.

Single prefill ABI samples: old frozen HOST 7.149745 seconds, new discovery
7.417684, fixed 6.817797, frozen 6.729598. Different stages are not equivalent
performance baselines. Even the two frozen samples are insufficient to claim
a stable speedup. This milestone establishes the descriptor reduction and
small-fixture correctness, not the production latency target.

Remote pipeline `lazy-row4k` completed with status 0, HCCL port 62600.
The 8K mixed-request regression is a separate follow-up and is not claimed here.
