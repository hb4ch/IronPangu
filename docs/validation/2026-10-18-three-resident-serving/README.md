# Three-resident serving plan — 2026-10-18

The current device-zero backend completed discovery, fixed-workspace and frozen
qualification with three compiled resident slots, 4096 uncaptured prefill tokens
and 65,536 context capacity. Six interleaved requests (three 8176-token retrieval
prompts plus arithmetic peers) matched all 39 reference tokens through EOS in
each stage. Decode graphs for one, two and three sequences were prepared.

Frozen arena: 40,492,359,704 bytes/rank. Workspace: 159,221,248 bytes/rank. All
eight ranks passed activation-residency, no execution allocation, no internal
completion-boundary and no workspace-growth checks. Actual batches are recorded
in `trace.json`.

Remote pipeline `device-zero-apc8k` exited 0; HCCL port 63400. Its inputs are
`device-zero-apc64k-source` and `device-zero-apc8k-fixture`; its frozen output is
`device-zero-apc8k-frozen-plan`, paired with `native-build-device-zero`.

This supplies capacity for two active HTTP requests plus one prefix snapshot.
The native qualification exercises three active residents; it does not itself
prove prefix caching. HTTP cache hits, isolation, eviction and state-copy
correctness are separate follow-up gates.
