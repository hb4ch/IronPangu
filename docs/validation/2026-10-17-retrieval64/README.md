# 64K full-model retrieval qualification — 2026-10-17

The native OpenPangu v2 92B TP8/EP8 decoder completed nine requests with 65,520
input tokens and 16 forced output tokens each. All early/middle/late retrieval
answers are correct, match the independent reference through its first EOS, and
are stable across three repetitions. **Every case fails full forced-continuation
parity after EOS.** This is not a passing full-token qualification or a production
64K serving release.

| Needle position | Answer | Mean native TTFT | Mean TPOT | Full forced parity |
| --- | --- | ---: | ---: | --- |
| Early | 5932 | 235.440 s | 20.693 ms | Fail after EOS |
| Middle | 6077 | 234.878 s | 20.697 ms | Fail after EOS |
| Late | 5479 | 234.858 s | 20.667 ms | Fail after EOS |

The native process returned 0 without timeout after 2,171.852 seconds. The wrapper
and independent comparison returned 1 for numerical mismatches; those failures
are preserved in `comparison.json`. TTFT includes native reset/prefill, excludes
model loading/tokenization/HTTP, and TPOT covers all forced output steps, including
post-EOS steps. No other NPU workload overlapped. CPU-only native builds overlapped
later requests; the first three requests preceded them. This is not an HTTP tail
latency, throughput-at-load, or matched OmniInfer performance comparison.

All eight final rank audits pass:

- Frozen arena: 14,038,223,872 bytes/rank.
- Planned shared workspace: 131,020,288 bytes/rank; zero growth allocations.
- Zero activation host transfers, rejected transfers, per-step device allocations
  or internal completion boundaries.
- Each rank records 36,856 T=16 graph replays and 136 decode replays; smaller
  family variants have their single warmup replay.

A during-run `npu-smi` sample showed 36,441–36,719 MB total HBM usage on devices
0–7 and 33,609 MB process memory per rank. This is a point sample, not peak HBM.
The bundle reserves two residents, but these nine measured requests ran serially;
64K concurrent HTTP pressure, cancellation, prefix reuse and sustained-memory
qualification remain open. Prefill around 235 seconds is insufficient for the
production performance target.

`reference.json` is the independent reference, never native output relabeled as
an oracle. `identities.json` pins the executable, original backend, reference,
input fixture, tokenizer and raw native output. `model.inferfabric` preserves the
compiled context/family configuration. Compact all-rank audits retain hashes of
their original prepared reports. Discovery, preparation and frozen short-run
audits are included separately; the short 529-token case also retains known
post-EOS numerical debt.

Remote artifacts are under
`/data/p00603624/inferfabric/openpangu92/family64-long` in Docker
`inferfabric-92b` on `7.156.99.58`. The production context admission gate remains
unchanged; runs above 512 still require the explicit experimental opt-in.
