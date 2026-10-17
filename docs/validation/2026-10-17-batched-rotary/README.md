# Batched device-resident BF16 rotary

2026-10-17: TP8/EP8 OpenPangu 92B, experimental context 4,096, two residents,
sixteen-token prefill. The backend now rotates the entire token/head matrix in
one vendor call, using one absolute-position coefficient row per token. It
returns the contiguous output directly, eliminating per-query output assembly.
No activation upload, readback, dtype conversion or fence is added. The compiler
recipe/viewer exposes the call count, coefficient layout and zero assembly copies.

## Verification

- Local compiler suite: 18 tests pass, including recipes for batched rotary.
- Native operator: 369,024 bit-exact rotary/tail checks and 75 graph replays;
  chunk sizes 1/2/4/8/16, heads 1/6/24, boundary positions through 65,535, reset
  and replay. Maximum error is zero. See [operator-probe.log](operator-probe.log).
- Full decoder: discovery, planned-workspace preparation and frozen execution
  each match all 123 expected tokens, with interleaved residents and shuffled
  pages. This includes the standard fixtures and the 529-token retrieval.
- Isolated long run: all 72 native tokens at 2,049/3,072/4,088 prompt tokens are
  identical to the previous implementation across three repetitions. The known
  3,072-token post-EOS reference mismatch remains and the qualifier/comparator
  return nonzero. This milestone does not promote the long-context variant.
- All eight native/HTTP audits pass: frozen arena, planned workspace, no growth,
  no activation transfers or rejected transfers, no retained step allocations
  or internal fences. Rotary preparations drop from 3,162 to 372 per rank.

Workspace remains **131,020,288 bytes/rank** and arena **12,611,250,176 bytes/rank**.
Process HBM was sampled at **32,208 MB/rank** (not a peak measurement). Every
resource plan was rediscovered and bound against the new backend fingerprint.

## Performance

Isolated native measurements, three repetitions; no reference/HTTP process
overlaps these timings. TTFT includes prefill/sampling but excludes loading,
tokenization and HTTP. TPOT uses individual generated-step durations.

| Prompt tokens | Before TTFT, ms | After TTFT, ms | Reduction | After TPOT, ms |
|---:|---:|---:|---:|---:|
| 2,049 | 7,958.508 | 7,219.530 | 9.29% | 20.046 |
| 3,072 | 11,762.103 | 10,627.366 | 9.65% | 20.034 |
| 4,088 | 15,742.541 | 14,243.359 | 9.52% | 20.057 |

The vLLM Rust 0.25.1 HTTP test passes complete standard text/SSE outputs, six
concurrent requests, queue/disconnect isolation, cancellation during peer prefill,
context/sampling rejection and graceful shutdown. With two residents and token
budget 32, a 4,088-token request reaches **15.246 s TTFT**; its short peer finishes
first at **1.006 s TTFT**, with 75.5–76.3 ms text-chunk intervals. Text chunks need
not be individual tokens. The pool records 17 admissions and zero final occupancy;
31 mixed rounds and 256 prefill graph replays/rank are observed. Multiple prepared
chunk sizes and true multi-sequence kernels remain necessary.

## Provenance

Rust compiler cross-built locally in WSL; native builds/runs inside Docker
`inferfabric-92b` on `7.156.99.58`. Native build is isolated in
`native-build-batched-rotary`; the preceding build remains available unchanged.

- Native SHA256: `581d123c08158ed1993d6decdab34603356bb612b6ec484f3617c03228091ad4`.
- Execution SHA256: `fa56a54d2c5623044a52e97592e9a098ce8af5aaad5d1d29e27f4ebb2622fa53`.
- Rust frontend SHA256: `a81c91652b82e4917b8db7e767138c51f75fc3f2baa10ce799d476ae37ec582d`.
- Same DSL/reference/settings as [the preceding sixteen-token run](../2026-10-17-window16/README.md).
- Remote raw artifacts: `rotary16-{source,discovery,workspace,preparation,bound,qualified,long,http}`
  under `/data/p00603624/inferfabric/openpangu92`; local copies in `.deploy/rotary16-evidence`.
- Discovery/preparation overlap the separate-device 16K reference job; their
  timing is not used for the isolated performance comparison above.

64K serving remains unqualified. The independent 16K early/middle/late reference
has completed with correct answers; the unchanged baseline's 16K discovery
exposes a 529-token post-EOS difference against the existing 4K reference. That
evidence is retained separately and is not attributed to this rotary change.
