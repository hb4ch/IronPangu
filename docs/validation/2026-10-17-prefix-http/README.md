# HTTP device prefix-cache qualification — 2026-10-17

The vLLM Rust 0.25.1 frontend passed the native 92B TP8/EP8 HTTP gate with
one active resident, one private snapshot resident, a 4,096-token compiled
context and a 32-token scheduler budget. The explicit prefill family is
2/4/8/16. This is not concurrent 64K qualification.

For each independent 2,049/3,072/4,088-token fixture, the gate checks cold A,
warm A, cold B, warm B and evicted A. A/B use distinct cache salts. All 15
responses match reference text through EOS and exact prompt/completion/total
usage. Warm responses report 2,048/3,008/4,032 cached tokens respectively.
The immutable reference is preserved in the adjacent
[device-continuation evidence](../2026-10-17-device-continuation/reference.json).
Known forced post-EOS mismatches are not resolved or waived by this gate.

| Input tokens | Cold TTFT range | Warm TTFT range |
| ---: | ---: | ---: |
| 2,049 | 7.30–7.85 s | 0.343–0.387 s |
| 3,072 | 11.48–11.72 s | 0.720–0.761 s |
| 4,088 | 15.20–15.46 s | 0.829–0.865 s |

These are HTTP receipt timings including snapshot copies/restores, with one
active request. Output chunk intervals are approximately 20 ms; chunks are not
necessarily tokens. Three ordinary fixtures also pass collected/SSE agreement,
queued isolation, disconnect recovery, context-overflow and sampling rejection.

All eight audits show frozen arenas of 12,620,067,840 bytes and planned workspace
of 131,020,288 bytes per rank, with zero workspace growth, activation host
transfers, rejected transfers, replay allocations or internal fences. The cache
reports six hits, 18,176 restored tokens and nine published snapshots. All 36
state admissions retire; final occupied pool bytes are zero. Completed arithmetic
histories account for all 28,195 scheduler tokens, excluding imported prefixes.

`serving.json` derives copy support from registered executable state layouts.
Unsupported row-quantized layouts remain disabled. Its executable is byte-identical
to the previous continuation bundle; normal workspace/arena binding reused the
matching preparation evidence. `identities.json` pins the backend, frontend,
executable, manifest, fixture and harness. `audits.json` preserves compact
all-rank reports and hashes of the original full reports.

The initial two attempts exposed harness/integration issues: cache salts were
still rejected by legacy admission; then zero-hit usage was expected to contain
an optional object that vLLM deliberately omits. Both were corrected. The final
run exited 0 with graceful worker shutdown.

Remaining gates include two active residents plus snapshots, branching prefixes,
cache cancellation/pressure/soak, 64K HTTP restores and matched OmniInfer load
performance. The current cache keeps one aligned checkpoint per request and
copies complete device continuation state; it is not zero-copy page sharing.
