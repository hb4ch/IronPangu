# Full 92B sequence decode graphs

Startup now prepares each exact resident decode width before sealing graphs.
Sequence positions, recurrent slots, lengths, cache append indices and page
tables remain device control inputs; graph replay does not bake in a request.
Mixed batches and long prefill remain uncaptured. Startup also exercises the
maximum packed token envelope with every resident represented, so its controls
and buffers are included in discovery before arena binding.

On 7.156.99.58 inside Docker `inferfabric-92b`, TP8/EP8, the existing
21-request fixture again produced all 159 reference tokens through first EOS.
The native command and wrapper exited 0. All eight final transfer audits have
zero activation host transfers. Every rank reports:

- `token_decode`: one startup replay.
- `sequence_decode_1`: four replays.
- `sequence_decode_2`: 60 replays.

Across the same 118 post-first-output records belonging to two-sequence,
two-token decode batches, median token interval fell from 225.040 ms in the
uncaptured batch run to 29.0525 ms. This includes request scheduling delays;
it is not isolated kernel time, an HTTP benchmark or a general throughput claim.
`tokens.jsonl` preserves per-request timing; `summary.json` records per-rank
graph counters and transfer totals.

Discovery remains unbound. Production still requires planning one maximum
workspace, running with that fixed workspace, binding the resulting arena and
verifying frozen execution. Rust/HTTP batch serving is not yet enabled.
