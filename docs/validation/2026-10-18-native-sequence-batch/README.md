# Full 92B native packed sequence execution

The native ABI now accepts packed token sequences with explicit resident,
position, length and sample flags. The decoder executes one shared forward,
using per-sequence recurrent history and attention/index tables, per-row cache
append addresses and rotary positions. Sampling selects each requested final
row on device, gathers rank consensus once and completes the owning stream
once for the batch output. Existing single-request decode graph execution is
unchanged.

## Full-model evidence

Run on 7.156.99.58, Docker `inferfabric-92b`, `/models/92B`, TP8/EP8,
`bulk-prefill4k-source`, `native-build-head-batch/program_engine_probe` with
preparation, packed prefill, resident interleaving and sequence batching enabled.
The probe continuously reuses two resident slots over the existing 21-request
fixture. It includes short prompts and lengths 529/2049/3072/4088.

- All 159 output tokens through first EOS match the previous qualified serial
  execution, keyed by request and output step; no missing or duplicate records.
- 156 output records belong to a multi-sequence forward.
- Mixed prefill/decode includes 4088 prompt tokens plus a peer decode token in
  one 4089-token forward. This is a test of the compiled total-token capacity;
  frontend scheduling policy is not implemented by this offline probe.
- All eight final transfer audits report zero activation host transfers.
- `tokens.jsonl` retains individual request timing and batch dimensions.

The shell wrapper had a trailing CR-only line, which failed after the native
command returned successfully (`set -e` proceeded to that final line). Thus
the SSH wrapper exit status was 1; native output validation and all final
transfer audits completed successfully. Do not interpret wrapper status as a
clean qualification-runner result. No full-model rerun was made solely for
this wrapper formatting issue.

## Remaining work

This new ABI is restricted to preparation mode pending batch arena planning.
Multi-sequence decode currently uses the uncaptured path and is not performance
qualified. The Rust frontend still needs this ABI and actual batch scheduling;
production readiness, batch decode graphs, frozen allocations, multi-chunk/APC,
64K correctness and serving TTFT/TPOT remain outstanding. Sampling currently
matches the existing program ABI's greedy-only contract.
