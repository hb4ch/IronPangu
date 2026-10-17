# 8K continuous HTTP serving — 2026-10-18

PASS with `IFARENA2`, TP8/EP8 OpenPangu 92B and vLLM Rust 0.25.1 on
7.156.99.58 inside `inferfabric-92b`. Qualifier exit 0. The runtime uses the
same frozen pool as the [native qualification](../2026-10-18-pooled-arena/README.md).

[HTTP results](results.json), [scheduler log](server.log), [rank audits](ranks.json).
The legacy raw `mode` string says serial decode; rank audits establish the
actual batched graph execution: 89 single-sequence and 41 two-sequence decode
replays per rank, including preparation. No prefill graphs exist.

- Four baseline prompts (12/15/14/529 tokens), collected and streamed outputs,
  eight queued concurrent requests, three 8,176-token retrieval prompts with
  early/middle/late needles, concurrent cancellation and short-request recovery.
- Output text matches the independent references through first EOS. Token usage
  checks enforce full prompt consumption and the scoped completion count.
- 27 admissions; two active residents; 12 mixed prefill/decode ticks; all 29,152
  completed input tokens accounted by request arithmetic histories.
- Frozen arena 39,572,690,960 bytes/rank, fixed workspace, zero workspace growth
  and activation host transfers on all eight ranks. Shared pages fully reclaimed
  after graceful shutdown. Sampling/context rejection checks pass.

Mixed-workload HTTP TTFT was 14.520/14.172/14.622 seconds for the three long
prompts. The concurrent short requests saw 6.088/6.043/6.334 seconds TTFT: a
running 4K forward is non-preemptible. These latency costs remain an optimization
target, not a production SLO claim. Stream text-chunk intervals mix decode,
prefill and scheduling waits and must not be reported as isolated TPOT. Native
pure-decode measurements are in the linked native qualification.

This proves two-chunk prefill at 8K through the serving frontend. It does not
qualify 16K/32K/64K output, APC or sustained-load robustness. The compiled 64K
capacity is still guarded by the experimental long-context opt-in.

Reproduce in `/data/p00603624/inferfabric/openpangu92`, with the established
CANN/vendor/HCCL environment and `INFERFABRIC_EXPERIMENTAL_LONG_CONTEXT=1`:

```sh
python3 scripts/qualify-program-http.py \
  --server /data/p00603624/inferfabric/inferfabric-server-sequence \
  --plan pooled8k-clean-frozen-plan --checkpoint /models/92B \
  --library native-build-head-batch/libinferfabric_program.so \
  --cases batched16-preparation-reference.json \
  --resident-cases retrieval-8k-reference.json \
  --max-num-seqs 2 --max-num-batched-tokens 4096 \
  --expect-prefill packed --respect-eos --check-arithmetic-history \
  --output pooled8k-http --port 18085
```

Use a fresh output directory. The earlier `sequence8k-http` and
`sequence-envelope-http` attempts failed with the previous binding policies;
they are not included in this pass.
