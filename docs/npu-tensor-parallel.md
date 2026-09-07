# Single-node tensor parallelism

The checkpoint DSL enables TP with matching prefill/decode mesh widths:

```text
prefill = 2 1 false
decode = 2 1 false
```

Use `examples/qwen35-2b-tp2.pangu`. The device argument is the first local device; rank r uses device+r. The current DSL/compiler accepts matching TP widths 1 or 2, CP=1 and SP=false. Startup checks the visible device count. Only TP=2 on devices 0 and 1 is hardware-qualified here.

```sh
export HCCL_OP_EXPANSION_MODE=HOST
export HCCL_CONNECT_TIMEOUT=120
export HCCL_EXEC_TIMEOUT=120
./pangu-native-server --native examples/qwen35-2b-tp2.pangu \
  /data/p00603624/models/qwen35 native-build/libpangu_acl.so 0 18081 \
  --max-model-len 2048 --max-num-seqs 4 --max-num-batched-tokens 16 \
  --memory-profile tp2-memory.json
```

Execute inside the NPU Docker container with the CANN environment sourced. Rust is cross-compiled locally; HCCL and ACL code is C++.

Every linear projection partitions its output rows across ranks. Matrix weights are uploaded as physical row shards; the tied embedding remains replicated for lookup and is sliced for the LM head matmul. HCCL AllGather and captured D2D layout copies reconstruct canonical `[batch, output]` tensors. Attention, norms, convolution and recurrent states are replicated. This is a compute/weight-sharding correctness baseline with communication after each projection, rather than the design's future column/row-parallel fusion and sharded attention state. No speedup or end-to-end memory halving is claimed.

Each rank has a dedicated device thread and captures its full model graph at startup. The Rust coordinator sends the same selected slots and tokens to every rank and waits for all replays. Rank zero owns request sampling, stop decisions and HTTP output. Continuous admission, mixed prefill/decode, cancellation and per-slot reset use the same coordinator. There is no runtime graph recapture or eager fallback. HCCL HOST expansion is mandatory on this CANN 9.1 image: the default AI_CPU path fails capture with unjoined streams (107025). The per-communicator override did not resolve that failure; the process environment must be set before HCCL initialization.

Memory reports include rank/world, physical resident weight bytes and per-rank observed HBM growth. Peer reports use `.rankN.json`. The measurement baseline already includes context and communicator initialization; subsequent HCCL capture resources are included in observed growth. These are sampled watermarks, not exact transient allocation traces.

Qualification on 2026-09-07:

- Two-rank matmul: CPU, eager and graph outputs matched exactly for both replicated and physically sharded weights, using three changing inputs and nonconstant matrices.
- Actual 24-layer Qwen3.5-2B, context 2048, four slots: startup eager/graph logits and state equality, final-position execution and inactive-lane preservation passed on both ranks.
- Chat/completion, SSE, overflow, overlapping admission, 12 concurrent mixed requests, isolated seeded output equivalence, sampling penalties, cancellation and slot reuse passed.
- Resident weights: 2,391,145,152 bytes per rank, versus 3,763,862,208 for TP=1. Observed startup growth was about 5.39 GiB per rank, excluding the initialized context/communicator baseline.

See `validation-tp-2026-09-07`. The design's optimized row-parallel reductions, distributed sampling, native PD, CP and SP remain future work.
