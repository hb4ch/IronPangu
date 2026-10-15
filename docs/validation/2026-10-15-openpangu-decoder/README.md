# First complete 92B diagnostic decoder

This milestone wires all 46 main layers, embeddings, attention, both dense MLPs, all 44 routed/shared MoE blocks, mHC pre/post and final merge, MoME histories, sandwich/block normalization and the vocabulary projection. The real checkpoint is loaded directly with the TP8/EP8 load manifests. All projection GEMMs execute through CANN on the eight NPUs; TP gathers and EP reductions use HCCL. Scalar normalization, mHC, routing, attention softmax/accumulation, RoPE and MoME state currently execute in C++. This is an intentionally simple numerical integration baseline, not the final device-resident or graph-captured serving implementation.

The executable accepts token IDs, with no Python/PyTorch model execution:

```bash
source /usr/local/Ascend/cann/set_env.sh
export HCCL_OP_EXPANSION_MODE=HOST HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=120
native-build/openpangu_decode load-plan /models/92B 8 \
  148901 1344 198 16 10 16 30374 148902 148901 98390 198 148906
```

These checkpoint-tokenizer IDs encode `<|message_start|>user\n1+1=?<|message_end|><|message_start|>assistant\n</think>`. The full native run returned `1 + 1 = 2<|message_end|>`; output IDs/logits are recorded in `arithmetic-tokens.jsonl`. Every main-layer residual was checked for finite values. This demonstrates a complete model path, not numerical parity or general quality; independent reference comparison is still pending at this milestone.

The runner rejects prompts plus generated tokens exceeding 512. Within this bound every causal token is inside the SWA-512 window and sparse top-2048 set, so no learned selection can remove any causal key. The runner therefore uses the exact full candidate set and does not claim learned-indexer or long-context support. It uses absorbed MLA for every token, including prompt tokens. That has different BF16 rounding from an unabsorbed prefill reference and must be qualified.

Limitations: one request, serial token prefill, greedy decoding, C++ scalar/state math, synchronous temporary allocations, no integrated ACLgraph or Rust frontend, no request cancellation/recovery, and no production performance claim. Existing independently qualified graph paths are unchanged. Routed MLP currently rounds SwiGLU to BF16 before dynamic quantization; matching the reference fused quantization boundary is an open numerical integration task. Process-wide failure exit is development-only.
