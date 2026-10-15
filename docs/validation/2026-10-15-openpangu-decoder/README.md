# Full 92B native decoder qualification

The complete 46-layer OpenPangu V2 92B decoder runs on eight Ascend NPUs with TP8/EP8, direct checkpoint loading and HCCL collectives. The final device-kernel revision generates all **33 tokens, including EOS, identically to the offline OmniInfer reference** across arithmetic, geography and Chinese prompts. See `device/results.json`, `reference/results.json` and `comparison.json` for complete sequences and selected-token log-probability differences. This is a full-model generation test, not a component probe or a production qualification.

## Execution and reference scope

Projection GEMMs, dynamic W8A8 quantization, fused routed-expert activation quantization, dense/shared SwiGLU, RMSNorm, mHC pre/Sinkhorn/post/final merge, MoME updates, learned routing and paged absorbed MLA attention execute through native CANN/vendor operators. Only selected experts execute on their owning EP rank. Every generated token must agree across all eight ranks; every main-layer residual is checked for finite values. Native loading and execution use C++ with no Torch/Python model execution. The test driver uses Python only for tokenizer access, process control and result comparison.

The final mHC merge uses the same pre operator with four learned pre rows and twenty zero padding rows; unused post/residual outputs are discarded. MoME and attention retain device state across tokens. The two vendor packages share a library SONAME, so training mHC symbols are resolved from an explicitly opened library handle.

This diagnostic runner accepts a single request, serial token prefill, greedy decoding, and **at most 512 prompt-plus-output tokens**. At this bound all causal keys fit both SWA-512 and sparse top-2048; the full candidate set is exact and learned indexer selection cannot exclude a key. This does **not** qualify learned indexing or longer contexts. Host-side RoPE, expert combination, sampling, frequent transfers and temporary allocations remain. Full-decoder ACLgraph, Rust frontend serving, batching, non-greedy decoding, recovery and performance gates remain open. Existing component graph tests are separate evidence; no complete-model graph capture is claimed.

## Reproduce in the NPU Docker container

Use the checkpoint-bound TP8/EP8 manifests described in the sibling load qualification report, the pinned vendor packages below, and 16 free devices for simultaneous native/reference runs (or run sequentially with a reference device override).

```bash
# Regenerate initialization programs with the current locally cross-built exporter.
bash scripts/qualify-openpangu-load.sh /models/92B \
  examples/openpangu/92b-tp8-ep8.inferfabric load-qualification
bash scripts/qualify-openpangu-decoder.sh /models/92B load-qualification/plan decoder-device-qualified
# Offline reference only; defaults to logical devices 8..15.
bash scripts/reference-openpangu-decoder.sh
python3 scripts/compare-openpangu-decoder.py \
  decoder-device-qualified/results.json reference-results.json comparison.json
```

The native driver snapshots the executable before running all three cases. `device/artifacts.sha256` identifies that executable and checkpoint config/tokenizer; `device/dependencies.txt` records linked libraries. Source build uses CMake Release and `-Wall -Wextra -Werror`. Native runtime code and all NPU builds/tests run inside Docker.

Reference: vLLM 0.25.1+empty (`752a3a504485790a2e8491cacbb35c137339ad34`), OmniInfer (`0c7e75a7b206550229210ebf16399f08a88625bc`), CANN 9.1.0, driver 25.5.1, custom training/inference operators 1.6.1.dev20260923+9.1.0.pta2.12.0. Image: `registry-cbu.huawei.com/omniai_omniinfer_dev/ai-infra-infer-1.0.3-a3-arm-develop-202610090945-debug:0.0.1`, image SHA `bf260616632829c9367f88f6a63b5d385f9cddb220333d69235d1cde1b1fb0cb`. Checkpoint metadata SHA: `e6ab8e354f31f844671a0d6013d686e86b4ebc2ba8f3e893d509c2c170d015d8`.

The reference explicitly enables the OmniInfer platform/model plugins, disables the inherited NCCL path, and initializes its declared local/layer communication groups after distributed startup. Its MoE strategy is `allreduce` with sequence parallelism disabled. An exploratory run incorrectly paired `allgather_reducescatter` with disabled sequence parallelism and produced divergent rank states and incorrect text; those results are excluded. The checked-in reference script/config reproduce the corrected configuration. Torch is used only by this independent offline reference, never by the native stack.

Token equality on three prompts establishes a narrow end-to-end correctness baseline. It does not establish bitwise logits, a broad numerical tolerance, task accuracy, long-context behavior, or production performance. The native serial absorbed-MLA path and the reference prefill path have different BF16 rounding; their measured log-probability differences are retained rather than hidden.

## Historical bring-up milestones

The following records describe earlier revisions; current behavior is described above.

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

## Three-prompt full-decoder smoke qualification

The next integration revision uses CANN `aclnnDequantSwigluQuant` directly for routed-expert activation quantization, removing the extra BF16 rounding before INT8 quantization. Every generated token is checked for agreement across all eight ranks. JSON records now include top-five log probabilities.

The real-checkpoint runs completed all three cases, including EOS:

| Prompt | Generated text (EOS omitted here) |
|---|---|
| `1+1=?` | `1 + 1 = 2` |
| `What is the capital of France?` | `The capital of France is Paris.` |
| `请用一句话介绍北京。` | `北京是中国的首都，拥有悠久的历史、丰富的文化遗产和现代化的城市风貌。` |

Reproduce inside Docker with `bash scripts/qualify-openpangu-decoder.sh /models/92B load-plan decoder-qualification`. The Python driver only tokenizes/detokenizes and invokes the native binary; model loading and execution are C++. It checks expected answer content, while the executable checks finite residuals and rank consensus. These are smoke checks, not proof of broad accuracy or reference-logit parity. The native executable dependency record contains no Python or Torch libraries. The prior arithmetic file records the earlier activation-rounding baseline; `native/` records the fused-quantization revision.
