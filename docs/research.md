# InferFabric: upstream research notes

Research date: 2026-09-06. Primary sources only. These notes establish upstream facts; they do not claim InferFabric implementation or Ascend validation.

## Rust frontend boundary

The experimental vLLM Rust frontend replaces the API-server process and uses the existing engine/core boundary. Its code has moved into vLLM's `rust/` directory; the feature-parity roadmap explicitly says it remains experimental. The existing engine is not thereby rewritten in Rust. [Roadmap](https://github.com/vllm-project/vllm/issues/44280)

The CLI example explicitly launches a managed headless Python vLLM engine alongside the Rust HTTP frontend, or connects to an already running engine. Therefore a Python-free InferFabric needs its own Rust launcher, scheduler, cache manager, executor, sampling path, and Ascend backend. Reuse/adapt HTTP, tokenization, request, streaming, and cancellation code where separable; replace the engine client contract. This is a proposed engineering boundary, not an existing upstream capability. [CLI example](https://github.com/vllm-project/vllm/blob/main/rust/src/cmd/examples/README.md)

The original RFC motivates eliminating frontend CPU overhead and GIL restrictions, but removing Python does not itself prove lower device inference latency. Measure host scheduling/dispatch overhead separately. [RFC](https://github.com/vllm-project/vllm/issues/40846)

## Qwen3.5-2B is a hybrid model

Pinned configuration: `bf4df5f05ef9c33020b38c73a676d85ad35d2c35`. Text model: hidden size 2048, 24 layers, MLP intermediate 6144, BF16, 248320 vocabulary with tied embeddings. Layer pattern repeats three linear-attention layers then one full-attention layer: 18 GDN and 6 full-attention layers. Full attention has 8 query heads, 2 KV heads, head dimension 256; output gating enabled. Linear attention has 16 key and 16 value heads, 128-dimensional heads, convolution kernel width 4. RoPE is partial (0.25), theta 10000000; RMS norm epsilon 1e-6; recurrent cache default FP32. A vision encoder is present. [Pinned config](https://huggingface.co/Qwen/Qwen3.5-2B/blob/bf4df5f05ef9c33020b38c73a676d85ad35d2c35/config.json)

The model-specific overview calls the linear layers Gated DeltaNet and lists a dense FFN. Do not misclassify the 2B checkpoint as MoE because the shared family introduction mentions MoE. First-demo text-only scope is reasonable but must be explicit; it is not complete multimodal support. [Model card](https://huggingface.co/Qwen/Qwen3.5-2B)

Reference implementation maintains both convolution state and recurrent state, with different chunk-prefill and single-token recurrent paths. It uses Q/K L2 normalization, gating, and gated normalization; faithful implementation needs these details in addition to ordinary attention/MLP. The decoder uses a dense `Qwen3_5MLP` and zero-centered RMS norm. [Transformers reference](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modular_qwen3_5.py)

Design implication: paged KV applies to six full-attention layers. Eighteen GDN layers need per-request recurrent matrices and convolution history. Chunk boundaries and P/D handoff must preserve all three state categories and a common consumed-token frontier. Page tables alone cannot resume this model. This is our deduction from the model and reference code.

## Scheduling and disaggregation

vLLM V1 chunked prefill prioritizes decode tokens, fills remaining token budget with prefill, and splits a prefill that cannot fit. [Optimization guide](https://docs.vllm.ai/en/v0.22.1/configuration/optimization/)

InferFabric should implement the semantics in Rust. In physically disaggregated mode, the P scheduler chunks/interleaves prefills and D continuously admits ready transferred requests at iteration boundaries. Mixed prefill/decode token packing matters for a colocated reference mode; it should not force prefill compute onto the disaggregated D pool. Continuous batching means dynamically admitting and retiring requests per iteration, not waiting for an entire fixed batch to complete. These are proposed design choices.

vLLM documents separate instances for P and D and a connector transferring cache/results. It motivates independent TTFT/ITL tuning and reduced prefill interference, and cautions against treating P/D as an automatic throughput improvement. [P/D guide](https://docs.vllm.ai/en/latest/features/disagg_prefill/)

The April hybrid-disaggregation article describes distinct full-attention and recurrent-state layouts, homogeneous/heterogeneous TP transfer issues, and release-after-transfer completion. It still listed GDN as future work at that time. [April article](https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg)

That limitation is historical: the August article explicitly reports Qwen3.5 GDN P/D support through PR #41869 and highlights async transfer/block-freeing races fixed in #48481 and #45357. Its performance evidence is a large MoE Qwen checkpoint on GB200/NVIDIA; neither its kernels, benchmarks, nor NIXL/CUDA path establish availability on Ascend or a Python-free backend. Use the state-lifetime lessons, not the hardware performance numbers, for InferFabric. [August article](https://vllm.ai/blog/2026-08-06-qwen35-25k-tps)

## Parallelism design deductions

TP=2 is a natural first distributed configuration because full attention has two KV heads, while query, GDN heads, and MLP dimensions divide evenly. TP>2 requires explicit KV replication or a verified alternative partition rule. SP should denote sharding tokenwise activations around normalization/MLP boundaries, whereas CP partitions attention context; they need separate DSL layout types and lowering rules.

Full-attention CP needs correct global softmax combination across context shards. GDN cannot be treated as independent chunks with ordinary KV exchange: recurrence requires ordered boundary-state propagation or a mathematically valid associative scan. A first correct CP lowering may propagate state sequentially; label this a correctness implementation with limited speedup, and reserve parallel scan for a later validated optimization. P/D with different TP/CP layouts needs an explicit state redistribution plan. Start with identical P/D TP layouts to isolate correctness.

Proposed validation should compare monolithic/chunked prefill, eager/graph decode, colocated/P-D, TP1/TP2, and CP/SP variants on identical prompt tokens and model revision. Include forced non-contiguous KV pages, request insertion/removal, cancellation during transfer, and recurrent-state contamination checks.
