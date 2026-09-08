# NPU sampling — 7 September 2026

Native serving now supports temperature, top-k, top-p, min-p, seeded sampling, repetition/presence/frequency penalties, and minimum output length. Temperature zero selects NPU greedy argmax. All logit processing, filtering, random-number generation and categorical token selection run through C++ ACLNN on the NPU. Only the selected token ID returns to Rust; serving no longer downloads the full vocabulary logits for CPU argmax.

The supplied checkpoint has no `generation_config.json`. Its `README.md`, lines 704–708 and 967–975, recommends these defaults for non-thinking text generation:

```json
{"temperature":1.0,"top_k":20,"top_p":1.0,"min_p":0.0,"presence_penalty":2.0,"frequency_penalty":0.0,"repetition_penalty":1.0}
```

The native frontend uses these defaults. If a checkpoint `generation_config.json` is present, its supported values override the model-card fallback. Explicit request parameters win, including zero penalties, `top_k: 0` (no top-k restriction), and `temperature: 0`. Thinking-mode recommendations differ; select them explicitly when enabling thinking. The weight/model files were not modified.

## Execution and semantics

- Repetition penalty covers tokens seen in the prompt or generated output. Positive logits are divided by the penalty; negative logits are multiplied. Presence and frequency penalties cover generated output only.
- Stop/EOS tokens are masked until `min_tokens` has been generated. Normal stopping remains in the Rust frontend/backend.
- FP32 temperature scaling precedes a stable descending NPU sort. Top-k retains ties at the kth score. Min-p removes scores below the maximum plus `log(min_p)` before nucleus normalization. Top-p retains the prefix whose exclusive probability sum is less than p, including the token that crosses the threshold.
- CANN generates a seeded uniform table on the device once per request. A mutable draw index advances once per generated token; prefill consumes no draws. An NPU inverse-CDF calculation samples the filtered distribution and gathers the original token ID.
- Model, greedy-sampler and stochastic-sampler graphs are captured at startup. Sampling parameters, counts, masks and draw indices use stable buffers. No per-token device allocation, operator preparation or graph capture is introduced.
- The installed tensor-seeded random/multinomial paths failed retained-executor qualification. The working implementation uses `aclnnInplaceUniform` outside capture once per request, including its workspace query and temporary workspace lifetime, then graph-replays the sampling pipeline. This is an explicit per-request preparation cost.
- Same seed, prompt and parameters reproduce output on this qualified build/device setup. This does not promise identical random streams to Python vLLM, PyTorch, or another CANN release. Omitted seeds select a fresh host seed; the random draws themselves are generated on device.

Context 128 and one active request remain the current serving limits. Logprobs, logit bias, structured output and other advanced constraints are still rejected explicitly. Native PD and continuous batching remain separate work.

## Validation

[Recorded reports](validation-sampling-2026-09-07/) include:

- Both logical devices: 12 synthetic distribution cases at vocabulary sizes 32 and 248320; temperature, top-k with ties, top-p, min-p, penalties and masking. Maximum absolute probability error against an independent CPU reference: **1.49012e-8**.
- Native greedy results agree with the CPU reference; fixed draw metadata yields identical eager/graph samples. Request-seed reset reproduces the random table, and a different seed changes it.
- 2,048 uniform draws from four supported tokens: counts `[485, 521, 529, 513]` on the qualified device, with no draws outside support.
- Actual model HTTP tests: same-seed repeatability, four distinct outputs from four seeds, seeded SSE/collected agreement, omitted/default parameter agreement, greedy seed independence, and minimum-token suppression.
- Existing native health/chat/completion/reset/cancellation/overlap tests pass. Local workspace tests, frontend tests and the vendored text-layer tests pass (67 text tests; one explicitly ignored).

Rust is still compiled locally in WSL. C++ builds, RNG qualification and model execution run only inside `inferfabric-npu`.

```bash
# Inside the NPU container after sourcing CANN:
./sampler-build/sampler_probe 0
./sampler-build/sampler_probe 1
./inferfabric-native-server --sampling-smoke-test http://127.0.0.1:18081
./inferfabric-native-server --native-smoke-test http://127.0.0.1:18081
```

The existing SSH tunnel exposes the container API on this Windows machine at `http://127.0.0.1:18081/v1`. Example request body for `/chat/completions`:

```json
{
  "model": "inferfabric-qwen35",
  "messages": [{"role": "user", "content": "Invent a name for a tiny dragon."}],
  "max_tokens": 24,
  "seed": 42,
  "chat_template_kwargs": {"enable_thinking": false}
}
```

Omitting sampling fields uses the text defaults above. Change the seed for another sample, or set `temperature: 0` for greedy decoding.
