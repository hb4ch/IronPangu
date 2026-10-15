# Native 92B checkpoint loading, TP8 / EP8

Roadmap OP-03 / OP-08 loader milestone. This is full main-model weight loading, not full decoder execution or an inference-quality result.

The locally cross-compiled Rust `export_openpangu_load_plan` audits the checkpoint, binds every required tensor and compiles the TP/EP DSL. It emits one compact `IFLOAD01` manifest per rank and the human-readable placement plan. C++ `ResidentWeights` consumes those manifests inside Docker and reads the original safetensors payload ranges directly into device allocations, with 4 MiB source staging per rank. It retains raw canonical tensor layouts and applies only declared BF16-to-FP32 casts. Projection transposes, gamma folding, absorbed MLA matrices and other operator-specific preparation remain executor work.

The 92B checkpoint at `/models/92B` has metadata fingerprint `e6ab8e354f31f844671a0d6013d686e86b4ebc2ba8f3e893d509c2c170d015d8`. Main layers are included; the three MTP layers remain excluded. On each of ranks 0–7:

- 9,956 assigned tensors, including owner-only routed experts and the plan's replicated / TP output-row weights.
- 12,885,336,434 source bytes.
- 12,945,669,860 resident bytes after declared FP32 casts.
- Every uploaded byte is read back and compared with its prepared host chunk in qualification mode.

Eight loader processes run concurrently, each selecting its assigned device. They do not synchronize their completion or establish an eight-rank inference-ready barrier. Resident allocations are retained until that rank's entire load completes, then released through RAII. A failed constructor releases previously loaded tensors. Execution workspaces, transformed copies, caches, activations and graphs are not included in these weight totals.

Host staging is bounded independently of checkpoint size: 4 MiB input, 8 MiB conversion storage and 8 MiB verification readback per process. No full-model host tensor copy or intermediate fixture export is required. The readback buffer is currently allocated even when verification is disabled.

Manifest validation rejects unsupported versions/dtypes, truncated/trailing data, zero/overflowing or inconsistent shapes, mismatched byte totals, duplicate tensor names, malformed paths, and out-of-file ranges. Every canonical shard path must remain directly inside the checkpoint root. Thirteen binary-manifest cases exercise the parser before ACL initialization. The recorded metadata fingerprint is provenance, not a runtime cryptographic payload check; immutable checkpoint storage / load-time fingerprint revalidation and payload integrity policy remain release work. This development format does not mark the compiled plan executable.

Reproduce after the local cross build and transfer, inside the remote Docker container:

```bash
bash scripts/qualify-openpangu-load.sh /models/92B \
  examples/openpangu/92b-tp8-ep8.inferfabric load-qualification
```

The script uses the checkpoint's explicitly allowed stale index-size exception; all other audit/binding checks remain enabled. Python runs only the malformed binary-manifest test driver, never loading or executing model tensors. Rust compilation remains local; C++ build and all NPU activity stay in Docker.

Next integration gate: compose all 46 decoder layers, real attention/indexer/cache updates, final mHC merge and logits; compare layer traces and generated output with a pinned reference; then connect the Rust frontend. Passing these loader checks does not establish any of those gates.
