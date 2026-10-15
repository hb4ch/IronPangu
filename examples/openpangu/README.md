# OpenPangu metadata-only reference

`92b.reference-config.json` is copied verbatim from `D:/omni/omniinfer/tools/quant/jointfix/tests/fixtures/pangu_92B/config.json`, at reference revision `e746a4df403b0e15d450e8c44569920232c4e11f` (copied 2026-10-15). It is a configuration fixture, not the deployed checkpoint or its weight payload. It declares 46 main layers, three next-token prediction layers, 256 routed experts, top-8 routing, MoME and four-stream mHC. The SWA list includes layer 46; the inspector retains it in raw metadata without counting it as a 47th main layer.

The default Pangu attention path gives SWA precedence and otherwise selects sparse MLA when `index_topk` is positive. The alternate high-throughput backend uses a different selection rule involving `dsa_layers`. The report labels its rule; backend profiles are not assumed interchangeable.

No real 505B configuration/header fixture was found in the reference locations inspected. Large-dimension behavior is covered by explicitly synthetic tests, not labeled as 505B checkpoint qualification. Audit both actual target checkpoints before closing OP-00.

From local WSL:

```sh
cargo run -p inferfabric-cli -- inspect-openpangu \
  examples/openpangu/92b.reference-config.json .deploy/openpangu-config.json
cargo run -p inferfabric-cli -- inspect-openpangu-checkpoint \
  /path/to/local/openpangu-checkpoint .deploy/openpangu-checkpoint.json
```

The first command needs only a config. The second reads config, an optional `model.safetensors.index.json`, and safetensors **headers only**. Without an index it expects `model.safetensors`. It validates contiguous tensor ranges, byte counts, duplicate keys, shard ownership and root-confined filenames. All indexed tensors, including MTP/quantization tensors, are counted without guessing exclusions. Supported stored dtypes: BOOL, integer 8/16/32/64-bit, BF16/F16/F32/F64 and F8_E4M3/F8_E5M2. Packed quantized weights are counted in their actual storage dtype; unknown dtypes fail explicitly. This does not implement quantized math or bind tensor names/shapes to the complete model contract.

Limits: 16 MiB per metadata/header/auxiliary file, 256 MiB aggregate shard headers, 4096 shards, one million tensors and rank at most 16. Duplicate JSON keys and arithmetic/range overflow are rejected. Standard tokenizer/template/generation sidecars are fingerprinted when present; their semantics are not yet validated. No Python model code, ACL initialization, network access or device discovery occurs.

Reports preserve the raw config, summarize main layers, record sharing declarations and list unresolved production prerequisites. `executable=false`, `hardware_qualified=false` and `payload_verified=false` are intentional. Metadata fingerprints identify headers, sizes, index and included sidecars, **not weight contents**. Identical metadata with different payload bytes can produce the same fingerprint.

Stored-byte totals are not peak HBM, packed-format parameter counts, or rank-placement estimates. Replication, transformed weights, load staging, caches, MoME state, graph buckets, workspace, communication and allocator reserves remain OP-00/03/08/09 work.
