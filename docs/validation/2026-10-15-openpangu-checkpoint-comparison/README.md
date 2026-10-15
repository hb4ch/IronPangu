# OpenPangu 92B checkpoint comparison

Date: 2026-10-15. Read-only inspection in Docker; no NPU allocation or interference with existing workloads.

## Result and acceptance

Both discovered candidates have the same core 92B architecture and complete tensor layout as the current checkpoint. The user clarified that architecture compatibility is sufficient; byte-for-byte weight identity is not required. Continue implementation against the existing `/mnt/sdb_data/models/92B` checkpoint and its explicit execution settings. There is no need to transfer another 100 GB checkpoint.

Before stopping the scan, all 50 shards of the first candidate completed and matched the baseline SHA-256 hashes, including all main and MTP payloads. Nine shards of the openresource candidate completed and matched; its remaining scan was canceled after the user clarified acceptance. The temporary inspection container and hash workers were stopped. [Machine-readable evidence](comparison.json) preserves the completed checks and config differences. No NPU was occupied.

Core architecture: 46 main layers plus three MTP layers, hidden size 2560, 48 attention heads, 256 routed experts with top-8 routing, one shared expert, mHC, MoME, mixed sparse/sliding-window MLA, learned sinks, and dynamic W8A8 quantization. Missing configuration fields still require deliberate execution semantics; architectural compatibility does not erase those differences.

## Locations

The originally supplied path was absent on all six alternate hosts (`7.208.149.157`, `7.208.150.77`, `7.208.150.76`, `7.208.150.107`, `7.208.151.138`, `7.208.150.115`). Docker rejected an exact read-only bind with `bind source path does not exist` on each host. SSH authentication succeeded on all six.

On `7.208.150.77`, two candidates were found under a different bucket:

- `/mnt/sfs_turbo/bucket-szc-nlp-guiyang/s00562949/models/global_step_26-iterative-int8-0626`
- `/mnt/sfs_turbo/bucket-szc-nlp-guiyang/s00562949/models/Pangu92B_step26_INT8_openresource/global_step_26-iterative-int8-0626`

These are candidates matching the requested checkpoint basename, not proof of the provenance of the absent original path. The baseline is `/mnt/sdb_data/models/92B` on `7.156.99.58`.

Inspection used a temporary container on `7.208.150.77`, image `97532f3ce2d7`, with a read-only root filesystem, no network, no NPU devices, and `/mnt/sfs_turbo` bound read-only at `/storage`. Python standard-library code read JSON, safetensors headers, and bytes for SHA-256; no model code, Torch, or inference was invoked.

## Metadata and sampled payloads

Both candidates match the baseline's complete tensor inventory: 73,790 names, shard ownership, shapes, dtypes, absolute offsets, and lengths, totaling 104,347,285,362 tensor bytes in 50 shards. Canonical inventory SHA-256 is `363f7c817ac9c40c1ee931fbba51cc97b49a3ca38095b4b80d8282adda8700e9`: JSON object keyed by tensor name, each value containing `shard`, `dtype`, `shape`, `offset`, and `bytes`, serialized with sorted keys and compact separators.

The three previously qualified projections (layer-0 attention q_b, layer-0 dense down, layer-2 expert-0 gate) and their scales match the baseline's payload hashes in both candidates. This sample check alone does not establish full weight identity.

## Configuration differences matter

Neither candidate is an identical checkpoint package to the baseline.

The first candidate has `model_type=pangu_v2_moe` and architecture `PanguV2MoEForCausalLM`; it lacks the baseline's `auto_map`, `pad_token_id`, `rope_interleave`, and explicit `indexer_types`. Its tokenizer, tokenizer config, generation config, special token map, and chat template hashes match the baseline.

The openresource candidate uses the released OpenPangu architecture but lacks `indexer_types`, `param_sink_with_value`, and `rope_interleaved`. It adds `sliding_window=512` and `_attn_implementation=eager`, and declares Transformers 5.0.0 instead of 4.48.2. The tokenizer bytes and generation config match, but tokenizer config differs and the standalone chat template is absent.

The baseline explicitly alternates full and shared sparse indexers. The current binder defaults an unspecified sparse indexer mode to `full`; therefore identical weight bytes would not make these configs interchangeable execution contracts. Do not silently copy config fields or change the baseline's shared-index plan on the basis of weight identity. Preserve checkpoint config identity in the plan and qualify any alternate semantics separately.

## TP=8 / EP=8 status

The existing [TP8/EP8 milestone](../2026-10-15-openpangu-tp8-ep8/README.md) remains the qualified baseline: eight colocated TP/EP ranks, 32 experts owned per rank, real-weight component pipeline with HCCL and ACL graph replay at batch sizes 1 and 4. This comparison does not turn that component qualification into complete 92B decoding or a production serving endpoint.

The current binder also requires an explicit `param_sink_with_value=true`; the openresource config does not satisfy that contract as supplied. This is a source-level compatibility finding, not an inference test on that package.

## Reproduce full shard identity checks

Inside a container with the checkpoint mounted read-only, run:

```bash
cd /checkpoint
sha256sum model-*.safetensors
```

Compare the complete 50 filename/hash pairs with `baseline-shards.sha256`. The remote candidate scan uses eight concurrent standard-library SHA-256 readers with 8 MiB buffers. Hashes cover entire safetensors files, including headers and all main/MTP tensor payloads. No deserialization of executable checkpoint code is needed.
