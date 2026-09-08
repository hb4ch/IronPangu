# Metadata fixture provenance

`qwen35-config.json` and `qwen35-header.json` were read from the user-supplied checkpoint at `/data/p00603624/models/qwen35` inside the development container on 7 September 2026. They contain configuration and safetensors descriptors only; no tensor payload bytes.

The original header is 76,648 bytes with SHA-256 `ccba2c1f645fe59268f89ee7ea552e02ce6bf0c8a087bf332fb0e3bd99bfee9d`. The shard payload is 4,548,144,832 bytes. Preserve header bytes exactly so offset validation uses the original prefix size.
