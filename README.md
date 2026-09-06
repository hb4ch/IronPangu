# Iron Pangu

An experimental Rust LLM inference engine for Huawei Ascend, with necessary C++/Ascend C kernels and no Python execution dependency. Its central experiment is compiling a model-architecture and parallelism DSL into inference programs with explicit communication.

The first target is end-to-end **Qwen3.5-2B text generation**, with disaggregated prefill/decode, ACL Graph decode, chunked prefill, continuous batching, and paged attention. TP and SP/CP are part of the compiler design and distributed validation plan.

Read the [design document](docs/design.md). This repository currently contains the design and research, not a working inference engine.

[Primary-source research](docs/research.md) records the vLLM Rust frontend boundary and the pinned Qwen model configuration.
