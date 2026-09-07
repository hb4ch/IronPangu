# Pinned vLLM Rust frontend

Source: vLLM v0.25.1, commit `752a3a504485790a2e8491cacbb35c137339ad34`.
Copied from `/opt/vllm/rust` in Docker image `19cc2ebef117` on 2026-09-07.
The image's Rust source tree was clean according to `git status --porcelain -- rust`.
Original source archive SHA-256: `077ac274826ee3d83be2ab5e93261be6f58cf8b7b9e7801deb4a2828564591bd`.
Upstream license is preserved in `LICENSE`.

Local adaptations:

- `vllm-llm`: in-process `GenerationBackend`, backend metadata, a backend-neutral stream with explicit early-close errors, and unsupported-request errors.
- `vllm-text` / `vllm-chat`: metadata independent of an engine-core connection; optional administration client; request-error classification.
- `vllm-server`: `serve_with_backend`; existing routes, chat processing and SSE implementation retained. Engine-core-only administration returns an explicit unsupported error for the local backend.
- Cargo package workspace anchors keep this snapshot separate from Iron Pangu's core workspace.

The original Rust workspace includes command, managed-engine and Python-binding packages. They are not dependencies of `frontend/Cargo.toml` and are not built or launched by Iron Pangu. The deployed server uses the Rust HF/Jinja/tokenizer path with language-only configuration. No Python engine, interpreter embedding, or managed launcher is used. The Rust engine-core protocol crate remains a type dependency; the in-process path creates no engine-core connection.

The integration's resolved dependencies are pinned in `frontend/Cargo.lock`; the original workspace lockfile is retained separately. To review an upgrade, compare against the original tag and rerun stream, request-validation, and HTTP tests before changing this pin.
