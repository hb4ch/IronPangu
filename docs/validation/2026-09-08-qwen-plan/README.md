# Qwen3.5-2B full-model inspection

Open [model.html](model.html) in a browser. It is self-contained and works offline. The file embeds the complete [typed plan](model.typed-plan.json); no model weights or NPU are needed to inspect it.

This is the actual canonical text-model DAG: 24 layers (18 Delta, 6 Full), 640 operators, 929 tensor dependency edges, 320 checkpoint weight bindings and 48 persistent state tensors. Layer buttons include embedding and the output head. Select operators for tensor shapes, attributes and weight/state bindings; follow producer/consumer buttons across layers. Global search locates operators through layer names, tensor names, weight/state references and attributes. The tied embedding has both embedding and output-head consumers. Zoom and scroll inspect large layer DAGs; Fit DAG gives an overview.

The checkpoint metadata came from the existing local typed-plan capture of `/data/p00603624/models/qwen35`, whose config and safetensors header are also preserved in the compiler's metadata fixtures. This reference was regenerated with the current canonical compiler, retaining those checkpoint descriptors. No tensor payload was copied or read for this visualization. Vision and MTP are excluded from the text contract, as recorded in the checkpoint metadata.

Reproduce locally (Rust in WSL):

```sh
cargo run -p inferfabric-cli -- explain-model \
  docs/validation/2026-09-08-qwen-plan/model.typed-plan.json \
  --html docs/validation/2026-09-08-qwen-plan/model.html
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

For a new checkpoint, first use `compile-checkpoint DSL CHECKPOINT_DIR OUTPUT.json`, then pass that typed JSON to `explain-model`. The inspector checks the graph against canonical Qwen operations, inputs, states and memory formulas. It supports previously emitted metadata identities when the mathematical graph is unchanged; it does not attest weight payloads or native binary provenance.

Validation: workspace tests and Clippy passed. Local headless Edge checked every layer's node and edge coverage, finite SVG coordinates, full-attention state links, cross-layer producer navigation, last-layer weight search, tied embedding consumers, binding coverage, and fit/zoom. Results are in [ui-check.json](ui-check.json). The rendered page was also visually reviewed at 1500×1100.

Scope: this view presents the checkpoint-bound mathematical plan, not the final native kernel schedule. Memory numbers are logical payload formulas (weights, recurrent/conv per request, KV per page). Dynamic tensor dimensions remain symbolic. Native activation offsets, workspace sizes, kernel fusion, stream/rank placement and measured peaks require a future native physical-plan export; this view does not invent them.
