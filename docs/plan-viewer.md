# Hierarchical execution-plan inspector

Open the generated [92B inspector](validation/2026-10-16-reusable-dsl/execution.html)
in a browser. It is a self-contained HTML artifact with no server, CDN, fonts,
JavaScript packages or network access required at viewing time. GitHub displays
HTML as source; download the file and open it locally to use the viewer.

The overview shows the input stage, 46 collapsed decoder layers and the output
stage. Select a layer to see its immediate operations and collapsed modules.
Select a module and choose **Expand module**, or double-click its card. Breadcrumbs,
the sidebar and **Up** return to enclosing scopes. Only the current scope's
children are rendered: the qualified 92B plan has at most 19 visible DAG cards.

Selecting an operation shows:

- Its compiler-validated logical output type/width and the types of its inputs.
- Concrete weight bindings, placement and other operation attributes.
- Clickable producers and consumers, including dependencies crossing module boundaries.
- DSL UTF-8 byte spans and the complete expansion stack.

Edges aggregate real dependencies between collapsed modules. Selecting a card
highlights its incoming/outgoing edges. Search matches operation names, instance
paths and attribute values, including weights; results navigate directly to the
containing scope. **Fit**, **+** and **−** control diagram scale. Browser history
and URL fragments preserve scope/operation navigation. **Esc** goes up; **/**
focuses search.

Shapes are exported by Rust's existing `Graph::validate()` type checker, not
reconstructed by a separate browser implementation. Tensor shapes here are the
logical flattened per-invocation widths of `ascend_token_v1`; expert widths and
state handles are distinguished. The graph does not encode activation dtype,
physical buffer layouts, stream schedules or execution timings, so the viewer
does not invent them. Module grouping comes from expansion provenance; the
92B overview recognizes `DecoderBlock` instances.

## Generate

`compile-token` now includes `execution.html` in its atomically published bundle,
alongside the binary, JSON, DOT and per-rank initialization plans. This does not
change executable binary formats or native runtime behavior.

To inspect a previously saved graph without checkpoint access or NPU occupation,
run locally in WSL:

```sh
cargo run -p inferfabric-compiler --example view_token_plan -- \
  docs/validation/2026-10-16-reusable-dsl/execution.json execution.html
```

The renderer validates the graph before exporting. Source metadata is retained,
and embedded JSON is escaped so graph labels cannot terminate the data block.

## Verification

```sh
cargo test -p inferfabric-compiler
cargo clippy -p inferfabric-compiler --all-targets -- -D warnings
npm install --prefix .deploy/plan-view-test --no-audit --no-fund playwright@1.64.1
node scripts/test-plan-viewer.cjs execution.html
```

The browser check requires an installed Playwright Chromium browser, or set
`PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH` to an existing Chrome/Edge executable.
An optional third argument names an existing screenshot output directory.

The 92B browser check visits all 413 module scopes, compares every scope's rendered
edges with the underlying graph, verifies overview collapse, nested expansion,
search, producer navigation, browser back, input/output stages and narrow viewport
layout. It checks a real projection's validated width and weight binding, and
requires zero JavaScript errors and zero HTTP requests. Overview and attention
screenshots were also visually inspected. This is viewer validation; the recorded
NPU numerical qualification remains the same previously tested execution program.
