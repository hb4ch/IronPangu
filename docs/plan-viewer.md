# Hierarchical execution-plan inspector

Open the generated [92B inspector](validation/2026-10-16-reusable-dsl/execution.html)
in a browser. It is a self-contained HTML artifact with no server, CDN, fonts,
JavaScript packages or network access required at viewing time. GitHub displays
HTML as source; download the file and open it locally to use the viewer.

The overview shows the plan's immediate scopes and operations, in expansion
order. Labels come from recorded instance paths and module names, without
recognizing particular architectures or guessing input/decoder/output stages.
Select a scope to see its immediate operations and collapsed child scopes.
Select a module and choose **Expand module**, or double-click its card. Breadcrumbs,
the sidebar and **Up** return to enclosing scopes. Only the current scope's
children are rendered, with at most 24 items per page. **View dependencies**
switches the root overview to its DAG. **Previous** and **Next** paginate both
overview cards and DAGs, including flat graphs with no source metadata.

Selecting an operation shows:

- Its compiler-validated logical output type/width and the types of its inputs.
- Concrete weight bindings, placement and other operation attributes.
- Clickable producers and consumers, including dependencies crossing module boundaries.
- DSL UTF-8 byte spans and the complete expansion stack.

Edges aggregate real dependencies between collapsed modules. Selecting a card
highlights its incoming/outgoing edges. Search matches operation names, instance
paths and attribute values, including weights; results navigate directly to the
containing scope and page. Page-boundary dependencies remain clickable even
when their producer/consumer is not on the current page. **Fit**, **+** and **−**
control diagram scale. Browser history and URL fragments preserve scope, page
and operation navigation. **Esc** goes up; **/**
focuses search.

Shapes are exported by Rust's existing `Graph::validate()` type checker, not
reconstructed by a separate browser implementation. Tensor shapes here are the
logical flattened per-invocation widths of `ascend_token_v1`; expert widths and
state handles are distinguished. The graph does not encode activation dtype,
physical buffer layouts, stream schedules or execution timings, so the viewer
does not invent them. Binding roles are exported as compiler metadata; the browser
does not recognize operator-specific binding attributes. Its generic typed-handle
styling likewise uses output type metadata, not an operator-name list.

Every expansion frame is preserved, including compile-time iteration frames.
Scope identity uses the full ancestry of module/instance pairs, so equal local
instance names under different parents or module types do not collide. Labels
remove a parent's instance prefix only when it actually matches. Operations
without provenance remain direct children of the plan. This viewer is generic
over the supported token execution IR; it does not add support for unrelated IR
formats or backend operator contracts.

## Generate

**Physical plan** switches the current scope to per-rank call order and activation
storage views. Select a call to inspect implementation selection, alternatives,
cost provenance, workspace bounds, lifetimes, collective signatures and opaque
host/device boundaries. Unknown costs/HBM remain unknown; this initial physical
artifact is inspection-only. See [the physical design and status](native-physical-planning.md).

`compile-token` now includes `execution.html` in its atomically published bundle,
alongside `physical.json`, the binary, JSON, DOT and per-rank initialization plans. This does not
change executable binary formats or native runtime behavior.

To inspect a previously saved graph without checkpoint access or NPU occupation,
run locally in WSL:

```sh
cargo run -p inferfabric-compiler --example view_token_plan -- \
  docs/validation/2026-10-16-reusable-dsl/execution.json execution.html
```

The renderer validates the graph before exporting. Source metadata is retained,
and embedded JSON is escaped so graph labels cannot terminate the data block.
An optional third argument to `view_token_plan` writes the physical JSON separately.

## Verification

```sh
cargo test -p inferfabric-compiler
cargo clippy -p inferfabric-compiler --all-targets -- -D warnings
npm install --prefix .deploy/plan-view-test --no-audit --no-fund playwright@1.64.1
node scripts/test-plan-viewer.cjs execution.html
node scripts/test-plan-viewer.cjs execution.html --flat
cargo run -p inferfabric-compiler --example view_token_plan -- \
  crates/inferfabric-compiler/tests/fixtures/viewer-scopes.json renamed.html
node scripts/test-plan-viewer.cjs renamed.html
```

The browser check requires an installed Playwright Chromium browser, or set
`PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH` to an existing Chrome/Edge executable.
An optional third argument names an existing screenshot output directory.
`--flat` writes a synthetic fixture under `.deploy/` with renamed operations and
all expansion metadata removed, then tests the viewer against it.

The browser check derives expectations from each supplied artifact. The 92B run
visits 460 scopes including the root and iteration frames, across 462 pages. The
flat 2,109-operation fixture covers 88 pages; a seven-operation fixture covers six
scopes with arbitrary module names, uneven nesting and repeated local instance
names. Checks compare all rendered edges with the underlying graph, preserve
every provenance frame, and exercise pagination, nested expansion, search,
dependency links, browser history, shapes, bindings and narrow viewport layout.
All runs require zero JavaScript errors and zero HTTP requests. Rust tests also
check deterministic HTML output, validation and safe embedding. Overview and
operation screenshots were visually inspected. This is viewer validation; the recorded
NPU numerical qualification remains the same previously tested execution program.

### Device-residency inspection

Current generated plans use `serial-device-resident-v1`. Every activation has
`location: device`; host and opaque implementation candidates are rejected
before cost selection. Operation details distinguish control-input uploads from
immutable-parameter preparation. Request-output byte bounds appear in plan
limitations. No activation transfer or implicit offload is legal. The displayed
host activation payload peak is therefore zero; this does not mean total host
memory or total device memory is zero. Workspace, duplicated expert banks and
persistent state still need a complete prepared memory plan.

## Layer execution view

Opening a layer/module now shows its complete selected API inventory, named
communication primitives and payload shapes, and scope input/output value shapes.
The execution table shows every input and output shape/dtype beside its selected
kernel, placement, and communication contract. Weight bindings and attributes
remain expandable within the row; dependencies and arena offsets remain available
in instruction details. Nested scopes come from source metadata.

The communication filter includes actual HCCL primitives, not local expert
computation. Kernel lists are not truncated. Instruction tables show up to 100
rows per page within a selected scope; the model overview stays hierarchical.
Shapes describe recorded SSA values and structured fields. Missing structured
field shapes and vendor-internal kernel variants are explicitly identified as
unavailable; API choices are not presented as vendor tiling decisions.

Regenerate an existing offline viewer with the current compiler template while
preserving its complete embedded plan and bound-arena metadata:

```sh
python3 scripts/refresh-token-viewer.py old-execution.html new-execution.html
node scripts/test-token-viewer.cjs new-execution.html
```

The same template is embedded in freshly compiled plans. The Node test executes
the actual viewer JavaScript against the recorded plan and checks dense/expert
layer navigation, full kernel inventory, collective filtering, shapes and rank
selection. It is a rendering-logic check, not a browser screenshot test.
