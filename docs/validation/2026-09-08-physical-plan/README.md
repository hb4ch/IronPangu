# Local physical-plan qualification — 2026-09-08

Target: `cpu-f32-v1`, Rust 1.98.1 in local WSL. No remote machine, NPU, CANN, or Python model execution was used.

- 44 workspace tests passed across the full suite and the final targeted alignment/planner/CLI checks; 6 frontend tests passed. Workspace and frontend Clippy passed with warnings denied; formatting passed.
- Ten new tests cover physical arena base alignment, planner semantics, binary-only execution, deterministic source reordering, invalid graphs, corrupt/re-signed malformed binaries, input/error state isolation, state swapping, constant-only execution, rectangular matmul tails, safe HTML embedding, and end-to-end CLI export. Several cases share a test function.
- The CLI integration test deletes the logical source before explaining and executing the produced binary in fresh processes.
- The reference residual graph yields `[10,13,2,2]`, then `[20,26,4,4]` after persistent-state continuation. Independent hand calculation is recorded in the Rust test.
- Eight source operations become six executable calls: one folded constant and one removed dead operation. Unused source constants are also dropped. The aligned arena is 192 bytes versus 384 bytes for separate aligned intermediate allocations.
- Browser checks passed for all 12 displayed DAG/binding/commit nodes, all 11 buffer rows, selected-kernel inspection, memory-edge toggling, search highlighting, and fit/zoom controls. The HTML was also visually inspected. The normal UI automation helper could not start in the Windows sandbox; verification used an isolated local headless Edge profile instead.

Files:

- `demo.ifplan`: actual checksummed executable CPU plan.
- `ir/00-logical.json`: authoring/interchange graph.
- `ir/01-typed.json`: resolved/topologically ordered graph before optimization.
- `ir/02-optimized.json`: folded/pruned graph with compact IDs.
- `ir/03-physical.json`: kernels, launch schedule, offsets, lifetimes and dependencies.
- `ir/04-bundle-manifest.json`: binary identity and execution binding requirements.
- `results.json`: two invocation reports.
- `plan.html`: self-contained interactive view decoded from `demo.ifplan`.
- `ui-check.json`: browser interaction results.

These artifacts qualify the CPU plan/execute boundary. They do not establish new DSL parser support, foreign-kernel export, generic Qwen binary execution, multistream/rank execution, or Ascend qualification. See `docs/physical-planning.md` for the native integration contract.
