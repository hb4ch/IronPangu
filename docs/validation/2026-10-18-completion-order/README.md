# Request completion follows resource release — 2026-10-18

The program worker now retains the terminal response until native resident
release and active-request removal have succeeded. Nonterminal streaming outputs
are unchanged. If release fails, the request receives an error instead of a
success already sent to the client. Cancellation and closed-consumer cleanup
remain bounded; no extra device fence or activation transfer is introduced.
This fixes a verified completion-order issue, not an attributed explanation for
the historical one-FD excursion.

Local tests explicitly observe that terminal output is absent during release,
then inject release success/failure through both batched and nonbatched workers.
All 31 frontend tests pass (one NPU-only test is ignored locally), strict Clippy
passes and the ARM frontend is cross-built in local WSL. Native library and
frozen program are unchanged; their hashes and the new frontend hash are in
`summary.json`.

The complete 92B NPU HTTP qualification passes with the new frontend:

- 24 four-client waves, each containing a rotating 64K request and short peers;
  120 EOS-scoped reference load/recovery outputs and 24 deliberate disconnects.
- Post-wave RSS growth 71.816 MiB, below the unchanged 128 MiB limit; threads
  remain 663 and FDs remain 76 in every sample. The load phase lasts 279.4 seconds.
- All 171 admissions and 1,785,397 input tokens are accounted for; shared pages
  drain to zero. Resource audits on all eight ranks retain the frozen arena and
  fixed workspace, zero activation-host transfers, zero execution allocations
  or internal completion boundaries, and zero retained transient metadata.
- Decode graphs replay; no long-prefill graph replays. SSE/collected agreement,
  mixed short-request progress, queue isolation, overflow rejection and graceful
  shutdown pass.

The initial mixed 64K cases have long TTFT 9.740–9.857 seconds and short TTFT
742–784 ms. Prefix caching is disabled in this run. These samples must not be
attributed to the completion-order fix or compared as a controlled speedup against
runs that also publish prefix snapshots. A finite repeated-load pass is not a
long-duration production soak. Prefix restoration on the updated frontend and
full-model transport-level slow-client qualification remain separate gates.

## Reproduction

Docker `inferfabric-92b` on 7.156.99.58; workspace
`/data/p00603624/inferfabric/openpangu92`; normal CANN/vendor environment, HOST
expansion, HCCL base port 44800 and explicit decode-graph/long-context flags.
Run `scripts/qualify-program-http.py` with server
`../inferfabric-server-release-order`, plan `bulk-domains8k-frozen-plan`, checkpoint
`/models/92B`, library `native-build-bulk-domains/libinferfabric_program.so`,
`batched16-preparation-reference.json`, `retrieval-64k-reference.json`, two active
sequences, 4096 token budget, packed prefill, first-EOS and arithmetic-history
checks, port 18085, 24 load rounds, four clients, 128 MiB RSS gate and 300-second
request inactivity timeout. Output: `release-order64k-load-http`.

`results.json` is the completed terminal report. Intermediate progress files are
preserved for provenance; they are not independently passing reports. The full
raw evidence is in `release-order64k-load-http/`.
