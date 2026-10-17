# 64K repeated-load gate: transient FD growth — 2026-10-18

**Failed resource gate; do not label this run passed.** The qualified v9 bulk-domain
backend completes 24 four-client waves and 120 reference-checked load/recovery
responses, with one deliberate disconnect per wave. Each wave includes a rotating
64K retrieval case and three short cases. Prefix caching is disabled, so the
long requests recompute their input. The EOS-scoped output comparisons pass.

Post-wave RSS grows 73.961 MiB, below the unchanged 128 MiB limit. Threads remain
663. FD count is 76 except at wave 14, which observes 77; the final sample returns
to 76. The unchanged gate checks the maximum across all samples, so it fails.
The observations span at least 261 seconds between first and final wave samples.
This does not establish a long-duration memory bound.

The old sampler records counts only. It cannot identify whether the extra FD is
a closing HTTP socket, another temporary resource or a leak. No diagnosis is
claimed. The qualifier now records each FD target and matching TCP state from
`/proc`, retaining enumerated descriptors that close during observation as null.
Its count and acceptance rule remain unchanged. Thirteen helper tests pass,
including socket attribution and the close-during-observation race.

Raw evidence is `bulk-domains64k-load-http/`; `load-progress.json` contains all
completed waves but is deliberately **not** a passing terminal report. No
`results.json` was published. `resource-growth.json` is the authoritative failure.
Later terminal resource checks were not reached and must not be inferred passed.
The process exits 1 after reporting the resource-growth failure.

Reproduction: same frozen plan, library and frontend as
`../2026-10-18-bulk-domains/README.md`, HCCL port 44400, HTTP 18085;
`qualify-program-http.py --load-rounds 24 --load-concurrency 4
--load-rss-growth-limit-mib 128 --resident-cases retrieval-64k-reference.json`,
with packed prefill, first-EOS, arithmetic-history checks, two active sequences,
4096 token budget and 300-second request inactivity timeout. No prefix-case flag.
The diagnostic rerun uses the same settings and the new FD identity recorder;
it does not widen the gate or change model execution.
