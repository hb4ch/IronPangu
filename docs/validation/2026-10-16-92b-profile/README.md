# Full 92B decode profile — 2026-10-16

See [findings and ordered optimization plan](../../92b-decode-optimization-plan.md).

## Saved evidence

- `profile-analysis.json`: every rank's lightweight timings, eight-device task
  summaries, overlap-corrected task interval unions, host-counter conversion and
  SHA-256 hashes of the parsed SQLite sources.
- `profile-final-{arithmetic,geography,chinese}/`: two captured full decoder steps
  per request, all eight rank CSVs, full output records, correctness result and
  transfer audits for the whole request. All 33 output records match the resident
  reference exactly; all rank activation-host-transfer and rejection counts zero.
- `profile-vendor-92b/`: lightweight timers collected alongside CANN tracing and
  its full-generation correctness result. Vendor timing is strongly perturbed.

Final lightweight instrumented native binary SHA-256:
`b0637ce911240a2c1040ee39a19564e02476f82c14a6a7d1bf2cc7d3687cdc2d`.
Execution binary:
`46fa67de15080a6ccf370ea852f2d7a6744b532603bdf73ded7f59c34a082513`.
The binary hash was read after the three final runs, before any further rebuild.
The vendor capture preceded the added HCCL host API scopes; its executable was
subsequently replaced and no binary hash was retained. Its parsed source database
hashes are preserved. An explicit standard-header include was added after these
measurements; this does not change the execution algorithm. Future runner outputs
include input/binary hashes automatically.

The 3.8 GB CANN capture remains inside `inferfabric-92b` on `7.156.99.58` at:

```
/data/p00603624/inferfabric/openpangu92/profile-vendor-92b/vendor/PROF_000001_20261016154157559_00102431IFLPIBPN
```

Only small evidence is checked into Git. Export produced parsed task/API SQLite
and a timeline; warnings about absent summary/cluster-analysis data mean those
reports are not claimed as successful. Host ApiData stores 100 MHz counter ticks;
the analyzer converts using `host/info.json`, not by treating ticks as ns. Device
task duration columns used here are normalized ns. Kernel summaries count AI_CORE
parents only; task unions deduplicate overlapping mixed-kernel children.

## Reproduce inside the configured Docker workspace

Source the CANN and both omni custom-operator environments first. Build native
C++ with `cmake --build native-build --target openpangu_decode -j4`.

```sh
export HCCL_OP_EXPANSION_MODE=HOST HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=120
python3 scripts/profile-native-decode.py \
  --binary native-build/openpangu_decode --plan resident-dsl-plan \
  --checkpoint /models/92B --cases decoder-resident-qualified/results.json \
  --case chinese --output NEW_HOST_PROFILE
# Repeat for arithmetic and geography; each output directory must be new.
python3 scripts/profile-native-decode.py \
  --binary native-build/openpangu_decode --plan resident-dsl-plan \
  --checkpoint /models/92B --cases decoder-resident-qualified/results.json \
  --case chinese --output NEW_VENDOR_PROFILE --vendor
/usr/local/Ascend/cann/tools/profiler/bin/msprof \
  --export=on --output=NEW_VENDOR_PROFILE/vendor
python3 scripts/analyze-native-decode-profile.py \
  --vendor NEW_VENDOR_PROFILE/vendor --host NEW_HOST_PROFILE \
  --output profile-analysis.json
```

The native environment variables are `INFERFABRIC_PROFILE_DIR` and optional
`INFERFABRIC_PROFILE_VENDOR`. Without a profile directory, no profiling rendezvous
is added and the timer maps remain inactive. The current capture is deliberately
bounded to two decode steps after the first emitted output. The runner verifies
that both iterations were captured on every rank. CANN 9.1 parsed SQLite schema
is required by this analyzer; it is not a universal profiler importer.

The final source and harness were rebuilt and re-run once more after the header
include and provenance checks were added. `profile-provenance-check/` contains the
17-record Chinese repeat, all eight transfer audits/timers and captured input
hashes; it passes exact-record comparison and residency checks. It is excluded
from the three-case timing aggregate above. Instrumentation commit: `eb9a92a`.
Local validation: 42 compiler tests, compiler Clippy with warnings denied, Python
syntax/evidence checks and the local AArch64 compiler cross-build passed.
