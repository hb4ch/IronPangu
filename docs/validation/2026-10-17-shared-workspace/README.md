# Shared workspace qualification — 2026-10-17

The implementation uses a hardware-discovery pass followed by fresh preparation
with one fixed maximum shared workspace. C8 discovery and fixed preparation
passed the full 92B TP8/EP8 arithmetic, geography and Chinese fixtures, including
complete token/logit/top-five records and graph replay.

All eight ranks planned 130,916,352 workspace bytes. The new C8 frozen arena is
12,408,071,232 bytes per rank, versus 12,523,057,728 previously: 114,986,496 bytes
(109.66 MiB) removed. Final bound graph/eager performance and fresh BF16 comparison
are being collected before publishing the measurement milestone.

Local planner/binder regressions pass; native tests cover 16 invalid manifest
contracts, fixed-address reuse and overflow rejection. See
[the protocol](../../shared-workspace-planning.md).
