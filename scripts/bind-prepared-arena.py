#!/usr/bin/env python3
"""Bind measured per-rank slots to a new immutable execution bundle.

Run after a full prepared audit in the same backend binary. This preserves the
qualified exact-size/LIFO reuse policy; it does not claim optimal packing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def fingerprint(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return str(value)


def bind(plan, audits, output):
    program = (plan / "execution.bin").read_bytes()
    expected = fingerprint(program)
    ranks = len(list(plan.glob("rank-*.bin")))
    if not ranks:
        raise ValueError("no rank initialization plans")
    arenas = []
    for rank in range(ranks):
        audit = json.loads((audits / f"rank-{rank}.json").read_text())
        arena = audit["arena"]
        if arena["program_fingerprint"] != expected:
            raise ValueError("preparation audit belongs to another program")
        if arena["format"] != "inferfabric.frozen-arena.v1":
            raise ValueError("unsupported arena format")
        if not any(g["replays"] for g in audit["graphs"]):
            raise ValueError("hardware binding requires a observed captured path")
        end = 0
        for index, slot in enumerate(arena["slots"]):
            if (slot["id"] != index or slot["offset"] % 512 or
                    slot["offset"] < end or slot["bytes"] <= 0):
                raise ValueError("invalid arena slot")
            end = slot["offset"] + slot["bytes"]
        if not end or end != arena["bytes"] or end > 1 << 40:
            raise ValueError("invalid arena extent")
        arenas.append(arena)
    if len({a["backend_fingerprint"] for a in arenas}) != 1:
        raise ValueError("inconsistent backend binary")
    if output.exists():
        raise ValueError("output must not exist")
    shutil.copytree(plan, output)
    for rank, arena in enumerate(arenas):
        lines = [f"IFARENA1 {expected} {arena['backend_fingerprint']} "
                 f"{arena['bytes']} {len(arena['slots'])}"]
        lines.extend(f"{s['offset']} {s['bytes']}" for s in arena["slots"])
        bindings = arena["instruction_slots"]
        lines.append(f"bindings {len(bindings)}")
        for instruction, slots in sorted(bindings.items(), key=lambda pair: int(pair[0])):
            lines.append(f"{instruction} {len(slots)} " + " ".join(map(str, slots)))
        (output / f"arena-rank-{rank}.txt").write_text("\n".join(lines) + "\n")
    metadata = {
        "format": "inferfabric.bound-arena.v1",
        "execution_sha256": hashlib.sha256(program).hexdigest(),
        "policy": "fixed disjoint slots; exact-size LIFO lease reuse on owning stream",
        "scope": "prepared tensors and vendor workspace; checkpoint allocations separate",
        "ranks": arenas,
    }
    (output / "arena.json").write_text(json.dumps(metadata, indent=2) + "\n")
    html = (output / "execution.html").read_text()
    pattern = r'(<script type="application/json" id="plan-data">)(.*?)(</script>)'
    match = re.search(pattern, html, re.S)
    if not match:
        raise ValueError("plan viewer data missing")
    data = json.loads(match[2])
    data["arena"] = metadata
    safe = json.dumps(data).replace("<", "\\u003c")
    html = html[:match.start(2)] + safe + html[match.end(2):]
    (output / "execution.html").write_text(html)
    print(json.dumps({"output": str(output), "bytes_per_rank": [a["bytes"] for a in arenas]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("audits", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    bind(args.plan, args.audits, args.output)
