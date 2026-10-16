#!/usr/bin/env python3
"""Plan one maximum shared workspace per rank from hardware discovery audits.

Discovery is not bindable. A fresh preparation run must consume this plan and
exercise capture before binding its arena. Exceeding the sampled bound fails.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

spec = importlib.util.spec_from_file_location('arena_binding', Path(__file__).with_name('bind-prepared-arena.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def plan(source, output, audits):
    program = module.fingerprint((source / 'execution.bin').read_bytes())
    ranks = len(list(source.glob('rank-*.bin')))
    if not ranks or output.exists() or (source / 'arena.json').exists():
        raise ValueError('requires an unbound source and a new output directory')
    records = []
    for rank in range(ranks):
        requests, backends, evidence = {}, set(), []
        for directory in audits:
            path = directory / f'rank-{rank}.json'
            raw = path.read_bytes()
            audit = json.loads(raw)
            arena, workspace = audit['arena'], audit['workspace']
            if arena['program_fingerprint'] != program:
                raise ValueError('workspace discovery belongs to another program')
            if not audit['steps'] or not any(g['replays'] for g in audit['graphs']):
                raise ValueError('workspace discovery must exercise token execution and graph replay')
            backends.add(arena['backend_fingerprint'])
            observed = workspace['instruction_requests']
            if max(observed.values(), default=0) != workspace['peak_requested_bytes']:
                raise ValueError('inconsistent workspace peak')
            for instruction, size in observed.items():
                if not isinstance(size, int) or size < 0 or size > 1 << 40:
                    raise ValueError('invalid workspace requirement')
                requests[instruction] = max(requests.get(instruction, 0), size)
            evidence.append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()})
        if len(backends) != 1:
            raise ValueError('discovery mixes backend binaries')
        records.append({'rank': rank, 'bytes': max(requests.values(), default=0),
                        'backend_fingerprint': next(iter(backends)),
                        'program_fingerprint': program, 'instruction_requests': requests,
                        'evidence': evidence})
    if len({r['backend_fingerprint'] for r in records}) != 1:
        raise ValueError('ranks use different backend binaries')
    shutil.copytree(source, output)
    for record in records:
        (output / f"workspace-rank-{record['rank']}.txt").write_text(
            f"IFWORKSPACE1 {program} {record['backend_fingerprint']} {record['bytes']}\n")
    metadata = {'format': 'inferfabric.shared-workspace.v1',
                'policy': 'one permanent lease on owning stream; no growth; fail on overflow',
                'scope': 'maximum observed vendor/compiled-kernel workspace across supplied discovery runs',
                'ranks': records}
    (output / 'workspace.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'workspace_bytes_per_rank': [r['bytes'] for r in records]}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('audits', type=Path, nargs='+')
    args = parser.parse_args()
    plan(args.source, args.output, args.audits)
