#!/usr/bin/env python3
"""Compare native identity/shuffled page access across boundaries in one engine.

This is layout equivalence against the serial backend, not external model-quality
or learned long-context indexer qualification. Python only drives an offline probe.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('binary', 'plan', 'checkpoint', 'bindings', 'cases', 'output'):
        p.add_argument('--' + key, required=True, type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.plan / 'serving.json').read_text())
    context = manifest['variants'][0]['max_context']
    steps = 8
    lengths = sorted({n for n in (1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 255, 480, context-steps) if 0 < n <= context-steps})
    source = json.loads(args.cases.read_text())
    requests = []
    for i, length in enumerate(lengths):
        seed = source[i % len(source)]['input_ids']
        prompt = (seed * ((length + len(seed)-1) // len(seed)))[:length]
        for _ in range(2):
            requests.append(str(steps) + ' ' + ' '.join(map(str, prompt)) + '\n')
    request_file = args.output / 'requests.txt'
    request_file.write_text(''.join(requests))
    env = dict(os.environ, INFERFABRIC_DECODE_GRAPH='1',
               INFERFABRIC_PAGE_BINDINGS=str(args.bindings.resolve()),
               INFERFABRIC_ALTERNATE_PAGE_LAYOUTS='1',
               INFERFABRIC_PREPARED_AUDIT_DIR=str((args.output/'prepared').resolve()),
               INFERFABRIC_TRANSFER_AUDIT_DIR=str((args.output/'transfers').resolve()))
    env.pop('INFERFABRIC_PREPARE_ARENA', None)
    with (args.output/'stderr.log').open('w') as err:
        result = subprocess.run([str(args.binary.resolve()), str(args.plan), str(args.checkpoint), str(request_file)],
                                env=env, stdout=subprocess.PIPE, stderr=err, text=True, timeout=1800)
    (args.output/'tokens.jsonl').write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f'probe failed: {result.returncode}')
    records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
    if len(records) != len(requests)*steps:
        raise RuntimeError('incomplete boundary records')
    measurements = []
    for i, length in enumerate(lengths):
        pair = [[r for r in records if r['request'] == 2*i + layout] for layout in range(2)]
        for rows in pair:
            if [r['step'] for r in rows] != list(range(steps)):
                raise RuntimeError('invalid output step order')
        if [r['token'] for r in pair[0]] != [r['token'] for r in pair[1]]:
            raise RuntimeError(f'page layout changed output at prompt length {length}')
        measurements.append(dict(prompt_tokens=length, output_tokens=steps,
            layouts=[dict(layout=name, engine_ttft_ms=rows[0]['elapsed_us']/1000,
                          tpot_ms=statistics.mean(r['elapsed_us']/1000 for r in rows[1:]))
                     for name, rows in zip(('identity', 'shuffled'), pair)]))
    for rank in range(len(manifest['identity']['devices'])):
        transfer = json.loads((args.output/'transfers'/f'rank-{rank}.json').read_text())
        prepared = json.loads((args.output/'prepared'/f'rank-{rank}.json').read_text())
        if transfer['activation_host_transfers'] or transfer['rejected']:
            raise RuntimeError('device residency violation')
        if not prepared['arena']['frozen'] or not prepared['workspace']['planned'] or prepared['workspace']['growth_allocations']:
            raise RuntimeError('unplanned arena/workspace')
        if any(s['device_allocations'] or s['completion_boundaries'] for s in prepared['steps']):
            raise RuntimeError('graph allocation/internal fence')
    report = dict(scope='serial full-model layout equivalence; no external long-context correctness claim',
                  exact_layout_equivalence=True, requests=len(requests), output_tokens=len(records),
                  bindings_sha256=hashlib.sha256(args.bindings.read_bytes()).hexdigest(),
                  requests_sha256=hashlib.sha256(request_file.read_bytes()).hexdigest(), measurements=measurements)
    (args.output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
