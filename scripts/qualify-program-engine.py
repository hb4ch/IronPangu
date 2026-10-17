#!/usr/bin/env python3
"""Offline regression client; production engine/frontend never depend on Python.

Alternate complete model fixtures in one loaded native rank group. Engine TTFT
includes reset and prefill, excludes model loading/tokenization/network. TPOT
includes native sampling and rank consensus. This is not an HTTP load benchmark.
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
    for name in ('binary', 'plan', 'checkpoint', 'cases', 'output'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--prepare', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = json.loads(args.cases.read_text()) * 3
    requests = args.output / 'requests.txt'
    requests.write_text(''.join(f"{len(c['output_ids'])} " + ' '.join(map(str, c['input_ids'])) + '\n' for c in cases))
    env = dict(os.environ, INFERFABRIC_DECODE_GRAPH='1',
               INFERFABRIC_PREPARED_AUDIT_DIR=str((args.output / 'prepared').resolve()),
               INFERFABRIC_TRANSFER_AUDIT_DIR=str((args.output / 'transfers').resolve()))
    env.pop('INFERFABRIC_PREPARE_ARENA', None)
    if args.prepare:
        env['INFERFABRIC_PREPARE_ARENA'] = '1'
    with (args.output / 'stderr.log').open('w') as err:
        result = subprocess.run([str(args.binary.resolve()), str(args.plan), str(args.checkpoint), str(requests)],
                                env=env, stdout=subprocess.PIPE, stderr=err, text=True, timeout=1800)
    (args.output / 'tokens.jsonl').write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f'native engine exited {result.returncode}; inspect stderr.log')
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
    expected_total = sum(len(c['output_ids']) for c in cases)
    if len(rows) != expected_total:
        raise RuntimeError('incomplete or unexpected output records')
    measurements = []
    for request, case in enumerate(cases):
        actual = [r for r in rows if r['request'] == request]
        if [r['step'] for r in actual] != list(range(len(case['output_ids']))) or [r['token'] for r in actual] != case['output_ids']:
            raise RuntimeError(f"request {request} {case['case']}: full output mismatch")
        measurements.append(dict(request=request, case=case['case'], prompt_tokens=len(case['input_ids']),
                                 output_tokens=len(actual), engine_ttft_ms=actual[0]['elapsed_us']/1000,
                                 tpot_ms=statistics.mean(r['elapsed_us']/1000 for r in actual[1:])))
    for rank in range(8):
        audit = json.loads((args.output / 'transfers' / f'rank-{rank}.json').read_text())
        if audit['activation_host_transfers'] or audit['rejected']:
            raise RuntimeError('residency violation')
        prepared = json.loads((args.output / 'prepared' / f'rank-{rank}.json').read_text())
        if not args.prepare and (not prepared['workspace']['planned'] or prepared['workspace']['growth_allocations']):
            raise RuntimeError('workspace growth or missing fixed plan')
    report = dict(format='inferfabric.persistent-program-qualification.v1', requests=len(cases),
                  output_tokens=expected_total, exact_fixture_tokens=True,
                  fixture_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                  scope='one loaded TP8/EP8 group, serial token graph; no packed prefill or long-context claim',
                  measurements=measurements)
    (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
