#!/usr/bin/env python3
"""Run the cross-built full-model Rust consumer test; retain all-rank audits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'plan', 'checkpoint', 'library', 'cases', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, INFERFABRIC_DECODE_GRAPH='1')
    env.pop('INFERFABRIC_PREPARE_ARENA', None)
    for key in ('plan', 'checkpoint', 'library', 'cases'):
        env['INFERFABRIC_TEST_' + key.upper()] = str(getattr(args, key).resolve())
    for key in ('prepared', 'transfer'):
        env['INFERFABRIC_' + key.upper() + '_AUDIT_DIR'] = str((args.output/key).resolve())
    command = [str(args.binary.resolve()), '--ignored', '--exact',
               'native_backend::program_backpressure_test::npu_bounded_consumer_resumption_and_cancellation', '--nocapture']
    began = time.monotonic()
    with (args.output/'test.log').open('w') as log:
        try:
            result = subprocess.run(command, env=env, stdout=log, stderr=log, timeout=1200)
        except subprocess.TimeoutExpired:
            (args.output/'process.json').write_text(json.dumps(dict(completed=False, timed_out=True))+'\n')
            raise
    (args.output/'process.json').write_text(json.dumps(dict(completed=True, timed_out=False,
        returncode=result.returncode, elapsed_seconds=time.monotonic()-began), indent=2)+'\n')
    logs = (args.output/'test.log').read_text()
    conserved=re.search(r'bounded consumer: (\d+) EOS-scoped tokens conserved',logs)
    if result.returncode or 'test result: ok. 1 passed; 0 failed' not in logs or not conserved:
        raise RuntimeError('full-model backpressure test did not pass')
    manifest = json.loads((args.plan/'serving.json').read_text())
    ranks = []
    for rank in range(len(manifest['identity']['devices'])):
        audit = json.loads((args.output/'prepared'/f'rank-{rank}.json').read_text())
        transfer = json.loads((args.output/'transfer'/f'rank-{rank}.json').read_text())
        if not audit['steps'] or not any(graph['replays'] > 1 for graph in audit['graphs']):
            raise RuntimeError('backpressure test did not exercise replay on every rank')
        if not audit['arena']['frozen'] or not audit['workspace']['planned'] or audit['workspace']['growth_allocations']:
            raise RuntimeError('backpressure test escaped frozen resource plans')
        if transfer['activation_host_transfers'] or transfer['rejected'] or any(
                step['device_allocations'] or step['completion_boundaries'] for step in audit['steps']):
            raise RuntimeError('backpressure test violated device residency or replay contract')
        ranks.append(dict(rank=rank, arena_bytes=audit['arena']['bytes'], graphs=audit['graphs']))
    pool = re.search(r'shared state pool .*remaining_occupied_bytes=(\d+)', logs)
    if not pool or int(pool[1]):
        raise RuntimeError('state ownership was not reclaimed')
    report = dict(scope='native worker bounded consumer; not HTTP socket backpressure',
                  reference_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                  binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                  library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                  execution_sha256=hashlib.sha256((args.plan/'execution.bin').read_bytes()).hexdigest(),
                  output_tokens_conserved=int(conserved[1]), reference_scope='through_first_eos',
                  test_queue_capacity=2, production_queue_capacity=64,
                  peer_reference=True, cancellation=True, slot_reuse=True,
                  remaining_occupied_bytes=0, ranks=ranks)
    (args.output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
