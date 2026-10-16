#!/usr/bin/env python3
"""Time flushed native output records locally; no model execution in Python.

TPOT excludes the first output: (last receipt - first receipt)/(outputs - 1).
Includes decode, NPU selection/consensus, output serialization and pipe delivery.
Process-to-first-output includes checkpoint loading and prefill; it is not warm TTFT.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import statistics
import subprocess
import time


def stats(values):
    ordered = sorted(values)
    return {'intervals': len(values), 'mean_ms': statistics.mean(values),
            'median_ms': statistics.median(values),
            'p95_ms': ordered[math.ceil(.95 * len(ordered)) - 1],
            'min_ms': ordered[0], 'max_ms': ordered[-1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--cases', required=True, help='Qualified results JSON containing input_ids/output_ids')
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=900)
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout <= 0:
        parser.error('positive repeats/timeout required')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    fixtures = json.loads(Path(args.cases).read_text())
    results = []
    intervals = []
    for repeat in range(args.repeats):
        for case in fixtures:
            name = f"{case['case']}-{repeat}"
            command = [args.binary, args.plan, args.checkpoint, '32', *map(str, case['input_ids'])]
            rows, receipts, pending = [], [], b''
            start = time.perf_counter_ns()
            with (out / f'{name}.stderr').open('wb') as error:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=error,
                                           start_new_session=True)
                selector = selectors.DefaultSelector()
                selector.register(process.stdout, selectors.EVENT_READ)
                try:
                    while selector.get_map():
                        if (time.perf_counter_ns() - start) / 1e9 > args.timeout:
                            raise TimeoutError(name)
                        for key, _ in selector.select(1):
                            data = os.read(key.fileobj.fileno(), 65536)
                            stamp = time.perf_counter_ns()
                            if not data:
                                selector.unregister(key.fileobj)
                                continue
                            pending += data
                            while b'\n' in pending:
                                line, pending = pending.split(b'\n', 1)
                                if not line.startswith(b'{'):
                                    continue
                                row = json.loads(line)
                                rows.append(row)
                                receipts.append(stamp)
                                if len(rows) == 1:
                                    print(json.dumps({'case': name, 'first_output_s': (stamp-start)/1e9}), flush=True)
                    code = process.wait(timeout=30)
                    if code or pending or len(rows) < 2:
                        raise RuntimeError(f'{name}: incomplete native run, exit={code}')
                finally:
                    selector.close()
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    process.stdout.close()
            if [r['token'] for r in rows] != case['output_ids']:
                raise RuntimeError(f'{name}: output differs from qualified native sequence')
            elapsed = [(b-a)/1e6 for a,b in zip(receipts,receipts[1:])]
            if any(t <= 0 for t in elapsed):
                raise RuntimeError('coalesced output records; cannot resolve token intervals')
            intervals.extend(elapsed)
            result = {'case': case['case'], 'repeat': repeat, 'prompt_tokens': len(case['input_ids']),
                      'output_tokens': len(rows), 'process_to_first_output_s': (receipts[0]-start)/1e9,
                      'decode_span_s': (receipts[-1]-receipts[0])/1e9, 'tpot': stats(elapsed),
                      'interval_ms': elapsed, 'receipt_offset_ns': [t-start for t in receipts],
                      'exact_qualified_tokens': True, 'outputs': rows}
            results.append(result)
            (out/'runs.json').write_text(json.dumps(results, indent=2)+'\n')
            print(json.dumps({k:v for k,v in result.items() if k not in ('outputs','receipt_offset_ns','interval_ms')}), flush=True)
    summary = {'format': 'inferfabric.native-tpot.v1', 'binary_sha256': hashlib.sha256(Path(args.binary).read_bytes()).hexdigest(),
               'execution_plan_sha256': hashlib.sha256((Path(args.plan)/'execution.bin').read_bytes()).hexdigest(),
               'clock': 'perf_counter_ns on the NPU host inside Docker; flushed stdout receipts',
               'method': 'exclude first output; includes native decode, selection, consensus and output delivery',
               'batch_size': 1, 'repeats': args.repeats, 'runs': len(results), 'pooled_tpot': stats(intervals),
               'decode_tokens_per_second': 1000/statistics.mean(intervals),
               'notes': ['Fresh native process per case; loading/prefill excluded from TPOT.',
                         'No warm persistent server, ACLgraph or concurrent batching; short-context baseline.',
                         'P95 uses nearest-rank empirical percentile; small sample, not a production SLO.']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
