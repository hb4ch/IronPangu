#!/usr/bin/env python3
"""Summarize exported CANN kernel tables; accepts native or Torch profiler trees."""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--iterations', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error('--iterations must be positive and match the recorded window')
    tables = sorted(Path(args.root).rglob('ai_core_op_summary.db'))
    if not tables:
        raise ValueError('No exported CANN kernel tables; run msprof --export first')
    reports = []
    for table in tables:
        with sqlite3.connect(f'file:{table}?mode=ro', uri=True) as connection:
            rows = connection.execute('''
                select g.op_type,count(*),sum(t.duration_time)
                from ge_summary g join task_time t
                on g.task_id=t.task_id and g.stream_id=t.stream_id and g.batch_id=t.batch_id
                where t.task_type='AI_CORE'
                group by g.op_type order by sum(t.duration_time) desc
            ''').fetchall()
        reports.append({
            'table': str(table),
            'sha256': hashlib.file_digest(table.open('rb'), 'sha256').hexdigest(),
            'kernels': [{'type': op, 'count_per_iteration': count / args.iterations,
                         'summed_ms_per_iteration': duration / 1e6 / args.iterations}
                        for op, count, duration in rows],
        })
    result = {'format': 'inferfabric.cann-kernel-summary.v1',
              'iterations': args.iterations, 'devices': reports,
              'caveats': ['Inclusive task sums overlap across streams and are not TPOT.',
                          'Profiler overhead and cross-rank waits distort latency.',
                          'Iteration count must be established from the capture window.']}
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
