#!/usr/bin/env python3
"""Report observed collective algorithms and payloads from an exported CANN trace."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3


def summarize(table):
    sizes = {'FP32': 4, 'FP16': 2, 'BFP16': 2, 'INT8': 1, 'INT16': 2,
             'INT32': 4, 'INT64': 8, 'UINT8': 1}
    with sqlite3.connect(f'file:{table}?mode=ro', uri=True) as connection:
        rows = connection.execute('''
            select op_type,alg_type,data_type,"count",rank_size,count(*)
            from HCCLOpSingleDevice
            group by op_type,alg_type,data_type,"count",rank_size
            order by op_type,alg_type,data_type,"count",rank_size
        ''').fetchall()
    groups = []
    for op, algorithm, dtype, elements, ranks, calls in rows:
        if elements < 0 or int(elements) != elements or ranks <= 0:
            raise ValueError('invalid collective extent')
        groups.append(dict(operation=op, algorithm=algorithm, dtype=dtype,
            elements_per_call=int(elements), rank_count=ranks, calls=calls,
            input_bytes_per_call=int(elements)*sizes[dtype] if dtype in sizes else None))
    return dict(table=str(table), sha256=hashlib.sha256(table.read_bytes()).hexdigest(),
                collectives=groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tables = sorted(args.root.rglob('hccl_single_device.db'))
    if not tables:
        raise ValueError('no exported collective tables')
    result = dict(format='inferfabric.serving-collectives.v1',
        devices=[summarize(table) for table in tables],
        caveats=['Algorithms are profiler-reported labels, not proof of a particular executor implementation.',
                 'Input payload bytes are not measured link traffic or critical-path time.',
                 'Unknown dtype sizes are null, never zero.'])
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
