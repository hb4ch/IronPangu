#!/usr/bin/env python3
"""Summarize one exported CANN capture without treating overlapping sums as latency."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def union_us(intervals):
    total = 0.0
    end = None
    for start, stop in sorted(intervals):
        if stop < start:
            raise ValueError('negative task interval')
        total += stop - start if end is None else max(0, stop - max(start, end))
        end = stop if end is None else max(end, stop)
    return total


def summarize(task_file, api_file, communication_file):
    devices = defaultdict(lambda: defaultdict(list))
    with task_file.open() as stream:
        for row in csv.DictReader(stream):
            devices[row['Device_id']][row['kernel_type']].append(
                (float(row['task_start(us)']), float(row['task_stop(us)'])))
    result = []
    for device, kinds in sorted(devices.items()):
        intervals = [i for values in kinds.values() for i in values]
        result.append({'device': device,
                       'task_span_ms': (max(i[1] for i in intervals) - min(i[0] for i in intervals)) / 1000,
                       'any_task_union_ms': union_us(intervals) / 1000,
                       'kinds': {kind: {'count': len(values),
                                        'sum_ms': sum(b-a for a, b in values) / 1000,
                                        'union_ms': union_us(values) / 1000}
                                 for kind, values in sorted(kinds.items())}})
    with api_file.open() as stream:
        apis = sorted(csv.DictReader(stream), key=lambda row: float(row['Time(us)']), reverse=True)
    with communication_file.open() as stream:
        communication = list(csv.DictReader(stream))
    return {'format': 'inferfabric.serving-timeline.v1', 'devices': result,
            'host_api_inclusive': apis, 'communication_inclusive': communication,
            'sources': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (task_file, api_file, communication_file)},
            'caveats': ['API and communication durations include waits and overlap.',
                        'Task kinds overlap; unions must not be added across kinds or ranks.',
                        'Notify waits are not useful compute. Device task span excludes some host work.',
                        'Absolute timestamps are parsed as floating-point microseconds; submicrosecond precision is limited.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    files = []
    for stem in ('task_time', 'api_statistic', 'communication_statistic'):
        matches = list(args.root.rglob(stem + '_*.csv'))
        if len(matches) != 1:
            raise ValueError(f'expected exactly one {stem} table, found {len(matches)}')
        files.append(matches[0])
    args.output.write_text(json.dumps(summarize(*files), indent=2) + '\n')


if __name__ == '__main__':
    main()
