#!/usr/bin/env python3
"""Compare native JSONL request outputs with an independent token reference.

Writes mismatches as evidence before returning nonzero. Does not qualify device
residency or model readiness; use the separate native transfer/arena audits.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def compare(cases, records, repetitions, eos):
    if repetitions < 1 or not cases:
        raise ValueError('empty reference or invalid repetitions')
    groups = {i: [] for i in range(len(cases) * repetitions)}
    for row in records:
        if 'request' not in row:
            if {'step', 'token', 'top5'} <= row.keys():
                continue  # Existing sampler diagnostics, not a second output.
            raise ValueError('unknown native record')
        if row['request'] not in groups:
            raise ValueError('unexpected request ID')
        elapsed = row.get('elapsed_us')
        if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('invalid step duration')
        groups[row['request']].append(row)
    results = []
    for i, case in enumerate(cases):
        expected = case['output_ids']
        if not expected or not case['input_ids']:
            raise ValueError('empty reference request')
        runs = [groups[r * len(cases) + i] for r in range(repetitions)]
        for run in runs:
            if [r['step'] for r in run] != list(range(len(expected))):
                raise ValueError('missing, duplicate or out-of-order output steps')
        outputs = [[r['token'] for r in run] for run in runs]
        end = expected.index(eos) + 1 if eos in expected else len(expected)
        intervals = [r['elapsed_us'] / 1000 for run in runs for r in run[1:]]
        results.append(dict(
            case=case.get('case', str(i)), prompt_tokens=len(case['input_ids']),
            reference_output_ids=expected, native_output_ids_by_repetition=outputs,
            exact_all_tokens=all(v == expected for v in outputs),
            exact_through_reference_eos=all(v[:end] == expected[:end] for v in outputs),
            stable_repetitions=all(v == outputs[0] for v in outputs),
            repetitions=repetitions,
            ttft_ms_mean=statistics.mean(run[0]['elapsed_us'] / 1000 for run in runs),
            tpot_samples=len(intervals),
            tpot_ms_mean=statistics.mean(intervals) if intervals else None,
            tpot_ms_median=statistics.median(intervals) if intervals else None))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'tokens', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--repetitions', type=int, default=3)
    p.add_argument('--eos', type=int, required=True)
    args = p.parse_args()
    reference = args.reference.read_bytes()
    tokens = args.tokens.read_bytes()
    records = [json.loads(line) for line in tokens.decode().splitlines() if line.startswith('{')]
    result = dict(format='inferfabric.reference-comparison.v1',
                  reference_sha256=hashlib.sha256(reference).hexdigest(),
                  tokens_sha256=hashlib.sha256(tokens).hexdigest(),
                  timing_scope='native step durations; interleaved runs include peer work',
                  cases=compare(json.loads(reference), records, args.repetitions, args.eos))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    return 0 if all(c['exact_all_tokens'] for c in result['cases']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
