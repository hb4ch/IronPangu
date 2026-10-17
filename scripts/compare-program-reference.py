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
    if any(c.get('reference_intervention') for c in cases):
        raise ValueError('diagnostic intervention is not an independent reference')
    if any(c.get('qualification', {}).get('output_is_placeholder') for c in cases):
        raise ValueError('placeholder outputs are not a reference; run the independent reference first')
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


def check_answers(cases, results, decode, eos):
    """Independent task correctness: token parity can reproduce a wrong answer."""
    def answer(ids):
        end = ids.index(eos) if eos in ids else len(ids)
        return decode(ids[:end]).strip()
    for case, result in zip(cases, results):
        expected = case.get('qualification', {}).get('expected_answer')
        if expected is None:
            continue
        reference = answer(result['reference_output_ids'])
        native = [answer(ids) for ids in result['native_output_ids_by_repetition']]
        result['retrieval'] = dict(expected_answer=expected, reference_answer=reference,
                                  native_answers=native, reference_correct=reference == expected,
                                  native_correct=all(value == expected for value in native))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'tokens', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--repetitions', type=int, default=3)
    p.add_argument('--eos', type=int, required=True)
    p.add_argument('--tokenizer', type=Path, help='Required for cases with expected textual answers')
    args = p.parse_args()
    reference = args.reference.read_bytes()
    tokens = args.tokens.read_bytes()
    records = [json.loads(line) for line in tokens.decode().splitlines() if line.startswith('{')]
    cases = json.loads(reference)
    result = dict(format='inferfabric.reference-comparison.v1',
                  reference_sha256=hashlib.sha256(reference).hexdigest(),
                  tokens_sha256=hashlib.sha256(tokens).hexdigest(),
                  timing_scope='native step durations; interleaved runs include peer work',
                  cases=compare(cases, records, args.repetitions, args.eos))
    if any('expected_answer' in c.get('qualification', {}) for c in cases):
        if args.tokenizer is None:
            p.error('--tokenizer is required to verify expected textual answers')
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(args.tokenizer))
        result['tokenizer_sha256'] = hashlib.sha256(args.tokenizer.read_bytes()).hexdigest()
        check_answers(cases, result['cases'], lambda ids: tokenizer.decode(ids, skip_special_tokens=True), args.eos)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    return 0 if all(c['exact_all_tokens'] and c.get('retrieval', {}).get('reference_correct', True)
                    and c.get('retrieval', {}).get('native_correct', True) for c in result['cases']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
