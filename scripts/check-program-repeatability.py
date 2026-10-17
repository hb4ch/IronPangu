#!/usr/bin/env python3
"""Check repeated sampler outputs without reading device activations.

This complements the native correctness/resource qualifier. Identical sampled
tokens alone can conceal drifting logits. References are never rewritten.
"""
import argparse
import hashlib
import json
from pathlib import Path


def compare(cases, diagnostics, repetitions, eos=None):
    if not cases or repetitions < 2:
        raise ValueError('requires fixtures and at least two repetitions')
    groups = [[] for _ in range(len(cases) * repetitions)]
    for row in diagnostics:
        request = row['request']
        if type(request) is not int or not 0 <= request < len(groups):
            raise ValueError('unexpected diagnostic request')
        groups[request].append(row)
    results = []
    for index, case in enumerate(cases):
        expected = case['output_ids']
        if not expected or case.get('qualification', {}).get('output_is_placeholder'):
            raise ValueError('empty or placeholder reference')
        count = expected.index(eos) + 1 if eos is not None and eos in expected else len(expected)
        runs = [groups[index + repeat * len(cases)] for repeat in range(repetitions)]
        for run in runs:
            if not count <= len(run) <= len(expected) or [r['generation_step'] for r in run] != list(range(len(run))):
                raise ValueError('missing, duplicate or unordered diagnostic steps')
        samples = [[{key: row[key] for key in ('token', 'logit', 'top5')}
                    for row in run[:count]] for run in runs]
        differences = [dict(repetition=repeat, generation_step=step,
                            first=samples[0][step], actual=value)
                       for repeat, run in enumerate(samples[1:], 1)
                       for step, value in enumerate(run) if value != samples[0][step]]
        results.append(dict(case=case['case'], compared_tokens_per_run=count,
                            ignored_post_eos_records=sum(len(run)-count for run in runs),
                            stable=not differences, differences=differences))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cases', 'diagnostics', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--eos', type=int, help='Compare only through this EOS; ignore forced suffix records')
    args = parser.parse_args()
    case_bytes, diagnostic_bytes = args.cases.read_bytes(), args.diagnostics.read_bytes()
    results = compare(json.loads(case_bytes), json.loads(diagnostic_bytes), args.repetitions, args.eos)
    passed = all(case['stable'] for case in results)
    report = dict(format='inferfabric.sampler-repeatability.v1', passed=passed,
                  scope='through_first_eos' if args.eos is not None else 'full_reference_length',
                  criterion='exact sampled token, logit and top-five diagnostics across repeated requests',
                  reference_sha256=hashlib.sha256(case_bytes).hexdigest(),
                  diagnostics_sha256=hashlib.sha256(diagnostic_bytes).hexdigest(), cases=results)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=passed, cases=len(results), output=str(args.output))))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
