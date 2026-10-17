#!/usr/bin/env python3
"""Inspect output-only sampler diagnostics up to the first differing token.

This is not a qualification gate or a comparison of intermediate activations.
Scores after a divergent autoregressive history are deliberately not compared.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def analyze(case, records, actual_input_ids):
    if case.get('reference_intervention') or case.get('qualification', {}).get('output_is_placeholder'):
        raise ValueError('an independent reference is required')
    if actual_input_ids != case['input_ids']:
        raise ValueError('diagnostic prompt differs from reference')
    outputs = [row for row in records if 'request' in row]
    diagnostics = [row for row in records if 'top5' in row and 'request' not in row]
    if not outputs or len(outputs) != len(diagnostics) or not case['input_ids']:
        raise ValueError('missing or ambiguous output diagnostics')
    if any(row['request'] != 0 for row in outputs) or [row['step'] for row in outputs] != list(range(len(outputs))):
        raise ValueError('requires one request with contiguous output steps')
    if len(outputs) > len(case['output_ids']) or len(case['logprobs']) != len(case['output_ids']):
        raise ValueError('reference output/logprob extent mismatch')
    for index, (row, diag) in enumerate(zip(outputs, diagnostics)):
        if diag['step'] != len(case['input_ids']) - 1 + index or diag['token'] != row['token']:
            raise ValueError('diagnostic position/token differs from its output')
        entries = diag['top5']
        if not entries or len({v['token'] for v in entries}) != len(entries):
            raise ValueError('empty or duplicate native top-k')
        if any(not math.isfinite(v['logprob']) for v in entries):
            raise ValueError('nonfinite native logprob')
    first = next((i for i, row in enumerate(outputs) if row['token'] != case['output_ids'][i]), None)
    result = dict(format='inferfabric.token-divergence.v1', case=case.get('case'),
                  inspected_outputs=len(outputs), reference_outputs=len(case['output_ids']),
                  matching_prefix_tokens=len(outputs) if first is None else first,
                  first_difference=None, scope='output logprobs at equal token history; not full-model qualification')
    if first is not None:
        expected, actual = case['output_ids'][first], outputs[first]['token']
        native = {v['token']: v['logprob'] for v in diagnostics[first]['top5']}
        reference = {int(k): v['logprob'] for k, v in case['logprobs'][first].items()}
        if any(not math.isfinite(value) for value in reference.values()):
            raise ValueError('nonfinite reference logprob')
        def gap(scores):
            return scores[actual] - scores[expected] if actual in scores and expected in scores else None
        result['first_difference'] = dict(output_index=first,
            position=diagnostics[first]['step'], expected_token=expected, actual_token=actual,
            native_topk=diagnostics[first]['top5'], reference_topk=case['logprobs'][first],
            native_actual_minus_expected_logprob=gap(native),
            reference_actual_minus_expected_logprob=gap(reference))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'tokens', 'requests', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--case', required=True)
    args = parser.parse_args()
    reference, tokens = args.reference.read_bytes(), args.tokens.read_bytes()
    cases = [c for c in json.loads(reference) if c['case'] == args.case]
    if len(cases) != 1:
        raise ValueError('case must identify exactly one reference')
    rows = [json.loads(line) for line in tokens.decode().splitlines() if line.startswith('{')]
    requests = args.requests.read_bytes()
    lines = requests.decode().splitlines()
    if len(lines) != 1:
        raise ValueError('requires exactly one diagnostic request')
    fields = list(map(int, lines[0].split()))
    if len(fields) < 2 or fields[0] < 1:
        raise ValueError('invalid diagnostic request')
    result = analyze(cases[0], rows, fields[1:])
    if result['inspected_outputs'] != fields[0]:
        raise ValueError('incomplete diagnostic output budget')
    result['request_sha256'] = hashlib.sha256(requests).hexdigest()
    result.update(reference_sha256=hashlib.sha256(reference).hexdigest(),
                  tokens_sha256=hashlib.sha256(tokens).hexdigest())
    with args.output.open('x') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
