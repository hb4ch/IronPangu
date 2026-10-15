#!/usr/bin/env python3
"""Compare complete native sequences with a pinned offline reference run."""
import argparse
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('native')
parser.add_argument('reference')
parser.add_argument('output')
args = parser.parse_args()
native = json.loads(Path(args.native).read_text())
reference = json.loads(Path(args.reference).read_text())
if len(native) != len(reference) or not native:
    raise RuntimeError('case count mismatch')
rows = []
for actual, expected in zip(native, reference):
    if actual['input_ids'] != expected['input_ids']:
        raise RuntimeError('prompt tokens differ')
    if len(actual['steps']) != len(actual['output_ids']):
        raise RuntimeError('native step count mismatch')
    if len(expected['logprobs']) != len(expected['output_ids']):
        raise RuntimeError('reference logprob count mismatch')
    same = actual['output_ids'] == expected['output_ids']
    differences = []
    if same:
        for token, step, ref in zip(actual['output_ids'], actual['steps'], expected['logprobs']):
            if token != step['token']:
                raise RuntimeError('native step token mismatch')
            value = next(x['logprob'] for x in step['top5'] if x['token'] == token)
            delta = abs(value - ref[str(token)]['logprob'])
            if not math.isfinite(delta):
                raise RuntimeError('nonfinite logprob')
            differences.append(delta)
    rows.append({
        'case': actual['case'], 'exact_token_match': same,
        'native_tokens': len(actual['output_ids']),
        'reference_tokens': len(expected['output_ids']),
        'max_selected_logprob_abs_delta': max(differences) if differences else None,
    })
result = {
    'exact_sequences': all(x['exact_token_match'] for x in rows), 'cases': rows,
    'scope': 'greedy token equality on these prompts; not bitwise logit equality or broad quality qualification',
}
Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
if not result['exact_sequences']:
    raise SystemExit(1)
