#!/usr/bin/env python3
"""Offline text/JSON test driver. All model execution is the native C++ binary."""
import argparse
import json
import os
import pathlib
import subprocess
from tokenizers import Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--plan', required=True)
parser.add_argument('--binary', default='native-build/openpangu_decode')
parser.add_argument('--output', required=True)
args = parser.parse_args()
output = pathlib.Path(args.output)
output.mkdir(parents=True, exist_ok=True)
tokenizer = Tokenizer.from_file(str(pathlib.Path(args.checkpoint) / 'tokenizer.json'))
cases = [('arithmetic', '1+1=?', '2'),
         ('geography', 'What is the capital of France?', 'Paris'),
         ('chinese', '请用一句话介绍北京。', '北京')]
results = []
for name, prompt, anchor in cases:
    rendered = '<|message_start|>user\n' + prompt + '<|message_end|><|message_start|>assistant\n</think>'
    ids = tokenizer.encode(rendered, add_special_tokens=False).ids
    command = [args.binary, args.plan, args.checkpoint, '32', *map(str, ids)]
    audit_dir = output / (name + '-transfers')
    audit_dir.mkdir(exist_ok=True)
    env = dict(os.environ, INFERFABRIC_TRANSFER_AUDIT_DIR=str(audit_dir.resolve()),
               INFERFABRIC_PREPARED_AUDIT_DIR=str((output / (name + '-prepared')).resolve()))
    with (output / (name + '.stderr')).open('w') as error:
        run = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=error, timeout=900, env=env)
    (output / (name + '.jsonl')).write_text(run.stdout)
    if run.returncode:
        raise RuntimeError(f'{name}: native decoder failed: {run.returncode}')
    rows = [json.loads(line) for line in run.stdout.splitlines() if line.startswith('{')]
    tokens = [row['token'] for row in rows]
    for rank in range(8):
        audit = json.loads((audit_dir / f'rank-{rank}.json').read_text())
        if audit['activation_host_transfers'] != 0 or audit['rejected'] != 0:
            raise RuntimeError('residency violation')
        boundaries = {x['role']: x for x in audit['boundaries']}
        result_bytes = boundaries['request_output']['d2h_bytes']
        if result_bytes != len(tokens) * (144 if rank == 0 else 64):
            raise RuntimeError('unexpected output transfer volume')
        if (boundaries['control_input']['h2d_bytes'] == 0
                or boundaries['parameter_preparation']['h2d_bytes'] == 0
                or boundaries['parameter_preparation']['d2h_bytes'] == 0):
            raise RuntimeError('missing transfer instrumentation')
        if boundaries['control_input']['d2h_bytes'] != 0:
            raise RuntimeError('control input readback')
    text = tokenizer.decode(tokens, skip_special_tokens=False)
    if not tokens or anchor not in text:
        raise RuntimeError(f'{name}: expected answer content missing: {text!r}')
    result = {'case': name, 'prompt': prompt, 'input_ids': ids, 'output_ids': tokens,
              'text': text, 'steps': rows, 'anchor_check': 'passed'}
    results.append(result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
(output / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
