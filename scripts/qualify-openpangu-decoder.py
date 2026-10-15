#!/usr/bin/env python3
"""Offline text/JSON test driver. All model execution is the native C++ binary."""
import argparse
import json
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
    with (output / (name + '.stderr')).open('w') as error:
        run = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=error, timeout=900)
    (output / (name + '.jsonl')).write_text(run.stdout)
    if run.returncode:
        raise RuntimeError(f'{name}: native decoder failed: {run.returncode}')
    rows = [json.loads(line) for line in run.stdout.splitlines() if line.startswith('{')]
    tokens = [row['token'] for row in rows]
    text = tokenizer.decode(tokens, skip_special_tokens=False)
    if not tokens or anchor not in text:
        raise RuntimeError(f'{name}: expected answer content missing: {text!r}')
    result = {'case': name, 'prompt': prompt, 'input_ids': ids, 'output_ids': tokens,
              'text': text, 'steps': rows, 'anchor_check': 'passed'}
    results.append(result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
(output / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
