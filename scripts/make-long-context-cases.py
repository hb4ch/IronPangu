#!/usr/bin/env python3
"""Deterministic token-exact retrieval inputs for the offline reference runner.

output_ids are length placeholders, NEVER an oracle. Obtain reference outputs
with reference-openpangu-decoder.sh before running native qualification.
"""
import argparse
import hashlib
import json
from pathlib import Path


def make_cases(encode, contexts, positions, output_tokens=16):
    if output_tokens < 1 or not contexts or not positions:
        raise ValueError('nonempty context/position lists and positive output budget required')
    if len(set(contexts)) != len(contexts) or len(set(positions)) != len(positions):
        raise ValueError('duplicate context or position')
    if any(p not in ('early', 'middle', 'late') for p in positions):
        raise ValueError('unknown needle position')
    head = encode('<|message_start|>user\nRead the archive and retrieve the requested record.\n')
    filler = encode('Routine maintenance entry: the ventilation inspection completed without changes.\n')
    tail = encode('\nWhat is the access code for record ORCHID? Reply with the code only.'
                  '<|message_end|><|message_start|>assistant\n</think>')
    if not head or not filler or not tail:
        raise ValueError('tokenizer produced an empty template')
    cases = []
    for context in contexts:
        if not 1 <= context <= 65536:
            raise ValueError('context must be in 1..65536')
        for position in positions:
            # Distinct deterministic answers prevent a constant-output backend
            # from passing the positional suite. Keep the answer outside filler.
            code = str(1000 + int(hashlib.sha256(f'{context}:{position}'.encode()).hexdigest()[:8], 16) % 9000)
            needle = encode(f'\nRecord ORCHID: the access code is {code}.\n')
            count = context - output_tokens - len(head) - len(tail) - len(needle)
            if count < 3:
                raise ValueError('context cannot fit template and generation budget')
            before = {'early': 0, 'middle': count // 2, 'late': count}[position]
            fill = lambda n: (filler * ((n + len(filler) - 1) // len(filler)))[:n]
            ids = head + fill(before) + needle + fill(count - before) + tail
            start = len(head) + before
            assert len(ids) + output_tokens == context
            cases.append(dict(case=f'retrieval-{context}-{position}', input_ids=ids,
                              output_ids=[0] * output_tokens,
                              qualification=dict(total_context=context, needle_position=position,
                                                 needle_token_span=[start, start + len(needle)],
                                                 expected_answer=code, output_is_placeholder=True)))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--contexts', type=int, nargs='+', default=[8192, 16384, 32768, 65536])
    parser.add_argument('--positions', nargs='+', choices=['early', 'middle', 'late'], default=['early', 'middle', 'late'])
    parser.add_argument('--output-tokens', type=int, default=16)
    args = parser.parse_args()
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    cases = make_cases(lambda text: tokenizer.encode(text, add_special_tokens=False).ids,
                       args.contexts, args.positions, args.output_tokens)
    # Do not silently replace previously pinned reference inputs.
    with args.output.open('x', encoding='utf-8') as out:
        json.dump(cases, out, ensure_ascii=False, indent=2)
        out.write('\n')
    print(json.dumps(dict(cases=len(cases), output=str(args.output),
                          tokenizer_sha256=hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
                          input_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
