#!/usr/bin/env python3
"""Validate completed native batch geometry without reading any activations."""
import argparse
import json
from pathlib import Path


def validate(events, requests, budget=4096):
    positions = [0] * len(requests)
    samples = [0] * len(requests)
    chunks = [[] for _ in requests]
    decode_us = []
    decode_batches = []
    batches = 0
    for event in events:
        if event.get('event') != 'sequence_batch':
            continue
        batches += 1
        members = event['members']
        if not members or len({m['resident'] for m in members}) != len(members) or len({m['request'] for m in members}) != len(members):
            raise ValueError('empty or duplicate batch member')
        if not 0 < event['tokens'] <= budget or sum(m['tokens'] for m in members) != event['tokens']:
            raise ValueError('batch token budget/accounting')
        pure_decode = True
        remaining = budget
        for m in members:
            r, count = m['request'], m['tokens']
            if not 0 <= r < len(requests) or count <= 0 or m['position'] != positions[r]:
                raise ValueError('invalid request or noncontiguous history')
            prompt, outputs = requests[r]
            if positions[r] < prompt:
                pure_decode = False
                # This probe fills the remaining batch budget in resident order.
                if count != min(prompt - positions[r], remaining):
                    raise ValueError('prefill was fragmented instead of filling the batch')
                chunks[r].append(count)
            elif count != 1:
                raise ValueError('decode must consume one token per request')
            positions[r] += count
            remaining -= count
            if bool(m['sample']) != (positions[r] >= prompt):
                raise ValueError('sampling before prompt completion or missing sample')
            samples[r] += int(bool(m['sample']))
            if samples[r] > outputs:
                raise ValueError('request continued after its output budget')
        if event['elapsed_us'] < 0:
            raise ValueError('negative duration')
        if pure_decode:
            decode_us.append(event['elapsed_us'])
            decode_batches.append(dict(sequences=len(members), elapsed_us=event['elapsed_us']))
    if not batches or positions != [p + n - 1 for p, n in requests] or samples != [n for _, n in requests]:
        raise ValueError('incomplete batch trace')
    return dict(batches=batches, prefill_chunks=chunks, pure_decode_batch_us=decode_us,
                pure_decode_batches=decode_batches,
                completed_input_tokens=sum(positions),
                timing_scope='completed native ABI call including sampling/consensus; pure-decode batches exclude competing prefill')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('requests', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    requests = [(len(row.split()) - 1, int(row.split()[0])) for row in args.requests.read_text().splitlines()]
    events = [json.loads(line) for line in args.trace.read_text().splitlines() if line.startswith('{')]
    args.output.write_text(json.dumps(validate(events, requests), indent=2) + '\n')
