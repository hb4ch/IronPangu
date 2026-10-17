#!/usr/bin/env python3
"""Prepare deterministic native serving fixtures from independent references.

This is an offline test client, never a source of generated reference answers.
It derives page permutations and EOS/context limits from the compiled plan.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random


def prepare(plan, references, output, peers=None, peer_case=None):
    program = json.loads((plan / 'device-program.json').read_text())
    manifest = json.loads((plan / 'serving.json').read_text())
    cases = json.loads(references.read_text())
    if not cases:
        raise ValueError('empty reference fixture')
    sources = [references]
    if peers is not None:
        peer_cases = json.loads(peers.read_text())
        if peer_case is not None:
            peer_cases = [c for c in peer_cases if c['case'] == peer_case]
            if len(peer_cases) != 1:
                raise ValueError('peer case must identify exactly one reference')
        if not peer_cases:
            raise ValueError('empty peer reference fixture')
        cases = [c for index, case in enumerate(cases) for c in (case, peer_cases[index % len(peer_cases)])]
        sources.append(peers)
    elif peer_case is not None:
        raise ValueError('peer case requires peer reference file')
    requests, expected = [], []
    for index, case in enumerate(cases):
        if case.get('reference_intervention'):
            raise ValueError('diagnostic intervention is not an independent reference')
        prompt, tokens = case['input_ids'], case['output_ids']
        if not prompt or not tokens or case.get('qualification', {}).get('output_is_placeholder'):
            raise ValueError('empty or placeholder reference')
        if any(type(t) is not int or t < 0 or t > 0xffffffff for t in prompt + tokens):
            raise ValueError('invalid token ID')
        if program['eos'] in tokens:
            tokens = tokens[:tokens.index(program['eos']) + 1]
        if len(tokens) > 128 or len(prompt) + len(tokens) - 1 > program['max_context']:
            raise ValueError('fixture exceeds probe/context bounds')
        requests.append(str(len(tokens)) + ' ' + ' '.join(map(str, prompt)) + '\n')
        expected.extend(dict(request=index, step=step, token=token) for step, token in enumerate(tokens))
    rng, bindings = random.Random(1729), []
    for state in manifest['states']:
        storage = state['storage']
        if storage['kind'] != 'pages':
            continue
        ops = [i for i in program['instructions'] if i['logical_node'] == state['logical_node']
               and i['op'] in ('cache_append', 'cache_norm_rope_append', 'cache_c8_norm_rope_append', 'index_cache_append')]
        if len(ops) != 1 or storage['page_tokens'] <= 0 or state['capacity_tokens'] % storage['page_tokens']:
            raise ValueError('invalid cache binding/capacity')
        for rank in state['ranks']:
            pages = list(range(state['capacity_tokens'] // storage['page_tokens']))
            rng.shuffle(pages)
            bindings.append(f"{rank} {ops[0]['name']} " + ' '.join(map(str, pages)) + '\n')
    if not bindings:
        raise ValueError('fixture requires paged state')
    files = {'requests.txt': ''.join(requests),
             'reference.jsonl': ''.join(json.dumps(r) + '\n' for r in expected),
             'page-bindings.txt': ''.join(bindings), 'cases.json': json.dumps(cases)}
    provenance = dict(scope='through_first_eos', page_permutation_seed=1729,
        sources=[dict(name=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources],
        plan_sha256=hashlib.sha256((plan / 'device-program.json').read_bytes()).hexdigest(),
        files_sha256={name: hashlib.sha256(text.encode()).hexdigest() for name, text in files.items()})
    output.mkdir()  # Never overwrite earlier evidence.
    for name, text in files.items():
        (output / name).write_text(text, encoding='utf-8')
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return dict(requests=len(cases), expected_tokens=len(expected), output=str(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'references', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--peers', type=Path)
    parser.add_argument('--peer-case')
    args = parser.parse_args()
    print(json.dumps(prepare(args.plan, args.references, args.output, args.peers, args.peer_case)))
