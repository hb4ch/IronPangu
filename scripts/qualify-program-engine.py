#!/usr/bin/env python3
"""Offline regression client; production engine/frontend never depend on Python.

Alternate complete model fixtures in one loaded native rank group. Engine TTFT
includes reset and prefill, excludes model loading/tokenization/network. TPOT
includes native sampling and rank consensus. This is not an HTTP load benchmark.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import random
import re
import subprocess
import time


def run_engine(command, env, output, timeout):
    """Persist progress even if a long qualification times out or crashes."""
    started = time.monotonic()
    with (output / 'stderr.log').open('w') as err, (output / 'tokens.jsonl').open('w') as out:
        try:
            result = subprocess.run(command, env=env, stdout=out, stderr=err, timeout=timeout)
        except subprocess.TimeoutExpired:
            (output / 'process.json').write_text(json.dumps(dict(completed=False, timed_out=True,
                timeout_seconds=timeout, elapsed_seconds=time.monotonic()-started), indent=2)+'\n')
            raise RuntimeError('native qualification timed out; partial tokens.jsonl and stderr.log retained') from None
    (output / 'process.json').write_text(json.dumps(dict(completed=True, timed_out=False,
        returncode=result.returncode, timeout_seconds=timeout, elapsed_seconds=time.monotonic()-started), indent=2)+'\n')
    if result.returncode:
        raise RuntimeError(f'native engine exited {result.returncode}; inspect tokens.jsonl and stderr.log')
    return (output / 'tokens.jsonl').read_text()


def validate_uncaptured_prefill(program, prepared, cases, copy_boundary=None):
    if not program.get('uncaptured_prefill') or program['prefill_tokens'] != 4096 or program.get('prefill_variants'):
        raise RuntimeError('invalid uncaptured prefill contract')
    graphs = prepared['graphs']
    if len(graphs) != 1 or graphs[0]['name'] != 'token_decode' or graphs[0]['replays'] < 1:
        raise RuntimeError('uncaptured prefill must preserve decode graph and forbid all prefill graphs')
    # The engine explicitly prepares the maximum shape twice without capture.
    forwards, tokens = 2, 8192
    for case in cases:
        segments = [len(case['input_ids'])] if copy_boundary is None else [copy_boundary] + [len(case['input_ids']) - copy_boundary] * 2
        for remaining in segments:
            if remaining < 1:
                raise RuntimeError('invalid uncaptured prefill segment')
            while remaining:
                query = min(4096, remaining)
                if query > 1:  # Single-row tails use the ordinary token entry.
                    forwards += 1
                    tokens += query
                remaining -= query
    if prepared.get('uncaptured_prefill_forwards') != forwards or prepared.get('uncaptured_prefill_tokens') != tokens:
        raise RuntimeError('uncaptured model forward/token count mismatch')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'plan', 'checkpoint', 'cases', 'output'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--experimental-long-context', action='store_true', help='Explicitly opt in to unqualified context above 512 for hardware qualification')
    p.add_argument('--output-diagnostics', action='store_true', help='Record existing sampler top-five request outputs; no activation readback')
    p.add_argument('--check-binding-limits', action='store_true', help='Reject oversized bindings through both ABIs, then verify valid retries and inference')
    p.add_argument('--packed-prefill', action='store_true', help='require the compiled packed chunk path')
    p.add_argument('--permuted-pages', action='store_true', help='qualify rank/component-specific noncontiguous device cache bindings')
    p.add_argument('--interleave', action='store_true', help='interleave resident histories token by token, including slot reuse while peers are live')
    p.add_argument('--copy-boundary', type=int, help='Copy complete device state at this prefix boundary; compare independent source/destination continuations and reference')
    p.add_argument('--timeout-seconds', type=int, default=1800, help='Whole native run deadline; raise explicitly for the 64K qualification ladder')
    args = p.parse_args()
    if args.timeout_seconds < 1:
        p.error('--timeout-seconds must be positive')
    if args.copy_boundary is not None and (args.copy_boundary<1 or not args.permuted_pages or args.prepare or args.interleave or args.output_diagnostics):
        p.error('--copy-boundary requires positive boundary, frozen permuted pages, no interleave/diagnostics')
    args.output.mkdir(parents=True, exist_ok=False)
    cases = json.loads(args.cases.read_text()) * 3
    if args.copy_boundary is not None and any(len(c['input_ids'])<=args.copy_boundary for c in cases):
        p.error('copy boundary must precede every fixture suffix')
    if any(c.get('qualification', {}).get('output_is_placeholder') for c in cases):
        raise ValueError('placeholder outputs are not a reference; run the independent reference first')
    requests = args.output / 'requests.txt'
    requests.write_text(''.join(f"{len(c['output_ids'])} " + ' '.join(map(str, c['input_ids'])) + '\n' for c in cases))
    env = dict(os.environ, INFERFABRIC_DECODE_GRAPH='1',
               INFERFABRIC_PREPARED_AUDIT_DIR=str((args.output / 'prepared').resolve()),
               INFERFABRIC_TRANSFER_AUDIT_DIR=str((args.output / 'transfers').resolve()))
    env.pop('INFERFABRIC_PREPARE_ARENA', None)
    env.pop('INFERFABRIC_ALTERNATE_PAGE_LAYOUTS', None)
    env.pop('INFERFABRIC_INTERLEAVE_RESIDENTS', None)
    env.pop('INFERFABRIC_PACKED_PREFILL', None)
    env.pop('INFERFABRIC_OUTPUT_DIAGNOSTICS',None)
    env.pop('INFERFABRIC_TEST_COPY_BOUNDARY',None)
    env.pop('INFERFABRIC_STATE_COPY_AUDIT_DIR',None)
    if args.copy_boundary is not None:
        env['INFERFABRIC_TEST_COPY_BOUNDARY']=str(args.copy_boundary)
        env['INFERFABRIC_STATE_COPY_AUDIT_DIR']=str((args.output/'state-copies').resolve())
    env.pop('INFERFABRIC_EXPERIMENTAL_LONG_CONTEXT',None)
    if args.experimental_long_context:
        env['INFERFABRIC_EXPERIMENTAL_LONG_CONTEXT']='1'
    env.pop('INFERFABRIC_TEST_BINDING_LIMITS',None)
    if args.check_binding_limits:
        if not args.permuted_pages:
            raise ValueError('binding-limit checks require --permuted-pages')
        env['INFERFABRIC_TEST_BINDING_LIMITS']='1'
    if args.output_diagnostics:
        env['INFERFABRIC_OUTPUT_DIAGNOSTICS']='1'
    if args.packed_prefill:
        env['INFERFABRIC_PACKED_PREFILL'] = '1'
    if args.interleave:
        env['INFERFABRIC_INTERLEAVE_RESIDENTS'] = '1'
    if args.prepare:
        env['INFERFABRIC_PREPARE_ARENA'] = '1'
    if args.permuted_pages:
        manifest = json.loads((args.plan / 'serving.json').read_text())
        program = json.loads((args.plan / 'device-program.json').read_text())
        executable = program['instructions']
        randomizer = random.Random(1729)
        lines = []
        for state in manifest['states']:
            storage = state['storage']
            if storage['kind'] != 'pages':
                continue
            lowered = [i for i in executable if i['logical_node'] == state['logical_node'] and i['op'] in ('cache_append', 'cache_norm_rope_append', 'cache_c8_norm_rope_append', 'index_cache_append')]
            if len(lowered) != 1:
                raise RuntimeError('state does not map to exactly one executable cache binding')
            component = lowered[0]['name']
            for rank in state['ranks']:
                pages = list(range(state['capacity_tokens'] // storage['page_tokens']))
                randomizer.shuffle(pages)
                lines.append(f"{rank} {component} " + ' '.join(map(str, pages)) + '\n')
        binding_file = args.output / 'page-bindings.txt'
        binding_file.write_text(''.join(lines))
        env['INFERFABRIC_PAGE_BINDINGS'] = str(binding_file.resolve())
    else:
        env.pop('INFERFABRIC_PAGE_BINDINGS', None)
    stdout = run_engine([str(args.binary.resolve()), str(args.plan), str(args.checkpoint), str(requests)],
                        env, args.output, args.timeout_seconds)
    binding_rejections=0
    if args.check_binding_limits:
        checks=re.findall(r'binding limits: (\d+) invalid requests rejected; valid retries passed',
                          (args.output/'stderr.log').read_text())
        expected=len(lines)*(4 if program.get('shared_pages',False) else 2)
        if checks!=[str(expected)] or not expected:
            raise RuntimeError('binding-limit checks did not execute all expected rejections/retries')
        binding_rejections=expected
    records = [json.loads(line) for line in stdout.splitlines() if line.startswith('{')]
    rows = [row for row in records if 'request' in row]
    diagnostics=[row for row in records if 'top5' in row]
    if len(rows)+len(diagnostics)!=len(records) or (diagnostics and not args.output_diagnostics):
        raise RuntimeError('unexpected native output records')
    if args.output_diagnostics:
        if len(diagnostics)!=len(rows) or any(a['token']!=b['token'] for a,b in zip(diagnostics,rows)):
            raise RuntimeError('sampler diagnostic/output correspondence mismatch')
        (args.output/'output-diagnostics.json').write_text(json.dumps([
            dict(request=row['request'],case=cases[row['request']]['case'],generation_step=row['step'],**diag)
            for row,diag in zip(rows,diagnostics)],indent=2)+'\n')
    expected_total = sum(len(c['output_ids']) for c in cases)
    if len(rows) != expected_total:
        raise RuntimeError('incomplete or unexpected output records')
    if args.interleave and len({r.get('resident') for r in rows}) < 2:
        raise RuntimeError('interleaving did not exercise multiple resident slots')
    measurements = []
    for request, case in enumerate(cases):
        actual = [r for r in rows if r['request'] == request]
        if [r['step'] for r in actual] != list(range(len(case['output_ids']))) or [r['token'] for r in actual] != case['output_ids']:
            raise RuntimeError(f"request {request} {case['case']}: full output mismatch")
        measurement=dict(request=request, case=case['case'], prompt_tokens=len(case['input_ids']),
                         output_tokens=len(actual), tpot_ms=statistics.mean(r['elapsed_us']/1000 for r in actual[1:]))
        measurement['restored_suffix_ttft_ms' if args.copy_boundary is not None else 'engine_ttft_ms']=actual[0]['elapsed_us']/1000
        measurements.append(measurement)
    for rank in range(8):
        audit = json.loads((args.output / 'transfers' / f'rank-{rank}.json').read_text())
        if audit['activation_host_transfers'] or audit['rejected']:
            raise RuntimeError('residency violation')
        prepared = json.loads((args.output / 'prepared' / f'rank-{rank}.json').read_text())
        if args.packed_prefill and json.loads((args.plan / 'device-program.json').read_text()).get('uncaptured_prefill'):
            validate_uncaptured_prefill(json.loads((args.plan / 'device-program.json').read_text()), prepared, cases, args.copy_boundary)
        elif args.packed_prefill:
            program = json.loads((args.plan / 'device-program.json').read_text())
            family = program.get('prefill_variants', [])
            counts = family or [program['prefill_tokens']]
            expected = {f'packed_prefill_{n}' if family else 'packed_prefill': 1 for n in counts}
            for case in cases:
                segments=[len(case['input_ids'])] if args.copy_boundary is None else [args.copy_boundary]+[len(case['input_ids'])-args.copy_boundary]*2
                for remaining in segments:
                    for n in reversed(counts):
                        name = f'packed_prefill_{n}' if family else 'packed_prefill'
                        expected[name] += remaining // n
                        remaining %= n
            graphs = [g for g in prepared['graphs'] if g['name'].startswith('packed_prefill')]
            actual = {g['name']: g['replays'] for g in graphs}
            if min(counts) <= 1 or sum(expected.values())<=len(expected) or len(graphs)!=len(expected) or actual != expected:
                raise RuntimeError('packed prefill graph did not execute every expected chunk')
        if not args.prepare and (not prepared['workspace']['planned'] or prepared['workspace']['growth_allocations']):
            raise RuntimeError('workspace growth or missing fixed plan')
        if not args.prepare and (not prepared['arena']['frozen'] or any(
                s['device_allocations'] or s['completion_boundaries'] for s in prepared['steps'])):
            raise RuntimeError('unfrozen arena, graph allocation or internal fence')
        if args.copy_boundary is not None:
            copied=json.loads((args.output/'state-copies'/f'rank-{rank}.json').read_text())
            manifest=json.loads((args.plan/'serving.json').read_text())
            required={s['logical_node'] for s in manifest['states'] if rank in s['ranks']}
            if copied['source']!=0 or copied['destination']!=1 or copied['boundary']!=args.copy_boundary or {op['logical_node'] for op in copied['operations']}!=required:
                raise RuntimeError('copy plan omits declared continuation state or has wrong boundary')
            if any(op['primitive']!='D2D' or op['bytes']<=0 for op in copied['operations']) or not prepared['state_copy_operations'] or not prepared['state_copy_bytes']:
                raise RuntimeError('complete device copies were not exercised')
            if prepared['state_copy_operations']!=len(copied['operations'])*len(cases) or prepared['state_copy_bytes']!=sum(op['bytes'] for op in copied['operations'])*len(cases):
                raise RuntimeError('device copy counters disagree with the repeated fixed-boundary plan')
    report = dict(format='inferfabric.persistent-program-qualification.v1', requests=len(cases),
                  output_tokens=expected_total, exact_fixture_tokens=True,
                  fixture_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                  permuted_pages=args.permuted_pages,
                  binding_limit_rejections=binding_rejections,
                  interleaved_residents=args.interleave,
                  packed_prefill=args.packed_prefill,
                  copied_prefix_tokens=args.copy_boundary,
                  scope='one loaded TP8/EP8 group, ' + ('compiled packed prefill and serial decode' if args.packed_prefill else 'serial token graph') + '; no long-context claim',
                  measurements=measurements)
    (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
