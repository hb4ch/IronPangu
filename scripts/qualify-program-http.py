#!/usr/bin/env python3
"""Offline client of the vLLM Rust frontend; no Python inference engine.

Checks full fixture text, SSE agreement, queue isolation and disconnect recovery.
Receipt intervals are HTTP chunk intervals, not necessarily one token per chunk.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request

def expected_ids(case, eos, respect_eos):
    if case.get('qualification', {}).get('output_is_placeholder'):
        raise ValueError('placeholder outputs are not an independent reference')
    ids = case['output_ids']
    if not ids:
        raise ValueError('empty reference output')
    return ids[:ids.index(eos)+1] if respect_eos and eos in ids else list(ids)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('server', 'plan', 'checkpoint', 'library', 'cases', 'output'):
        p.add_argument('--' + key, required=True, type=Path)
    p.add_argument('--port', type=int, default=18085)
    p.add_argument('--max-num-seqs', type=int)
    p.add_argument('--max-num-batched-tokens', type=int)
    p.add_argument('--expect-prefill', choices=('packed','serial'), help='verify actual graph use, excluding startup warmup')
    p.add_argument('--resident-cases', type=Path, help='serial oracle fixtures for concurrent long-prefill/short-decode qualification')
    p.add_argument('--check-overload', action='store_true', help='Burst long streaming requests beyond admission capacity, check 503/retry metadata and recovery')
    p.add_argument('--request-timeout-seconds', type=int, default=120, help='Socket inactivity timeout, not a whole-run deadline; raise for 64K prefill')
    p.add_argument('--respect-eos', action='store_true', help='Test normal serving through first EOS; preserve full references and report the narrower comparison scope')
    args = p.parse_args()
    if args.request_timeout_seconds < 1:
        p.error('--request-timeout-seconds must be positive')
    if args.check_overload and (not args.resident_cases or not args.max_num_seqs):
        p.error('--check-overload requires --resident-cases and --max-num-seqs')
    args.output.mkdir(parents=True, exist_ok=False)
    cases = json.loads(args.cases.read_text())
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.checkpoint / 'tokenizer.json'))
    eos = json.loads((args.plan/'device-program.json').read_text())['eos']
    def expected_text(case):
        ids = expected_ids(case, eos, args.respect_eos)
        expected_answer = case.get('qualification', {}).get('expected_answer')
        if expected_answer is not None:
            through_eos = expected_ids(case, eos, True)
            if tokenizer.decode(through_eos, skip_special_tokens=True).strip() != expected_answer:
                raise ValueError('independent reference failed its retrieval answer')
        return tokenizer.decode(ids, skip_special_tokens=True)
    for case in cases:
        expected_text(case)
    if args.resident_cases:
        for case in json.loads(args.resident_cases.read_text()):
            expected_text(case)
    url = f'http://127.0.0.1:{args.port}'

    def request(case, stream=False, disconnect=False, temperature=0):
        body = dict(model='inferfabric-program', prompt=case['input_ids'],
                    max_tokens=len(case['output_ids']), temperature=temperature, stream=stream, ignore_eos=not args.respect_eos)
        if stream:
            body['stream_options'] = {'include_usage': True}
        began = time.perf_counter()
        req = urllib.request.Request(url + '/v1/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=args.request_timeout_seconds) as response:
            if not stream:
                value = json.load(response)
                return value['choices'][0]['text'], {'total_ms': (time.perf_counter()-began)*1000, 'usage': value['usage']}
            chunks, receipts, usage, finished = [], [], None, False
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                data = line[6:].strip()
                if data == b'[DONE]':
                    finished = True
                    break
                value = json.loads(data)
                if value.get('usage'):
                    usage = value['usage']
                for choice in value.get('choices', []):
                    text = choice.get('text', '')
                    if text:
                        chunks.append(text)
                        receipts.append(time.perf_counter())
                        if disconnect:
                            return ''.join(chunks), {'cancelled': True}
            if not finished or not receipts:
                raise RuntimeError('incomplete SSE response')
            return ''.join(chunks), {'ttft_ms': (receipts[0]-began)*1000,
                                    'text_chunks': len(receipts),
                                    'chunk_intervals_ms': [(b-a)*1000 for a,b in zip(receipts,receipts[1:])],
                                    'total_ms': (time.perf_counter()-began)*1000, 'usage': usage}

    env = dict(os.environ, INFERFABRIC_DECODE_GRAPH='1',
               INFERFABRIC_PREPARED_AUDIT_DIR=str((args.output/'prepared').resolve()),
               INFERFABRIC_TRANSFER_AUDIT_DIR=str((args.output/'transfers').resolve()))
    env.pop('INFERFABRIC_PREPARE_ARENA', None)
    command = [str(args.server.resolve()), '--program', str(args.plan), str(args.checkpoint), str(args.library), str(args.port)]
    for name in ('max_num_seqs', 'max_num_batched_tokens'):
        value = getattr(args, name)
        if value is not None:
            command += ['--'+name.replace('_','-'), str(value)]
    with (args.output / 'server.log').open('w') as log:
        server = subprocess.Popen(command, env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 900
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'server startup failed: {server.returncode}')
                try:
                    with urllib.request.urlopen(url + '/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('frontend readiness')
                time.sleep(1)
            results = []
            for case in cases:
                expected = expected_text(case)
                text, collected = request(case)
                streamed, metrics = request(case, stream=True)
                if text != expected or streamed != text:
                    raise RuntimeError(f"{case['case']} text mismatch: {text!r} / {streamed!r} / {expected!r}")
                for observed in (collected, metrics):
                    if observed['usage']['prompt_tokens'] != len(case['input_ids']) or observed['usage']['completion_tokens'] != len(expected_ids(case, eos, args.respect_eos)):
                        raise RuntimeError('HTTP token usage does not match complete fixture')
                results.append(dict(case=case['case'], text=text, collected=collected, streaming=metrics))
            residents = args.max_num_seqs or json.loads((args.plan/'serving.json').read_text()).get('resident_sequences', 1)
            # Fit the bounded queue even before the worker takes its first job.
            # Overload is a separate gate; it must not mask isolation/progress.
            concurrent_cases = (cases * 2)[:4 * residents]
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(concurrent_cases)) as executor:
                futures = [executor.submit(request, c) for c in concurrent_cases]
                for case, future in zip(concurrent_cases, futures):
                    text, _ = future.result()
                    if text != expected_text(case):
                        raise RuntimeError('queued request isolation')
            request(cases[-1], stream=True, disconnect=True)
            text, _ = request(cases[0])
            if text != expected_text(cases[0]):
                raise RuntimeError('disconnect recovery')
            mixed = None
            if args.resident_cases:
                long_case = max(json.loads(args.resident_cases.read_text()), key=lambda c: len(c['input_ids']))
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    long = executor.submit(request, long_case, True)
                    time.sleep(0.25)
                    request(cases[-1], stream=True, disconnect=True)
                    short_text, short_metrics = request(cases[0], stream=True)
                    if long.done():
                        raise RuntimeError('long prefill blocked short request: no independent resident progress')
                    if short_text != expected_text(cases[0]):
                        raise RuntimeError('resident cancellation/reset corrupted short request')
                    long_text, long_metrics = long.result()
                    if long_text != expected_text(long_case):
                        raise RuntimeError('resident cancellation/reset corrupted long peer')
                    mixed = dict(long_prompt_tokens=len(long_case['input_ids']), short_finished_before_long=True,
                                 cancellation_during_peer_prefill=True, short=short_metrics, long=long_metrics)
            try:
                request(cases[0], temperature=0.7)
            except urllib.error.HTTPError as e:
                if e.code != 400:
                    raise
            else:
                raise RuntimeError('unsupported sampling was silently accepted')
            context = json.loads((args.plan/'serving.json').read_text())['variants'][0]['max_context']
            try:
                request(dict(input_ids=[cases[0]['input_ids'][0]]*context, output_ids=[0]))
            except urllib.error.HTTPError as e:
                if e.code != 400:
                    raise
            else:
                raise RuntimeError('context overflow was silently accepted')
            overload = None
            if args.check_overload:
                long_case = max(json.loads(args.resident_cases.read_text()), key=lambda c:len(c['input_ids']))
                count = args.max_num_seqs * 8
                start = threading.Barrier(count)
                body = json.dumps(dict(model='inferfabric-program', prompt=long_case['input_ids'],
                    max_tokens=len(long_case['output_ids']), temperature=0, stream=True, ignore_eos=True)).encode()
                def submit_burst(_):
                    start.wait(timeout=30)
                    req = urllib.request.Request(url+'/v1/completions', data=body, headers={'Content-Type':'application/json'})
                    try:
                        response = urllib.request.urlopen(req, timeout=max(300, args.request_timeout_seconds))
                        return response.status, dict(response.headers), None, response
                    except urllib.error.HTTPError as error:
                        with error:
                            return error.code, dict(error.headers), json.load(error), None
                responses = []
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
                        for future in concurrent.futures.as_completed([pool.submit(submit_burst,i) for i in range(count)]):
                            responses.append(future.result())
                    accepted = sum(status==200 for status,_,_,_ in responses)
                    rejected = [r for r in responses if r[0]!=200]
                    if not accepted or not rejected:
                        raise RuntimeError('burst did not exercise both acceptance and overload')
                    for status,headers,payload,_ in rejected:
                        headers = {k.lower():v for k,v in headers.items()}
                        if status!=503 or headers.get('retry-after')!='1' or payload['error']['code']!='server_overloaded':
                            raise RuntimeError(f'incorrect overload response: {status} {headers} {payload}')
                    overload = dict(submitted=count,accepted=accepted,rejected=len(rejected),status=503,
                                    error_code='server_overloaded',retry_after_seconds=1)
                finally:
                    for _,_,_,response in responses:
                        if response is not None: response.close()
                # Accepted streams are deliberately disconnected; then verify a
                # fresh request after their queued work has observed cancellation.
                deadline = time.monotonic()+30
                while True:
                    try:
                        text, _ = request(cases[0])
                        break
                    except urllib.error.HTTPError as error:
                        if error.code!=503 or time.monotonic()>=deadline: raise
                        error.close()
                        time.sleep(0.1)
                if text!=expected_text(cases[0]):
                    raise RuntimeError('output changed after overload/disconnect recovery')
                overload['recovery_output_correct'] = True
            report = dict(frontend='vLLM Rust 0.25.1', mode='resident request scheduler; compiled prefill and serial decode',
                          full_fixture_text=not args.respect_eos, sse_agreement=True, queued_isolation=True,
                          reference_comparison_scope='through_first_eos' if args.respect_eos else 'full_forced_continuation',
                          fixture_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                          resident_fixture_sha256=hashlib.sha256(args.resident_cases.read_bytes()).hexdigest() if args.resident_cases else None,
                          request_socket_timeout_seconds=args.request_timeout_seconds,
                          disconnect_recovery=True, unsupported_sampling_rejected=True, results=results,
                          concurrent_requests=len(concurrent_cases), mixed_residents=mixed,
                          context_overflow_rejected=True,
                          max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens)
            if overload is not None: report['overload'] = overload
        finally:
            if server.poll() is None:
                server.send_signal(signal.SIGTERM)
                try:
                    server.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=30)
                    raise RuntimeError('graceful shutdown timed out')
        if server.returncode:
            raise RuntimeError(f'frontend shutdown failed: {server.returncode}')
        report['graceful_sigterm_shutdown'] = True
        manifest = json.loads((args.plan/'serving.json').read_text())
        report['prefill_graph_replays'] = []
        for rank in range(len(manifest['identity']['devices'])):
            audit = json.loads((args.output/'prepared'/f'rank-{rank}.json').read_text())
            transfers = json.loads((args.output/'transfers'/f'rank-{rank}.json').read_text())
            program = json.loads((args.plan/'device-program.json').read_text())
            names = {f'packed_prefill_{n}' for n in program.get('prefill_variants', [])} or ({'packed_prefill'} if program['prefill_tokens']>1 else set())
            packed = [g for g in audit['graphs'] if g['name'].startswith('packed_prefill')]
            if len(packed)!=len(names) or {g['name'] for g in packed} != names:
                raise RuntimeError('HTTP prepared graphs differ from the declared family')
            replays = sum(g['replays'] for g in packed)
            if args.expect_prefill=='packed' and replays<=len(names):
                raise RuntimeError('HTTP did not exercise packed prefill beyond warmup')
            if args.expect_prefill=='serial' and replays>len(names):
                raise RuntimeError('HTTP exceeded serial-only prefill budget')
            report['prefill_graph_replays'].append(replays)
            if not audit['arena']['frozen'] or not audit['workspace']['planned'] or audit['workspace']['growth_allocations']:
                raise RuntimeError('HTTP execution violated frozen resource plans')
            if transfers['activation_host_transfers'] or transfers['rejected'] or any(
                    s['device_allocations'] or s['completion_boundaries'] for s in audit['steps']):
                raise RuntimeError('HTTP execution introduced host transfers, allocations or internal fences')
        if args.resident_cases:
            logs = (args.output/'server.log').read_text()
            stats = re.search(r'program scheduler stopped max_active=(\d+) mixed_ticks=(\d+) consumed_tokens=(\d+) token_budget=(\d+)', logs)
            if not stats or int(stats[1]) < 2:
                raise RuntimeError('scheduler did not confirm two active residents')
            if int(stats[4]) >= 2 and not int(stats[2]):
                raise RuntimeError('scheduler did not confirm mixed prefill/decode work')
            report['scheduler'] = dict(zip(('max_active','mixed_ticks','consumed_tokens','token_budget'), map(int,stats.groups())))
        if manifest.get('shared_pages', False):
            logs=(args.output/'server.log').read_text()
            pool=re.search(r'shared state pool admissions=(\d+) cross_partition_bindings=(\d+) peak_occupied_bytes=(\d+) remaining_occupied_bytes=(\d+)',logs)
            if not pool or not int(pool[1]) or not int(pool[2]) or int(pool[4]):
                raise RuntimeError('shared pages were not exercised across partitions or were not fully reclaimed')
            report['shared_pool']=dict(zip(('admissions','cross_partition_bindings','peak_occupied_bytes','remaining_occupied_bytes'),map(int,pool.groups())))
        (args.output / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
