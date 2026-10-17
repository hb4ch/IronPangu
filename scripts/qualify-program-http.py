#!/usr/bin/env python3
"""Offline client of the vLLM Rust frontend; no Python inference engine.

Checks full fixture text, SSE agreement, queue isolation and disconnect recovery.
Receipt intervals are HTTP chunk intervals, not necessarily one token per chunk.
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request

def process_resources(pid, proc_root=Path('/proc')):
    """Linux process-wide counters (including native rank threads), no device calls."""
    root = proc_root / str(pid)
    fields = {}
    for line in (root / 'status').read_text().splitlines():
        key, _, value = line.partition(':')
        if key in ('VmRSS', 'VmHWM', 'Threads'):
            parts = value.split()
            if len(parts) != (1 if key == 'Threads' else 2) or (key != 'Threads' and parts[1] != 'kB'):
                raise ValueError(f'invalid process status field {key}')
            fields[key] = int(parts[0])
    if set(fields) != {'VmRSS', 'VmHWM', 'Threads'}:
        raise ValueError('missing live process resource counters')
    return dict(rss_bytes=fields['VmRSS']*1024, peak_rss_bytes=fields['VmHWM']*1024,
                threads=fields['Threads'], file_descriptors=sum(1 for _ in (root/'fd').iterdir()))


def resource_growth(samples, rss_limit_mib):
    """Compare every post-wave sample to the first completed wave, not startup."""
    if len(samples) < 2 or not math.isfinite(rss_limit_mib) or rss_limit_mib < 0:
        raise ValueError('resource gate needs two waves and a finite nonnegative RSS limit')
    first = samples[0]
    growth = {key: max(0, max(s[key] for s in samples)-first[key])
              for key in ('rss_bytes', 'threads', 'file_descriptors')}
    return dict(baseline=first, final=samples[-1], maximum_growth=growth,
                rss_limit_bytes=int(rss_limit_mib*1024*1024),
                passed=(growth['rss_bytes'] <= rss_limit_mib*1024*1024
                        and growth['threads'] == 0 and growth['file_descriptors'] == 0),
                scope='post-wave process counters; not continuous peak or device memory sampling')

def expected_ids(case, eos, respect_eos):
    if case.get('qualification', {}).get('output_is_placeholder'):
        raise ValueError('placeholder outputs are not an independent reference')
    ids = case['output_ids']
    if not ids:
        raise ValueError('empty reference output')
    return ids[:ids.index(eos)+1] if respect_eos and eos in ids else list(ids)


def validate_usage(case, metrics, eos, respect_eos):
    usage = metrics.get('usage')
    prompt = len(case['input_ids'])
    completion = len(expected_ids(case, eos, respect_eos))
    if not usage or any(usage.get(key) != value for key, value in (
        ('prompt_tokens', prompt), ('completion_tokens', completion),
        ('total_tokens', prompt + completion),
    )):
        raise ValueError('HTTP token usage does not match independent reference')


def validate_mixed_progress(prompt_tokens, chunk_tokens, long, short):
    progress = short['first_text_ms_since_run'] < long['first_text_ms_since_run']
    if prompt_tokens > 2 * chunk_tokens:
        if short['request_started_ms_since_run'] >= long['first_text_ms_since_run']:
            raise ValueError('mixed fixture did not submit the short request during long prefill')
        if not progress:
            raise ValueError('multi-chunk prefill blocked short first-output progress')
    return progress


def prefix_boundary(prompt_tokens, page_alignment, chunk_tokens=None):
    if prompt_tokens <= 0 or page_alignment <= 0 or (chunk_tokens is not None and chunk_tokens <= 0):
        raise ValueError('invalid prefix boundary inputs')
    quantum = math.lcm(page_alignment, chunk_tokens) if chunk_tokens is not None else page_alignment
    return (prompt_tokens - 1) // quantum * quantum


def validate_histories(records, manifest, consumed):
    def allowed(numerical_class, count):
        match = re.fullmatch(r'(.+):sequence-batch-v1:([0-9a-f]{64}):member=([0-9]+)', numerical_class)
        base = match[1] if match else numerical_class
        if match and int(match[3]) >= manifest['resident_sequences']:
            return False
        for variant in manifest['variants']:
            if variant['max_sequences'] != 1 or variant['numerical_class'] != base:
                continue
            variable = variant.get('phase') == 'prefill' and variant.get('packed') and not variant.get('graph_capable', True)
            if match and not variable:
                continue
            if variable and 0 < count <= min(variant['max_tokens'], variant['max_query_tokens']):
                return True
            if not variable and count == variant['max_tokens'] == variant['max_query_tokens']:
                return True
        return False
    context=max(v['max_context'] for v in manifest['variants'])
    sequences=set()
    total=0
    for record in records:
        sequence=record['sequence']
        if record['format']!='inferfabric.completed-arithmetic.v1' or sequence in sequences or not isinstance(sequence,int) or sequence<1:
            raise ValueError('invalid completed arithmetic identity')
        sequences.add(sequence)
        if not 0<=record['resident']<manifest['resident_sequences'] or record['history']['version']!=1:
            raise ValueError('invalid arithmetic resident/version')
        end=0
        for span in record['history']['spans']:
            count=span['query_tokens']
            if not allowed(span['numerical_class'],count) or count<=0 or span['start']!=end or not end<span['end']<=context or (span['end']-end)%count:
                raise ValueError('arithmetic history is not a complete declared invocation sequence')
            end=span['end']
        if not end:raise ValueError('empty completed arithmetic record')
        imported=record.get('imported_tokens',0)
        if not isinstance(imported,int) or not 0<=imported<=end:
            raise ValueError('invalid imported arithmetic prefix')
        for span in record['history']['spans']:
            if span['start']<imported<span['end'] and (imported-span['start'])%span['query_tokens']:
                raise ValueError('imported arithmetic prefix splits an invocation')
        total+=end-imported
    if not records or sequences!=set(range(1,len(records)+1)) or total!=consumed:
        raise ValueError('completed arithmetic does not account for scheduler work')
    return dict(request_histories=len(records),completed_input_tokens=total,all_scheduler_tokens_accounted=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('server', 'plan', 'checkpoint', 'library', 'cases', 'output'):
        p.add_argument('--' + key, required=True, type=Path)
    p.add_argument('--port', type=int, default=18085)
    p.add_argument('--max-num-seqs', type=int)
    p.add_argument('--max-num-batched-tokens', type=int)
    p.add_argument('--expect-prefill', choices=('packed','serial'), help='verify actual graph use, excluding startup warmup')
    p.add_argument('--resident-cases', type=Path, help='Test every serial oracle fixture with concurrent short decode and peer cancellation')
    p.add_argument('--prefix-cases', type=Path, help='Enable one snapshot slot; test cold/warm outputs, salt isolation and LRU replacement for every fixture')
    p.add_argument('--prefix-branch-cases', type=Path, help='Two independent references sharing a chunk prefix but different suffixes; requires --prefix-cases')
    p.add_argument('--check-overload', action='store_true', help='Burst long streaming requests beyond admission capacity, check 503/retry metadata and recovery')
    p.add_argument('--load-rounds', type=int, default=0, help='Repeated concurrent reference waves with disconnect/recovery between waves')
    p.add_argument('--load-concurrency', type=int, default=4, help='Clients per repeated-load wave, within bounded admission capacity')
    p.add_argument('--load-rss-growth-limit-mib', type=float, help='Require >=2 load waves; bound post-wave RSS growth from wave 1, with no thread/FD growth')
    p.add_argument('--host-allocator-probe', type=Path, help='Optional Linux preload diagnostic library, applied only to the server process')
    p.add_argument('--request-timeout-seconds', type=int, default=120, help='Socket inactivity timeout, not a whole-run deadline; raise for 64K prefill')
    p.add_argument('--respect-eos', action='store_true', help='Test normal serving through first EOS; preserve full references and report the narrower comparison scope')
    p.add_argument('--check-arithmetic-history', action='store_true', help='Require completed per-resident invocation histories to account for all scheduler work')
    args = p.parse_args()
    if args.request_timeout_seconds < 1:
        p.error('--request-timeout-seconds must be positive')
    if args.load_rounds < 0 or args.load_concurrency < 1:
        p.error('load rounds must be nonnegative and concurrency positive')
    if args.load_rss_growth_limit_mib is not None and (args.load_rounds < 2 or not math.isfinite(args.load_rss_growth_limit_mib) or args.load_rss_growth_limit_mib < 0):
        p.error('RSS gate requires >=2 load waves and a finite nonnegative limit')
    if args.prefix_cases and not args.max_num_seqs:
        p.error('--prefix-cases requires explicit --max-num-seqs and one additional compiled resident')
    if args.prefix_branch_cases and not args.prefix_cases:
        p.error('--prefix-branch-cases requires --prefix-cases')
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
    resident_cases = json.loads(args.resident_cases.read_text()) if args.resident_cases else []
    prefix_cases = json.loads(args.prefix_cases.read_text()) if args.prefix_cases else []
    if args.prefix_cases and not prefix_cases:
        p.error('--prefix-cases must contain independent references')
    for case in prefix_cases:
        expected_text(case)
    branch_cases = json.loads(args.prefix_branch_cases.read_text()) if args.prefix_branch_cases else []
    if args.prefix_branch_cases and (len(branch_cases) != 2 or branch_cases[0]['input_ids'] == branch_cases[1]['input_ids']):
        p.error('--prefix-branch-cases requires two distinct prompts')
    for case in branch_cases:
        expected_text(case)
    if args.resident_cases:
        if not resident_cases:
            p.error('--resident-cases must contain at least one independent reference')
        for case in resident_cases:
            expected_text(case)
    url = f'http://127.0.0.1:{args.port}'
    clock_origin = time.perf_counter()

    def request(case, stream=False, disconnect=False, temperature=0, cache_salt=None):
        body = dict(model='inferfabric-program', prompt=case['input_ids'],
                    max_tokens=len(expected_ids(case,eos,args.respect_eos)), temperature=temperature, stream=stream, ignore_eos=not args.respect_eos)
        if cache_salt is not None:
            body['cache_salt']=cache_salt
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
                                    'request_started_ms_since_run': (began-clock_origin)*1000,
                                    'first_text_ms_since_run': (receipts[0]-clock_origin)*1000,
                                    'text_chunks': len(receipts),
                                    'chunk_intervals_ms': [(b-a)*1000 for a,b in zip(receipts,receipts[1:])],
                                    'total_ms': (time.perf_counter()-began)*1000, 'usage': usage}

    env = dict(os.environ, INFERFABRIC_DECODE_GRAPH='1',
               INFERFABRIC_ARITHMETIC_AUDIT_DIR=str((args.output/'arithmetic').resolve()),
               INFERFABRIC_PREPARED_AUDIT_DIR=str((args.output/'prepared').resolve()),
               INFERFABRIC_TRANSFER_AUDIT_DIR=str((args.output/'transfers').resolve()))
    env.pop('INFERFABRIC_PREPARE_ARENA', None)
    if args.host_allocator_probe:
        probe=args.host_allocator_probe.resolve(strict=True)
        env['LD_PRELOAD']=str(probe)+(' '+env['LD_PRELOAD'] if env.get('LD_PRELOAD') else '')
        env['INFERFABRIC_HOST_ALLOCATOR_AUDIT']=str((args.output/'host-allocator.jsonl').resolve())
        env['INFERFABRIC_HOST_ALLOCATOR_EXECUTABLE']=str(args.server.resolve())
    command = [str(args.server.resolve()), '--program', str(args.plan), str(args.checkpoint), str(args.library), str(args.port)]
    if args.prefix_cases:
        command += ['--prefix-cache-slots','1','--prefix-cache-namespace','qualification']
        env['INFERFABRIC_STATE_COPY_AUDIT_DIR']=str((args.output/'state-copies').resolve())
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
                    validate_usage(case, observed, eos, args.respect_eos)
                results.append(dict(case=case['case'], text=text, collected=collected, streaming=metrics))
            residents = args.max_num_seqs or json.loads((args.plan/'serving.json').read_text()).get('resident_sequences', 1)
            # Fit the bounded queue even before the worker takes its first job.
            # Overload is a separate gate; it must not mask isolation/progress.
            concurrent_cases = (cases * 2)[:4 * residents]
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(concurrent_cases)) as executor:
                futures = [executor.submit(request, c) for c in concurrent_cases]
                for case, future in zip(concurrent_cases, futures):
                    text, metrics = future.result()
                    if text != expected_text(case):
                        raise RuntimeError('queued request isolation')
                    validate_usage(case, metrics, eos, args.respect_eos)
            request(cases[-1], stream=True, disconnect=True)
            text, _ = request(cases[0])
            if text != expected_text(cases[0]):
                raise RuntimeError('disconnect recovery')
            prefix_results=[]
            prefix_branch=None
            if prefix_cases:
                manifest=json.loads((args.plan/'serving.json').read_text())
                alignment=math.lcm(*(state['storage'].get('page_tokens',1) for state in manifest['states']))
                program=json.loads((args.plan/'device-program.json').read_text())
                quantum=min(program['prefill_tokens'],args.max_num_batched_tokens or program['prefill_tokens']) if program.get('uncaptured_prefill') else None
                for case in prefix_cases:
                    boundary=prefix_boundary(len(case['input_ids']),alignment,quantum)
                    if not boundary:raise RuntimeError('prefix fixture has no eligible checkpoint')
                    observations=[]
                    # One snapshot slot: switching salt must miss and evict the
                    # prior domain, even when its token IDs are identical.
                    for label,salt,cached in [('cold-a','a',0),('warm-a','a',boundary),('cold-b','b',0),('warm-b','b',boundary),('evicted-a','a',0)]:
                        text,metrics=request(case,stream=True,cache_salt=f"{case['case']}:{salt}")
                        if text!=expected_text(case):raise RuntimeError(f'prefix {label} output mismatch')
                        validate_usage(case,metrics,eos,args.respect_eos)
                        details=metrics['usage'].get('prompt_tokens_details') or {}
                        # vLLM omits the optional details object for zero hits.
                        if details.get('cached_tokens',0)!=cached:
                            raise RuntimeError(f'prefix {label} cached tokens {details} != {cached}')
                        observations.append(dict(mode=label,cached_tokens=cached,**metrics))
                    prefix_results.append(dict(case=case['case'],boundary=boundary,observations=observations))
                    (args.output/'prefix-progress.json').write_text(json.dumps(dict(complete=False,cases=prefix_results),indent=2)+'\n')
                if branch_cases:
                    source, target = branch_cases
                    shared = 0
                    for left, right in zip(source['input_ids'], target['input_ids']):
                        if left != right: break
                        shared += 1
                    boundary = prefix_boundary(len(source['input_ids']),alignment,quantum)
                    if not boundary or shared < boundary or boundary >= len(target['input_ids']):
                        raise RuntimeError('branch fixture must share the complete published prefix and have an uncached suffix')
                    observations=[]
                    for label,case,cached_tokens in [('source-cold',source,0),('different-suffix',target,boundary)]:
                        text,metrics=request(case,stream=True,cache_salt='qualification-distinct-suffix')
                        if text!=expected_text(case):raise RuntimeError(f'prefix branch {label} output mismatch')
                        validate_usage(case,metrics,eos,args.respect_eos)
                        actual=(metrics['usage'].get('prompt_tokens_details') or {}).get('cached_tokens',0)
                        if actual!=cached_tokens:raise RuntimeError(f'prefix branch {label} cached {actual} != {cached_tokens}')
                        observations.append(dict(mode=label,case=case['case'],cached_tokens=actual,**metrics))
                    prefix_branch=dict(shared_tokens=shared,published_boundary=boundary,observations=observations)
            mixed = None
            mixed_cases = []
            for long_case in resident_cases:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    long = executor.submit(request, long_case, True)
                    time.sleep(0.25)
                    request(cases[-1], stream=True, disconnect=True)
                    short_text, short_metrics = request(cases[0], stream=True)
                    short_first=not long.done()
                    # Completion order depends on requested output lengths.
                    # Check first-output progress during multi-chunk prefill.
                    compiled_chunk=json.loads((args.plan/'device-program.json').read_text())['prefill_tokens']
                    if short_text != expected_text(cases[0]):
                        raise RuntimeError('resident cancellation/reset corrupted short request')
                    long_text, long_metrics = long.result()
                    short_prefill_progress = validate_mixed_progress(len(long_case['input_ids']), compiled_chunk, long_metrics, short_metrics)
                    if long_text != expected_text(long_case):
                        raise RuntimeError('resident cancellation/reset corrupted long peer')
                    validate_usage(cases[0], short_metrics, eos, args.respect_eos)
                    validate_usage(long_case, long_metrics, eos, args.respect_eos)
                    record = dict(case=long_case['case'], long_prompt_tokens=len(long_case['input_ids']),
                                  short_finished_before_long=short_first, cancellation_during_peer_prefill=True,
                                  short_first_output_before_long=short_prefill_progress,
                                  short=short_metrics, long=long_metrics)
                    mixed_cases.append(record)
                    # Preserve completed cases if a subsequent long run fails.
                    # This is partial evidence, not the final passing report.
                    (args.output/'mixed-progress.json').write_text(json.dumps(
                        dict(complete=False, completed=mixed_cases, expected_cases=len(resident_cases)),
                        ensure_ascii=False, indent=2)+'\n')
            if mixed_cases:
                # Retain the legacy largest-case summary for evidence readers.
                mixed = max(mixed_cases, key=lambda c: c['long_prompt_tokens'])
            load = None
            if args.load_rounds:
                if args.load_concurrency > 4*residents:
                    raise RuntimeError('load concurrency exceeds conservative queue bound; use --check-overload separately')
                began=time.perf_counter();waves=[]
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.load_concurrency) as pool:
                    for round_id in range(args.load_rounds):
                        wave=[cases[(round_id+i)%len(cases)] for i in range(args.load_concurrency)]
                        if resident_cases:wave[0]=resident_cases[round_id%len(resident_cases)]
                        pending=[pool.submit(request,case,True) for case in wave]
                        observations=[]
                        for case,future in zip(wave,pending):
                            text,metrics=future.result()
                            if text!=expected_text(case):raise RuntimeError(f'load round {round_id} output mismatch')
                            validate_usage(case,metrics,eos,args.respect_eos)
                            observations.append(dict(case=case['case'],**metrics))
                        request(cases[0],stream=True,disconnect=True)
                        text,metrics=request(cases[0],stream=True)
                        if text!=expected_text(cases[0]):raise RuntimeError('load disconnect recovery output mismatch')
                        validate_usage(cases[0],metrics,eos,args.respect_eos)
                        waves.append(dict(round=round_id,requests=observations,recovery=metrics))
                        if args.load_rss_growth_limit_mib is not None:
                            waves[-1]['process_resources'] = process_resources(server.pid)
                            waves[-1]['resource_sample_time_ns'] = time.time_ns()
                        (args.output/'load-progress.json').write_text(json.dumps(dict(complete=False,rounds=waves),indent=2)+'\n')
                load=dict(rounds=args.load_rounds,concurrency=args.load_concurrency,
                          completed_reference_requests=args.load_rounds*(args.load_concurrency+1),
                          deliberate_disconnects=args.load_rounds,elapsed_seconds=time.perf_counter()-began,waves=waves)
                if args.load_rss_growth_limit_mib is not None:
                    growth = resource_growth([w['process_resources'] for w in waves], args.load_rss_growth_limit_mib)
                    (args.output/'resource-growth.json').write_text(json.dumps(growth,indent=2)+'\n')
                    load['resource_growth'] = growth
                    if not growth['passed']:
                        raise RuntimeError('repeated-load process resource growth exceeded the declared gate; see resource-growth.json')
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
                long_case = max(resident_cases, key=lambda c:len(c['input_ids']))
                count = args.max_num_seqs * 8
                start = threading.Barrier(count)
                body = json.dumps(dict(model='inferfabric-program', prompt=long_case['input_ids'],
                    max_tokens=len(expected_ids(long_case,eos,args.respect_eos)), temperature=0, stream=True, ignore_eos=not args.respect_eos)).encode()
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
            report = dict(frontend='vLLM Rust 0.25.1', mode='resident request scheduler; compiled prefill and native decode',
                          full_fixture_text=not args.respect_eos, sse_agreement=True, queued_isolation=True,
                          reference_comparison_scope='through_first_eos' if args.respect_eos else 'full_forced_continuation',
                          fixture_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                          resident_fixture_sha256=hashlib.sha256(args.resident_cases.read_bytes()).hexdigest() if args.resident_cases else None,
                          request_socket_timeout_seconds=args.request_timeout_seconds,
                          disconnect_recovery=True, unsupported_sampling_rejected=True, results=results,
                          concurrent_requests=len(concurrent_cases), mixed_residents=mixed,
                          mixed_resident_cases=mixed_cases,
                          prefix_cases=prefix_results,
                          prefix_branch=prefix_branch,
                          prefix_branch_fixture_sha256=hashlib.sha256(args.prefix_branch_cases.read_bytes()).hexdigest() if args.prefix_branch_cases else None,
                          prefix_fixture_sha256=hashlib.sha256(args.prefix_cases.read_bytes()).hexdigest() if args.prefix_cases else None,
                          context_overflow_rejected=True,
                          max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens)
            if overload is not None: report['overload'] = overload
            if load is not None: report['repeated_load'] = load
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
        if args.prefix_cases:
            logs=(args.output/'server.log').read_text()
            cached=re.search(r'prefix cache stopped hits=(\d+) restored_tokens=(\d+) published=(\d+) remaining=0',logs)
            if not cached or int(cached[1])<2*len(prefix_cases) or int(cached[2])<2*sum(c['boundary'] for c in prefix_results):
                raise RuntimeError('prefix cache did not confirm hits and shutdown retirement')
            report['prefix_cache_shutdown']=dict(zip(('hits','restored_tokens','published'),map(int,cached.groups())),remaining=0)
        manifest = json.loads((args.plan/'serving.json').read_text())
        report['prefill_graph_replays'] = []
        if args.prefix_cases: report['state_copy_totals'] = []
        for rank in range(len(manifest['identity']['devices'])):
            audit = json.loads((args.output/'prepared'/f'rank-{rank}.json').read_text())
            transfers = json.loads((args.output/'transfers'/f'rank-{rank}.json').read_text())
            if args.prefix_cases:
                copies=json.loads((args.output/'state-copies'/f'rank-{rank}.json').read_text())
                if not copies['operations'] or any(op['primitive']!='D2D' for op in copies['operations']) or audit.get('state_copy_operations',0)<=0:
                    raise RuntimeError('prefix state copy lacks device-only execution evidence')
                report['state_copy_totals'].append(dict(rank=rank,operations=audit['state_copy_operations'],bytes=audit['state_copy_bytes']))
            program = json.loads((args.plan/'device-program.json').read_text())
            uncaptured=program.get('uncaptured_prefill',False)
            names = set() if uncaptured else ({f'packed_prefill_{n}' for n in program.get('prefill_variants', [])} or ({'packed_prefill'} if program['prefill_tokens']>1 else set()))
            packed = [g for g in audit['graphs'] if g['name'].startswith('packed_prefill')]
            if len(packed)!=len(names) or {g['name'] for g in packed} != names:
                raise RuntimeError('HTTP prepared graphs differ from the declared family')
            replays = sum(g['replays'] for g in packed)
            executed=audit.get('uncaptured_prefill_forwards',0) if uncaptured else replays
            warmup=2+int(min(manifest['resident_sequences'],program['prefill_tokens'])>1) if uncaptured else len(names)
            if args.expect_prefill=='packed' and executed<=warmup:
                raise RuntimeError('HTTP did not exercise packed prefill beyond warmup')
            if args.expect_prefill=='serial' and executed>warmup:
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
        if args.check_arithmetic_history:
            logs=(args.output/'server.log').read_text()
            stats=re.search(r'program scheduler stopped .*consumed_tokens=(\d+)',logs)
            if not stats:raise RuntimeError('missing scheduler completion count')
            records=[json.loads(path.read_text()) for path in (args.output/'arithmetic').glob('request-*.json')]
            report['arithmetic_history']=validate_histories(records,manifest,int(stats[1]))
        (args.output / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
