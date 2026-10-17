#!/usr/bin/env python3
"""Offline client of the vLLM Rust frontend; no Python inference engine.

Checks full fixture text, SSE agreement, queue isolation and disconnect recovery.
Receipt intervals are HTTP chunk intervals, not necessarily one token per chunk.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.error
import urllib.request
from tokenizers import Tokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('server', 'plan', 'checkpoint', 'library', 'cases', 'output'):
        p.add_argument('--' + key, required=True, type=Path)
    p.add_argument('--port', type=int, default=18085)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = json.loads(args.cases.read_text())
    tokenizer = Tokenizer.from_file(str(args.checkpoint / 'tokenizer.json'))
    url = f'http://127.0.0.1:{args.port}'

    def request(case, stream=False, disconnect=False, temperature=0):
        body = dict(model='inferfabric-program', prompt=case['input_ids'],
                    max_tokens=len(case['output_ids']), temperature=temperature, stream=stream)
        if stream:
            body['stream_options'] = {'include_usage': True}
        began = time.perf_counter()
        req = urllib.request.Request(url + '/v1/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=120) as response:
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

    env = dict(os.environ, INFERFABRIC_DECODE_GRAPH='1')
    env.pop('INFERFABRIC_PREPARE_ARENA', None)
    command = [str(args.server.resolve()), '--program', str(args.plan), str(args.checkpoint), str(args.library), str(args.port)]
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
                expected = tokenizer.decode(case['output_ids'], skip_special_tokens=True)
                text, collected = request(case)
                streamed, metrics = request(case, stream=True)
                if text != expected or streamed != text:
                    raise RuntimeError(f"{case['case']} text mismatch: {text!r} / {streamed!r} / {expected!r}")
                results.append(dict(case=case['case'], text=text, collected=collected, streaming=metrics))
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(request, c) for c in cases]
                for case, future in zip(cases, futures):
                    text, _ = future.result()
                    if text != tokenizer.decode(case['output_ids'], skip_special_tokens=True):
                        raise RuntimeError('queued request isolation')
            request(cases[-1], stream=True, disconnect=True)
            text, _ = request(cases[0])
            if text != tokenizer.decode(cases[0]['output_ids'], skip_special_tokens=True):
                raise RuntimeError('disconnect recovery')
            try:
                request(cases[0], temperature=0.7)
            except urllib.error.HTTPError as e:
                if e.code != 400:
                    raise
            else:
                raise RuntimeError('unsupported sampling was silently accepted')
            report = dict(frontend='vLLM Rust 0.25.1', mode='persistent serial program',
                          full_fixture_text=True, sse_agreement=True, queued_isolation=True,
                          disconnect_recovery=True, unsupported_sampling_rejected=True, results=results)
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
        (args.output / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
