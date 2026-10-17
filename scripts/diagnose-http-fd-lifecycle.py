#!/usr/bin/env python3
"""Diagnose real HTTP connection cleanup without model/NPU execution.

Run in the container with the existing Rust frontend's synthetic mock mode.
This does not qualify model output or relax any serving resource gate.
"""
import argparse
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

spec = importlib.util.spec_from_file_location('http_qualify', Path(__file__).with_name('qualify-program-http.py'))
qualify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualify)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('server', 'model', 'tokenizer', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--port', type=int, default=18086)
    parser.add_argument('--endpoint', choices=('health', 'sse'), default='health')
    parser.add_argument('--rounds', type=int, default=200)
    parser.add_argument('--concurrency', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 10000 or not 1 <= args.concurrency <= 64 or not 1 <= args.port <= 65535:
        parser.error('invalid diagnostic bounds')
    args.output.mkdir()
    command = [str(args.server.resolve()), '--mock', str(args.model.resolve()),
               str(args.tokenizer.resolve()), str(args.port)]
    url = f'http://127.0.0.1:{args.port}/health'

    def health():
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status != 200:
                raise RuntimeError('health request failed')
            response.read()

    def request(_=None):
        if args.endpoint == 'health':
            return health()
        body = dict(model='inferfabric-mock', messages=[dict(role='user', content='hello world')],
                    temperature=0, max_tokens=4, stream=True)
        req = urllib.request.Request(url.replace('/health','/v1/chat/completions'),
                                     data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req, timeout=5) as response:
            for line in response:
                if line.strip() == b'data: [DONE]':
                    return  # Same completion boundary used by the serving qualifier.
                if line.startswith(b'data: '):
                    event = json.loads(line[6:])
                    if event.get('error'):
                        raise RuntimeError('mock SSE error')
            raise RuntimeError('missing SSE completion')

    with (args.output / 'server.log').open('w') as log:
        server = subprocess.Popen(command, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 30
            while True:
                if server.poll() is not None:
                    raise RuntimeError('mock frontend exited during startup')
                try:
                    health()
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('mock frontend readiness')
                    time.sleep(0.1)
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                list(pool.map(request, range(args.concurrency * 5)))
                time.sleep(0.2)
                baseline = qualify.process_resources(server.pid)
                observations = []
                for round_id in range(args.rounds):
                    list(pool.map(request, range(args.concurrency)))
                    began = time.monotonic()
                    sample = qualify.process_resources(server.pid)
                    observation = dict(round=round_id + 1, immediate=sample, cleanup_samples=[])
                    # A bounded diagnostic observation, not a retry-to-pass gate.
                    while sample['file_descriptors'] > baseline['file_descriptors'] and time.monotonic() - began < 2:
                        time.sleep(0.01)
                        sample = qualify.process_resources(server.pid)
                        observation['cleanup_samples'].append(dict(elapsed_ms=(time.monotonic()-began)*1000, resources=sample))
                    observations.append(observation)
                final = qualify.process_resources(server.pid)
            report = dict(scope='mock backend, production HTTP connection path; no model or NPU qualification',
                          command=command, baseline=baseline, final=final, observations=observations,
                          rounds=args.rounds, concurrency=args.concurrency, endpoint=args.endpoint,
                          immediate_fd_growth_rounds=[o['round'] for o in observations if o['immediate']['file_descriptors'] > baseline['file_descriptors']])
            (args.output / 'observations.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({k: report[k] for k in ('scope', 'endpoint', 'rounds', 'concurrency', 'immediate_fd_growth_rounds')}))
        finally:
            if server.poll() is None:
                server.send_signal(signal.SIGTERM)
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
                    raise RuntimeError('diagnostic frontend did not drain')
            (args.output / 'process.json').write_text(json.dumps(dict(returncode=server.returncode)) + '\n')
        if server.returncode:
            raise RuntimeError('diagnostic frontend shutdown failed')


if __name__ == '__main__':
    main()
