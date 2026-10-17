#!/usr/bin/env python3
"""Qualify release-error propagation and explicit process restart on real NPU.

Inject only an exclusive audit-file collision in this run's new output directory.
No model files, runtime fault hooks, device resets or external processes are touched.
This is a backend error-return test, not a hardware rank crash or automatic recovery.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import time
import urllib.error
import urllib.request


def validate_failure(events):
    if not any(isinstance(e, dict) and e.get('error') for e in events):
        raise ValueError('failure was not surfaced as a structured API error')
    if any(c.get('finish_reason') in ('stop', 'length') for e in events for c in e.get('choices', [])):
        raise ValueError('terminal success published despite release failure')


def validate_pool_drain(log):
    counts = re.findall(r'shared state pool .*remaining_occupied_bytes=(\d+)', log)
    if not counts or any(int(n) for n in counts):
        raise ValueError('logical shared-page leases were not drained')


def validate_fault_log(log):
    if 'program worker failed:' not in log or 'File exists (os error 17)' not in log:
        raise ValueError('backend failure does not match the injected exclusive-file collision')
    validate_pool_drain(log)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('server', 'plan', 'checkpoint', 'library', 'cases', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--port', type=int, default=18085)
    p.add_argument('--hccl-port', type=int, default=45200)
    args = p.parse_args()
    if not 1 <= args.port <= 65535 or not 1024 <= args.hccl_port <= 65000:
        p.error('invalid port bounds')
    args.output.mkdir()
    from tokenizers import Tokenizer
    case = json.loads(args.cases.read_text())[0]
    if case.get('qualification', {}).get('output_is_placeholder'):
        raise ValueError('independent reference required')
    eos = json.loads((args.plan / 'device-program.json').read_text())['eos']
    expected = case['output_ids'][:]
    if eos in expected:
        expected = expected[:expected.index(eos) + 1]
    if not expected:
        raise ValueError('empty reference')
    tokenizer = Tokenizer.from_file(str(args.checkpoint / 'tokenizer.json'))
    text = tokenizer.decode(expected, skip_special_tokens=True)
    base = f'http://127.0.0.1:{args.port}'
    body = dict(model='inferfabric-program', prompt=case['input_ids'],
                temperature=0, max_tokens=len(expected), ignore_eos=False)

    def health():
        try:
            with urllib.request.urlopen(base + '/health', timeout=2) as response:
                return response.status
        except urllib.error.HTTPError as error:
            return error.code

    def request(stream=False):
        req = urllib.request.Request(base + '/v1/completions', data=json.dumps({**body, 'stream':stream}).encode(),
                                     headers={'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                if not stream:
                    return response.status, [json.load(response)]
                events = []
                for line in response:
                    if line.startswith(b'data: ') and line[6:].strip() != b'[DONE]':
                        events.append(json.loads(line[6:]))
                return response.status, events
        except urllib.error.HTTPError as error:
            return error.code, [json.loads(error.read())]

    command = [str(args.server.resolve()), '--program', str(args.plan.resolve()),
               str(args.checkpoint.resolve()), str(args.library.resolve()), str(args.port),
               '--max-num-seqs', '2', '--max-num-batched-tokens', '4096']
    stages = []
    for index, name in enumerate(('fault', 'restart')):
        directory = args.output / name
        directory.mkdir()
        audit = directory / 'arithmetic'
        audit.mkdir()
        env = dict(os.environ, INFERFABRIC_ARITHMETIC_AUDIT_DIR=str(audit.resolve()),
                   HCCL_IF_BASE_PORT=str(args.hccl_port + index * 200))
        result = dict(stage=name, command=command)
        with (directory / 'server.log').open('w') as log:
            started = time.monotonic()
            server = subprocess.Popen(command, env=env, stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 900
                while True:
                    if server.poll() is not None:
                        raise RuntimeError('server exited during readiness')
                    try:
                        if health() == 200:
                            break
                    except OSError:
                        pass
                    if time.monotonic() >= deadline:
                        raise TimeoutError('readiness')
                    time.sleep(1)
                result['ready_seconds'] = time.monotonic() - started
                status, events = request()
                baseline = events[0]
                if status != 200 or baseline['choices'][0]['text'] != text or baseline['usage']['completion_tokens'] != len(expected):
                    raise RuntimeError('baseline reference output differs')
                result['baseline'] = baseline
                if name == 'fault':
                    ids = [int(f.stem.split('-')[1]) for f in audit.glob('request-*.json')]
                    if not ids:
                        raise RuntimeError('baseline did not complete its audit before response')
                    collision = audit / f'request-{max(ids)+1}.json'
                    with collision.open('x') as f:
                        json.dump({'injected_exclusive_file_collision': True}, f)
                    status, events = request(stream=True)
                    result.update(failure_http_status=status, failure_events=events, injection=collision.name)
                    validate_failure(events)
                    deadline = time.monotonic() + 30
                    while health() != 503:
                        if server.poll() is not None:
                            raise RuntimeError('service crashed instead of becoming unhealthy')
                        if time.monotonic() >= deadline:
                            raise TimeoutError('backend did not become unavailable')
                        time.sleep(0.05)
                    result['unhealthy_status'] = 503
                    status, events = request()
                    if status != 503:
                        raise RuntimeError('unhealthy backend accepted another request')
                    result.update(rejected_status=status, rejected_events=events)
                    if 'program worker failed:' not in (directory/'server.log').read_text():
                        raise RuntimeError('missing backend failure evidence')
            finally:
                if server.poll() is None:
                    server.send_signal(signal.SIGTERM)
                    try:
                        server.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait()
                        raise RuntimeError('failed backend did not shut down within 120 seconds')
                result['returncode'] = server.returncode
                (directory / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
            if server.returncode:
                raise RuntimeError('server shutdown failed')
            logs = (directory / 'server.log').read_text()
            if name == 'fault':
                validate_fault_log(logs)
            else:
                validate_pool_drain(logs)
            result['logical_pages_drained'] = True
            (directory / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        stages.append(result)
    report = dict(passed=True, scope='release-error return, unhealthy rejection, explicit process restart; not rank crash or automatic failover',
                  stages=stages, hashes={name: hashlib.sha256(path.read_bytes()).hexdigest() for name,path in
                  [('frontend',args.server),('library',args.library),('execution',args.plan/'execution.bin')]})
    (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(passed=True, scope=report['scope'], stages=len(stages))))


if __name__ == '__main__':
    main()
