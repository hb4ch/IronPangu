#!/usr/bin/env python3
"""HTTP-only allocator qualification; run inside the NPU Docker container.

No model execution occurs in Python. Requests use the vLLM Rust frontend.
Record dedicated allocation results, then compare the same corpus with pooling.
"""
import argparse
import concurrent.futures
import json
import re
import statistics
import subprocess
import time
import urllib.request
import urllib.error
from pathlib import Path


def memory():
    output = subprocess.check_output(['npu-smi', 'info'], text=True)
    return [int(value) for value in re.findall(r'\|\s*\d+\s+\d+\s*\|\s*\d+\s*\|\s*inferfabric[^|]*\|\s*(\d+)\s*\|', output)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:18081')
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference')
    parser.add_argument('--rounds', type=int, default=5)
    args = parser.parse_args()
    assert args.rounds >= 2
    prompts = ['The capital of France is', 'Once upon a time', 'Explain addition in one sentence.', 'red blue green ' * 160 + '\nThe colors are']
    corpus = [dict(model='inferfabric-qwen35', prompt=prompts[i % 4], max_tokens=12,
                   temperature=0 if i % 3 == 0 else 0.8, top_k=20, top_p=0.9,
                   seed=100+i, presence_penalty=0.3, frequency_penalty=0.1)
              for i in range(12)]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 180
    while True:
        try:
            with opener.open(args.url + '/health', timeout=5) as response:
                assert response.status == 200
            break
        except urllib.error.URLError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(1)
    def request(payload):
        start = time.monotonic()
        req = urllib.request.Request(args.url + '/v1/completions',
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with opener.open(req, timeout=180) as response:
            result = json.load(response)
        return dict(choices=result['choices'], usage=result['usage']), time.monotonic()-start
    outputs = [request(payload)[0] for payload in corpus]
    if args.reference:
        expected = json.loads(Path(args.reference).read_text())['outputs']
        assert outputs == expected, 'dedicated/arena output mismatch'
    memories, latencies = [], []
    for _ in range(args.rounds):
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            results = list(pool.map(request, corpus))
        assert [result[0] for result in results] == outputs, 'mixed/isolated output mismatch'
        latencies.extend(result[1] for result in results)
        memories.append(memory())
        assert memories[-1], 'no NPU process memory readings'
    assert all(len(sample) == len(memories[0]) for sample in memories), 'device process set changed'
    growth = [max(sample[rank] for sample in memories) - min(sample[rank] for sample in memories)
              for rank in range(len(memories[0]))]
    assert max(growth) <= 16, f'HBM did not plateau within 16 MiB: {growth}'
    latencies.sort()
    report = dict(status='passed', reference=args.reference, outputs=outputs,
        mixed_requests=len(corpus)*args.rounds, process_memory_mib=memories,
        memory_range_mib=growth, median_request_seconds=statistics.median(latencies),
        p95_request_seconds=latencies[int(0.95*(len(latencies)-1))])
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key != 'outputs'}))


if __name__ == '__main__':
    main()
