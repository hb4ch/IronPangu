#!/usr/bin/env python3
"""Summarize full-model C8 qualification from immutable plans and rank audits.

Run inside the NPU Docker workspace after benchmark-c8 qualification. No NPU
execution occurs here; missing, mismatched or incomplete evidence is an error.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def summarize(root, c8_stem="cache-c8", bf16_stem="cache-c8-bf16", require_planned_workspace=False):
    variants = {}
    for name, stem, cache_bytes, modes in [
        ('bf16', bf16_stem, 46 * 512 * 1152, ['tpot']),
        ('c8', c8_stem, 16 * 512 * 656 + 30 * 512 * 1152, ['tpot', 'eager']),
    ]:
        arena = read(root / f'{stem}-bound-plan/arena.json')
        measurements = {}
        for mode in modes:
            run_dir = root / f'{stem}-bound-{mode}'
            summary, runs = read(run_dir / 'summary.json'), read(run_dir / 'runs.json')
            assert summary['execution_plan_sha256'] == arena['execution_sha256']
            memories, steps = [], 0
            for run in runs:
                assert run['exact_qualified_tokens']
                folder = run_dir / f"{run['case']}-{run['repeat']}"
                for rank in range(8):
                    audit = read(folder / 'prepared' / f'rank-{rank}.json')
                    transfer = read(folder / 'transfers' / f'rank-{rank}.json')
                    if require_planned_workspace:
                        workspace = audit['workspace']
                        assert workspace['planned'] and workspace['growth_allocations'] == 0
                        assert workspace['slot'] == 0
                        assert workspace['peak_requested_bytes'] <= workspace['planned_bytes']
                        assert workspace['planned_bytes'] == arena['ranks'][rank]['slots'][0]['bytes']
                        for instruction in workspace['instruction_requests']:
                            assert 0 in arena['ranks'][rank]['instruction_slots'][instruction]
                    assert audit['device_allocations'] == 1
                    assert audit['device_bytes'] == arena['ranks'][rank]['bytes']
                    assert all(s['device_allocations'] == 0 and s['completion_boundaries'] == 0 for s in audit['steps'])
                    assert transfer['activation_host_transfers'] == 0 and transfer['rejected'] == 0
                    if mode == 'tpot':
                        assert any(g['replays'] for g in audit['graphs'])
                    steps += len(audit['steps'])
                    memories.append(read(folder / 'prepared' / f'memory-rank-{rank}.json'))
            measurements[mode] = {
                'summary': summary, 'runs': len(runs),
                'output_records': sum(len(r['outputs']) for r in runs),
                'rank_audits': len(memories), 'checked_token_steps': steps,
                'zero_token_allocations_and_internal_fences': True,
                'zero_activation_host_transfers': True,
                'hbm_used_after_generation_bytes_median': statistics.median(m['hbm_used_after_generation_bytes'] for m in memories),
                'hbm_growth_bytes_median': statistics.median(m['hbm_growth_bytes'] for m in memories),
                'memory_samples': memories,
            }
        variants[name] = {
            'persistent_cache_bytes_per_rank': cache_bytes,
            'shared_workspace': [{'planned_bytes': r['workspace']['planned_bytes'],
                                  'peak_requested_bytes': r['workspace']['peak_requested_bytes'],
                                  'slot': r['workspace']['slot'],
                                  'growth_allocations': r['workspace']['growth_allocations']}
                                 for r in arena['ranks']] if require_planned_workspace else None,
            'arena_bytes_per_rank': [r['bytes'] for r in arena['ranks']],
            'arena_slots_per_rank': [len(r['slots']) for r in arena['ranks']],
            'measurements': measurements,
        }
    binaries = {m['summary']['binary_sha256'] for v in variants.values() for m in v['measurements'].values()}
    assert len(binaries) == 1, 'baseline must use the same native binary'
    baseline_path = root / 'paired-linear-eager/results.json'
    bf16, c8 = read(baseline_path), read(root / 'cache-c8-preparation/results.json')
    comparisons = []
    for reference, candidate in zip(bf16, c8, strict=True):
        assert reference['input_ids'] == candidate['input_ids']
        common = 0
        for a, b in zip(reference['output_ids'], candidate['output_ids']):
            if a != b:
                break
            common += 1
        comparisons.append({
            'case': candidate['case'], 'bf16_tokens': len(reference['output_ids']),
            'c8_tokens': len(candidate['output_ids']), 'common_token_prefix': common,
            'tokens_identical': reference['output_ids'] == candidate['output_ids'],
            'full_records_identical': reference['steps'] == candidate['steps'],
            'max_common_prefix_logit_difference': max(abs(a['logit']-b['logit']) for a,b in zip(reference['steps'][:common],candidate['steps'][:common])),
            'bf16_text': reference['text'], 'c8_text': candidate['text'],
        })
    return {
        'format': 'inferfabric.packed-c8-qualification.v1',
        'scope': '46-layer 92B TP8/EP8, batch one, context 512, three short prompts; DSA C8/SWA BF16; no long-context indexer qualification',
        'bf16_fixture_sha256': hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        'kernel_checks': read(root / 'c8-probe.json'),
        'bf16_comparison': comparisons,
        'variants': variants,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--c8-stem', default='cache-c8')
    parser.add_argument('--bf16-stem', default='cache-c8-bf16')
    parser.add_argument('--require-planned-workspace', action='store_true')
    args = parser.parse_args()
    result = summarize(args.root, args.c8_stem, args.bf16_stem, args.require_planned_workspace)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for name, variant in result['variants'].items():
        m = variant['measurements']['tpot']
        print(json.dumps({'format': name, 'tpot_ms': m['summary']['pooled_tpot']['mean_ms'],
                          'cache_bytes_per_rank': variant['persistent_cache_bytes_per_rank'],
                          'arena_bytes_per_rank': variant['arena_bytes_per_rank'][0],
                          'median_hbm_bytes': m['hbm_used_after_generation_bytes_median']}))
