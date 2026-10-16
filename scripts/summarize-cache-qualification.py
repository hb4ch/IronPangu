import argparse,json,statistics
from pathlib import Path
parser=argparse.ArgumentParser(description='Summarize frozen-arena cache precision qualification; run inside the NPU Docker workspace.')
parser.add_argument('root',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
variants={}
for dtype,bits in [('bf16',16),('int8',8),('int4',4)]:
 root=args.root;plan=json.loads((root/f'cache-{dtype}-bound-plan/arena.json').read_text())
 modes=['tpot'] if bits==16 else ['tpot','eager'];measurements={}
 for mode in modes:
  run_dir=root/f'cache-{dtype}-bound-{mode}';summary=json.loads((run_dir/'summary.json').read_text());runs=json.loads((run_dir/'runs.json').read_text())
  memories=[];checked_steps=0
  for run in runs:
   for rank in range(8):
    folder=run_dir/f"{run['case']}-{run['repeat']}"
    audit=json.loads((folder/'prepared'/f'rank-{rank}.json').read_text())
    transfers=json.loads((folder/'transfers'/f'rank-{rank}.json').read_text())
    assert audit['device_allocations']==1
    assert audit['device_bytes']==plan['ranks'][rank]['bytes']
    assert all(s['device_allocations']==0 and s['completion_boundaries']==0 for s in audit['steps'])
    assert transfers['activation_host_transfers']==0 and transfers['rejected']==0
    checked_steps+=len(audit['steps'])
    memories.append(json.loads((folder/'prepared'/f'memory-rank-{rank}.json').read_text()))
  measurements[mode]={'tpot':summary['pooled_tpot'],'runs':len(runs),'output_records':sum(len(r['outputs']) for r in runs),'rank_audits':len(memories),'checked_token_steps':checked_steps,'zero_token_allocations_and_internal_fences':True,'zero_activation_host_transfers':True,'binary_sha256':summary['binary_sha256'],'execution_plan_sha256':summary['execution_plan_sha256'],'qualification_cases_sha256':summary['qualification_cases_sha256'],'hbm_used_after_generation_bytes_median':statistics.median(m['hbm_used_after_generation_bytes'] for m in memories),'hbm_growth_bytes_median':statistics.median(m['hbm_growth_bytes'] for m in memories),'memory_samples':memories}
 variants[dtype]={'persistent_cache_payload_plus_scales_bytes_per_rank':46*512*(576*bits//8+(4 if bits!=16 else 0)),'arena_bytes_per_rank':[r['bytes'] for r in plan['ranks']],'arena_slots_per_rank':[len(r['slots']) for r in plan['ranks']],'measurements':measurements}
result={'format':'inferfabric.cache-precision-qualification.v1','scope':'TP8/EP8, 46-layer 92B, context capacity 512, batch one, three short prompts; per-format exact replay records; not a broad quality benchmark','variants':variants}
args.output.write_text(json.dumps(result,indent=2)+'\n')
for dtype,v in variants.items():
 print(json.dumps({'dtype':dtype,'cache_bytes':v['persistent_cache_payload_plus_scales_bytes_per_rank'],'arena_bytes':v['arena_bytes_per_rank'][0],'tpot_ms':v['measurements']['tpot']['tpot']['mean_ms'],'hbm_bytes_median':v['measurements']['tpot']['hbm_used_after_generation_bytes_median']}))
