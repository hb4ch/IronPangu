#!/usr/bin/env python3
"""Offline OmniInfer performance reference; never imported by native serving."""
import argparse, json, os, time
from pathlib import Path
from vllm import LLM, SamplingParams
import omni_npu.worker.npu_worker as worker
_original_init = worker.init_worker_distributed_environment
def reference_distributed_init(*args, **kwargs):
    _original_init(*args, **kwargs)
    from omni_npu.v1.distributed.parallel_state_ext import ensure_layer_parallel_initialized
    ensure_layer_parallel_initialized(backend='hccl')
worker.init_worker_distributed_environment = reference_distributed_init

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True);p.add_argument('--cases',required=True)
    p.add_argument('--output',required=True);p.add_argument('--eager',action='store_true')
    p.add_argument('--profile',action='store_true',help='Profile two decode steps of the first case; timings include profiler overhead')
    p.add_argument('--repeats',type=int,default=2);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    cases=json.loads(Path(a.cases).read_text())
    profile_options = {'profiler_config': {'profiler': 'torch', 'torch_profiler_dir': str((out/'trace').resolve())}} if a.profile else {}
    model=LLM(model=a.checkpoint,trust_remote_code=True,tensor_parallel_size=8,
      enable_expert_parallel=True,max_model_len=512,max_num_seqs=1,
      max_num_batched_tokens=512,gpu_memory_utilization=.5,enforce_eager=a.eager,
      enable_prefix_caching=False,disable_custom_all_reduce=True,**profile_options)
    params=SamplingParams(temperature=0,max_tokens=32,logprobs=5)
    # Warm up the actual model/parallel configuration once before timing.
    model.generate([{'prompt_token_ids':cases[0]['input_ids']}],params,use_tqdm=False)
    # LLM.generate changes output_kind to FINAL_ONLY; use fresh streaming params.
    from vllm.sampling_params import RequestOutputKind
    params=SamplingParams(temperature=0,max_tokens=32,logprobs=5,output_kind=RequestOutputKind.CUMULATIVE)
    runs=[]
    for repeat in range(a.repeats):
      for case in cases:
        request=f"{case['case']}-{repeat}"
        model.llm_engine.add_request(request,{'prompt_token_ids':case['input_ids']},params)
        receipts=[];tokens=[]
        while model.llm_engine.has_unfinished_requests():
          outputs=model.llm_engine.step();stamp=time.perf_counter_ns()
          for result in outputs:
            if result.request_id!=request:raise RuntimeError('unexpected request')
            current=list(result.outputs[0].token_ids)
            if len(current)>len(tokens):
              if len(current)!=len(tokens)+1:raise RuntimeError('coalesced token results')
              receipts.append(stamp);tokens=current
              if a.profile and repeat==0 and case is cases[0]:
                if len(tokens)==1:model.start_profile()
                if len(tokens)==3:model.stop_profile()
        if tokens!=case['output_ids']:raise RuntimeError(f'output mismatch: {request}: {tokens}')
        intervals=[(b-a)/1e6 for a,b in zip(receipts,receipts[1:])]
        row={'case':case['case'],'repeat':repeat,'output_ids':tokens,'interval_ms':intervals,
             'mean_ms':sum(intervals)/len(intervals),'exact_tokens':True}
        runs.append(row);(out/'runs.json').write_text(json.dumps(runs,indent=2)+'\n')
        print(json.dumps(row),flush=True)
    intervals=[x for r in runs for x in r['interval_ms']]
    summary={'parallelism':{'tp':8,'ep':8},'batch':1,'context':512,'max_batched_tokens':512,
             'eager':a.eager,'devices':os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
             'profiled':a.profile,'latency_comparable':not a.profile,
             'hccl_expansion_mode_env':os.environ.get('HCCL_OP_EXPANSION_MODE'),
             'hccl_deterministic_env':os.environ.get('HCCL_DETERMINISTIC'),
             'mean_ms':sum(intervals)/len(intervals),'intervals':len(intervals),
             'method':'host perf_counter_ns at each offline engine.step output; exclude first token',
             'prefix_caching':False,'logprobs':5,'max_output_tokens':32}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary,flush=True)
if __name__=='__main__':main()
