# Offline reference only. Not part of InferFabric loading, inference or serving.
import json
import argparse
from pathlib import Path
from vllm import LLM, SamplingParams
from tokenizers import Tokenizer
# Image compatibility shim: initialize the reference's declared local/layer
# communication groups after the ordinary distributed startup.
import omni_npu.worker.npu_worker as worker
_original_distributed_init = worker.init_worker_distributed_environment
def _reference_distributed_init(*args, **kwargs):
    _original_distributed_init(*args, **kwargs)
    from omni_npu.v1.distributed.parallel_state_ext import ensure_layer_parallel_initialized
    ensure_layer_parallel_initialized(backend='hccl')
worker.init_worker_distributed_environment = _reference_distributed_init
if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='Offline reference only; never part of native serving')
    parser.add_argument('--cases',type=Path,help='Token-ID cases; expected output length sets generation length, not expected contents')
    parser.add_argument('--output',type=Path,default=Path('reference-results.json'))
    parser.add_argument('--max-model-len',type=int,default=512)
    parser.add_argument('--kv-cache-dtype',default='auto')
    parser.add_argument('--max-num-batched-tokens',type=int,help='Explicit chunked-prefill token budget for numerical comparison')
    parser.add_argument('--respect-eos',action='store_true',help='Stop at first EOS instead of spending compute on post-EOS tokens')
    args=parser.parse_args()
    tokenizer=Tokenizer.from_file('/models/92B/tokenizer.json')
    prompts=['1+1=?','What is the capital of France?','请用一句话介绍北京。']
    rendered=['<|message_start|>user\n'+p+'<|message_end|><|message_start|>assistant\n</think>' for p in prompts]
    cases=json.loads(args.cases.read_text()) if args.cases else [dict(prompt=p,input_ids=tokenizer.encode(t,add_special_tokens=False).ids) for p,t in zip(prompts,rendered)]
    for case in cases:
        count=len(case['output_ids']) if args.cases else 32
        if count<1 or not case['input_ids'] or len(case['input_ids'])+count>args.max_model_len:
            raise ValueError('reference request exceeds total context or has empty input/output')
    if args.max_num_batched_tokens is not None and args.max_num_batched_tokens<1:
        raise ValueError('reference token budget must be positive')
    model=LLM(model='/models/92B',trust_remote_code=True,tensor_parallel_size=8,enable_expert_parallel=True,max_model_len=args.max_model_len,max_num_seqs=1,max_num_batched_tokens=args.max_num_batched_tokens or min(args.max_model_len,512),enable_chunked_prefill=True,kv_cache_dtype=args.kv_cache_dtype,gpu_memory_utilization=0.5,enforce_eager=True,enable_prefix_caching=False,disable_custom_all_reduce=True)
    results=[]
    for case in cases:
        ids=case['input_ids']
        out=model.generate([{'prompt_token_ids':ids}],SamplingParams(temperature=0,max_tokens=len(case['output_ids']) if args.cases else 32,ignore_eos=bool(args.cases) and not args.respect_eos,logprobs=5))[0].outputs[0]
        results.append({'case':case.get('case',case.get('prompt')),'input_ids':ids,'output_ids':list(out.token_ids),'text':out.text,'logprobs':[{str(k):{'logprob':v.logprob,'rank':v.rank} for k,v in row.items()} for row in out.logprobs]})
        if 'qualification' in case:
            results[-1]['qualification'] = dict(case['qualification'], output_is_placeholder=False)
        if not args.cases:
            results[-1]['prompt']=case['prompt']
        print(json.dumps(results[-1],ensure_ascii=False),flush=True)
        args.output.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
