# Offline reference only. Not part of InferFabric loading, inference or serving.
import json
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
    tokenizer=Tokenizer.from_file('/models/92B/tokenizer.json')
    prompts=['1+1=?','What is the capital of France?','请用一句话介绍北京。']
    rendered=['<|message_start|>user\n'+p+'<|message_end|><|message_start|>assistant\n</think>' for p in prompts]
    model=LLM(model='/models/92B',trust_remote_code=True,tensor_parallel_size=8,enable_expert_parallel=True,max_model_len=512,max_num_seqs=1,max_num_batched_tokens=512,gpu_memory_utilization=0.5,enforce_eager=True,enable_prefix_caching=False,disable_custom_all_reduce=True)
    results=[]
    for prompt,text in zip(prompts,rendered):
        ids=tokenizer.encode(text,add_special_tokens=False).ids
        out=model.generate([{'prompt_token_ids':ids}],SamplingParams(temperature=0,max_tokens=32,logprobs=5))[0].outputs[0]
        results.append({'prompt':prompt,'input_ids':ids,'output_ids':list(out.token_ids),'text':out.text,'logprobs':[{str(k):{'logprob':v.logprob,'rank':v.rank} for k,v in row.items()} for row in out.logprobs]})
        print(json.dumps(results[-1],ensure_ascii=False),flush=True)
    with open('reference-results.json','w') as f:json.dump(results,f,ensure_ascii=False,indent=2)
