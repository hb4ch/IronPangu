import json,re
from pathlib import Path
root=Path('docs/validation/2026-10-16-integer-cache')
checks=[]
for dtype in ['int8','int4']:
 h=(root/dtype/'execution.html').read_text()
 match=re.search(r'<script[^>]*id="plan-data"[^>]*>(.*?)</script>',h,re.S)
 if not match: raise ValueError('missing plan data')
 d=json.loads(match.group(1));p=d['program'];ins=p['instructions'];recipes=d['recipes'];a=json.loads((root/dtype/'arena.json').read_text())
 assert len(ins)==2753
 count={op:sum(n['op']==op for n in ins) for op in ['cache_normalize','cache_quantize_append','cache_dequantize']}
 assert set(count.values())=={46}
 scratch=[]
 for rank in a['ranks']:
  ids=set()
  for i,n in enumerate(ins):
   if n['op']=='cache_quantize_append':
    assert n['attrs']['cache_dtype']==dtype
    assert recipes[i]['value_fields'][0]['dtype']==dtype
    assert recipes[i]['apis']==['if_cache_quantize']
   if n['op']=='cache_dequantize':
    assert ins[n['inputs'][0]]['op']=='cache_quantize_append'
    assert recipes[i]['apis']==['if_cache_dequantize']
    for sid in rank['instruction_slots'][str(i)]:
     if rank['slots'][sid]['bytes']==512*576*2:ids.add(sid)
  assert len(ids)==1,(dtype,ids)
  scratch.append(sorted(ids))
 script=re.findall(r'<script(?:\s[^>]*)?>(.*?)</script>',h,re.S)[-1]
 (Path('.deploy')/f'{dtype}-viewer.js').write_text(script)
 checks.append({'dtype':dtype,'physical_instructions':len(ins),'explicit_cache_instructions':count,'one_shared_full_cache_scratch_per_rank':True,'scratch_slots':scratch,'precision_filter_present':'Quantization / cache precision' in h})
(root/'plan-checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks))
