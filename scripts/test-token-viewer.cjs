// Exercise the real viewer script with recorded plan metadata and a minimal DOM.
const fs=require('node:fs'), vm=require('node:vm'), assert=require('node:assert/strict');
const file=process.argv[2];
const html=fs.readFileSync(file,'utf8');
const payload=html.match(/<script[^>]*id="plan-data"[^>]*>([\s\S]*?)<\/script>/)[1];
const scripts=[...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)];
const elements=new Map();
function element(id){if(!elements.has(id))elements.set(id,{value:id==='#filter'?'all':'',innerHTML:'',textContent:id==='#plan-data'?payload:'',querySelectorAll:()=>[],scrollIntoView(){}});return elements.get(id);}
const context=vm.createContext({document:{querySelector:element},window:{}});
vm.runInContext(scripts.at(-1)[1],context);
const api=context.window.planInspector, root=api.scope;
const layer=[...api.groups.values()].find(g=>g.parent===root&&g.logical.length>30);
assert(layer,'recorded plan needs a composite layer');api.go(layer);
let rendered=element('#content').innerHTML;
assert(rendered.includes('Complete selected kernel inventory'));
assert(rendered.includes('Communication primitives'));
assert(rendered.includes('Input → output value shapes'));
const logical=new Set(layer.logical);
const ids=api.data.program.instructions.map((_,i)=>i).filter(i=>logical.has(api.data.program.instructions[i].logical_node));
const apis=new Set(ids.flatMap(i=>api.data.recipes[i].apis));
for(const name of apis)assert(rendered.includes(name),`missing selected API: ${name}`);
for(const id of ids.filter(i=>api.data.recipes[i].apis.some(a=>/^Hccl/.test(a)))){
 assert(rendered.includes(`#${id}`),'missing collective identity');
 assert(rendered.includes(api.data.recipes[id].apis[0]));
}
assert(rendered.includes('Inputs'));assert(rendered.includes('Output #'));assert(!rendered.includes('other APIs'));
element('#filter').value='comm';vm.runInContext('render()',context);
const filtered=vm.runInContext('idsFor(scope).filter(matches)',context);
assert(filtered.length>0);
assert(filtered.every(i=>api.data.recipes[i].apis.some(a=>/^Hccl/.test(a))),'local expert work misclassified as communication');
element('#rank').onchange({target:{value:'7'}});
assert(element('#content').innerHTML.includes('rank 7'));
console.log(JSON.stringify({file,layer:layer.label,kernel_apis:apis.size,collectives:filtered.length,pass:true}));
const expertScope=[...api.groups.values()].find(g=>g.parent===root&&api.data.program.instructions.some((i,id)=>g.logical.includes(i.logical_node)&&api.data.recipes[id].apis.some(a=>/GroupedMatmul/.test(a))));
assert(expertScope);element('#filter').value='all';api.go(expertScope);
rendered=element('#content').innerHTML;
assert(rendered.includes('HcclAllReduce'));assert(rendered.includes('aclnnGroupedMatmulV5'));
assert(rendered.includes('owns experts'));
assert(rendered.includes('Weight bindings / attributes'));
console.log(JSON.stringify({expert_layer:expertScope.label,full_expert_kernel_and_collective_visible:true,pass:true}));

// Packed structured fields must expose storage offsets rather than just logical widths.
if(api.data.recipes.some(r=>r.value_fields?.some(f=>f.row_stride_bytes))) {
  api.go(layer);
  const packedHtml=element('#content').innerHTML;
  assert(packedHtml.includes('byte offset 512'));
  assert(packedHtml.includes('byte offset 640'));
  assert(packedHtml.includes('row stride 656 bytes'));
  assert(packedHtml.includes('aclnnAiInfraKvQuantSparseFlashAttention'));
}

if(api.data.arena?.ranks[7]?.workspace) {
  const details=vm.runInContext('memoryDetails(25)',context);
  assert(details.includes('Shared workspace:'));
  assert(details.includes('permanent slot 0'));
  assert(details.includes('Growth allocations: 0'));
  assert(details.includes('instruction demand'));
  if(api.data.arena.ranks[7].instruction_envelopes) {
    assert(details.includes('Frozen shared pool:'));
    assert(details.includes('Maximum request:'));
    assert(details.includes('Observed slot IDs:'));
    assert(details.includes('Live leases cannot overlap'));
  }
}
