"""Rebuild README diagrams and evidence plot. Requires matplotlib (pip install matplotlib)."""
from pathlib import Path
from html import escape
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/assets'
BG='#0c1625'; CARD='#152638'; FG='#eef4fc'; MUTED='#a7bbcf'; GREEN='#70ead2'; BLUE='#96b5ff'; LINE='#50667e'
def text(x,y,s,size=16,color=FG,weight=400):
 return f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" font-weight="{weight}">{escape(s)}</text>'
def box(x,y,w,h,title,lines,accent=GREEN):
 out=f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{CARD}"/>'
 out+=f'<rect x="{x}" y="{y+18}" width="3" height="24" rx="1.5" fill="{accent}"/>'
 out+=text(x+22,y+38,title,21,FG,600)
 for i,line in enumerate(lines):out+=text(x+22,y+72+i*26,line,16,MUTED)
 return out
def arrow(path,dashed=False):
 return f'<path d="{path}" fill="none" stroke="{LINE}" stroke-width="2" marker-end="url(#arrow)"'+(' stroke-dasharray="6 5"' if dashed else '')+'/>'
def save(name,title,desc,body,height):
 svg=f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 {height}" role="img" aria-labelledby="title desc">
<title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="{LINE}"/></marker></defs>
<rect width="1200" height="{height}" rx="16" fill="{BG}"/>
<g font-family="Segoe UI,Arial,sans-serif">{body}</g></svg>'''
 (OUT/name).write_text(svg,encoding='utf-8')

body=text(40,42,'THE GRAND IDEA / TARGET ARCHITECTURE',12,GREEN,600)
body+=text(40,82,'Turn a model into an inspectable execution program.',30,FG,600)
body+=text(40,121,'Compilation resolves decisions; serving executes a qualified plan.',18,MUTED)
body+=text(40,170,'PLAN & COMPILE',12,BLUE,600)
body+=box(40,190,340,150,'Describe the model',['Tensor math + explicit state','Checkpoint bindings + constants','Shapes + parallelism constraints'])
body+=box(430,190,340,150,'Optimize the whole graph',['Fusion + backend candidates','Layouts + communication','Schedule + memory lifetimes'])
body+=box(820,190,340,150,'Emit a native bundle',['Kernels + verified physical DAG','Resource contracts + identities','IR dumps + visual explanation'])
body+=arrow('M380 265H427')+arrow('M770 265H817')
body+=text(40,400,'LOAD & SERVE',12,GREEN,600)
body+=box(40,420,340,145,'vLLM Rust frontend',['Chat / completions + streaming','Tokenization + cancellation'])
body+=box(430,420,340,145,'Schedule requests',['Admission + token budgets','Bind tokens, positions and state'])
body+=box(820,420,340,145,'Execute on Ascend',['Load / prepare / capture','Native kernels + graph replay'])
body+=arrow('M380 490H427')+arrow('M770 490H817')+arrow('M990 340V417',True)
body+=text(40,610,'Current: native Qwen serving and graph replay. Planned: general v1 compiler, native bundles and generated backends.',15,MUTED)
save('inferfabric-vision.svg','InferFabric model-to-execution vision','Target architecture. A model is optimized into a native bundle; the Rust frontend and scheduler drive qualified Ascend execution. General compilation and native bundles remain planned.',body,644)

body=text(40,42,'TWO LEVELS OF OPTIMIZATION / PROPOSED DESIGN',12,GREEN,600)
body+=text(40,82,'Plan globally. Optimize each computation region locally.',29,FG,600)
body+=box(310,115,580,130,'InferFabric: semantic region + constraints',['Tensor operations, shapes, precision and effects','Fusion candidates and external layout / resource contracts'])
for x in [170,455,740,1025]:body+=arrow(f'M600 245V267H{x}V292')
for x,title,lines in [
 (40,'TileLang-Ascend',['Parameterized tile program','Passes + legal tuning space']),
 (325,'PTO',['Region-to-tile lowering','Instructions + synchronization']),
 (610,'Triton-Ascend',['Region-to-kernel lowering','Backend compilation']),
 (895,'Ascend C / ACLNN',['Generated / custom kernels','Vendor implementation calls'])]:
 body+=box(x,295,265,140,title,lines,BLUE)
for x in [170,455,740,1025]:body+=arrow(f'M{x} 435V463H600V487')
body+=box(310,490,580,130,'InferFabric: select, verify, finalize',['Compare kernels + conversions + communication cost','Finalize global memory, schedule and native bundle'])
body+=text(40,657,'Backends return compiled implementations, layouts, workspace bounds and qualification / cost evidence.',16,MUTED)
body+=text(40,687,'TileLang receives computation structure before tiling is fixed. PTO is a sibling backend, not a required intermediate.',15,MUTED)
save('inferfabric-backends.svg','Whole-model planning and backend optimization','Proposed design: InferFabric delegates computation regions to TileLang, PTO, Triton or Ascend C/ACLNN, then selects implementations and finalizes memory and scheduling. Generated backend routes are planned.',body,718)

# Pull raw byte counts from the recorded comparison, not rounded README numbers.
source=(ROOT/'docs/npu-activation-memory.md').read_text(encoding='utf-8')
labels=['TP2 rank 0 observed startup HBM growth','TP2 per-rank workspace bytes','TP2 per-rank temporary bytes']
values=[]
for label in labels:
 m=re.search(r'\| '+re.escape(label)+r' \| ([\d,]+) \| ([\d,]+) \|',source)
 assert m,label
 values.append(tuple(int(v.replace(',','')) for v in m.groups()))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':FG,'axes.labelcolor':MUTED,'xtick.color':MUTED,'ytick.color':MUTED,'svg.fonttype':'none','svg.hashsalt':'inferfabric-readme-v1'})
fig,axes=plt.subplots(1,3,figsize=(12,4.8),facecolor=BG)
fig.subplots_adjust(left=.075,right=.98,top=.64,bottom=.27,wspace=.42)
fig.text(.035,.92,'MEASURED EVIDENCE / DEDICATED VS POOLED ALLOCATION',fontsize=10,color=GREEN,weight='bold')
fig.text(.035,.82,'Explicit memory planning reduces device memory use.',fontsize=21,weight='bold')
for ax,(a,b),title in zip(axes,values,['Startup HBM growth\nRank 0','Operator workspace\nPer rank','Temporary allocations\nPer rank']):
 ax.set_facecolor(BG);scale=1e9 if a>1e9 else 1e6;unit='GB' if scale==1e9 else 'MB'
 nums=[a/scale,b/scale];ax.bar([0,1],nums,color=[LINE,GREEN],width=.56,zorder=3)
 ax.set_xticks([0,1],['Dedicated','Pooled']);ax.tick_params(axis='both',length=0)
 ax.set_ylim(0,max(nums)*1.27);ax.set_ylabel(unit,labelpad=5)
 ax.set_title(title,loc='left',fontsize=12,color=FG,pad=14)
 ax.yaxis.grid(True,color='#263548',linewidth=.7);ax.set_axisbelow(True)
 for spine in ax.spines.values():spine.set_visible(False)
 for i,v in enumerate(nums):ax.text(i,v+max(nums)*.04,f'{v:.3f}' if unit=='GB' else f'{v:.1f}',ha='center',fontsize=12,color=FG)
 ax.text(.5,-.35,f'{(1-b/a)*100:.1f}% lower',transform=ax.transAxes,ha='center',color=GREEN,weight='bold',fontsize=13)
fig.text(.035,.09,'Qwen3.5-2B · Ascend A3 / CANN 9.1.0 · TP2 · context 2048 · 4 sequence slots',fontsize=10,color=MUTED)
fig.text(.035,.035,'Recorded configuration comparison, not a framework speedup claim. GB / MB are decimal; panels are not additive.',fontsize=9,color=MUTED)
fig.savefig(OUT/'inferfabric-memory.svg',facecolor=BG,metadata={'Date':None,'Title':'InferFabric measured allocation comparison','Description':'Recorded dedicated and pooled byte counts from docs/npu-activation-memory.md.'})
fig.savefig(ROOT/'.deploy/inferfabric-memory-preview.png',facecolor=BG,dpi=150)
print('Generated two architecture diagrams and one evidence plot.')
