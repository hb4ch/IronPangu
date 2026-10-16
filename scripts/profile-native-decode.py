#!/usr/bin/env python3
"""Profile two warmed full decode iterations and verify complete native output."""
import argparse,csv,hashlib,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--binary',required=True);p.add_argument('--plan',required=True)
p.add_argument('--checkpoint',required=True);p.add_argument('--cases',required=True)
p.add_argument('--case',default='chinese');p.add_argument('--output',required=True)
p.add_argument('--vendor',action='store_true');a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
sources=[Path(a.binary),Path(a.plan)/'execution.bin',Path(a.cases)]
provenance={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
(out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
case=next(c for c in json.loads(Path(a.cases).read_text()) if c['case']==a.case)
if len(case['output_ids'])<3:raise ValueError('capture needs at least three outputs')
env=dict(os.environ,INFERFABRIC_PROFILE_DIR=str(out.resolve()),INFERFABRIC_TRANSFER_AUDIT_DIR=str((out/'transfers').resolve()))
env.pop('INFERFABRIC_PROFILE_VENDOR',None)
if a.vendor:env['INFERFABRIC_PROFILE_VENDOR']='1'
with (out/'stderr.txt').open('w') as err:
 r=subprocess.run([a.binary,a.plan,a.checkpoint,'32',*map(str,case['input_ids'])],env=env,text=True,stdout=subprocess.PIPE,stderr=err,timeout=900)
(out/'stdout.txt').write_text(r.stdout)
if r.returncode:raise RuntimeError(f'native exit {r.returncode}')
rows=[json.loads(x) for x in r.stdout.splitlines() if x.startswith('{"step"')]
if rows!=case['steps']:raise RuntimeError('profiled output differs from qualified native records')
for rank in range(8):
 with (out/f'rank-{rank}.csv').open() as file:
  timers=list(csv.DictReader(file))
 if not any(row['name']=='decode_total' and int(row['calls'])==2 for row in timers):
  raise RuntimeError('incomplete decode capture')
 audit=json.loads((out/'transfers'/f'rank-{rank}.json').read_text())
 if audit['activation_host_transfers'] or audit['rejected']:raise RuntimeError('residency violation')
(out/'correctness.json').write_text(json.dumps({'case':a.case,'tokens':len(rows),'complete_records_exact':True,'activation_host_transfers':0,'captured_decode_iterations':2,'vendor':a.vendor},indent=2)+'\n')
print(json.dumps({'output':str(out),'correctness':'passed','tokens':len(rows),'vendor':a.vendor}),flush=True)
