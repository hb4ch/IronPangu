#!/usr/bin/env python3
"""Aggregate native host timers and CANN 9.1 parsed SQLite profiling data.

Device task timestamps/durations are normalized ns in these tables. Host ApiData
uses architectural counter ticks: convert with info.json sysClockFreq (MHz).
API/node tables are separate inclusive views, not additive budget categories.
"""
import argparse,csv,hashlib,json,sqlite3,statistics
from collections import defaultdict
from pathlib import Path


def union_ns(intervals):
    end=None;total=0
    for start,stop in sorted(intervals):
        if end is None or start>end:total+=stop-start;end=stop
        elif stop>end:total+=stop-end;end=stop
    return total


def host_summary(path,iterations):
    ranks=[]
    for p in sorted(path.glob('rank-*.csv')):
        data=list(csv.DictReader(p.open()))
        ranks.append({'rank':int(p.stem.split('-')[1]),'entries':[
            {'kind':r['kind'],'name':r['name'],'calls_per_token':int(r['calls'])/iterations,
             'ms_per_token':int(r['host_duration_ns'])/1e6/iterations} for r in data]})
    if len(ranks)!=8:raise ValueError('expected eight complete rank timers')
    grouped=defaultdict(list)
    for rank in ranks:
        for r in rank['entries']:grouped[r['kind'],r['name']].append(r)
    mean=[{'kind':kind,'name':name,'calls_per_token':statistics.mean(x['calls_per_token'] for x in rows),
           'mean_rank_ms_per_token':statistics.mean(x['ms_per_token'] for x in rows),
           'min_rank_ms_per_token':min(x['ms_per_token'] for x in rows),
           'max_rank_ms_per_token':max(x['ms_per_token'] for x in rows)} for (kind,name),rows in grouped.items()]
    return {'path':str(path),'iterations':iterations,'ranks':ranks,'rank_mean':sorted(mean,key=lambda x:-x['mean_rank_ms_per_token'])}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vendor',required=True);p.add_argument('--host',action='append',required=True)
    p.add_argument('--output',required=True);p.add_argument('--iterations',type=int,default=2)
    a=p.parse_args();root=next(Path(a.vendor).glob('PROF_*'));iters=a.iterations
    result={'format':'inferfabric.decode-profile.v1','profile_root':str(root),'iterations':iters,
            'host_runs':[host_summary(Path(path),iters) for path in a.host],
            'vendor_host_timers':host_summary(Path(a.vendor).parent,iters),'devices':[]}
    artifacts=[]
    def connect(path):
        artifacts.append(path)
        return sqlite3.connect(f'file:{path}?mode=ro',uri=True)
    for device in sorted(root.glob('device_*')):
        c=connect(device/'sqlite/ascend_task.db')
        tasks=c.execute("select start_time,duration,device_task_type from AscendTask where device_task_type!='PLACE_HOLDER_SQE'").fetchall()
        start=min(t[0] for t in tasks);end=max(t[0]+t[1] for t in tasks)
        compute=[(s,s+d) for s,d,k in tasks if k in ['AI_CORE','MIX_AIC','MIX_AIV']]
        union=union_ns((s,s+d) for s,d,k in tasks)
        group=c.execute('select device_task_type,count(*),sum(duration) from AscendTask group by device_task_type order by sum(duration) desc').fetchall()
        cc=connect(device/'sqlite/ai_core_op_summary.db')
        kernels=cc.execute("select g.op_type,count(*),sum(t.duration_time) from ge_summary g join task_time t on g.task_id=t.task_id and g.stream_id=t.stream_id and g.batch_id=t.batch_id where t.task_type='AI_CORE' group by g.op_type order by sum(t.duration_time) desc").fetchall()
        hc=connect(device/'sqlite/hccl_single_device.db')
        hccl=hc.execute('select op_type,occurrences,total_time from HcclOpReport').fetchall()
        links=hc.execute('select hccl_name,count(*),sum(duration),sum(size) from HCCLTaskSingleDevice group by hccl_name').fetchall()
        result['devices'].append({'device':device.name,'window_ms':(end-start)/1e6,
            'compute_union_ms':union_ns(compute)/1e6,'all_task_union_ms':union/1e6,
            'no_recorded_task_ms':(end-start-union)/1e6,
            'task_types':[{'type':k,'count_per_token':n/iters,'summed_ms_per_token':ns/1e6/iters} for k,n,ns in group],
            'kernels':[{'type':k,'count_per_token':n/iters,'summed_ms_per_token':ns/1e6/iters} for k,n,ns in kernels],
            'collectives':[{'type':k,'count_per_token':int(n)/iters,'inclusive_ms_per_token':ns/1e6/iters} for k,n,ns in hccl],
            'communication_tasks':[{'type':k,'count_per_token':n/iters,'summed_ms_per_token':ns/1e6/iters,'bytes_per_token':size/iters} for k,n,ns,size in links]})
    info=json.loads((root/'host/info.json').read_text());freq=float(info['sysClockFreq'])
    if freq<=0:raise ValueError('invalid host counter frequency')
    result['host_counter_mhz']=freq
    hc=connect(root/'host/sqlite/api_event.db')
    apis=hc.execute("select id,count(*),sum(end-start) from ApiData where id!='0' group by id order by sum(end-start) desc limit 40").fetchall()
    result['vendor_host_api_rank_mean']=[{'name':name,'calls_per_rank_token':n/8/iters,'inclusive_ms_per_rank_token':ticks/freq/1000/8/iters} for name,n,ticks in apis]
    result['source_hashes']={str(path):hashlib.file_digest(path.open('rb'),'sha256').hexdigest() for path in artifacts}
    result['caveats']=['Host and device profiles are different runs; do not subtract their durations.',
                       'Node and API durations are inclusive alternative views, not additive.',
                       'Task sums overlap across streams; only union durations remove overlap.',
                       'No-recorded-task intervals are trace gaps, not a sampled hardware utilization counter.',
                       'Profiler overhead is measured by comparing host decode_total; use lightweight timings for prioritization.']
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'output':a.output,'devices':len(result['devices']),'device0':result['devices'][0]},indent=2))


if __name__=='__main__':main()
