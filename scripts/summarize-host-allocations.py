#!/usr/bin/env python3
"""Group surviving sampled allocations by module-relative stack, without scaling."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def parse(text):
    result=[];current=None
    for line in text.splitlines():
        words=line.split()
        if not words:continue
        if words[0]=='snapshot':
            if current is not None:raise ValueError('unfinished allocation snapshot')
            if len(words)!=4 or not words[2].startswith('rate=') or words[3]!='dropped=0':
                raise ValueError('invalid snapshot or dropped allocation samples')
            current=dict(time_ns=int(words[1]),rate=int(words[2].split('=')[1]),entries={})
        elif words[0]=='allocation':
            if current is None or len(words)<4:raise ValueError('allocation outside snapshot')
            identity=int(words[1]);size=int(words[2])
            if identity in current['entries'] or size<0:raise ValueError('invalid allocation identity/size')
            current['entries'][identity]=(size,tuple(words[3:]))
        elif words==['end']:
            if current is None:raise ValueError('end without snapshot')
            result.append(current);current=None
        else:raise ValueError('unknown allocation record')
    if current is not None:raise ValueError('truncated allocation snapshot')
    if len(result)<2:raise ValueError('need two complete allocation snapshots')
    return result


def compare(first,last):
    if first['rate']!=last['rate'] or last['time_ns']<=first['time_ns']:
        raise ValueError('incompatible sample windows')
    totals=defaultdict(lambda:[0,0,0,0])
    for offset,snapshot in ((0,first),(2,last)):
        for size,stack in snapshot['entries'].values():
            totals[stack][offset]+=size;totals[stack][offset+1]+=1
    rows=[dict(stack=list(stack),initial_bytes=v[0],initial_count=v[1],final_bytes=v[2],final_count=v[3],
               byte_growth=v[2]-v[0],count_growth=v[3]-v[1]) for stack,v in totals.items()]
    rows.sort(key=lambda row:row['byte_growth'],reverse=True)
    return dict(rate=first['rate'],first_time_ns=first['time_ns'],last_time_ns=last['time_ns'],
                scope='unscaled sampled live allocations; not exact total heap size or proof of leaks',
                sampled_byte_growth=sum(r['byte_growth'] for r in rows),stacks=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path);p.add_argument('--progress',type=Path)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    samples=parse(args.input.read_text())
    first,last=samples[0],samples[-1]
    if args.progress:
        waves=json.loads(args.progress.read_text())['rounds']
        if not all('allocation_snapshot_time_ns' in w for w in (waves[0],waves[-1])):
            raise ValueError('wave comparison requires acknowledged between-wave snapshots')
        indexed={s['time_ns']:s for s in samples}
        first,last=[indexed[w['allocation_snapshot_time_ns']] for w in (waves[0],waves[-1])]
    report=compare(first,last)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({**report,'stacks':report['stacks'][:5]},indent=2))


if __name__=='__main__':main()
