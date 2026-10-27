"""Sanitized Nsight activity/gap accounting; never reads TARGET_INFO tables.

Graph envelopes can include internal gaps. Their union is an upper bound on
device busy; node captures supply kernel/copy counts. Profiled wall is not S1.
"""
from collections import Counter,defaultdict
import argparse
import bisect
import hashlib
import json
from pathlib import Path
import sqlite3


def merge(rows):
    out=[]
    for lo,hi in sorted(rows):
        if not out or lo>out[-1][1]:out.append([lo,hi])
        else:out[-1][1]=max(out[-1][1],hi)
    return out


def overlap(lo,hi,rows):
    return sum(max(0,min(hi,b)-max(lo,a)) for a,b in rows if a<hi and b>lo)


def partition(lo,hi,categories,output):
    """Disjoint overlap accounting; uncovered time is explicitly unresolved."""
    rows={'output callback (±0.5ms)':output,**categories}
    clipped={key:[(max(lo,a),min(hi,b)) for a,b in spans if a<hi and b>lo]
             for key,spans in rows.items()}
    points=sorted({lo,hi}|{x for spans in clipped.values() for ab in spans for x in ab})
    totals=Counter()
    order=['output callback (±0.5ms)','sync','allocation','transfer','launch/update','other CUDA API']
    for a,b in zip(points,points[1:]):
        middle=(a+b)/2
        active=next((key for key in order if any(x<=middle<y for x,y in clipped.get(key,[]))),
                    'outside CUDA API (CPU cause unresolved)')
        totals[active]+=(b-a)/1e6
    assert abs(sum(totals.values())-(hi-lo)/1e6)<1e-6
    return dict(totals)


def analyze(root):
    db=root/'profile.sqlite'
    c=sqlite3.connect(f'file:{db}?mode=ro',uri=True)
    names={x[0] for x in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    r=json.loads((root/'run/receipt.json').read_text())
    # Only names referenced by activity records are retrieved, not all strings.
    runtime=[(a,b,cor,n) for a,b,cor,n in c.execute('SELECT r.start,r.end,r.correlationId,s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id')]
    bycor={cor:(a,b,n) for a,b,cor,n in runtime}
    start_api=next((a,b) for a,b,_,n in runtime if n=='cuProfilerStart')
    start_event=next(x['t'] for x in r['cache_events'] if x['event'].startswith('cudaProfilerStart'))
    offset=start_event-start_api[1]/1e9  # receipt time = ns/1e9 + offset; ±0.5 ms rounding
    scopes=[]
    for x in r.get('advance',[]):scopes.append((x['t'],x['t']+x['host_s'],x['d'],x.get('s0')))
    for x in r.get('force',[]):scopes.append((x['t'],x['t']+x['host_s'],'force',None))
    for x in r.get('outputs',[]):scopes.append((x['t_start'],x['t_start']+x['s'],'output:'+x['d'],None))
    scopes.sort();starts=[x[0] for x in scopes]
    def owner(ns):
        t=ns/1e9+offset;i=bisect.bisect_right(starts,t)-1
        if i>=0 and scopes[i][0]<=t<=scopes[i][1]:return scopes[i][2],scopes[i][3]
        return 'outside-instrumented-calls',None
    ops=[];kernels=[];kernel_totals=defaultdict(lambda:[0,0])
    for a,b,cor,name,grid,block,gid in c.execute('SELECT k.start,k.end,k.correlationId,s.value,k.gridX,k.blockX,k.graphId FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id'):
        host=bycor.get(cor);domain,step=owner(host[0]) if host else ('NO_CORRELATION',None)
        ops.append((a,b));kernels.append({'start':a,'end':b,'correlation':cor,'name':name,'grid_x':grid,'block_x':block,'graph_id':gid,'domain':domain,'step':step})
        kernel_totals[domain][0]+=1;kernel_totals[domain][1]+=b-a
    copies=defaultdict(lambda:[0,0,0]);copy_owners=defaultdict(lambda:[0,0,0])
    for a,b,cor,kind,size in c.execute('SELECT start,end,correlationId,copyKind,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY'):
        ops.append((a,b));copies[str(kind)][0]+=1;copies[str(kind)][1]+=size;copies[str(kind)][2]+=b-a
        host=bycor.get(cor);domain,step=owner(host[0]) if host else ('NO_CORRELATION',None)
        key=domain+':'+str(kind);copy_owners[key][0]+=1;copy_owners[key][1]+=size;copy_owners[key][2]+=b-a
    graph=[]
    if 'CUPTI_ACTIVITY_KIND_GRAPH_TRACE' in names:
        graph=list(c.execute('SELECT start,end FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE'));ops+=graph
    if 'CUPTI_ACTIVITY_KIND_MEMSET' in names:
        ops+=list(c.execute('SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMSET'))
    union=merge(ops);lo,hi=union[0][0],union[-1][1]
    # Disjoint API categories: overlap with a device-idle gap, not whole API wall.
    categories=defaultdict(list)
    for a,b,_,name in runtime:
        category=('allocation' if any(s in name for s in ('Alloc','Free','MemPool')) else
                  'sync' if any(s in name for s in ('Synchronize','WaitEvent')) else
                  'transfer' if 'Memcpy' in name else 'launch/update' if any(s in name for s in ('Launch','GraphExec','GraphInstantiate')) else 'other CUDA API')
        categories[category].append((a,b))
    categories={k:merge(v) for k,v in categories.items()}
    output=merge([((x['t_start']-offset)*1e9,(x['t_start']+x['s']-offset)*1e9) for x in r.get('outputs',[])])
    gaps=[]
    for left,right in zip(union,union[1:]):
        a,b=left[1],right[0]
        if b-a<=1e6:continue
        cover={k:overlap(a,b,v)/1e6 for k,v in categories.items()}
        out=overlap(a,b,output)/1e6
        label=max(cover,key=cover.get) if cover and max(cover.values())>0 else 'host outside CUDA API (Python/IO/unknown)'
        if out>(b-a)/2e6:label='output callback (clock aligned ±0.5ms)'
        gaps.append({'start_ms':a/1e6,'duration_ms':(b-a)/1e6,'dominant_observation':label,'api_overlap_ms':cover,'output_overlap_ms':out,
                     'partition_ms':partition(a,b,categories,output),
                     'confidence':'overlap only, not CPU causality; no Python sampling tables exported'})
    totals=Counter();parts=Counter()
    for x in gaps:
        totals[x['dominant_observation']]+=x['duration_ms'];parts.update(x['partition_ms'])
    result={'schema':'w44-perf.trace.v1','scope':__doc__,'arm':root.name,'raw_db':str(db),
            'clock_offset_receipt_s':offset,'clock_uncertainty_ms':0.5,'device_span_ms':(hi-lo)/1e6,
            'device_busy_union_or_graph_upper_ms':sum(b-a for a,b in union)/1e6,'kernel_count':len(kernels),
            'kernel_sum_ms':sum(x['end']-x['start'] for x in kernels)/1e6,'graph_envelopes':len(graph),
            'graph_sum_ms':sum(b-a for a,b in graph)/1e6,'kernel_domain_count_ns':dict(kernel_totals),
            'memcpy_kind_count_bytes_ns':dict(copies),'copy_domain_kind_count_bytes_ns':dict(copy_owners),
            'gaps_over_1ms':gaps,'gap_ms_by_dominant_observation':dict(totals),'gap_partition_ms':dict(parts),
            'runtime_api_count_ns':{n:[sum(1 for _,_,_,name in runtime if name==n),sum(b-a for a,b,_,name in runtime if name==n)] for n in {x[3] for x in runtime}}}
    return result,kernels


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    for arm in ('G32','G43','N32','N43','R43'):
        root=Path('<USER_HOME>/wrf_gpu2_lanes/w44-perf/gpu/runs')/arm
        result,kernels=analyze(root)
        (a.out/(arm+'.json')).write_text(json.dumps(result,indent=2)+'\n')
        # Sanitized activity identities only, never environment/process tables.
        (a.out/(arm+'_kernels.json')).write_text(json.dumps(kernels,separators=(',',':'))+'\n')
        print(arm,{k:result[k] for k in ('device_span_ms','device_busy_union_or_graph_upper_ms','kernel_count','graph_envelopes','kernel_domain_count_ns','gap_partition_ms')},flush=True)


if __name__=='__main__':main()
