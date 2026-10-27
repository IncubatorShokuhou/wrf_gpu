"""Attribute uncovered idle slices to existing same-thread NVTX/OSRT evidence.

Reads only activity records, callchain symbols referenced by selected activity,
and StringIds referenced by those records. No TARGET_INFO/environment queries.
Labels are observations; an enclosing thunk is not proof of a CPU stall cause.
"""
from collections import Counter,defaultdict
import bisect
import heapq
import json
from pathlib import Path
import sqlite3

from trace_analysis import merge
L=Path('<USER_HOME>/wrf_gpu2_lanes/w44-perf')
arm='G43'
c=sqlite3.connect('file:'+str(L/'gpu/runs'/arm/'profile.sqlite')+'?mode=ro',uri=True)
api=list(c.execute('SELECT r.start,r.end,r.globalTid,s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId'))
launches=Counter(t for a,b,t,n in api if 'LaunchKernel' in n or 'GraphLaunch' in n)
tids={t for t,n in launches.items() if n>5}
osrt=[(a,b,t,n,chain) for a,b,t,n,chain in c.execute('SELECT r.start,r.end,r.globalTid,s.value,r.callchainId FROM OSRT_API r JOIN StringIds s ON s.id=r.nameId') if t in tids]
nvtx=list(c.execute('SELECT n.start,n.end,n.globalTid,COALESCE(n.text,s.value) FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId WHERE n.end IS NOT NULL'))
nvtx=[(a,b,t,n) for a,b,t,n in nvtx if t in tids and n and n.startswith('Thunk:')]
trace=json.loads((L/'TRACE_FINAL/G43.json').read_text())
offset=trace['clock_offset_receipt_s']
receipt=json.loads((L/'gpu/runs/G43/run/receipt.json').read_text())
output=[((o['t_start']-offset)*1e9,(o['t_start']+o['s']-offset)*1e9) for o in receipt['outputs']]
covered=merge([(a,b) for a,b,_,_ in api]+output)
outside=[]
for gap in trace['gaps_over_1ms']:
    a=gap['start_ms']*1e6;b=a+gap['duration_ms']*1e6;at=a
    for x,y in covered:
        if y<=at:continue
        if x>=b:break
        if x>at:outside.append((at,min(x,b)))
        at=max(at,min(y,b))
        if at>=b:break
    if at<b:outside.append((at,b))
assert abs(sum(b-a for a,b in outside)/1e6-trace['gap_partition_ms']['outside CUDA API (CPU cause unresolved)'])<1e-5
scopes=Counter();waits=Counter();chains=Counter();observations=[]
nv_by_tid={tid:sorted((a,b,n) for a,b,t,n in nvtx if t==tid) for tid in tids}
nv_index={tid:0 for tid in tids};nv_active={tid:[] for tid in tids}
osrt.sort();os_index=0;os_expire=[];os_active={}
for a,b in outside:
    # Smallest fully-enclosing thunk on each known launch thread.
    found=[]
    for tid in tids:
        rows=nv_by_tid[tid];i=nv_index[tid];active=nv_active[tid]
        while i<len(rows) and rows[i][0]<=a:
            lo,hi,name=rows[i];heapq.heappush(active,(hi-lo,hi,name));i+=1
        nv_index[tid]=i
        while active and active[0][1]<b:heapq.heappop(active)
        if active:
            dur,end,name=active[0];found.append((name,tid,dur))
    label=found[0][0] if len(found)==1 else 'multiple launch-thread thunk scopes' if found else 'no enclosing launch-thread thunk'
    scopes[label]+=(b-a)/1e6
    touched=[]
    while os_index<len(osrt) and osrt[os_index][0]<b:
        row=osrt[os_index];os_active[os_index]=row;heapq.heappush(os_expire,(row[1],os_index));os_index+=1
    while os_expire and os_expire[0][0]<=a:
        _,i=heapq.heappop(os_expire);os_active.pop(i,None)
    for x,y,t,name,chain in os_active.values():
        overlap=max(0,min(b,y)-max(a,x))
        if overlap:
            waits[name]+=overlap/1e6;chains[chain]+=overlap/1e6;touched.append({'name':name,'tid':t,'overlap_ms':overlap/1e6,'chain':chain})
    observations.append({'start_ms':a/1e6,'s':(b-a)/1e6,'enclosing_scopes':[{'name':n,'tid':t,'span_ms':d/1e6} for n,t,d in found],'osrt':touched})
chain_rows={}
for chain,_ in chains.most_common(8):
    rows=list(c.execute('SELECT r.stackDepth,s.value,m.value FROM OSRT_CALLCHAINS r LEFT JOIN StringIds s ON r.symbol=s.id LEFT JOIN StringIds m ON r.module=m.id WHERE r.id=? ORDER BY r.stackDepth',(chain,)))
    chain_rows[str(chain)]=[{'depth':d,'symbol':s,'module':m} for d,s,m in rows]
result={'scope':__doc__,'arm':arm,'launch_thread_api_counts':dict(launches),'uncovered_ms':sum(b-a for a,b in outside)/1e6,'by_enclosing_thunk_ms':dict(scopes.most_common()),'osrt_overlap_ms_not_disjoint':dict(waits.most_common()),'top_osrt_callchains':chain_rows,'observations':observations}
(L/'CPU_GAP_SCOPES.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('observations','top_osrt_callchains')},indent=2))
print('callchains',chain_rows)

