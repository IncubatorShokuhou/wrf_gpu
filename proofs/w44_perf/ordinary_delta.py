"""E82/E195 ordinary-root census from actual executable bindings and NVTX.

No kernel-name time transfer across versions. Every source candidate comes from
that run's exact loaded runtime thunks + rich HLO. Generic shared symbols stay
shared; out-of-module launches stay unbound. NVTX and launch dimensions resolve
call sites where the existing trace permits it. Totals count each activity once.
"""
import argparse
from collections import Counter,defaultdict
import bisect
import hashlib
import heapq
import json
from pathlib import Path
import re
import sqlite3

from family_times import family,KV,TABLES

L=Path('<USER_HOME>/wrf_gpu2_lanes/w44-perf')
LINE=re.compile(r'^\s*(?:ROOT )?%([\w.\-]+) = (.*?) ([a-z][a-z0-9-]*)\(')
MOVES={'parameter','constant','bitcast','reshape','transpose','copy','tuple','get-tuple-element'}
CONTROL={'parameter','constant','add','subtract','multiply','compare','select','and','or','not','convert','maximum','minimum'}


def source_family(location):
    path=location.split(':',1)[0]
    # The coupling directory also contains physics adapters: its directory
    # name alone is not evidence of boundary/nesting work.
    if '/coupling/noahmp' in path:return 'Noah-MP'
    if path.endswith('/coupling/physics_couplers.py'):return 'physics preparation/coupling'
    if '/coupling/' in path and 'boundary' not in path:return 'physics preparation/coupling'
    return family(path)


def hlo_model(rec,thunks_dir=L/'ORDINARY_THUNKS'):
    p=Path(rec['hlo_file']);text=p.read_text()
    assert hashlib.sha256(text.encode()).hexdigest()==rec['hlo_sha256']
    tables={key:{} for key in TABLES};section=None;comp=None;entry=None
    comps={};instructions={}
    for line in text.splitlines():
        if line in tables:section=line;continue
        if section and line[:1].isdigit():
            k,v=line.split(' ',1);tables[section][int(k)]=v;continue
        section=None
        m=re.match(r'^(ENTRY )?%?([\w.\-]+) .*{\s*$',line)
        if m:
            comp=m[2];comps[comp]=[]
            if m[1]:entry=comp
        elif line=='}':comp=None
        elif comp and (m:=LINE.match(line)):
            item={'name':m[1],'shape':m[2],'opcode':m[3],'line':line,'computation':comp}
            comps[comp].append(item);instructions[m[1]]=item
    def sources(line):
        out=Counter()
        for sid in re.findall(r'stack_frame_id=(\d+)',line):
            fr={k:int(v) for k,v in KV.findall(tables['StackFrames'].get(int(sid),''))}
            loc={k:int(v) for k,v in KV.findall(tables['FileLocations'].get(fr.get('file_location_id'),''))}
            f=tables['FileNames'].get(loc.get('file_name_id'),'UNKNOWN').strip('"')
            fn=tables['FunctionNames'].get(loc.get('function_name_id'),'UNKNOWN').strip('"')
            out[f+':'+str(loc.get('line'))+':'+fn]+=1
        return out
    records={}
    for name,item in instructions.items():
        loc=sources(item['line']);ops=[item]
        if item['opcode']=='fusion':
            call=re.search(r'\bcalls=%?([\w.\-]+)',item['line'])
            assert call
            ops=comps[call[1]]
            for op in ops:
                if op['opcode'] not in ('parameter','constant'):loc.update(sources(op['line']))
        fams={source_family(s) for s in loc}
        physical=fams-{'unattributed','glue/other','layout','physics preparation/coupling'}
        if len(physical)==1:label=next(iter(physical))
        elif len(physical)>1:label='mixed/shared fusion'
        elif 'layout' in fams:label='layout-only source'
        elif 'physics preparation/coupling' in fams:label='physics preparation/coupling'
        elif 'glue/other' in fams:label='runtime/glue source'
        else:label='unattributed source'
        opc={op['opcode'] for op in ops}
        opclass='native Pallas' if item['opcode']=='custom-call' else 'fusion arithmetic'
        if opc<=MOVES:opclass='layout/view'
        elif opc<=MOVES|{'convert'}:opclass='layout/cast'
        elif opc<=CONTROL and all(re.match(r'^(?:[su]\d+|pred)\[\]',op['shape']) for op in ops):
            opclass='scalar control';label='scalar control'
        records[name]={'hlo_instruction':name,'family':label,'source_families':sorted(fams),
                       'sources':dict(loc),'operation_class':opclass,'opcodes':sorted(opc),
                       'shape':item['shape'],'computation':item['computation']}
    roots={item['name'] for item in comps[entry] if item['opcode']=='while'}
    assert roots
    runtime=json.loads((thunks_dir/f"{rec['version']}_{rec['program']}.json").read_text())
    assert runtime['actual_execution_binding']['runtime_thunks_field13_equal']
    sym=defaultdict(list)
    for t in runtime['thunks']:
        if t['kind']=='kernel':
            d=t['launch_dimensions_proto']
            try:
                gd=d['1'][0]['2'][0];bd=d['2'][0]['1'][0]
                dims=tuple(gd[str(k)][0] for k in (1,2,3))+tuple(bd[str(k)][0] for k in (1,2,3))
            except KeyError:dims=None
            sym[t['native_symbol']].append({**t,'dimensions':dims,'source':records[t['hlo_instruction']]})
    return {'roots':roots,'instructions':records,'symbols':sym,'native_symbols':set(runtime['native_symbols']),
            'binding':runtime['actual_execution_binding'],'copy_thunks':[t for t in runtime['thunks'] if t['kind']=='copy']}


def annotate(points,nvtx,models):
    # One monotone interval sweep per thread; no O(activities*ranges) scans.
    out={};bytid=defaultdict(list)
    for p in points:bytid[p[1]].append(p)
    root_names={r:domain for domain,m in models.items() for r in m['roots']}
    valid_names=set().union(*(set(m['instructions']) for m in models.values()))
    for tid,queries in bytid.items():
        spans=sorted((a,b,n) for a,b,t,n in nvtx if t==tid)
        i=0;allheap=[];rootheap=[];instheap=[]
        for at,_,key in sorted(queries):
            while i<len(spans) and spans[i][0]<=at:
                a,b,n=spans[i];i+=1
                row=(b-a,b,n)
                heapq.heappush(allheap,row)
                # HLO instruction names can collide across nested modules:
                # old d02 has an inner while.793, also the old d01 entry name.
                # The OUTERMOST proven entry range supplies the module scope.
                if n in root_names:heapq.heappush(rootheap,(-(b-a),b,n))
                if n in valid_names:heapq.heappush(instheap,row)
            for heap in (allheap,rootheap,instheap):
                while heap and heap[0][1]<at:heapq.heappop(heap)
            out[key]={'root_domain':root_names.get(rootheap[0][2]) if rootheap else None,
                      'hlo_instruction':instheap[0][2] if instheap else None,
                      'innermost_thunk':allheap[0][2] if allheap else None}
    return out


def census(arm,version,models):
    base=L/'gpu/runs'/arm;db=base/'profile.sqlite'
    c=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True)
    # Only activity-referenced strings, bounded lengths, no TARGET_INFO reads.
    api=list(c.execute('SELECT r.start,r.end,r.globalTid,r.correlationId,s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId WHERE length(s.value)<128'))
    bycor={r[3]:r for r in api}
    nv=[]
    for a,b,t,s in c.execute("SELECT n.start,n.end,n.globalTid,COALESCE(n.text,s.value) FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id WHERE n.end IS NOT NULL AND length(COALESCE(n.text,s.value))<256 AND COALESCE(n.text,s.value) LIKE 'Thunk:#hlo_op=%'"):
        m=re.fullmatch(r'Thunk:#hlo_op=([^#]+)#',s)
        if m:nv.append((a,b,t,m[1]))
    ge=list(c.execute('SELECT start,globalTid,graphNodeId,originalGraphNodeId FROM CUDA_GRAPH_NODE_EVENTS'))
    queries=[((a+b)/2,t,('api',cor)) for a,b,t,cor,n in api]
    queries +=[(a,t,('node',node)) for a,t,node,original in ge]
    ann=annotate(queries,nv,models)
    kernel=list(c.execute('SELECT k.start,k.end,k.correlationId,s.value,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,k.graphId,k.graphNodeId,k.registersPerThread FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName WHERE length(s.value)<256'))
    receipt=json.loads((base/'run/receipt.json').read_text())
    start_api=next(r for r in api if r[4]=='cuProfilerStart')
    start_event=next(r['t'] for r in receipt['cache_events'] if r['event'].startswith('cudaProfilerStart'))
    offset=start_event-start_api[1]/1e9
    scopes=sorted([(r['t'],r['t']+r['host_s'],r['d']) for r in receipt['advance']]+
                  [(r['t'],r['t']+r['host_s'],'nesting/force') for r in receipt['force']])
    starts=[r[0] for r in scopes]
    def owner(t):
        at=t/1e9+offset;i=bisect.bisect_right(starts,at)-1
        return scopes[i][2] if i>=0 and scopes[i][0]<=at<=scopes[i][1] else 'outside callback'
    graph_domains=defaultdict(set)
    for k in kernel:
        a=ann.get(('api',k[2]),{})
        if k[10] and a.get('root_domain'):graph_domains[k[10]].add(a['root_domain'])
    # Graph activities correlate to replay APIs, which can occur just AFTER
    # the while's recording range ends. Bind creation and clone lineage too;
    # never infer a program solely from a generic CUDA function name.
    adjacency=defaultdict(set)
    for at,tid,node,original in ge:
        gid=node>>32
        root=ann.get(('node',node),{}).get('root_domain')
        if root:graph_domains[gid].add(root)
        if original:
            og=original>>32;adjacency[gid].add(og);adjacency[og].add(gid)
    for k in kernel:
        if k[10] and k[11]:assert k[10]==k[11]>>32
    seen=set()
    for gid in set(adjacency)|set(graph_domains):
        if gid in seen:continue
        stack=[gid];component=set()
        while stack:
            x=stack.pop()
            if x in component:continue
            component.add(x);stack.extend(adjacency[x]-component)
        domains=set().union(*(graph_domains[x] for x in component))
        for x in component:graph_domains[x]|=domains
        seen|=component
    # Bind graph clones through their creation IDs when the source instruction is present.
    clones={node:original for _,_,node,original in ge if original}
    def node_annotation(node):
        seen=set()
        while node and node not in seen:
            seen.add(node);r=ann.get(('node',node),{})
            if r.get('root_domain') and r.get('hlo_instruction'):return r
            node=clones.get(node)
        return {}
    counts=defaultdict(lambda:[0,0]);details=defaultdict(lambda:[0,0]);method=Counter()
    witnesses={};helpers=defaultdict(lambda:[0,0]);unbound=Counter();copybins=Counter()
    for row in kernel:
        a,b,cor,sym,*_=row
        host=bycor[cor];scope=owner(host[0]);note=ann.get(('api',cor),{})
        domain=note.get('root_domain')
        if not domain and row[10] and len(graph_domains[row[10]])==1:domain=next(iter(graph_domains[row[10]]))
        candidates=[];binding='unbound';opclass='unknown'
        if scope=='nesting/force' and not domain:
            label='nesting/force call scope';binding='receipt callback scope'
        elif domain:
            m=models[domain];dims=tuple(row[4:10])
            candidates=[q for q in m['symbols'].get(sym,[]) if q['dimensions']==dims]
            if sym not in m['native_symbols'] and row[10]:
                label='graph helper outside model CUBIN';binding='bound graph; absent native ELF symbol'
                helpers[sym][0]+=1;helpers[sym][1]+=b-a
            elif candidates:
                if not row[10]:
                    exact=[q for q in candidates if q['hlo_instruction']==note.get('hlo_instruction')]
                    if exact:candidates=exact;binding='NVTX exact source instruction + runtime thunk/grid'
                else:
                    node=node_annotation(row[11])
                    exact=[q for q in candidates if node.get('root_domain')==domain and q['hlo_instruction']==node.get('hlo_instruction')]
                    if exact:candidates=exact;binding='graph clone creation + source instruction/runtime thunk'
                labels={q['source']['family'] for q in candidates}
                classes={q['source']['operation_class'] for q in candidates}
                opclass=next(iter(classes)) if len(classes)==1 else 'mixed operation classes'
                label=next(iter(labels)) if len(labels)==1 else 'mixed/shared fusion'
                if binding=='unbound':binding='same loaded module + exact symbol/grid; all runtime call sites retained'
            else:label='unbound in RK-loop scope';binding='NVTX root only; symbol/grid mismatch'
        else:
            label='outside proven RK-loop module';unbound[(scope,sym)]+=1
        counts[label][0]+=1;counts[label][1]+=b-a
        details[(scope,label,opclass)][0]+=1;details[(scope,label,opclass)][1]+=b-a
        method[binding]+=1
        # Witnesses retain *all* possible source sites, never split one event.
        key=(domain or scope,sym,tuple(row[4:10]),label,binding)
        old=witnesses.get(key,{'activities':0,'kernel_ns':0})
        witnesses[key]={'domain':domain,'callback_scope':scope,'symbol':sym,'dimensions':list(row[4:10]),
                        'label':label,'binding':binding,'possible_source_sites':[
                            {'hlo_instruction':q['hlo_instruction'],'source':q['source'],'thunk_ancestors':q['ancestors']} for q in candidates],
                        'activities':old['activities']+1,'kernel_ns':old['kernel_ns']+b-a}
    for a,b,cor,kind,size in c.execute('SELECT start,end,correlationId,copyKind,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY'):
        scope=owner(bycor[cor][0])
        copybins[(scope,kind,size)]+=1
    assert len(kernel)=={'N32':8292,'N43':10227,'R43':22308}[arm]
    assert sum(v[0] for v in counts.values())==len(kernel)
    tables={x[0] for x in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    memcpy_count=c.execute('SELECT count(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY').fetchone()[0]
    memset_count=c.execute('SELECT count(*) FROM CUPTI_ACTIVITY_KIND_MEMSET').fetchone()[0] if 'CUPTI_ACTIVITY_KIND_MEMSET' in tables else 0
    helper_grids=Counter(tuple(k[4:10]) for k in kernel if k[3]=='memcpy32_post')
    return {'arm':arm,'version':version,'count':len(kernel),'kernel_ns':sum(k[1]-k[0] for k in kernel),
            'family_count_ns':dict(counts),'by_scope_family_class':{':'.join(k):v for k,v in details.items()},
            'binding_methods':dict(method),'helper_count_ns':dict(helpers),
            'unbound_symbol_counts':{d+':'+n:v for (d,n),v in unbound.items()},
            'copy_scope_kind_size_counts':[{'scope':s,'kind':k,'bytes':z,'count':n} for (s,k,z),n in copybins.items()],
            'graph_module_bindings':{str(g):sorted(v) for g,v in graph_domains.items()},
            'device_activity_rows':{'kernels':len(kernel),'memcpy':memcpy_count,'memset':memset_count,
                                    'total':len(kernel)+memcpy_count+memset_count},
            'copy_helper_grid_counts':[{'dimensions':list(d),'count':n} for d,n in sorted(helper_grids.items())],
            'source_execution_bindings':{d:m['binding'] for d,m in models.items()}},list(witnesses.values())


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--thunks',type=Path,default=L/'ORDINARY_THUNKS');p.add_argument('--include-radiation',action='store_true');a=p.parse_args();a.out.mkdir(exist_ok=False)
    idx=json.loads((L/'ACTUAL_HLO/INDEX.json').read_text());models={}
    for rec in idx['records']:models[(rec['version'],rec['program'])]=hlo_model(rec,a.thunks)
    rows={}
    arms=[('N32','v032'),('N43','v043')]+([('R43','v043')] if a.include_radiation else [])
    for arm,version in arms:
        proof=json.loads((L/'gpu/runs'/arm/'run/wrfout/proofs/nested_pipeline_run.json').read_text())
        for d in ('d01','d02'):
            assert proof['metadata']['nested_aot']['domains'][d]['cheap_key']==models[(version,d)]['binding']['cheap_key']
        result,witnesses=census(arm,version,{d:models[(version,d)] for d in ('d01','d02')});rows[arm]=result
        (a.out/(arm+'_witnesses.json')).write_text(json.dumps(witnesses,indent=2)+'\n')
        print(arm,json.dumps({k:result[k] for k in ['count','family_count_ns','binding_methods','helper_count_ns']}),flush=True)
    families=rows['N32']['family_count_ns'].keys()|rows['N43']['family_count_ns'].keys()
    delta=[]
    for f in families:
        old=rows['N32']['family_count_ns'].get(f,[0,0]);new=rows['N43']['family_count_ns'].get(f,[0,0])
        delta.append({'family':f,'old_count':old[0],'new_count':new[0],'delta_count':new[0]-old[0],
                      'old_ms':old[1]/1e6,'new_ms':new[1]/1e6,'delta_ms':(new[1]-old[1])/1e6})
    delta.sort(key=lambda r:-r['delta_count']);assert sum(r['delta_count'] for r in delta)==1935
    (a.out/'DELTA.json').write_text(json.dumps({'scope':__doc__,'arms':rows,'delta':delta,
                  'supersedes_family_labels':'Earlier name-based same-module FAMILY_TIMES labels did not account for deduplicated native symbols and small eager executables; kernel totals are unchanged.'},indent=2)+'\n')


if __name__=='__main__':main()

