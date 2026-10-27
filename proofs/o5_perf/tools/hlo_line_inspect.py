import sys,re,json
sys.path.insert(0,'proofs/o5_perf')
import roofline_join as R
from collections import Counter
from pathlib import Path
prog=sys.argv[1]; names=sys.argv[2:]
path=Path(f'<USER_HOME>/wrf_gpu2_lanes/w44-perf/ACTUAL_HLO/v043_{prog}.hlo')
text=path.read_text().splitlines()
tables={k:{} for k in ('FileNames','FunctionNames','FileLocations','StackFrames')};sec=None
for line in text:
    if line in tables: sec=line;continue
    if sec and line[:1].isdigit():
        k,v=line.split(' ',1);tables[sec][int(k)]=v;continue
    if sec and not line[:1].isdigit(): sec=None
KV=re.compile(r"(\w+)=(-?\d+)")
def loc(fid):
    f=dict((k,int(v)) for k,v in KV.findall(tables['StackFrames'].get(fid,'')))
    l=dict((k,int(v)) for k,v in KV.findall(tables['FileLocations'].get(f.get('file_location_id'),'')))
    return tables['FileNames'].get(l.get('file_name_id'),'?').strip('"')+':'+str(l.get('line'))+':'+tables['FunctionNames'].get(l.get('function_name_id'),'?').strip('"')
comps=R.parse_hlo(path)
inst_comp={n:c for c,i in comps.items() for n in i}
raw={}
cur=None
for line in text:
    m=re.match(r'^(ENTRY )?%?([\w.\-]+) .*\{\s*$', line)
    if m and not line.startswith(' '): cur=m.group(2); raw[cur]=[]; continue
    if cur and line.startswith('  '): raw[cur].append(line)
for n in names:
    c=inst_comp[n]; i=comps[c][n]
    print('==',n,'in',c,'shape',i['shape'][:200],'op',i['op'],'kind',i['kind'],'calls',i['calls'])
    print('  operands:',[ (o, comps[c][o]['shape'][:60]) for o in i['operands'] if o in comps[c]][:60])
    print('  bytes r/w', R.instruction_bytes(i,c,comps))
    if i['calls'] in raw:
        body=raw[i['calls']]
        ops=Counter(); srcs=Counter()
        for l in body:
            p=R.parse_inst(l)
            if p: ops[p['op']]+=1
            for s in re.findall(r'stack_frame_id=(\d+)',l): srcs[loc(int(s))]+=1
        print('  body',len(body),'ops',ops.most_common(12))
        for s,cn in srcs.most_common(12): print('   ',cn,s)
