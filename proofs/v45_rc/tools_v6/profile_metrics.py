"""Numeric kernel durations inside explicit public ranges; no environment-table reads."""
import collections,hashlib,json,re,sqlite3,statistics,sys
from pathlib import Path
DB,ARM,OUT=map(Path,sys.argv[1:4]);cpu=sys.argv[4]=='smoke'
meta=json.loads((ARM/'PROFILE_CALLS.json').read_text());calls=meta['calls'];assert calls,'profile hook not executed'
programs=('d01','d02','fused_d01','fused_d02')
names=tuple('W45RC:step:'+n for n in programs);keys=tuple(x for n in names for x in (n,'TSL:'+n))
with sqlite3.connect('file:'+str(DB.resolve())+'?mode=ro',uri=True) as db:
 tables={x[0] for x in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
 assert 'NVTX_EVENTS' in tables
 labels={}
 for code,key in enumerate(keys):
  # Exact public strings, numeric IDs only; BLOB length handles embedded NUL safely.
  for (sid,) in db.execute('SELECT id FROM StringIds WHERE id>=0 AND length(CAST(value AS BLOB))<64 AND value=?',(key,)):
   labels[sid]=programs[code//2]
 ranges=collections.defaultdict(list)
 for code,key in enumerate(keys):
  for start,end in db.execute('SELECT start,end FROM NVTX_EVENTS WHERE length(CAST(text AS BLOB))<64 AND text=?',(key,)):
   if end and end>start:ranges[programs[code//2]].append((start,end))
 for sid,program in labels.items():
  for start,end in db.execute('SELECT start,end FROM NVTX_EVENTS WHERE textId=?',(sid,)):
   if end and end>start:ranges[program].append((start,end))
 for program in ranges:ranges[program]=sorted(set(ranges[program]))
 expected=collections.Counter(c['program'] for c in calls)
 assert all(len(ranges[n])==expected[n] for n in expected),('missing actual NVTX labels',expected,{n:len(v) for n,v in ranges.items()})
 result=dict(scope='single-launch GPU durations under node tracing; NOT S1; profiler adds host synchronization',
             explicit_NVTX_api=meta['api'],CPU_smoke=cpu,range_counts={n:len(v) for n,v in ranges.items()},kernels=[])
 if cpu:
  result['GPU_native_profile']='NOT_APPLICABLE_CPU_SMOKE'
 else:
  assert 'CUPTI_ACTIVITY_KIND_KERNEL' in tables
  offsets=collections.Counter();samples=collections.defaultdict(list);perroot=collections.defaultdict(lambda:collections.Counter())
  for call in calls:
   program=call['program'];start,end=ranges[program][offsets[program]];offsets[program]+=1
   rows=db.execute('SELECT shortName,end-start FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=?',(start,end)).fetchall()
   assert rows,('empty GPU program range',program)
   for kid,ns in rows:
    assert type(ns) is int and ns>0
    samples[(program,kid)].append(ns);perroot[(program,kid)][call['root']]+=ns
  for (program,kid),values in samples.items():
   # Only IDs already proven to be kernel-name IDs; never broad string queries.
   value=db.execute('SELECT value FROM StringIds WHERE id=? AND length(CAST(value AS BLOB))<=256',(kid,)).fetchone()
   assert value is not None,('unbounded kernel-name ID',kid)
   name=value[0]
   assert isinstance(name,str) and '\x00' not in name and re.fullmatch(r'[A-Za-z_.$][A-Za-z0-9_.$:<>(), *&\-]*',name),('invalid kernel-name ID',kid)
   values=sorted(values);rootvals=list(perroot[(program,kid)].values())
   result['kernels'].append(dict(program=program,kernel=name,string_id=kid,launches=len(values),single_launch_ns_min=values[0],
       single_launch_ns_median=statistics.median(values),single_launch_ns_p95=values[min(len(values)-1,int(.95*len(values)))],
       single_launch_ns_max=values[-1],observed_roots=len(rootvals),sum_per_root_ns_median=statistics.median(rootvals)))
  result['GPU_native_profile']='MEASURED; source/DAG-to-thunk binding required before attributing a kernel to the 664 diagnostic'
OUT.write_text(json.dumps(result,indent=2)+'\n')
print('PROFILE ranges',result['range_counts'],'kernels',len(result['kernels']),result['GPU_native_profile'])
