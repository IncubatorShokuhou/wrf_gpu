"""CPU-only caller measurement of the first cheap-key source fingerprint.

No model initialization, lowering, array read, executable load or GPU query.
Profiles a separate cold function call; cProfile time is diagnostic only.
"""
import argparse
import cProfile
import hashlib
import json
import os
from pathlib import Path
import pstats
import subprocess
import sys
import time

ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--src',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
assert os.environ['JAX_PLATFORMS']=='cpu' and os.environ['GPUWRF_JAX_CACHE']=='0'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
assert not subprocess.check_output(['git','-C',str(a.src),'status','--porcelain','--untracked-files=no','--','src'],text=True).strip()
sys.path.insert(0,str(a.src/'src'))
from gpuwrf.runtime import aot_cheap_key as ck
ck.source_fingerprint_hash.cache_clear()
t=time.perf_counter();digest=ck.source_fingerprint_hash();first=time.perf_counter()-t
t=time.perf_counter();again=ck.source_fingerprint_hash();hit=time.perf_counter()-t
ck.source_fingerprint_hash.cache_clear();pr=cProfile.Profile();pr.enable();profile_digest=ck.source_fingerprint_hash();pr.disable()
assert digest==again==profile_digest
stats=pstats.Stats(pr);rows=[]
for (filename,line,fn),(cc,nc,tt,ct,callers) in stats.stats.items():
    rows.append(dict(file=filename,line=line,function=fn,primitive_calls=cc,calls=nc,self_s=tt,cumulative_s=ct))
rows.sort(key=lambda r:-r['cumulative_s'])
result=dict(scope=__doc__,source=str(a.src),head=subprocess.check_output(['git','-C',str(a.src),'rev-parse','HEAD'],text=True).strip(),
            module_sha256=hashlib.sha256(Path(ck.__file__).read_bytes()).hexdigest(),fingerprint=digest,
            first_unprofiled_s=first,cached_unprofiled_s=hit,profiled_total_s=stats.total_tt,
            total_calls=stats.total_calls,top_cumulative=rows[:35],top_self=sorted(rows,key=lambda r:-r['self_s'])[:25])
a.out.write_text(json.dumps(result,indent=2)+'\n');print(a.src.name,first,hit,[(r['function'],r['cumulative_s']) for r in rows[:8]],flush=True)
