"""Fresh CPU processes: actual legacy walker vs uncached/cached candidate.

All modes use the same immutable candidate source tree. The legacy function is
extracted verbatim from the sealed v0.4.3 module. No lowering or GPU access.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def child(args):
    production=args.child.startswith('production-')
    assert os.environ['JAX_PLATFORMS']=='cpu' and os.environ['GPUWRF_JAX_CACHE']==('1' if production else '0')
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    sys.path.insert(0,str(args.src/'src'))
    from gpuwrf.runtime import aot_cheap_key as ck
    legacy_path=Path('<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3/src/src/gpuwrf/runtime/aot_cheap_key.py')
    if args.child=='legacy':
        module=ast.parse(legacy_path.read_text())
        fn=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='_gpuwrf_imports_in_file')
        namespace={}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(legacy_path),'exec'),ck.__dict__,namespace)
        ck._gpuwrf_imports_in_file=namespace[fn.name]
    if not production:
        ck._source_import_cache_dir=lambda: args.cache if args.child in ('prime','warm') else None
    calls=[];original=ast.parse
    def counted(source,*a,**kw):
        calls.append(kw.get('filename',a[0] if a else None))
        return original(source,*a,**kw)
    ast.parse=counted
    ck.source_fingerprint_hash.cache_clear()
    start=time.perf_counter();value=ck.source_fingerprint_hash();elapsed=time.perf_counter()-start
    print(json.dumps(dict(mode=args.child,fingerprint=value,fingerprint_s=elapsed,ast_parses=len(calls),
                         parsed_paths=calls,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                         module_sha256=hashlib.sha256(Path(ck.__file__).read_bytes()).hexdigest(),
                         legacy_module_sha256=hashlib.sha256(legacy_path.read_bytes()).hexdigest())))
    return


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--src',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path)
    p.add_argument('--child',choices=['legacy','uncached','prime','warm','production-prime','production-warm'])
    a=p.parse_args()
    if a.child:child(a);return
    assert a.out and not a.cache.exists()
    assert not subprocess.check_output(['git','-C',str(a.src),'status','--porcelain','--untracked-files=no','--','src'],text=True).strip()
    records=[]
    # Fixed complete sample, no trimming: 3 baseline, 3 optimized misses,
    # one population process, 3 fresh warm processes.
    for mode in ['legacy']*3+['uncached']*3+['prime']+['warm']*3:
        command=[sys.executable,__file__,'--src',str(a.src),'--cache',str(a.cache),'--child',mode]
        start=time.perf_counter();r=subprocess.run(command,capture_output=True,text=True,timeout=45);wall=time.perf_counter()-start
        assert r.returncode==0,r.stderr
        rec=json.loads(r.stdout.splitlines()[-1]);rec['process_wall_s']=wall;records.append(rec)
        print(mode,rec['fingerprint_s'],rec['ast_parses'],wall,flush=True)
    assert len({r['fingerprint'] for r in records})==1
    assert all(r['ast_parses']==0 for r in records if r['mode']=='warm')
    assert all(r['ast_parses']>0 for r in records if r['mode']!='warm')
    import statistics
    means={mode:{metric:statistics.mean(r[metric] for r in records if r['mode']==mode)
                 for metric in ('fingerprint_s','process_wall_s','peak_rss_kib')}
           for mode in ('legacy','uncached','prime','warm')}
    files=list(a.cache.glob('*.json'))
    result=dict(scope=__doc__,candidate_head=subprocess.check_output(['git','-C',str(a.src),'rev-parse','HEAD'],text=True).strip(),
                records=records,means=means,all_fingerprints_byte_equal=True,
                metadata_cache=dict(path=str(a.cache),entries=len(files),bytes=sum(f.stat().st_size for f in files)),
                saved_fingerprint_s=means['legacy']['fingerprint_s']-means['warm']['fingerprint_s'],
                saved_process_s=means['legacy']['process_wall_s']-means['warm']['process_wall_s'])
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('means','metadata_cache','saved_fingerprint_s','saved_process_s')},indent=2))


if __name__=='__main__':main()

