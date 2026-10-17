"""r-pkg040: the packaging change of jensen_ishmael/tables.py (1fb2128686) moves only WHERE the table is read.

Proves, for the receipt NATIVE_COMPILED_DEVIATIONS.json (measured at 02a42169):
  1. lookup_tables.npz bytes are unchanged (measured commit == packaging parent == packaging commit == HEAD ==
     working file == assets_manifest pin);
  2. the source delta is exactly the two documented substitutions (reverting them gives the measured bytes);
  3. the old and new loaders return bitwise-identical arrays, from the repo copy and from a user-cache copy;
  4. the program the receipt measured (operational CPU-interpret jit of full_column on fixtures/fp32_4.json) lowers to
     byte-identical StableHLO WITH debug locations, old vs new tables.py at the same path (separate processes).
CPU only.  usage: JAX_PLATFORMS=cpu python3 prove_tables_packaging.py OUT.json SCRATCH_DIR
"""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[3]
REL='src/gpuwrf/physics/jensen_ishmael/tables.py'
NPZ='src/gpuwrf/physics/jensen_ishmael/data/lookup_tables.npz'
PACKAGING='1fb2128686'
MEASURED_SHA='7c06e2345824a6f5482e0ab3577b2e2187b1f7e56b7d43aae6589266ccf89a64'  # receipt pin before r-pkg040
NEW_IMPORT=(b"from gpuwrf.assets import asset_path, ensure_asset\n\n"
            b"# > 10 MB: repo copy in a private checkout, else fetched on first use (gpuwrf.assets, fail-closed).\n"
            b"_ASSET=asset_path('src/gpuwrf/physics/jensen_ishmael/data/lookup_tables.npz')\n")
OLD_IMPORT=b"_ASSET=Path(__file__).with_name('data')/'lookup_tables.npz'\n"
NEW_LOAD,OLD_LOAD=b"np.load(ensure_asset(_ASSET),",b"np.load(_ASSET,"
sha=lambda b:hashlib.sha256(b).hexdigest()


def git(*args,text=False):
    return subprocess.check_output(['git',*args],cwd=ROOT,text=text)


def lower_child(out):
    """One process: lower the measured program and hash its StableHLO text with debug locations."""
    sys.path.insert(0,str(ROOT/'src'))
    import jax,jax.numpy as jnp
    assert jax.devices()[0].platform=='cpu'
    from gpuwrf.kernels.phys_jensen import full_column
    from gpuwrf.physics.jensen_ishmael.registry import output_mapping
    import gpuwrf.physics.jensen_ishmael.tables as tables
    selection=(*output_mapping(),'th')
    original=json.loads((ROOT/'proofs/s2_mpmore/jensen55/fixtures/fp32_4.json').read_text())
    fields={n:jnp.asarray(a,jnp.float32) for n,a in original['input'].items()}
    fn=jax.jit(lambda f,d,i:full_column(f,d,i,interpret=True,output_names=selection))
    start=time.monotonic()
    lowered=fn.lower(fields,jnp.float32(original['dt']),jnp.int32(original['it']))
    text=lowered.as_text(debug_info=True)
    Path(out).write_text(json.dumps(dict(tables_sha256=sha(Path(tables.__file__).read_bytes()),
        stablehlo_debug_sha256=sha(text.encode()),stablehlo_debug_characters=len(text),
        tables_py_in_locations=REL.rsplit('/',1)[1] in text,table_loads=tables._host_tables.cache_info().misses,
        lower_s=time.monotonic()-start)))


def main(out,scratch):
    receipt=json.loads((ROOT/'proofs/s2_mpmore/jensen55/NATIVE_COMPILED_DEVIATIONS.json').read_text())
    measured=receipt['source_commit']
    parent=git('rev-parse',PACKAGING+'^',text=True).strip()
    head=git('rev-parse','HEAD',text=True).strip()
    result=dict(producer='r-pkg040',CPU=True,GPU=False,head=head,measured_commit=measured,
                packaging_commit=git('rev-parse',PACKAGING,text=True).strip(),packaging_parent=parent)
    # 1. table bytes
    pin=json.loads((ROOT/'src/gpuwrf/assets_manifest.json').read_text())['assets'][NPZ]
    blobs={c:git('rev-parse',f'{c}:{NPZ}',text=True).strip() for c in (measured,parent,PACKAGING,head)}
    digests={c:sha(git('show',f'{c}:{NPZ}')) for c in (measured,parent,PACKAGING,head)}
    working=sha((ROOT/NPZ).read_bytes())
    result['table']=dict(path=NPZ,git_blob=blobs,sha256=digests,working_sha256=working,manifest_pin=pin,
        identical=len(set(blobs.values()))==1 and len(set(digests.values()))==1==len({working,pin['sha256'],*digests.values()}))
    # 2. source delta
    old_src,new_src=git('show',f'{measured}:{REL}'),(ROOT/REL).read_bytes()
    assert sha(old_src)==MEASURED_SHA,'measured commit does not carry the pinned tables.py'
    assert new_src.count(NEW_IMPORT)==1 and new_src.count(NEW_LOAD)==1
    reverted=new_src.replace(NEW_IMPORT,OLD_IMPORT).replace(NEW_LOAD,OLD_LOAD)
    result['source']=dict(path=REL,measured_sha256=sha(old_src),current_sha256=sha(new_src),
        substitutions=[[OLD_IMPORT.decode(),NEW_IMPORT.decode()],[OLD_LOAD.decode(),NEW_LOAD.decode()]],
        reverted_sha256=sha(reverted),reverted_equals_measured=reverted==old_src,
        changed_between_measured_and_head=git('log','--format=%h %s',f'{measured}..{head}','--',REL,text=True).strip().splitlines())
    # 3. loaded arrays (old loader vs new loader; repo copy and user-cache copy)
    sys.path.insert(0,str(ROOT/'src'))
    import numpy as np
    import gpuwrf.assets as assets
    import gpuwrf.physics.jensen_ishmael.tables as tables
    old_ns=dict(__file__=str(ROOT/REL),__name__='_measured_jensen_tables')
    exec(compile(old_src,str(ROOT/REL),'exec'),old_ns)
    old,new=old_ns['_host_tables'](),tables._host_tables()
    same=lambda a,b:set(a)==set(b) and all(a[n].dtype==b[n].dtype and a[n].shape==b[n].shape and a[n].tobytes()==b[n].tobytes() for n in a)
    cache=Path(scratch)/'tables';os.environ['GPUWRF_TABLES_DIR']=str(cache)
    cached=assets._cache_path(pin);cached.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/NPZ,cached)
    resolved=assets.ensure_asset(cached)
    with np.load(resolved,allow_pickle=False) as data:from_cache={n:data[n] for n in data.files}
    result['arrays']=dict(names=sorted(old),repo_old_vs_new_bitwise=same(old,new),cache_vs_old_bitwise=same(from_cache,old),
        cache_path_pattern='<GPUWRF_TABLES_DIR>/<sha256>/<asset>',cache_resolved=str(resolved).startswith(str(cache)),
        new_loader_path=str(tables._ASSET.relative_to(ROOT)))
    shutil.rmtree(cache)
    # 4. lowered program, old vs new tables.py at the SAME path, one process each
    env=dict(os.environ,JAX_PLATFORMS='cpu',GPUWRF_JAX_CACHE='0',PYTHONDONTWRITEBYTECODE='1')
    env.pop('GPUWRF_TABLES_DIR')
    lows={}
    for variant,src in (('new',new_src),('measured',old_src),('new_again',new_src)):
        try:
            if src is not new_src:(ROOT/REL).write_bytes(src)
            path=Path(scratch)/f'lower_{variant}.json'
            subprocess.run([sys.executable,__file__,'--lower',str(path)],env=env,check=True,cwd=scratch)
            lows[variant]=json.loads(path.read_text())
        finally:
            (ROOT/REL).write_bytes(new_src)
    assert sha((ROOT/REL).read_bytes())==sha(new_src)
    result['lowered']=dict(program='jit(full_column interpret=True, operational outputs) on fixtures/fp32_4.json',variants=lows,
        identical=len({v['stablehlo_debug_sha256'] for v in lows.values()})==1,
        sources_seen=[v['tables_sha256'] for v in lows.values()])
    result['verdict']='IDENTICAL' if (result['table']['identical'] and result['source']['reverted_equals_measured'] and
        result['arrays']['repo_old_vs_new_bitwise'] and result['arrays']['cache_vs_old_bitwise'] and result['lowered']['identical']) else 'DIFFERS'
    Path(out).write_text(json.dumps(result,indent=1)+'\n')
    print(result['verdict'])
    return result['verdict']=='IDENTICAL'


if __name__=='__main__':
    if sys.argv[1]=='--lower':lower_child(sys.argv[2])
    else:sys.exit(0 if main(sys.argv[1],sys.argv[2]) else 1)
