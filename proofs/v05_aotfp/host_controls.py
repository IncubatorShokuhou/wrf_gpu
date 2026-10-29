"""Host-only supported AOT caller admission controls; never forecast fidelity."""
from pathlib import Path
import argparse,copy,dataclasses,hashlib,json,os,pickle,subprocess,sys,traceback
SOURCE=Path(os.environ['V05_AOTFP_SOURCE'])
ROOT=Path(os.environ['V05_AOTFP_ARTIFACT_DIR']);ROOT.mkdir(parents=True,exist_ok=True)
assert os.environ['JAX_PLATFORMS']=='cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
sys.path[:0]=[str(SOURCE/'src'),str(SOURCE)]
import gpuwrf
import jax,jax.numpy as jnp,pytest
from gpuwrf.runtime import aot_executable as ax,aot_precompile as ap,aot_cheap_key as ck,domain_tree as dt
from gpuwrf.integration import init_kernels, nested_pipeline
from gpuwrf import cli
from tests import test_aot_executable as h
assert all(d.platform=='cpu' for d in jax.devices())
assert Path(ax.__file__).resolve().is_relative_to(SOURCE)
FIELDS=['matching','cuda_driver','xla_flags','empty']

def case(kind,field):
    folder=ROOT/(kind+'_'+field);folder.mkdir(exist_ok=False); cache=folder/'cache'
    records=[]; executions=[]
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('GPUWRF_NESTED_AOT','1');mp.setenv('GPUWRF_AOT_VERIFY','0');mp.setenv('GPUWRF_ADVANCE_CHUNK_LOOP','fori')
        mp.setenv('GPUWRF_JAX_CACHE','0');mp.setenv('XLA_FLAGS','');mp.setenv('XLA_PYTHON_CLIENT_ALLOCATOR','cuda_async')
        mp.setattr(ap,'version_cache_tag',lambda:'w4-host-negative-control')
        dt.set_process_aot_reuse(False)
        jax.clear_caches()
        compiled,original_args,_=h._compile_dropping_graph(n_dead=0)
        blob,good=ax.serialize(compiled)
        original=ap.load_domain_blob
        original_private=ap._load_domain_blob
        paths_fn=ap._aot_blob_paths
        def prepare(name,kw):
            cheap=kw.get('cheap_key');hlo=kw.get('hlo_sha256')
            paths=paths_fn(name,cache,cheap_key=cheap,hlo_sha256=hlo);assert paths is not None
            fp=dict(good.fingerprint)
            if field=='empty':fp={}
            elif field!='matching':fp[field]=str(fp.get(field))+'-MISMATCH'
            meta=dataclasses.replace(good,fingerprint=fp,cheap_key=cheap,key_schema=ck.KEY_SCHEMA if cheap else None,hlo_sha256=hlo or good.hlo_sha256)
            paths[0].parent.mkdir(parents=True,exist_ok=True)
            paths[0].write_bytes(blob);paths[1].write_bytes(pickle.dumps(meta))
            return paths
        def record(name,kw,call,status):
            records.append(dict(name=name,cheap_key=kw.get('cheap_key'),hlo_sha256=kw.get('hlo_sha256'),loaded=status['loaded'],error=status.get('error'),source=status.get('source')))
            assert (call is not None)==status['loaded']
            if field!='matching' and call is not None:
                raise AssertionError('SUPPORTED CALLER ACCEPTED MISMATCH '+kind+' '+field)
        def public(name,*args,**kw):
            assert kw.get('check_fingerprint',True) is True
            prepare(name,kw)
            actual=dict(kw,return_status=True); actual.pop('cache_dir',None)
            call,status=original(name,cache,**actual)
            record(name,kw,call,status)
            if call is not None:
                # Execute the actual CPU binding first. Physics is outside this host admission test.
                result=call(*original_args);assert result is not None
                def payload(*a,**k):
                    executions.append(name)
                    return (a[0],a[1]) if kind=='fused' else a[0]
                return payload,status
            return None,status
        mp.setattr(ap,'load_domain_blob',public)
        def run_boundary():
            nl=h._aot_namelist();carry=h._aot_carry(1)
            mp.setattr(dt,'_advance_chunk_fori',h._make_aot_advance_like())
            mp.setattr(dt,'_advance_chunk',lambda carry,*a,**k:carry)
            advance=dt._operational_advance_factory(h._fake_tree(nl))
            return advance('d01',carry,start_step=1,n_steps=5)
        def cli_boundary():
            inp=folder/'input';inp.mkdir();(inp/'namelist.input').write_bytes((SOURCE/'tests/fixtures/wn3_20260227_cadence.namelist').read_bytes())
            def pipeline(config):
                run_boundary()
                return {'verdict':'PIPELINE_GREEN','metadata':{}}
            mp.setattr(nested_pipeline,'execute_nested_pipeline',pipeline)
            rc=cli.main(['run','--input-dir',str(inp),'--output-dir',str(folder/'output'),'--domains-from-namelist','--hours','3','--force-gpu-run'])
            assert rc==0,rc
        if kind in ('run','bundle'):
            cli_boundary()
        elif kind=='init':
            fn=jax.jit(lambda x:x+x)
            init_kernels.InitKernelCache(cache)(fn,jnp.arange(4.))
        elif kind=='fused':
            mp.setattr(dt,'_advance_chunk',lambda carry,*a,**k:carry)
            mp.setattr(dt,'build_child_boundary_package',lambda child,*a,**k:child)
            program=dt._build_fused_cascade_program(parent_name='d02',parent_namelist=h._aot_namelist(),parent_cadence=7,child_names=('d03',),child_namelists=(h._aot_namelist(),),child_weights=(jnp.asarray([1.]),),child_bdy_widths=(5,),child_ratios=(1,),child_cadences=(7,))
            program(h._FusedCarry(jnp.asarray([1.])),(h._FusedCarry(jnp.asarray([2.])),),1,(1,))
        elif kind=='prefetch':
            def private(name,*args,**kw):
                prepare(name,kw)
                call,status=original_private(name,cache,**kw)
                record(name,kw,call,status)
                return call,status
            mp.setattr(ap,'_load_domain_blob',private)
            # Restore normal public loader: exercise actual speculative and consume paths.
            mp.setattr(ap,'load_domain_blob',original)
            key='a'*64
            with ap.prefetch_domain_blobs({'d01':key},cache,dev=jax.devices()[0],max_workers=1) as prefetch:
                prefetch.futures['d01'].result(timeout=10)
                call,status=ap.load_domain_blob('d01',cache,cheap_key=key,dev=jax.devices()[0],return_status=True)
                assert status['loaded']==(field=='matching'),status
                if call is not None:call(*original_args);executions.append('prefetch')
            assert not ap._prefetched_loads and not ap._active_prefetches
        else:raise AssertionError(kind)
    assert records, 'caller never reached real loader'
    if field=='matching':
        assert any(r['loaded'] for r in records) and executions,(records,executions)
    else:
        assert all(not r['loaded'] and 'fingerprint mismatch' in (r['error'] or '') for r in records),records
        assert not executions
    result=dict(kind=kind,field=field,verdict='PASS',actual_platform='cpu',loaded=field=='matching',loader_records=records,binding_executions=len(executions),scope='actual caller/parser/strict loader; tiny CPU native and model payload stubs; no GPU or scientific result')
    (folder/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',choices=['bundle']);parser.add_argument('--field',choices=FIELDS);a=parser.parse_args()
    if a.worker:
        print(json.dumps(case('bundle',a.field)));return
    rows=[]
    for kind in ['run','init','fused','prefetch']:
        for field in FIELDS:
            row=case(kind,field);rows.append(row);print('CONTROL',kind,field,'PASS',flush=True)
    (ROOT/'DIRECT_RESULTS.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':
    try:main()
    except BaseException:
        (ROOT/'FAILURE.txt').write_text(traceback.format_exc())
        raise
