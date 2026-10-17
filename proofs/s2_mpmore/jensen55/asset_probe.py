"""Actual clean-archive reader, selected table identity, and host key cost."""
from pathlib import Path
import dataclasses,hashlib,importlib.util,io,json,os,resource,subprocess,sys,tarfile,tempfile,time
import numpy as np

tree,out=Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve()
assert (tree/'.git').is_file() and os.environ['JAX_PLATFORMS']=='cpu'
sys.path.insert(0,str(tree/'src'))
from gpuwrf.physics.jensen_ishmael import tables
from gpuwrf.runtime.aot_baked_tables import baked_table_hash
from gpuwrf.runtime.operational_mode import OperationalNamelist
from gpuwrf.contracts.grid import GridSpec
lane=Path('<USER_HOME>/wrf_gpu2_lanes/s2-mpmore')
asset='src/gpuwrf/physics/jensen_ishmael/data/lookup_tables.npz'
reader='src/gpuwrf/physics/jensen_ishmael/tables.py'
with tempfile.TemporaryDirectory(prefix='jensen_clean_archive_',dir=lane) as directory:
    archive=subprocess.check_output(['git','archive','HEAD',reader,asset],cwd=tree)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:tar.extractall(directory,filter='data')
    archive=None
    file=Path(directory)/reader
    spec=importlib.util.spec_from_file_location('jensen_clean_archive_tables',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    start=time.monotonic();raw=module._host_tables();first_load=time.monotonic()-start
    for dtype in (np.float32,np.float64):
        a,b=module.get_tables(dtype),tables.get_tables(dtype)
        assert set(a)==set(b)
        for name in a:assert a[name].dtype==b[name].dtype and a[name].tobytes()==b[name].tobytes(),name
    digest=hashlib.sha256((Path(directory)/asset).read_bytes()).hexdigest()
    assert digest=='4cbe72b01c7f509a05309e89e9b8e82223ddf0c8cc833f5b16b0ed1237741b76'
    raw_bytes=sum(a.nbytes for a in raw.values());compressed_bytes=(Path(directory)/asset).stat().st_size
    module._host_tables.cache_clear();raw=None;a=None;b=None

nl=OperationalNamelist(GridSpec.canary_3km_template(),None,None,18.,mp_physics=55,
                      ra_sw_physics=0,ra_lw_physics=0,sf_sfclay_physics=0,cu_physics=0)
def fn(x):return x
start=time.monotonic();identity=baked_table_hash(fn,nl);first_hash=time.monotonic()-start
times=[]
for _ in range(7):
    start=time.monotonic();assert baked_table_hash(fn,nl)==identity;times.append(time.monotonic()-start)
result=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=tree,text=True).strip(),
            CPU=True,GPU=False,clean_archive_actual_reader_BYTE=True,selected_modes=['REAL4','REAL8'],
            asset_sha256=digest,compressed_bytes=compressed_bytes,raw_host_array_bytes=raw_bytes,
            first_archive_host_load_s=first_load,first_selected_host_key_s=first_hash,
            warm_selected_host_key_median_ms=1000*float(np.median(times)),
            peak_process_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            scope='Host-only identity work outside timesteps; CPU process RSS includes verification copies, not GPU VRAM')
out.write_text(json.dumps(result,indent=2)+'\n');print(result)
