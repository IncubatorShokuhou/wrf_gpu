"""The native FSNO wiring repair preserves STC1/2 and the vector program."""
import hashlib,importlib.util,inspect,json,os,re,sys
from pathlib import Path
tree,out=map(Path,sys.argv[1:3]);sys.path[:0]=[str(tree/'src')]
os.environ.update(GPUWRF_NOAHMP_NATIVE_REAL='1',GPUWRF_NOAHMP_LAYER_SELECT='1',GPUWRF_NOAHMP_LAYER_LISTS='1')
import jax
from gpuwrf.physics.noahmp import energy,noahmp_driver
from gpuwrf.io.noahmp_options import NoahMPOptions
assert jax.devices()[0].platform=='cpu'
source=inspect.getsource(energy.noahmp_energy_canopy)
new="            native_bare_kw={**bare_kw,'fsno':va[26]} if 'fsno' in bare_kw else bare_kw\n            bare = _bare_flux(*ba,**native_bare_kw)"
assert source.count(new)==1
old=source.replace(new,'            bare = _bare_flux(*ba,**bare_kw)')
namespace=dict(vars(energy));exec(compile(old,energy.__file__,'exec'),namespace)
before=namespace['noahmp_energy_canopy'];after=energy.noahmp_energy_canopy
path=tree/'tests/v025/b_noahmp/test_snow_wrf_order_oracle.py'
spec=importlib.util.spec_from_file_location('stc_reuse_columns',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
columns=json.loads((path.parent/'fixtures/swiss_snow_wrf_order.json').read_text())['columns'][:4]
c=module._build(columns);report={}
def canonical(jp):
    text=re.sub(r'name_and_src_info=[^\]\n]*','',str(jp))
    return re.sub(r'/[^ \n:]*\.py:\d+(:\d+)?','',text)
try:
    for mode,native in ((1,True),(2,True),(3,False)):
        os.environ['GPUWRF_NOAHMP_COLUMN_KERNELS']='1' if native else '0'
        opts=None if mode==1 else NoahMPOptions(opt_stc=mode)
        hashes=[]
        for function in (before,after):
            noahmp_driver.noahmp_energy_canopy=function;jax.clear_caches()
            fn=lambda land,forcing:noahmp_driver.noah_mp_step(land,forcing,c['static'],c['dt'],
                energy_params=c['ep'],rad_params=c['rp'],**({'options':opts} if opts is not None else {}))
            hashes.append(hashlib.sha256(canonical(jax.make_jaxpr(fn)(c['land'],c['forcing']).jaxpr).encode()).hexdigest())
        report[f'stc{mode}_native{int(native)}']={'before':hashes[0],'after':hashes[1],'exact':hashes[0]==hashes[1]}
        assert hashes[0]==hashes[1],report
finally:
    noahmp_driver.noahmp_energy_canopy=after
    out.write_text(json.dumps(report,indent=2)+'\n')
