"""Execute actual product cap statements against untouched pristine blocks."""
import ast,json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def matrix(engine=jnp):
    from gpuwrf.physics.noahmp import energy
    fx=json.loads((ROOT/'proofs/u2_land/surface_caps_oracle.json').read_text())
    tree=ast.parse(Path(energy.__file__).read_text());outputs=[]
    for kind,name in ((1,'_vege_flux'),(2,'_bare_flux')):
        fn=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==name)
        start=next(i for i,node in enumerate(fn.body) if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='snow_clamp' for t in node.targets))
        last='gh' if kind==1 else 'ghb';end=next(i for i in range(start,len(fn.body)) if isinstance(fn.body[i],ast.Assign)
            and any(isinstance(t,ast.Name) and t.id==last for t in fn.body[i].targets))
        for stc in (1,2,3):
            cases=[c for c in fx['cases'] if (c['kind'],c['opt_stc'])==(kind,stc)]
            values=engine.asarray([c['inputs'] for c in cases],engine.float32)
            variables=('tg','irg','shg','evg','gh') if kind==1 else ('tgb','irb','shb','evb','ghb')
            def compute(values):
                namespace=dict(vars(energy));namespace.update(zip(fx['keys'],values.T));namespace.update(jnp=engine,opt_stc=stc)
                namespace.update(zip(variables,(namespace[k] for k in ('temp','ir','sh','ev','gh'))))
                namespace.update(cir_g=namespace['cir'],csh_g=namespace['csh'],cev_g=namespace['cev'],pahg=namespace['pah'],pahb=namespace['pah'],eair=namespace['eah'])
                exec(compile(ast.Module(body=fn.body[start:end+1],type_ignores=[]),str(ROOT/'src/gpuwrf/physics/noahmp/energy.py'),'exec'),namespace)
                return engine.stack([namespace[k] for k in variables],axis=-1)
            outputs.extend(np.asarray(jax.jit(compute)(values) if engine is jnp else compute(values)))
    return np.asarray(outputs),fx


def test_original_active_fractional_snow_caps():
    actual,fx=matrix();expected=np.asarray([c['expected'] for c in fx['cases']],np.float32)
    assert actual.dtype==np.float32 and np.isfinite(actual).all()
    np.testing.assert_allclose(actual,expected,rtol=3.e-6,atol=2.e-6)
