"""Compare default initializer lowerings to the unmodified helper at HEAD."""
from __future__ import annotations

import ast
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.physics import mynn_pbl as P


def main():
    root=Path(__file__).resolve().parents[2];out=Path(sys.argv[1]);out.mkdir(exist_ok=False,parents=True)
    baseline=subprocess.check_output(['git','show','HEAD:src/gpuwrf/physics/mynn_pbl.py'],cwd=root,text=True)
    node=next(n for n in ast.parse(baseline).body if isinstance(n,ast.FunctionDef)and n.name=='mynn_coldstart_init_columns')
    # Experimental snapshots already include the candidate at HEAD. The
    # baseline commit is explicit, supplied by the caller for immutable reuse.
    if len(sys.argv)>2:
        baseline=subprocess.check_output(['git','show',sys.argv[2]+':src/gpuwrf/physics/mynn_pbl.py'],cwd=root,text=True)
        node=next(n for n in ast.parse(baseline).body if isinstance(n,ast.FunctionDef)and n.name=='mynn_coldstart_init_columns')
    namespace=dict(P.__dict__)
    exec(compile(ast.Module(body=[node],type_ignores=[]),P.__file__,'exec'),namespace)
    old=namespace['mynn_coldstart_init_columns']
    capture=json.loads((root/'tests/v025/b_phys/fixtures/mynn_closure_carries_wrf.json').read_text())['rows'][0]
    records=[]
    for dtype in (jnp.float32,jnp.float64):
        with jax.enable_x64(dtype==jnp.float64):
            values={name:jnp.asarray(np.asarray(value)[:2],dtype=dtype) for name,value in capture['state'].items()}
            state=P.MynnPBLColumnState(**{n:values.get(n)for n in P.MynnPBLColumnState.__slots__ if n in values})
            ust=jnp.asarray(capture['flux']['ustar'][:2],dtype=dtype)
            land=jnp.asarray(capture['flux']['xland'][:2],dtype=dtype)
            args=(state,ust,capture['dx'],land)
            texts=[jax.jit(f).lower(*args).compiler_ir('hlo').as_hlo_text() for f in (old,P.mynn_coldstart_init_columns)]
            tag=str(np.dtype(dtype))
            for name,text in zip(('old','candidate'),texts):(out/(tag+'_'+name+'.hlo')).write_text(text)
            equal=texts[0]==texts[1]
            if not equal:(out/(tag+'.diff')).write_text(''.join(difflib.unified_diff(texts[0].splitlines(True),texts[1].splitlines(True))))
            records.append({'dtype':tag,'raw_lower_hlo_equal':equal,'sha256':[hashlib.sha256(t.encode()).hexdigest()for t in texts]})
    report={'records':records,'passed':all(r['raw_lower_hlo_equal']for r in records),'scope':'default helper lower HLO; CPU no execution/forecast/GPU'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    assert report['passed'],records


if __name__=='__main__':main()
