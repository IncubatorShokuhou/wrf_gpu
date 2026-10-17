"""Finite-first, independent-WRF numeric gates for TEMF CPU reference work."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import jax
import numpy as np

from gpuwrf.physics.bl_temf import temf_column
from gpuwrf.physics.sf_temf import temf_surface


def numeric_gate(got, expected, *, state_scale=None, dt=None):
    got, expected = np.asarray(got), np.asarray(expected)
    if got.shape != expected.shape or not np.isfinite(got).all() or not np.isfinite(expected).all():
        raise AssertionError('TEMF oracle/candidate shape or nonfinite failure')
    scale = np.float32(np.max(np.abs(expected)))
    ulp = float(np.spacing(scale)) if scale else 0.
    tol = 8.*ulp
    if state_scale is not None:
        tol = max(tol, 8.*float(np.spacing(np.float32(state_scale)))/float(dt))
    error = float(np.max(np.abs(got.astype(float)-expected.astype(float))))
    return error <= tol, error, tol, error/ulp if ulp else 0. if error==0 else float('inf')


def compare(path, kind):
    data = load(path)
    records = data['records']
    fn = temf_surface if kind=='surface' else temf_column
    states = dict(rublten='u',rvblten='v',rthblten='theta',rqvblten='qv',rqcblten='qc')
    integer = lambda k,v: np.asarray(v, np.int32 if k=='kpbl' else np.float32)
    inputs = [{k: {ck:integer(ck,cv) for ck,cv in v.items()} if k=='carry' else integer(k,v)
               for k,v in rec['inputs'].items()} for rec in records]
    batch = jax.tree.map(lambda *xs: np.stack(xs), *inputs)
    t = time.perf_counter()
    executable = jax.jit(jax.vmap(lambda kw: fn(**kw))).lower(batch).compile()
    compile_s = time.perf_counter()-t
    out = jax.tree.map(np.asarray, executable(batch))
    failures=[]; worst={}; compared=0; nonzero={}; byte_mismatches={}
    for i, rec in enumerate(records):
        assert out.keys()==rec['outputs'].keys(), (out.keys(),rec['outputs'].keys())
        for k, got in out.items():
            exp = np.asarray(rec['outputs'][k], np.float32)
            scale = float(np.max(np.abs(inputs[i][states[k]]))) if kind=='pbl' and k in states else None
            ok,err,tol,ulps=numeric_gate(got[i], exp, state_scale=scale, dt=rec['inputs'].get('dt'))
            if k=='kpbl':
                ok = bool(np.array_equal(got[i],exp))
            dtype=np.int32 if k=='kpbl' else np.float32
            actual_bytes=np.atleast_1d(got[i]).astype(dtype).view(np.uint32)
            reference_bytes=np.atleast_1d(rec['outputs'][k]).astype(dtype).view(np.uint32)
            byte_mismatches[k]=byte_mismatches.get(k,0)+int(np.count_nonzero(actual_bytes!=reference_bytes))
            if not ok:
                failures.append(dict(field=k,name=rec['name'],step=rec['step'],error=err,tol=tol,ulps=ulps))
            worst[k]=max(worst.get(k,0.),ulps)
            compared += int(exp.size)
            nonzero[k]=nonzero.get(k,0)+int(np.count_nonzero(exp))
    memory = executable.memory_analysis()
    return dict(backend=jax.devices()[0].platform, source_file=str(Path(fn.__code__.co_filename).resolve()),
                records=len(records), scalar_comparisons=compared, compile_s=compile_s,
                cpu_compiled_memory=dict(argument_bytes=memory.argument_size_in_bytes,
                                         output_bytes=memory.output_size_in_bytes,
                                         temp_bytes=memory.temp_size_in_bytes),
                worst_ulp=worst,nonzero=nonzero,byte_mismatches=byte_mismatches,failures=failures)


def load(path):
    path=Path(path)
    if path.suffix!='.npz':
        return json.loads(path.read_text())
    import hashlib
    d=json.loads(path.with_suffix('.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==d['fixture_sha256'], 'TEMF fixture hash differs'
    with np.load(path,allow_pickle=False) as arrays:
        records=[dict(name=str(n),step=int(s),inputs={'carry':{}},outputs={}) for n,s in zip(arrays['names'],arrays['steps'])]
        for key in arrays.files:
            if '__' not in key:
                continue
            parts=key.split('__')
            for i,record in enumerate(records):
                obj=record[parts[0]]
                if len(parts)==3:
                    obj=obj[parts[1]]
                obj[parts[-1]]=arrays[key][i].tolist()
        if 'inputs__carry__te_temf' not in arrays:
            for record in records:
                record['inputs'].pop('carry')
    d['records']=records
    return d


if __name__=='__main__':
    result=compare(sys.argv[1],sys.argv[2])
    Path(sys.argv[3]).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    raise SystemExit(bool(result['failures']))
