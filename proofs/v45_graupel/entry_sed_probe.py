"""G04: causal diagnostic for WRF mp8's retained entry L_qg column gate.

WRF mp8 sets L_qg TRUE at entry (:1917) and may clear it (:3302), while
mp38 also resets it TRUE (:3269). Graupel fall is gated by ANY(L_qg).
Intervention restores pre-sedimentation qg/Ng/ppt only when no entry graupel
survived. This is a diagnostic, not an efficient production implementation.
"""
import importlib.util
import inspect
import json
import os
from pathlib import Path
import textwrap
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=Path('<USER_HOME>/wrf_gpu2_lanes/v45-graupel')/os.environ.get('GRAUPEL_EXPERIMENT','G04')

def patch(tc):
    src=textwrap.dedent(inspect.getsource(tc._thompson_source_sink_body))
    anchors={
        '    state = _reset_mp8_graupel_number(state)\n':
        '    graupel_entry = state.qg\n    state = _reset_mp8_graupel_number(state)\n',
        '    if sediment:\n':
        '    qg_before_sed, ng_before_sed = state.qg, state.Ng\n    if sediment:\n',
        '    if not (sediment and _sed_prep_fused_enabled()):\n':
        '''    if sediment and _mixed_phase_wrf_enabled():
        qg_sed_on = jnp.any((graupel_entry > R1) & (qg_before_sed > R1), axis=-1, keepdims=True)
        state = state.replace(qg=jnp.where(qg_sed_on, state.qg, qg_before_sed),
                              Ng=jnp.where(qg_sed_on, state.Ng, ng_before_sed))
        precip['graupel'] = jnp.where(jnp.squeeze(qg_sed_on, axis=-1), precip['graupel'], 0.)
    if not (sediment and _sed_prep_fused_enabled()):
'''}
    for old,new in anchors.items():
        assert src.count(old)==1,old
        src=src.replace(old,new)
    env=dict(vars(tc));exec(compile(src,tc.__file__,'exec'),env)
    return env['_thompson_source_sink_body']

def main():
    assert os.environ['JAX_PLATFORMS']=='cpu'
    from gpuwrf._fast_defaults import apply_fast_path_defaults
    apply_fast_path_defaults()
    from gpuwrf.physics import thompson_column as tc
    spec=importlib.util.spec_from_file_location('real_column_replay',ROOT/'proofs/v043/w43-conv/thompson/run_oracle.py')
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    ref.DRV=Path('<USER_HOME>/wrf_gpu2_lanes/v45-graupel/G02/driver');ref.EXE=ref.DRV/'thompson_column_driver.exe'
    source=dict(np.load('<USER_HOME>/wrf_gpu2_lanes/v45-graupel/G01/columns.npz'))
    target=list(source['names']).index('cpu_h72_y47_x46')
    density_sweep=os.environ.get('GRAUPEL_EXPERIMENT')=='G05'
    scales=np.linspace(1.16,1.20,129,dtype=np.float32) if density_sweep else np.linspace(.8,1.2,257,dtype=np.float32)
    n=len(source['names'])
    z={k:np.concatenate((v,np.repeat(v[target:target+1],len(scales),axis=0))) for k,v in source.items()}
    z['in_qg'][:]=0
    if density_sweep:
        temp=z['in_th'][n:]*z['in_pii'][n:]
        z['in_p'][n:]*=scales[:,None]
        z['in_pii'][n:]=(z['in_p'][n:]/np.float32(1e5))**np.float32(2/7)
        z['in_th'][n:]=temp/z['in_pii'][n:]
        for key in ('in_qc','in_qr','in_nr'):z[key][n:]/=scales[:,None]
    else:z['in_qr'][n:]*=scales[:,None]
    kind='density' if density_sweep else 'rain'
    z['names']=np.array([*source['names'],*(f'trace_{kind}_scale_{s:.8f}' for s in scales)])
    idx=np.arange(len(z['names']));OUT.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    wrf=ref.run_wrf(z,idx,6.,'g04')
    old=ref.run_port(z,idx,6.)
    original=tc._thompson_source_sink_body
    try:
        tc._thompson_source_sink_body=patch(tc)
        fixed=ref.run_port(z,idx,6.)
    finally:tc._thompson_source_sink_body=original
    np.savez_compressed(OUT/'probe.npz',**z,**{f'wrf_{k}':v for k,v in wrf.items()},
        **{f'old_{k}':v for k,v in old.items()},**{f'fixed_{k}':v for k,v in fixed.items()})
    rows=[]
    for i,name in enumerate(z['names']):
        a,b,c=wrf['qg'][i],old['qg'][i],fixed['qg'][i]
        rows.append(dict(name=str(name),wrf_max=float(a.max()),old_max=float(b.max()),fixed_max=float(c.max()),
            old_zero_mismatches=int(np.count_nonzero((a>0)!=(b>0))),
            fixed_zero_mismatches=int(np.count_nonzero((a>0)!=(c>0))),
            old_abs=float(np.max(np.abs(a-b))),fixed_abs=float(np.max(np.abs(a-c)))))
    summary=dict(source_commit=os.environ['GRAUPEL_COMMIT'],rows=rows,elapsed_s=time.monotonic()-start,
        old_zero_mismatches=sum(r['old_zero_mismatches'] for r in rows),
        fixed_zero_mismatches=sum(r['fixed_zero_mismatches'] for r in rows),
        old_max_abs=max(r['old_abs'] for r in rows),fixed_max_abs=max(r['fixed_abs'] for r in rows))
    (OUT/'probe.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2),flush=True)
    for row in rows:
        if row['old_zero_mismatches'] or row['fixed_zero_mismatches']:print(row,flush=True)

if __name__=='__main__':main()
