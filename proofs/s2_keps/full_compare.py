"""Full-column WRF comparison, fixed REAL32 forward-error gate.

State/diagnostic fields: 16 ulps of the reference field scale. Tendencies:
16 input-state ulps / dt (EEPS temperature additionally / min(exner)). Flux
diagnostics propagate the same state-rounding bound through sum(rho*dz)/dt.
This treats cancellation in (new-old)/dt explicitly; no post-result tuning.
Both candidate and reference must be finite. Integer KPBL is exact.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import jax
import numpy as np
from gpuwrf.physics.pbl_keps import keps_column


def inputs(rec):
    i=rec['input'];n=rec['n'];s=rec['surface']
    a=lambda k:np.asarray(i[k][:n],np.float32)
    common=dict(u=a('u'),v=a('v'),theta=a('theta'),qv=a('qv'),qc=a('qc'),rho=a('rho'),dz=a('dz'),
                exner=a('exner'),raten=a('raten'),tke=a('tke'),epsilon=a('epsilon'),
                tke_adv=a('tke_adv'),epsilon_adv=a('epsilon_adv'),dt=np.float32(rec['dt']),
                step=np.int32(rec['step']),hfx=np.float32(s['hfx']),qfx=np.float32(s['qfx']),ustar=np.float32(s['ust']))
    if rec['scheme']==17:
        common.update(tpe=a('tpe'),prandtl=a('prandtl'),tpe_adv=a('tpe_adv'),
                      p8w=np.asarray(i['p8w'][:n+1],np.float32),roughness=np.float32(s['znt']),
                      br=np.float32(s['br']),fm=np.float32(s['fm']),fhh=np.float32(s['fhh']))
    else:
        common.update(temperature=a('temperature'),pressure=a('pressure'),qi=a('qi'),qr=a('qr'),qs=a('qs'),qg=a('qg'),
                      rmol=np.float32(s['rmol']),wspd=np.float32(s['wspd']),tsk=np.float32(s['tsk']),
                      qsfc=np.float32(s['qsfc']),psfc=np.float32(s['psfc']),dx=np.float32(rec['dx']))
    return common


def compare(got,rec):
    errors={};n=rec['n'];i=rec['input'];dt=rec['dt']
    parent={'rublten':'u','rvblten':'v','rthblten':'theta','rqvblten':'qv','rqcblten':'qc','rqiblten':'qi',
            'wu':'u','wv':'v','wt':'theta'}
    for name,values in rec['output'].items():
        if name=='kpbl' and rec['scheme']==17: continue  # KEPS does not return KPBL.
        if rec['scheme']==16 and name in ('tpe','prandtl','tpe_adv'):
            # EEPS never owns these KEPS-only buffers; verify them separately at caller wiring.
            continue
        assert name in got,name
        candidate=np.atleast_1d(np.asarray(got[name]));ref=np.asarray(values,np.float32)
        assert candidate.shape==ref.shape,(name,candidate.shape,ref.shape)
        assert np.isfinite(candidate).all() and np.isfinite(ref).all(),(rec['name'],rec['step'],name,'nonfinite')
        if name=='kpbl':
            assert np.array_equal(candidate,ref),(rec['name'],rec['step'],name,candidate,ref)
            continue
        assert candidate.dtype==np.float32,(name,candidate.dtype)
        scale=np.float32(np.max(np.abs(ref)));tol=16*float(np.spacing(scale)) if scale else 0.
        if name in parent:
            old=np.asarray(i[parent[name]][:n],np.float32)
            # Include the returned mean state when its exponent differs from the incoming one.
            maximum=max(float(np.max(np.abs(old))),float(np.max(np.abs(old+dt*ref))) if name.startswith('r') else 0.)
            quantum=16*float(np.spacing(np.float32(maximum))) if maximum else 0.
            tol=max(tol,quantum/dt)
            if name=='rthblten' and rec['scheme']==16:
                tscale=np.float32(np.max(np.abs(np.asarray(i['temperature'][:n],np.float32))))
                tol=max(tol,16*float(np.spacing(tscale))/dt/float(np.min(i['exner'][:n])))
            if name in ('wu','wv','wt'):
                tol=max(tol,quantum/dt*float(np.sum(np.asarray(i['rho'][:n])*np.asarray(i['dz'][:n]))))
        error=float(np.max(np.abs(candidate.astype(float)-ref.astype(float))))
        ratio=error/tol if tol else (0. if error==0. else float('inf'))
        if ratio>1.: errors[name]=dict(error=error,tolerance=tol,ratio=ratio)
    return errors


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('fixture',type=Path);ap.add_argument('receipt',type=Path)
    ap.add_argument('--scheme',type=int,choices=(16,17))
    args=ap.parse_args();assert os.environ.get('JAX_PLATFORMS')=='cpu';assert jax.default_backend()=='cpu'
    fixture=json.loads(args.fixture.read_text());assert fixture['init']=='zero'
    functions={17:keps_column}
    if args.scheme!=17:
        from gpuwrf.physics.pbl_eeps import eeps_column
        functions[16]=eeps_column
    rows=[r for r in fixture['records'] if args.scheme is None or r['scheme']==args.scheme]
    failures=[];count=0
    for scheme in sorted({r['scheme'] for r in rows}):
        for n in sorted({r['n'] for r in rows if r['scheme']==scheme}):
            selected=[r for r in rows if r['scheme']==scheme and r['n']==n]
            batch={k:np.stack([inputs(r)[k] for r in selected]) for k in inputs(selected[0])}
            result=jax.jit(jax.vmap(lambda kw:functions[scheme](**kw)))(batch)
            for idx,rec in enumerate(selected):
                err=compare({k:v[idx] for k,v in result.items()},rec)
                count+=1
                if err:failures.append(dict(name=rec['name'],step=rec['step'],fields=err))
    receipt=dict(status='FAIL' if failures else 'PASS',records=count,failures=failures,
                 backend='cpu',xla_flags=os.environ.get('XLA_FLAGS',''),
                 scope='full-column zero-initialized pristine WRF; not coupled/GPU qualification')
    args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status=receipt['status'],records=count,failed_records=len(failures))))
    assert not failures,failures[:2]


if __name__=='__main__':main()
