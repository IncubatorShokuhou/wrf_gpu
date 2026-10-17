"""Independent full-original MSKF comparison, with literal or compiled candidate.

Literal mode disables only CMT in the candidate and qualifies all REAL
thermal/moisture/cloud diagnostics against the original CMT-ON run. CMT
does not feed back into those original outputs; its DOUBLE island is
qualified separately on captures of the complete original caller.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from compare_kfcup_parcel import parse


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--oracle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--literal',action='store_true')
    p.add_argument('--literal-cmt',action='store_true')
    p.add_argument('--calm-reporting',action='store_true',help='separate documented exactly-calm upstream regression')
    p.add_argument('--no-cmt',action='store_true')
    p.add_argument('--limit',type=int)
    args=p.parse_args()
    assert os.environ.get('JAX_PLATFORMS')=='cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    oracle=json.loads(args.oracle.read_text())
    if args.literal:
        from literal_kfcup import executor,intrinsic
        fn,tables=executor(args.source,scheme='mskf')
    else:
        sys.path.insert(0,str(args.source/'src'))
        import jax
        import jax.numpy as jnp
        from gpuwrf.kernels.phys_mskf_column import mskf_fresh_column
        assert all(d.platform=='cpu' for d in jax.devices())
        if args.calm_reporting:
            from types import SimpleNamespace
            from gpuwrf.diagnostics.census import DeviceCensus,GUARDS,EVENTS,WORK,count_guard
            from gpuwrf.runtime.finite_state_guard import NonFiniteStateError,assert_state_finite_at_boundary
        fn=jax.jit(lambda *a:mskf_fresh_column(*a[:9],dt=a[9][0],dx=a[9][1],pblh=a[9][2],
            hfx=a[9][3],ust=a[9][4],zol=a[9][5],cudt=0.,cmt=not args.no_cmt,interpret=True))
    aliases={'CLDFRA_DP_KF':'CLDFRA_DP','CLDFRA_SH_KF':'CLDFRA_SH','DUDT':'RUCUTEN','DVDT':'RVCUTEN'}
    result=dict(source=str(args.source),oracle=str(args.oracle),literal=args.literal,
                cmt=args.literal_cmt if args.literal else not args.no_cmt,rows=[],gpu_executed=False,
                source_sha256=hashlib.sha256((args.source/'src/gpuwrf/kernels/phys_mskf_column.py').read_bytes()).hexdigest())
    save={}
    for row in oracle['records'][:args.limit]:
        fields,scalars=parse(row['stdout'])
        profiles=tuple(np.asarray(fields[key],np.float32) for key in ('T','QV','P','DZ','RHO','W0AVG','U','V'))
        n=len(profiles[0])
        control=np.asarray([scalars.get('CTRL_DT',54.),scalars.get('CTRL_DX',9000.),
            scalars.get('CTRL_PBLH',1000.),scalars.get('CTRL_HFX',150.),scalars.get('CTRL_UST',.3),
            scalars.get('CTRL_ZOL',-.1)],np.float32)
        started=time.monotonic()
        with np.errstate(all='ignore'):
            if args.literal:
                inputs=tuple(np.pad(a,(1,64-n-1)) for a in profiles)
                actual=fn(*inputs,control[0],control[1],n,
                    tables['THE0K'],tables['TTAB'],tables['QSTAB'],tables['ALU'],False,True,True,
                    np.float32(0.),control[2],control[3],control[4],control[5],
                    np.float32(0.),args.literal_cmt,cmt_tables=fn.harmonics)
                actual={key:np.asarray(value)[1:n+1] if np.ndim(value) else np.asarray(value)
                        for key,value in actual.items()}
                pi=np.asarray(fields['PI'],np.float32) if 'PI' in fields else intrinsic('powf',
                    profiles[2]/np.float32(1e5),np.float32(287.)/np.float32(1004.5))
                actual['RTHCUTEN']=actual['DTDT']/pi
            else:
                assert 'PI' in fields,'original independent PI capture is required for compiled mode'
                actual={key:np.asarray(value) for key,value in jax.block_until_ready(
                    fn(*profiles,np.asarray(fields['PI'],np.float32),control)).items()}
        calm=(args.calm_reporting and row['background_wind']==0.
              and not np.isfinite(np.asarray(fields['RQVCUTEN'])).all())
        reporting=None
        if calm:
            assert not args.literal,'native full-CPS execution is required for this reporting qualification'
            qv=jnp.asarray((profiles[1]+np.float32(54.)*actual['RQVCUTEN']).reshape(-1,1,1))
            before=np.asarray(qv).view(np.uint32).copy()
            expected=int(np.count_nonzero(~np.isfinite(actual['RQVCUTEN'])))
            assert expected>0,'calm singularity was repaired or not exercised'
            with jax.enable_x64(True):
                census=DeviceCensus(jnp.zeros((len(GUARDS),len(EVENTS)),jnp.uint64),jnp.zeros(len(WORK),jnp.uint64))
                census=count_guard(census,'boundary.qv',qv,rejected=jnp.zeros_like(qv,dtype=bool))
            count=int(census.guards[GUARDS.index('boundary.qv'),EVENTS.index('nonfinite')])
            repaired=int(census.guards[GUARDS.index('boundary.qv'),EVENTS.index('repaired')])
            assert count==expected and repaired==0
            raised=False
            try:
                assert_state_finite_at_boundary(SimpleNamespace(qv=qv),domain='mskf-original-calm',step=1,enabled=True)
            except NonFiniteStateError:
                raised=True
            assert raised,'finite-state guard did not report the native upstream singularity'
            np.testing.assert_array_equal(np.asarray(qv).view(np.uint32),before)
            reporting=dict(nonfinite=count,repaired=repaired,guard_raised=raised,bits_unchanged=True)
        differences={};compared=0;violations={};nonfinite=0
        for key,value in actual.items():
            if key in ('DUDT','DVDT') and ((args.literal and not args.literal_cmt) or args.no_cmt):continue
            name=aliases.get(key,key)
            truth=fields.get(name,scalars.get(name))
            if truth is None:continue
            truth=np.asarray(truth,np.float32);value=np.asarray(value,np.float32)
            assert value.shape==truth.shape,(key,value.shape,truth.shape)
            finite=np.isfinite(truth)&np.isfinite(value)
            if not calm:
                assert np.isfinite(truth).all(),('nonfinite primary original',key,row)
                assert np.isfinite(value).all(),('nonfinite primary candidate',key,row)
            nonfinite+=int(np.count_nonzero(~np.isfinite(value)))
            changed=int(np.count_nonzero(value.view(np.uint32)!=truth.view(np.uint32)))
            compared+=value.size
            if changed:
                differences[name]=dict(changed=changed,max_abs=float(np.max(np.abs(value[finite]-truth[finite]))) if finite.any() else None)
            limit=max(1e-10 if 'CUTEN' in name else 1e-7,float(np.max(np.abs(truth[finite])))*128*np.finfo(np.float32).eps) if finite.any() else 0.
            bad=int(np.count_nonzero(finite&(np.abs(value-truth)>limit)))
            if bad:violations[name]=dict(count=bad,limit=float(limit))
            save[f"w{row['background_wind']}_case{row['case']}_{key}"]=value
        out=dict(case=row['case'],wind=row['background_wind'],elapsed_s=time.monotonic()-started,
                 compared=compared,differences=differences,violations=violations,candidate_nonfinite=nonfinite,
                 regime='expected upstream calm singularity' if calm else 'finite supported primary',reporting=reporting)
        result['rows'].append(out)
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        np.savez(args.output.with_suffix('.npz'),**save)
        print(json.dumps(out,allow_nan=False),flush=True)


if __name__=='__main__':main()
