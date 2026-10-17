"""Full original CAM-ZM REAL caller comparison, CPU only, fixed float32 bound."""
from __future__ import annotations
import argparse
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
    args=p.parse_args()
    assert os.environ.get('JAX_PLATFORMS')=='cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    sys.path.insert(0,str(args.source/'src'))
    import jax
    from gpuwrf.kernels.phys_camzm_column import camzm_fresh_column
    if args.literal:
        from literal_camzm import executor
        from gpuwrf.physics.bl_camuw import ESTBL
        literal=executor(args.source)
        estbl=np.asarray(ESTBL+(-99999.,)*6,np.float64)
    assert all(d.platform=='cpu' for d in jax.devices())
    record=dict(source=str(args.source),oracle=str(args.oracle),rows=[],gpu_executed=False,
        gate='REAL32 outputs: finitefirst;128eps32*columnpeak;indices EXACT; original source events disclosed')
    saved={};calls={}
    for row in json.loads(args.oracle.read_text())['records']:
        fields,scalars=parse(row['stdout'])
        names=('T_PHY','QV','P','P8W','PI_PHY','U_PHY','V_PHY','Z','Z_AT_W','TKE_PBL','TH','QC','QI')
        inputs=tuple(np.asarray(fields['IN_'+key],np.float32) for key in names)
        if len(inputs[3])==len(inputs[0]):
            # Older mass-only driver dumps omit the final interface. The
            # independently captured original CONVR inputs retain all faces.
            core,_=parse(row['stderr'])
            assert 'ZM_CORE_IN_paph' in core and 'ZM_CORE_IN_zi' in core
            faces=np.asarray(core['ZM_CORE_IN_paph'],np.float32)[::-1]
            heights=np.asarray(core['ZM_CORE_IN_zi'],np.float32)[::-1]
            np.testing.assert_array_equal(faces[:-1],inputs[3])
            np.testing.assert_array_equal(heights[:-1],inputs[8])  # original fixture HT=0
            parts=list(inputs);parts[3]=faces;parts[8]=heights;inputs=tuple(parts)
        n=len(inputs[0]);arrays=inputs+(np.asarray(fields.get('IN_CLDFRA',np.zeros(n)),np.float32),np.asarray(fields['IN_DZ8W'],np.float32))
        ctrl=np.asarray([54.,9000.,0.,1.,100800. if row['case']==3 else 100000.,0.,1500.,6.,150.,1e-4,.3,1.,1.,inputs[0][0],.3],np.float32)
        controls=('DT','DX','CUDT','ITIMESTEP','PSFC','HT','PBLH','KPBL','HFX','QFX','UST','MAVAIL','XLAND','TSK','TPERT_CAMUWPBL')
        if 'CTRL_DT' in scalars:
            ctrl=np.asarray([scalars['CTRL_'+key] for key in controls],np.float32)
        pbl=row['pbl'];sf=row['surface']
        if pbl not in calls and not args.literal:
            def one(*a,_pbl=pbl,_sf=sf):
                c=a[15]
                return camzm_fresh_column(*a[:15],dt=c[0],dx=c[1],cudt=c[2],step=c[3],psfc=c[4],terrain=c[5],
                    pblh=c[6],kpbl=c[7],hfx=c[8],qfx=c[9],ust=c[10],mavail=c[11],xland=c[12],tsk=c[13],
                    tpert_camuw=c[14],pbl=_pbl,sfclay=_sf,interpret=True)
            calls[pbl]=jax.jit(one)
        started=time.monotonic()
        if args.literal:
            padded=tuple(np.pad(a,(1,64-a.shape[0]-1)) for a in arrays)
            with np.errstate(all='ignore'):
                raw=literal(padded,ctrl,estbl,kx=n,pbl=pbl)
            actual={key:np.asarray(value)[1:n+2 if key=='CMFMC' else n+1] if np.ndim(value) else np.asarray(value)
                    for key,value in raw.items()}
        else:
            actual={key:np.asarray(value) for key,value in jax.block_until_ready(calls[pbl](*arrays,ctrl)).items()}
        required=('RUCUTEN','RVCUTEN','RTHCUTEN','RQVCUTEN','RQCCUTEN','RQICUTEN','RQCNCUTEN','RQINCUTEN',
                  'DLF_OUT','MU_OUT','MD_OUT','ZMDT','ZMDQ','ZMDICE','ZMDLIQ','CMFMC','ZMFLXPRC','ZMFLXSNW',
                  'RAINCV','PRATEC','CAPE_OUT','CUTOP','CUBOT','PCONVB','PCONVT','RLIQ_OUT')
        assert all(key in actual and (key in fields or key in scalars) for key in required)
        result=dict(case=row['case'],pbl=pbl,elapsed_s=time.monotonic()-started,fields={},events={})
        for key,value in actual.items():
            if key in ('ENTROPY_FAILED','DLF_CLIPPED','VAPOR_FLOORED'):
                result['events'][key]=int(value);continue
            truth=fields.get(key,scalars.get(key))
            if truth is None:continue
            truth=np.asarray(truth,np.float32)
            # Original history driver dumps its mass slice of CMFMC (not top W).
            if key=='CMFMC':value=value[:len(truth)]
            assert value.shape==truth.shape,(key,value.shape,truth.shape)
            assert np.isfinite(truth).all() and np.isfinite(value).all(),key
            error=float(np.max(np.abs(value-truth)))
            limit=float(128*np.finfo(np.float32).eps*np.max(np.abs(truth))) if key not in ('CUTOP','CUBOT') else 0.
            result['fields'][key]=dict(max_abs=error,limit=limit,pass_=error<=limit,
                byte_values=int(np.count_nonzero(value.view(np.uint32)==truth.view(np.uint32))))
            saved[f"case{row['case']}_pbl{pbl}_{key}"]=value
        assert result['events']['ENTROPY_FAILED']==0
        record['rows'].append(result)
        args.output.write_text(json.dumps(record,indent=2)+'\n');np.savez(args.output.with_suffix('.npz'),**saved)
        print(json.dumps(result),flush=True)


if __name__=='__main__':main()
