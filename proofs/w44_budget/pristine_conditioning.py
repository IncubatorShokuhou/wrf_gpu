"""Original-only one-ULP input conditioning; no candidate-dependent bound.

Preregistered variants: twenty fixed random +/-1-ULP patterns on nonzero
continuous operands, two alternating theta patterns, and QKE +/-1 ULP. Exact
zeros/categorical XLAND remain fixed. All columns of both WN3 day/night sets
are measured (24 perturbations plus the nominal = 25 members). The approved
floor is max over levels of member-to-member Dtke spread, per column. This
does not relax any carried-byte rule.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure_original(O,exe,work,cols,nominal_dtke,dt,dx,nominal_pbl=None):
    """The same fixed 25-member protocol on any actual same-column inputs.

    ``nominal_pbl`` (NAME -> nominal original array): the same members also give
    the per-column PBL-tendency floor, max over levels of member spread
    (critic F1, pre-registered); returned as conditioning['pbl_floor'].
    """
    nominal=np.asarray(nominal_dtke,np.float64)
    low=nominal.copy();high=nominal.copy();envelope=np.zeros_like(nominal)
    pbl={k:np.asarray(v,np.float64) for k,v in (nominal_pbl or {}).items()}
    pbl_low={k:v.copy() for k,v in pbl.items()};pbl_high={k:v.copy() for k,v in pbl.items()}
    rng=np.random.default_rng(20261024);rows=[]
    for variant in range(24):
        inputs={k:v.copy() for k,v in cols.items()}
        for key,value in inputs.items():
            if key=='xland' or (variant in (20,21) and key!='th') or (variant in (22,23) and key!='qke'):
                continue
            if variant<20:sign=rng.integers(0,2,size=value.shape,dtype=np.int8)*2-1
            elif variant in (20,21):sign=np.broadcast_to((np.arange(value.shape[-1])%2)*2-1,value.shape)*(1 if variant==20 else -1)
            else:sign=np.ones_like(value)*(1 if variant==22 else -1)
            moved=np.nextafter(value,np.where(sign>0,np.float32(np.inf),np.float32(-np.inf)),dtype=np.float32)
            inputs[key]=np.where(value==0,value,moved)
            assert np.isfinite(inputs[key]).all()
        values,_=O.run(exe,work/str(variant),inputs,dt,dx)
        assert all(np.isfinite(v).all() for v in values.values())
        value=np.asarray(values['dqke'],np.float64);error=np.abs(value-nominal)
        low=np.minimum(low,value);high=np.maximum(high,value);envelope=np.maximum(envelope,error)
        for name in pbl:
            member=np.asarray(values[name.lower()],np.float64)
            pbl_low[name]=np.minimum(pbl_low[name],member);pbl_high[name]=np.maximum(pbl_high[name],member)
        rows.append(dict(variant=variant,max_abs_Dtke_change=float(error.max())))
    return (high-low).max(-1),dict(members=25,seed=20261024,variants=rows,
        max_abs_Dtke_change=float(envelope.max()),max_Dtke_spread=float((high-low).max()),
        columns=[dict(column=i,floor=float((high-low)[i].max())) for i in range(len(nominal))],
        pbl_floor={name:(pbl_high[name]-pbl_low[name]).max(-1).tolist() for name in pbl})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    assert os.environ.get('JAX_PLATFORMS')=='cpu' and not Path('/tmp/wrf_gpu2_quiet').exists()
    helper=args.source/'tests/v025/t_cfix_mynn/f20_oracle.py'
    spec=importlib.util.spec_from_file_location('original_f20',helper);O=importlib.util.module_from_spec(spec);spec.loader.exec_module(O)
    extra=('dqke','qwt','qshear','qbuoy','qdiss','edmf_qt','edmf_thl','edmf_ent','edmf_qc','sub_thl3d','sub_sqv3d','det_thl3d','det_sqv3d')
    O.CALLER=(args.reference/'export_on/caller.f90').read_text()
    O.OUT3=(*O.OUT3,*extra)
    args.out.mkdir(parents=True,exist_ok=False)
    exe=O.build(args.out/'original')
    old=json.loads((args.reference/'export_on/provenance.json').read_text());new=json.loads((args.out/'original/provenance.json').read_text())
    assert old['caller_sha256']==new['caller_sha256']
    assert sorted(old['objects'].values())==sorted(new['objects'].values()),'original objects changed'
    fixture=helper.parent/'fixtures/f20_columns.npz';meta=json.loads(fixture.with_suffix('.json').read_text())
    record=dict(schema='wrf_gpu2.w44_budget.original_ulp.v2',scope=__doc__,seed=20261024,members=25,
        runner_sha256=sha(__file__),floor_definition='max_k(max_member Dtke(k) - min_member Dtke(k)), including nominal',
        fixture_sha256=sha(fixture),caller_sha256=new['caller_sha256'],objects=new['objects'],cases={})
    with np.load(fixture) as f:
        for name in ('wn3_d03_day','wn3_d03_night'):
            cols={k.split('__',1)[1]:f[k].copy() for k in f.files if k.startswith(name+'__')}
            cols['qsfc']=cols['qv'][:,0].copy();cols['znt']=np.full_like(cols['tsk'],np.float32(.1))
            with np.load(args.reference/(name+'.npz')) as ref:baseline={k:ref[k].copy() for k in ref.files}
            envelope=np.zeros_like(baseline['dqke'],dtype=np.float64)
            low=baseline['dqke'].astype(np.float64).copy();high=low.copy()
            rows=[]
            rng=np.random.default_rng(20261024)
            for variant in range(24):
                inputs={k:v.copy() for k,v in cols.items()}
                for key,value in inputs.items():
                    if key=='xland' or (variant in (20,21) and key!='th') or (variant in (22,23) and key!='qke'):
                        continue
                    if variant<20:
                        sign=rng.integers(0,2,size=value.shape,dtype=np.int8)*2-1
                    elif variant in (20,21):
                        sign=np.broadcast_to((np.arange(value.shape[-1])%2)*2-1,value.shape)*(1 if variant==20 else -1)
                    else:
                        sign=np.ones_like(value)*(1 if variant==22 else -1)
                    moved=np.nextafter(value,np.where(sign>0,np.float32(np.inf),np.float32(-np.inf)),dtype=np.float32)
                    inputs[key]=np.where(value==0,value,moved)
                    assert np.isfinite(inputs[key]).all()
                result,_=O.run(exe,args.out/'runs'/name/str(variant),inputs,meta[name]['dt'],meta[name]['dx'])
                assert all(np.isfinite(v).all() for v in result.values())
                error=np.abs(result['dqke'].astype(np.float64)-baseline['dqke'].astype(np.float64))
                envelope=np.maximum(envelope,error)
                low=np.minimum(low,result['dqke'].astype(np.float64));high=np.maximum(high,result['dqke'].astype(np.float64))
                rows.append(dict(variant=variant,max_abs_Dtke_change=float(error.max())))
            spread=high-low
            np.savez(args.out/(name+'.npz'),Dtke_envelope=envelope,Dtke_profile_envelope=envelope.max(-1),
                     Dtke_member_spread=spread,Dtke_profile_floor=spread.max(-1))
            record['cases'][name]=dict(columns=len(cols['th']),variants=rows,
                Dtke_profile_envelope=envelope.max(-1).tolist(),Dtke_profile_floor=spread.max(-1).tolist(),
                max_abs_Dtke_change=float(envelope.max()),max_Dtke_spread=float(spread.max()))
            (args.out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    record['verdict']='ORIGINAL_CONDITIONING_MEASURED_NOT_GATE_AMENDMENT'
    (args.out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    assert not any(k=='jax' or k.startswith('gpuwrf') for k in sys.modules)
    print(json.dumps({k:v['max_abs_Dtke_change'] for k,v in record['cases'].items()}))


if __name__=='__main__':main()
