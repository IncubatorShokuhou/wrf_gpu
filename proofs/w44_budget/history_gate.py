"""Pre-registered MYNN history gates, including approved E100 Dtke floor.

PBL tendencies (critic cr-w44-budget F1, pre-registered 2026-10-24 before any
re-score): the same relative rule as the other terms, 5 % of the reference
column maximum, paired with a per-column REAL floor measured by the SAME
original-only 25-member one-ULP protocol as the approved Dtke floor
(max over levels of max_member - min_member, nominal included). No floor
means zero floor (strictest). The former absolute 1e-3 bound was vacuous.

Amendment (cr-w44-budget VERDICT_DELTA 00:37Z, manager 00:35Z), pre-registered
before any production GPU array: RQVBLTEN/RQCBLTEN/RQIBLTEN additionally get a
per-level REAL quantization floor 4*spacing_f32(M)/dt with
M = max(|x_in|, |x_in + r_ref*dt|), x_in the term's own admitted step input
mixing ratio and r_ref the ORIGINAL reference tendency (never the candidate);
0 where M < 2**-126. The GPU mini showed exactly 1 local ulp(qi)/dt; the CPU
twin association error reaches 3 local ulp(qv)/dt, hence k=4. The column-max
form was rejected (vacuous for RQIBLTEN). RTH/RU/RV are unchanged.
"""
import numpy as np

TKE=('qSHEAR','qBUOY','qDISS','qWT','Dtke')
EDMF=('edmf_a','edmf_w','edmf_qt','edmf_thl','edmf_ent','edmf_qc','sub_thl3D','sub_sqv3D','det_thl3D','det_sqv3D')
PBL=('RTHBLTEN','RQVBLTEN','RQCBLTEN','RQIBLTEN','RUBLTEN','RVBLTEN')
QUANT={'RQVBLTEN':'qv','RQCBLTEN':'qc','RQIBLTEN':'qi'}
QUANT_K=4


def quantization_floor(x_in,r_ref,dt,k=QUANT_K):
    """Per-level k*spacing_f32(max(|x|,|x+r*dt|))/dt, 0 below the smallest normal."""
    x=np.asarray(x_in,np.float64);r=np.asarray(r_ref,np.float64)
    m=np.maximum(np.abs(x),np.abs(x+r*dt))
    floor=k*np.spacing(m.astype(np.float32)).astype(np.float64)/dt
    return np.where(m<2.0**-126,0.,floor)


def evaluate(actual,reference,dtke_floor,*,edmf=True,pbl_floor=None,inputs=None,dt=None):
    assert set(actual)==set(reference)==set(TKE+EDMF+PBL),'missing/unexpected term'
    floor=np.asarray(dtke_floor,np.float64)
    assert floor.ndim==1 and np.isfinite(floor).all() and np.all(floor>=0),'invalid original floor'
    if pbl_floor is None:pbl_floor={name:np.zeros_like(floor) for name in PBL}
    assert set(pbl_floor)==set(PBL),'missing/unexpected PBL floor'
    pbl_floor={name:np.asarray(pbl_floor[name],np.float64) for name in PBL}
    for name,value in pbl_floor.items():
        assert value.shape==floor.shape and np.isfinite(value).all() and np.all(value>=0),f'invalid original PBL floor {name}'
    quant={}
    if inputs is not None:
        assert dt is not None and np.isfinite(dt) and dt>0,'invalid dt'
        assert set(inputs)==set(QUANT.values()),'missing/unexpected quantization input'
        for name,key in QUANT.items():
            x=np.asarray(inputs[key])
            assert x.shape==np.asarray(reference[name]).shape and np.isfinite(x).all(),f'invalid quantization input {key}'
            quant[name]=quantization_floor(x,reference[name],dt)
    # Reject either side before max/ratio/zero tests (E200).
    for name in actual:
        a,r=np.asarray(actual[name]),np.asarray(reference[name])
        assert a.shape==r.shape and a.ndim==2 and len(a)==len(floor),name
        assert np.isfinite(a).all() and np.isfinite(r).all(),f'nonfinite {name}'
    fields={};bounds={}
    for name in actual:
        a,r=np.asarray(actual[name],np.float64),np.asarray(reference[name],np.float64)
        d=np.abs(a-r)
        if name=='Dtke':
            bound=np.maximum(.05*np.abs(r),floor[:,None])
        elif name in PBL:
            relative=.05*np.max(np.abs(r),axis=-1)[:,None]*np.ones_like(r)
            ensemble=pbl_floor[name][:,None]*np.ones_like(r)
            q=quant.get(name,np.zeros_like(r))
            bound=np.maximum(np.maximum(relative,ensemble),q)
            binding=dict(relative=int(np.sum((relative>=ensemble)&(relative>=q))),
                         ensemble=int(np.sum((ensemble>relative)&(ensemble>=q))),
                         quantization=int(np.sum((q>relative)&(q>ensemble))))
        else:
            bound=np.broadcast_to(.05*np.max(np.abs(r),axis=-1)[:,None],r.shape)
        passed=bool(np.all(d<=bound))
        if name in EDMF[6:] or (not edmf and name in EDMF):
            passed=passed and not np.count_nonzero(a) and not np.count_nonzero(r)
        if name in TKE:
            passed=passed and not np.count_nonzero(a[:,-1]) and not np.count_nonzero(r[:,-1])
        columns=[dict(column=i,max_abs=float(d[i].max()),reference_max_abs=float(np.abs(r[i]).max()),
                      floor=float(floor[i]) if name=='Dtke' else float(pbl_floor[name][i]) if name in PBL else None,
                      max_excess=float(np.maximum(d[i]-bound[i],0).max()),passed=bool(np.all(d[i]<=bound[i])))
                 for i in range(len(a))]
        fields[name]=dict(max_abs=float(d.max()),rms=float(np.sqrt(np.mean(d*d))),passed=bool(passed),columns=columns)
        if name in PBL:fields[name]['binding_points']=binding
        bounds[name]=bound
    residual=lambda v:np.asarray(v['Dtke'],float)-(np.asarray(v['qSHEAR'],float)+np.asarray(v['qBUOY'],float)+np.asarray(v['qWT'],float)-np.asarray(v['qDISS'],float))
    d=np.abs(residual(actual)-residual(reference));bound=sum(bounds[n] for n in TKE)
    closure=dict(max_abs=float(d.max()),reference_max_abs=float(np.abs(residual(reference)).max()),
                 passed=bool(np.all(d<=bound)),note='compare original residual; never require artificial zero closure')
    return dict(terms=fields,closure=closure,passed=bool(all(v['passed'] for v in fields.values()) and closure['passed']))
