"""Pristine-WRF legacy19/21 column and component qualification gates."""
from pathlib import Path
import json
import os
import time

import jax
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)
DATA = Path(__file__).resolve().parents[2] / 'proofs/s2_mpmore/nssl1mom/fixtures'


def load(variant, mode, case):
    return json.loads((DATA/f'v{variant}_{mode}_{case}.json').read_text())


def canonical_stage(d, stage, key='an'):
    """Index remap only: original masses1..8,19volume9/CCNA10 ->16/18."""
    from gpuwrf.physics.nssl2mom.indices import LVH, LCCNA, NA
    native = np.asarray(d['stages'][stage][key])
    out = np.zeros((NA+1,)+native.shape[1:], native.dtype)
    for il in range(1, min(len(native), 8)+1):
        out[il] = native[il-1]
    if len(native) == 10:
        out[LVH] = native[8]
        out[LCCNA] = native[9]
    else:
        assert len(native) == 7
    return out


FUN = {}
COST = {}


@pytest.mark.parametrize('variant',[19,21])
def test_original_same_input_fma_control_seal(variant):
    import hashlib
    p=DATA.parent/'fma';manifest=json.loads((p/'manifest.json').read_text())
    assert manifest['source_sha256']=='29f42e76fbeca027a3ffb2ac805b148aaa1425c7f99ed1d8f19a90ebc1d015f8'
    assert manifest['compiled_objects']['noFMA']['fma_opcode_rows']==0
    assert manifest['compiled_objects']['FMA']['fma_opcode_rows']>0
    original=load(variant,'fp32',16)
    fields=('TH_IN','QV_IN','QC_IN','QR_IN','QI_IN','QS_IN','QH_IN','QHL_IN','CCW_IN','CRW_IN',
            'CCI_IN','CSW_IN','CHW_IN','CHL_IN','CN_IN','VHW_IN','VHL_IN','PII','P','DZ','W','DN')
    raw=b''.join(np.asarray(original['columns'][key],np.float32).tobytes() for key in fields)
    values=[]
    for row in manifest['rows']:
        if row['variant']!=variant:continue
        file=p/f"v{variant}_{row['mode']}.json"
        assert hashlib.sha256(file.read_bytes()).hexdigest()==row['output_sha256']
        assert row['same_input_bits'] and row['input_bytes_sha256']==hashlib.sha256(raw).hexdigest()
        control=json.loads(file.read_text())
        for field in fields:assert control['columns'][field]==original['columns'][field]
        for field in ('CASE','KX','DT','ITIMESTEP','CCN_IS_CCNA','QCCN'):
            assert control['scalars'][field]==original['scalars'][field]
        if row['mode']=='noFMA':
            assert control['columns']==original['columns'] and control['scalars']==original['scalars']
        values.append(control['columns']['QC_OUT'][0])
    assert len(values)==2
    bound=manifest['accepted_bound'][str(variant)]
    assert (bound['min'],bound['max'])==(min(values),max(values))


@pytest.mark.parametrize('variant', [19, 21])
@pytest.mark.parametrize('mode', ['fp64', 'fp32'])
@pytest.mark.parametrize('case', range(1, 18))
def test_original_one_moment_sedimentation(variant, mode, case):
    from gpuwrf.physics.nssl2mom.constants import get_one_moment_constants
    from gpuwrf.physics.nssl2mom.indices import FP32, FP64
    from gpuwrf.physics.nssl2mom.sediment import sediment1d
    d = load(variant, mode, case)
    prec = FP32 if mode == 'fp32' else FP64
    dtype = prec.nR
    C = get_one_moment_constants(mode, variant)
    dt = float(d['scalars']['DT'])
    key = (variant, mode, dt)
    if key not in FUN:
        FUN[key] = jax.jit(lambda an,t0,t7,dn,dz: sediment1d(an,t0,t7,dn,dz,dt,C,prec))
    s0 = canonical_stage(d, 'S0')
    s1 = canonical_stage(d, 'S1')
    np.testing.assert_array_equal(s0, s1)  # WRF ipconc>0 excludes calcnfromq
    cols = [np.asarray(d['stages']['S0'][k], dtype) for k in ('t0','t7','dn1','dz2d')]
    got,xf = jax.tree.map(np.asarray, FUN[key](s1.astype(dtype),*cols))
    ref = canonical_stage(d, 'S2')
    xref = canonical_stage(d, 'S2', 'xfall')
    for kind,p,r in [('an',got,ref),('xfall',xf,xref)]:
        assert np.isfinite(p).all() and np.isfinite(r).all(), (variant,mode,case,kind)
        for il in range(1,len(r)):
            scale = np.max(np.abs(r[il]))
            # Existing independently frozen sedimentation stage bounds.
            tolerance = 1e-10*np.abs(r[il])+1e-14*scale if mode=='fp64' else 5e-6*scale
            assert np.all(np.abs(p[il]-r[il]) <= tolerance), (variant,mode,case,kind,il,np.max(np.abs(p[il]-r[il])),scale)


@pytest.mark.parametrize('variant',[19,21])
@pytest.mark.parametrize('case',range(1,18))
def test_original_one_moment_preparation(variant,case):
    from gpuwrf.physics.nssl2mom.constants import get_one_moment_constants
    from gpuwrf.physics.nssl2mom.indices import FP64,LC,LI,LR,LS,LH,LHL
    from gpuwrf.physics.nssl2mom.gs_a import gs_part_a,A_TO_B_KEYS
    d=load(variant,'fp64',case);s=d['stages']['S0'];C=get_one_moment_constants('fp64',variant)
    G=gs_part_a(canonical_stage(d,'S2'),*[np.asarray(s[k]) for k in ('t0','t7','t00','t77','pn','wn','dn1','dz2d')],float(d['scalars']['DT']),C,FP64)
    with np.load(DATA.parent/'gs'/f'v{variant}_fp64_{case}_G1.npz') as ref:
        _assert_gs(d,G,C,ref,variant,case,set(A_TO_B_KEYS)-{'tqvcon'})


def _assert_gs(d,G,C,ref,variant,case,names):
        idx=ref['kgs'].astype(int)-1
        np.testing.assert_array_equal(np.flatnonzero(np.asarray(G['gathered'])),idx)
        bad=[]
        for flat in ref.files:
            name,sep,key=flat.partition('__')
            if name not in names or name not in G:continue
            mine=G[name]
            if sep:
                fields=tuple(map(int,key.split('_')));key=fields[0] if len(fields)==1 else fields
                if key not in mine:continue
                mine=mine[key]
            elif isinstance(mine,dict):continue
            p=np.asarray(mine);r=ref[flat]
            if name=='pinit':
                # Original pinit is a full nz background array, not gathered.
                assert np.all(r==p),(variant,case,'original zero background pinit')
                continue
            if r.ndim==0 and p.ndim>0:continue  # original last-point scalar
            if p.ndim==0 and r.ndim>0:p=np.broadcast_to(p,r.shape)
            if p.ndim>0 and p.shape[-1]==len(d['columns']['TH_IN']):p=p[...,idx]
            if name in ('axx','bxx','cdxgs') and sep:
                il=fields[0]
                if il not in G['qx']:continue
                has=np.asarray(G['qx'][il])[idx]>C.qxmin[il]
                p=np.where(has,p,0.);r=np.where(has,r,0.)
            if p.shape!=r.shape:
                bad.append((flat,'shape',p.shape,r.shape));continue
            scale=max(float(np.max(np.abs(r))) if r.size else 0.,1e-300)
            err=np.abs(p.astype(float)-r.astype(float));tol=1e-10*np.abs(r)+1e-13*scale
            if not np.all(np.isfinite(p)) or not np.all(np.isfinite(r)) or np.any(err>tol):
                bad.append((flat,float(np.max(err)) if err.size else 0.,scale))
        assert not bad,(variant,case,bad[:25])


@pytest.mark.parametrize('variant',[19,21])
@pytest.mark.parametrize('case',range(1,18))
def test_original_one_moment_collection_freezing_deposition(variant,case):
    from gpuwrf.physics.nssl2mom.constants import get_one_moment_constants
    from gpuwrf.physics.nssl2mom.indices import FP64
    from gpuwrf.physics.nssl2mom.gs_a import gs_part_a
    from gpuwrf.physics.nssl2mom.gs_b import gs_part_b
    from gpuwrf.physics.nssl2mom.gs_a import A_TO_B_KEYS
    from gpuwrf.physics.nssl2mom.gs_c import G2_KEYS_CONSUMED
    d=load(variant,'fp64',case);s=d['stages']['S0'];C=get_one_moment_constants('fp64',variant);dt=float(d['scalars']['DT'])
    G=gs_part_a(canonical_stage(d,'S2'),*[np.asarray(s[k]) for k in ('t0','t7','t00','t77','pn','wn','dn1','dz2d')],dt,C,FP64)
    G=gs_part_b(G,C,FP64,dt)
    with np.load(DATA.parent/'gs'/f'v{variant}_fp64_{case}_G2.npz') as ref:
        # Match both declared stage interfaces; dnmx/ihlr are scratch locals
        # reassigned before use or inactive-hail diagnostics, not state.
        _assert_gs(d,G,C,ref,variant,case,(set(A_TO_B_KEYS)|set(G2_KEYS_CONSUMED))-{'tqvcon'})


@pytest.mark.parametrize('variant',[19,21])
@pytest.mark.parametrize('case',range(1,18))
def test_original_one_moment_mass_and_saturation(variant,case):
    from gpuwrf.physics.nssl2mom.constants import get_one_moment_constants
    from gpuwrf.physics.nssl2mom.indices import FP64
    from gpuwrf.physics.nssl2mom.gs_a import gs_part_a
    from gpuwrf.physics.nssl2mom.gs_b import gs_part_b
    from gpuwrf.physics.nssl2mom.gs_c import gs_part_c
    d=load(variant,'fp64',case);s=d['stages']['S0'];C=get_one_moment_constants('fp64',variant);dt=float(d['scalars']['DT'])
    an=canonical_stage(d,'S2');cols=[np.asarray(s[k]) for k in ('t0','t7','t00','t77','pn','wn','dn1','dz2d')]
    G=gs_part_a(an,*cols,dt,C,FP64);G=gs_part_b(G,C,FP64,dt)
    zero=np.zeros_like(cols[0])
    got,temps=gs_part_c(G,an,cols[0],zero,zero,zero,zero,zero,zero,cols[1],zero,zero,C,FP64,dt,hail_enabled=variant==19)
    got=np.asarray(got);ref=canonical_stage(d,'S3')
    assert np.isfinite(got).all() and np.isfinite(ref).all()
    for il in range(1,len(ref)):
        scale=max(np.max(np.abs(ref[il])),np.max(np.abs(an[il])))
        err=np.abs(got[il]-ref[il]);tol=1e-10*np.abs(ref[il])+1e-13*scale
        assert np.all(err<=tol),(variant,case,il,float(np.max(err)),float(scale),np.flatnonzero(err>tol)[:20])


@pytest.mark.parametrize('variant',[19,21])
@pytest.mark.parametrize('mode',['fp64','fp32'])
@pytest.mark.parametrize('case',range(1,18))
def test_original_one_moment_cleanup(variant,mode,case):
    from gpuwrf.physics.nssl2mom.constants import get_one_moment_constants
    from gpuwrf.physics.nssl2mom.indices import FP32,FP64
    from gpuwrf.physics.nssl2mom.cleanup import smallvalues
    d=load(variant,mode,case);prec=FP32 if mode=='fp32' else FP64;C=get_one_moment_constants(mode,variant)
    a=canonical_stage(d,'S3');np.testing.assert_array_equal(a,canonical_stage(d,'S4')) # NUCOND early return
    s=d['stages']['S0'];dtype=prec.nR
    got,_=smallvalues(a.astype(dtype),np.asarray(s['t0'],dtype),np.asarray(s['dn1'],dtype),np.asarray(s['wn'],dtype),np.asarray(s['t77'],dtype),d['scalars']['DT'],C,prec)
    got=np.asarray(got);ref=canonical_stage(d,'S5')
    assert np.isfinite(got).all()
    for il in range(1,len(ref)):
        scale=max(np.max(np.abs(ref[il])),np.max(np.abs(a[il])))
        tol=1e-10*np.abs(ref[il])+1e-13*scale if mode=='fp64' else 5e-6*scale
        assert np.all(np.abs(got[il]-ref[il])<=tol),(variant,mode,case,il)


NAMES={'th':'TH','qv':'QV','qc':'QC','qr':'QR','qi':'QI','qs':'QS','qg':'QH','qh':'QHL',
       'qndrop':'CCW','qnr':'CRW','qni':'CCI','qns':'CSW','qng':'CHW','qnh':'CHL','qnn':'CN',
       'qvolg':'VHW','qvolh':'VHL'}


@pytest.mark.parametrize('variant',[19,21])
@pytest.mark.parametrize('mode',['fp64','fp32'])
@pytest.mark.parametrize('case',range(1,18))
def test_complete_original_one_moment_column(variant,mode,case):
    from gpuwrf.physics.nssl2mom.constants import get_one_moment_constants
    from gpuwrf.physics.nssl2mom.indices import FP32,FP64
    from gpuwrf.physics.nssl2mom.driver import nssl2mom_column
    d=load(variant,mode,case);dt=float(d['scalars']['DT']);prec=FP32 if mode=='fp32' else FP64
    C=get_one_moment_constants(mode,variant);dtype=prec.nR
    f={k:np.asarray(d['columns'][v+'_IN'],dtype) for k,v in NAMES.items()}
    f.update({k:np.asarray(d['columns'][v],dtype) for k,v in [('pii','PII'),('p','P'),('rho','DN'),('w','W'),('dz','DZ')]})
    key=('full',variant,mode,dt)
    if key not in FUN:
        fn=jax.jit(lambda fields,first:nssl2mom_column(fields,dt,C,prec,first_step=first,hail_enabled=variant==19))
        start=time.monotonic();FUN[key]=fn.lower(f,np.bool_(True)).compile()
        memory=FUN[key].memory_analysis()
        COST[str(key)]={'CPU_compile_s':time.monotonic()-start,'arguments_B':memory.argument_size_in_bytes,
                        'outputs_B':memory.output_size_in_bytes,'temps_B':memory.temp_size_in_bytes}
        if os.environ.get('GPUWRF_MPMORE_COST_JSON'):
            Path(os.environ['GPUWRF_MPMORE_COST_JSON']).write_text(json.dumps(COST,indent=2)+'\n')
    out,pr,_=jax.tree.map(np.asarray,FUN[key](f,np.bool_(d['scalars']['ITIMESTEP']==1)))
    r32=load(variant,'fp32',case);r64=load(variant,'fp64',case)
    for k,v in NAMES.items():
        assert out[k].dtype==np.dtype(dtype),(variant,mode,case,k,out[k].dtype)
        p=np.asarray(out[k],float);ref=np.asarray(d['columns'][v+'_OUT'],float)
        assert np.isfinite(p).all() and np.isfinite(ref).all(),(variant,mode,case,k)
        if mode=='fp64':
            scale=max(np.max(np.abs(ref)),np.max(np.abs(d['columns'][v+'_IN'])))
            bad=np.abs(p-ref)>1e-10*np.abs(ref)+1e-13*scale
        else:
            a=np.asarray(r32['columns'][v+'_OUT']);b=np.asarray(r64['columns'][v+'_OUT'])
            atol=1e-6*max(np.max(np.abs(a)),np.max(np.abs(b)),1e-30)
            bad=~((np.abs(p-a)<=2e-3*np.abs(a)+atol)|(np.abs(p-b)<=2e-3*np.abs(b)+atol))
            if k=='qvolg':
                mass=np.maximum.reduce([np.abs(out['qg']),np.abs(r32['columns']['QH_OUT']),np.abs(r64['columns']['QH_OUT'])])
                bad&=mass>=1e-10  # inherited NSSL volume dust criterion; all mass remains gated
            if case==16 and k=='qc':
                # Standing compiler-contraction policy: independently measured
                # original WRF REAL4 spread, identical raw inputs in both arms.
                control=json.loads((DATA.parent/'fma/manifest.json').read_text())['accepted_bound'][str(variant)]
                assert control['case']==16 and control['zero_based_level']==0
                bad[0]=not (control['min']<=p[0]<=control['max'])
        assert not bad.any(),(variant,mode,case,k,np.flatnonzero(bad)[:20],p[bad],ref[bad])
    for k in ('qndrop','qnr','qni','qns','qng','qnh','qvolh'):
        assert np.count_nonzero(out[k])==0,(variant,mode,case,k)
    if variant==21:
        for k in ('qh','qvolg','qnn'):assert np.count_nonzero(out[k])==0,(mode,case,k)
    else:
        expected=np.asarray(d['columns']['CN_OUT'],dtype)
        assert out['qnn'].tobytes()==expected.tobytes(),(mode,case,'original heldCN bytes')
    for k in ('rainncv','snowncv','grplncv','hailncv'):
        p=float(pr[k]);assert np.isfinite(p)
        if mode=='fp64':
            ref=d['scalars'][k.upper()]
            assert abs(p-ref)<=1e-9*abs(ref)+1e-15,(variant,case,k,p,ref)
        else:
            a=r32['scalars'][k.upper()];b=r64['scalars'][k.upper()]
            assert min(abs(p-a),abs(p-b))<=2e-3*max(abs(a),abs(b))+1e-12,(variant,case,k,p,a,b)
