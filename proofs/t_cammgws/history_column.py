"""Active 44-level CAM-MG history column builder, copied verbatim from
proofs/t_rainnc/cam_column.py::history_fields (lane t-rainnc) for t-cammgws screening.
History reconstruction is NOT the exact post-RK caller savepoint.
"""
import hashlib

import numpy as np


def history_fields(path, tree):
    from gpuwrf.io.netcdf_lock import Dataset
    from gpuwrf.kernels.phys_cammg import SURFACE_FIELDS
    from gpuwrf.physics.cammg.tables import get_globals
    with np.load(tree/'proofs/s2_cammg/fixtures/public.npz') as bank:
        keys=[k.split('_input_',1)[1] for k in bank.files if k.startswith('c03_s1_input_')]
    with Dataset(path) as d:
        get=lambda n:np.asarray(d.variables[n][0],dtype=np.float32)
        qc=get('QCLOUD');nz=qc.shape[0]
        # Select an active interior column from the measured history.
        column=qc.sum(axis=0);column[:5]=column[-5:]=0;column[:,:5]=column[:,-5:]=0
        iy,ix=np.unravel_index(np.argmax(column),column.shape)
        at=lambda n:get(n)[...,iy,ix]
        fields={k:np.zeros((),np.float32) if k in SURFACE_FIELDS else np.zeros(nz+1,np.float32) for k in keys}
        g=get_globals(nz)
        p=at('P')+at('PB');th=at('T')+np.float32(300)
        pi=(p/np.float32(1.e5))**np.float32(287/1004.5)
        zw=(at('PH')+at('PHB'))/np.float32(9.81)
        qv=at('QVAPOR')
        for key,name in [('qv_curr','QVAPOR'),('qc_curr','QCLOUD'),('qi_curr','QICE'),('qr_curr','QRAIN'),('qs_curr','QSNOW'),
                         ('nc3d','QNCLOUD'),('ni3d','QNICE'),('cldfra_mp','CLDFRA_MP'),('cldfra_mp_all','CLDFRA_MP_ALL')]:
            if name in d.variables:fields[key][:nz]=at(name)
        for key,value in [('th',th),('t_phy',th*pi),('pi_phy',pi),('p_hyd',p),
                          ('alt',np.float32(287)*(th*pi)*(np.float32(1)+np.float32(461.6/287)*qv)/p),
                          ('z_sea_level',np.float32(.5)*(zw[:-1]+zw[1:]))]:fields[key][:nz]=value
        fields['z_at_w']=zw
        pf=np.zeros(nz+1,np.float32);pf[1:-1]=np.sqrt(p[:-1]*p[1:]);pf[0]=p[0]*p[0]/pf[1];pf[-1]=p[-1]*p[-1]/pf[-2]
        fields['p8w']=pf
        for key,name in [('ht','HGT'),('xland','XLAND'),('snowh','SNOWH'),('qfx','QFX')]:
            if name in d.variables:fields[key]=np.asarray(at(name),np.float32)
        return fields,dict(path=str(path),y=int(iy),x=int(ix),nz=nz,
                           reconstructed=['p_hyd uses total history pressure','p8w log-pressure interpolation',
                                          'missing held fields zero; cold first step'],
                           inputs_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in fields.items()})
