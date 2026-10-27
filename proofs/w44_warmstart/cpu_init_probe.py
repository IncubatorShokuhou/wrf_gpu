"""Compare the actual native loader at t0 with the guarded original-WRF twin.

Reports full-grid errors against WRF, without choosing new tolerances. CPU
interpretation is initialization evidence; the GPU/D6 gates stay separate.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from netCDF4 import Dataset


def main():
    assert os.environ['JAX_PLATFORMS']=='cpu'
    root,output=map(Path,sys.argv[1:3])
    sys.path[:0]=[str(root/'src'),str(root/'proofs/o1_nlbind')]
    import cpu_safe
    cpu_safe.install()
    import jax
    assert jax.devices()[0].platform=='cpu'
    from gpuwrf.backends import pallas as facade
    facade.cuda_native=lambda:False
    from gpuwrf.contracts import state as contract
    contract._gpu_device=lambda:jax.devices('cpu')[0]
    from gpuwrf.integration import nested_pipeline as pipeline
    case=Path('<DATA_ROOT>/wrf_gpu2/scratch/w44-warmstart/native_may04_00_newforcing')
    reference=case.parent/'cpu_native_may04_00_guarded_3r/cpu'
    output.mkdir(exist_ok=False,parents=True)
    config=pipeline.NestedPipelineConfig(case,output/'unused_run',output/'unused_proofs',hours=3,max_dom=3)
    started=time.perf_counter()
    hierarchy,bundles,_,run_start,dt,carries=pipeline._load_domains(config,('d01','d02','d03'))
    loaded=time.perf_counter()-started
    mapping={'U':'u','V':'v','W':'w','P':'p_perturbation','PH':'ph_perturbation','MU':'mu_perturbation',
             'QVAPOR':'qv','QCLOUD':'qc','QRAIN':'qr','QICE':'qi','QSNOW':'qs','QGRAUP':'qg','QKE':'qke',
             'QNICE':'Ni','QNRAIN':'Nr','SST':'sst','TSK':'t_skin'}
    records=[];packages={}
    for domain,carry in carries.items():
        state=carry.state
        assert carry.precipitation_buckets is None and carry.ruc_land is None
        packages[domain]={'inactive_prec_acc_absent':True,'inactive_ruc_absent':True}
        with Dataset(reference/f'wrfout_{domain}_2026-05-04_00:00:00')as ds:
            ds.set_auto_maskandscale(False)
            for field,attr in mapping.items():
                expected=np.asarray(ds[field][0]);actual=np.asarray(getattr(state,attr)).astype(expected.dtype)
                assert expected.shape==actual.shape,(domain,field,expected.shape,actual.shape)
                assert np.isfinite(expected).all()and np.isfinite(actual).all(),(domain,field)
                delta=actual.astype(float)-expected.astype(float)
                where=np.unravel_index(np.argmax(np.abs(delta)),delta.shape)
                records.append({'domain':domain,'field':field,'byte_equal':expected.tobytes()==actual.tobytes(),
                    'different_cells':int(np.count_nonzero(expected!=actual)),'max_abs':float(np.max(np.abs(delta))),
                    'rmse':float(np.sqrt(np.mean(delta*delta))),'worst_index':list(map(int,where)),
                    'wrf_value':float(expected[where]),'port_value':float(actual[where])})
            for field,attr in [('SFROFF','sfcrunoff'),('UDROFF','udrunoff')]:
                expected=np.asarray(ds[field][0]);actual=(np.asarray(getattr(carry.noahmp_land,attr))*1000).astype(expected.dtype)
                delta=actual.astype(float)-expected.astype(float)
                records.append({'domain':domain,'field':field,'byte_equal':expected.tobytes()==actual.tobytes(),
                                'max_abs':float(np.max(np.abs(delta))),'rmse':float(np.sqrt(np.mean(delta*delta)))})
            # State carries absolute prognostic theta; WRF history stores the
            # perturbation. Reproduce WRF's physical caller REAL addition.
            field='THM'if int(ds.USE_THETA_M)else'T'
            expected=np.asarray(ds[field][0],np.float32)+np.float32(300.)
            actual=np.asarray(state.theta,np.float32)
            delta=actual.astype(float)-expected.astype(float)
            records.append({'domain':domain,'field':field+'+300_REAL','byte_equal':expected.tobytes()==actual.tobytes(),
                            'max_abs':float(np.max(np.abs(delta))),'rmse':float(np.sqrt(np.mean(delta*delta)))})
    report={'platform':jax.default_backend(),'source':str(root),'load_s':loaded,'records':records,'packages':packages,
            'byte_equal_fields':sum(r['byte_equal']for r in records),'fields':len(records),
            'guarded_cpu_sha256':'1c1269fb10b7aa135bf42417f1aa7f7a36f7137ea40cd82769316e8ac5bb4918',
            'reference':str(reference),'scope':'actual 3nest initialization vs WRF; report-only exact storage/raw errors, no new tolerance',
            'CPU_ISA':os.environ.get('XLA_FLAGS'),'GPU_qualified':False}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
