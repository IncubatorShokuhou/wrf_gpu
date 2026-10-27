"""Actual three-domain CPU loader admission, before requesting a CUDA mini."""
import json,os,sys,time
from pathlib import Path
import numpy as np


def main():
    root,case,output=map(Path,sys.argv[1:4]);assert os.environ['JAX_PLATFORMS']=='cpu'
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
    from gpuwrf.io.cycling_init import admit_cycling_input
    from gpuwrf.io.wrfbdy_coverage import require_wrfbdy_coverage
    for dom in (1,2,3):admit_cycling_input(case/f'wrfinput_d{dom:02}')
    coverage=require_wrfbdy_coverage(case,3)
    config=pipeline.NestedPipelineConfig(case,output.parent/'unused',output.parent/'unused_proofs',hours=3,max_dom=3)
    start=time.perf_counter();_,bundles,_,_,dt,carries=pipeline._load_domains(config,('d01','d02','d03'))
    rows=[]
    for name,carry in carries.items():
        s=carry.state;assert s.theta.shape==(44,24,24),(name,s.theta.shape)
        for field in ('theta','qv','qc','p','u','v','w','qke'):
            assert np.isfinite(np.asarray(getattr(s,field))).all(),(name,field)
        assert np.max(np.asarray(s.qc))>0,name
        assert carry.precipitation_buckets is None and carry.ruc_land is None
        rows.append(dict(domain=name,shape=list(s.theta.shape),qc_max=float(np.max(np.asarray(s.qc))),finite=True,
                         cycling=bundles[name].namelist.cycling,dt=dt[name]))
    report=dict(passed=True,load_seconds=time.perf_counter()-start,domains=rows,coverage=coverage,
                source=str(root),CPU_interpretation=True,GPU_executed=False,
                scope='actual full loader + stored cycling inventories/Noah-MP masks/geometry, not GPU forecast fidelity')
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
