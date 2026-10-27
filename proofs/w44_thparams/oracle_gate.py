"""Unchanged frozen mixed-phase increment rule shared by CLI and pytest."""
import numpy as np

FIELDS=("qv","qc","qr","qi","qs","qg","ni","nr","th")
TOL={**{v:1e-2 for v in FIELDS},"ni":5e-2,"precip":1e-2}

def score_arrays(z,idx,w,raw):
    assert all(np.isfinite(v).all() for v in raw.values())
    port={name:raw[str(i)].astype(np.float64) for i,name in enumerate(FIELDS[:8])}
    port["th"]=(raw["10"].astype(np.float32)/z["in_pii"][idx].astype(np.float32)).astype(np.float64)
    port["precip"]=sum(raw[str(i)].astype(np.float64) for i in (16,17,18,19))
    rows={}
    for name in FIELDS:
        x=z["in_"+name][idx].astype(np.float64)
        ref=w[name].astype(np.float64)
        dw=ref-x
        scale=np.max(np.abs(dw))
        floor=2*np.spacing(np.max(np.abs(ref),axis=1).astype(np.float32)).astype(np.float64)
        floor+=(1e-12 if name in ("qc","qr","qi","qs","qg") else 0.)+1e-4*scale
        error=np.max(np.abs(port[name]-ref),axis=1)
        excess=np.maximum(0.,error-floor)/np.maximum(np.max(np.abs(dw),axis=1),1e-30)
        rows[name]=dict(max_excess=float(np.max(excess)), tolerance=TOL[name],
                        failed_columns=int(np.count_nonzero(excess>TOL[name])),maxabs=float(np.max(error)))
    ref=w["rainncv"].astype(np.float64)
    floor=2*np.spacing(ref.astype(np.float32)).astype(np.float64)+1e-4*np.max(np.abs(ref))
    error=np.abs(port["precip"]-ref)
    excess=np.maximum(0.,error-floor)/np.maximum(ref,1e-30)
    rows["precip"]=dict(max_excess=float(np.max(excess)), tolerance=TOL["precip"],
                        failed_columns=int(np.count_nonzero(excess>TOL["precip"])),maxabs=float(np.max(error)))
    return rows
