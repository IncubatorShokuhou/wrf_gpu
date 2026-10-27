"""Finite-first bound for the manager-authorized ON-default round-off class.

The measured CPU AV_C class is <=2 WRF REAL ulp; Nt_c adds a localized <=3-ulp
Nr class (NT07). Cloudw is explicitly computed
in REAL32 then widened by phys_thompson_full._namespace.cloud; disclose BOTH
the carrier's float64 ulp and the original work-precision ulp. Other float64
outputs must remain byte-identical. This is regression, not a WRF oracle.
"""
import numpy as np

FIELDS=("qv","qc","qr","qi","qs","qg","Ni","Nr","Ns","Ng","T","p","rho","dz","w",
        "cloudw","graupel","ice","rain","snow")


def ulps(a,b):
    unsigned=np.uint32 if a.dtype==np.float32 else np.uint64
    sign=unsigned(1 << (a.dtype.itemsize*8-1))
    def ordered(x):
        bits=x.view(unsigned)
        return np.where(bits & sign,~bits,bits | sign)
    x,y=ordered(a),ordered(b)
    return np.maximum(x,y)-np.minimum(x,y)


def compare(reference, candidate, *, nt_c=False):
    assert len(reference)==len(candidate)==len(FIELDS)
    rows=[]
    for name,a,b in zip(FIELDS,reference,candidate,strict=True):
        a,b=np.asarray(a),np.asarray(b)
        assert a.dtype==b.dtype and a.shape==b.shape,name
        assert np.isfinite(a).all() and np.isfinite(b).all(),f"nonfinite {name}"
        if a.tobytes()==b.tobytes():
            continue
        assert a.dtype in (np.float32,np.float64),name
        distance=ulps(a,b)
        changed=distance!=0
        error=np.abs(a.astype(np.float64)-b.astype(np.float64))
        relative=np.divide(error,np.abs(a),out=np.zeros_like(error),where=a!=0)
        relative[(a==0)&(error!=0)]=np.inf
        widened=(name=="cloudw" and a.dtype==np.float64
                 and np.array_equal(a,a.astype(np.float32).astype(np.float64))
                 and np.array_equal(b,b.astype(np.float32).astype(np.float64)))
        work=ulps(a.astype(np.float32),b.astype(np.float32)) if widened else distance
        limit=2 if a.dtype==np.float32 or widened else 0
        if nt_c and name=="Nr":
            # NT07 isolates the newly traced cloud-moment rounding; NT02
            # measures <=3 ulp in Nr, all other fields retain the 2-ulp class.
            limit=3
        rows.append(dict(field=name,storage_dtype=str(a.dtype),changed_cells=int(changed.sum()),
                         max_abs=float(error.max()),max_rel=float(relative.max()),
                         max_storage_ulp=int(distance.max()),max_work_ulp=int(work.max()),
                         widened_REAL32=widened,limit_work_ulp=limit,pass_roundoff=bool(work.max()<=limit)))
    return dict(byte_exact=not rows,within_roundoff=all(row["pass_roundoff"] for row in rows),differences=rows)
