"""Parse exact native RAD_FLG preprocessing and band/transport savepoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from parse import operands, read_array

KG = (10,8,12,7,12,5,2,3,4,4,3,5,2,10,12,7,7,8)
PRE = ("p","t","q","o3","iwc","diameter","lwc","radius","rain","graupel","cloud","inhomogeneity")


def main(input_path: Path, final_path: Path, reference: Path, out: Path):
    if final_path.read_bytes() != reference.read_bytes():
        raise ValueError("Diagnostic writes changed native final outputs")
    inputs = operands(input_path)
    nc, n1, _ = inputs["fields"].shape
    nz = n1 - 1
    pre = {key: [] for key in PRE}
    pre.update({key: [] for key in ("albedo","emissivity","surface_t","coszen","solar_constant")})
    bands = {key: np.zeros((nc,18) + shape,np.float32) for key,shape in {
        "dz":(nz,),"trp":(nz,),"ice":(nz,6),"water":(nz,6),"bf":(n1,),"bs":(),
    }.items()}
    bands["active"] = np.zeros((nc,18),bool)
    transfer = []
    with Path(str(final_path) + ".capture").open("rb") as f:
        for col in range(nc):
            assert int(read_array(f,(),np.int32)) == n1
            as_ = read_array(f,(6,))
            u0,solcon,tsk = read_array(f,(3,))
            ee = read_array(f,(12,))
            assert np.all(as_==as_[0]) and np.all(ee==ee[0])
            for key,value in (("albedo",as_[0]),("emissivity",ee[0]),("surface_t",tsk),("coszen",u0),("solar_constant",solcon)):
                pre[key].append(value)
            for key in PRE:
                pre[key].append(read_array(f,(n1 if key in ("p","t","q","o3") else nz,)))
            for band in range(1 if u0>1.e-4 else 7,19):
                assert tuple(read_array(f,(2,),np.int32)) == (nz,band)
                b = band-1
                bands["active"][col,b]=True
                bands["dz"][col,b]=read_array(f,(nz,))
                bands["trp"][col,b]=read_array(f,(nz,))
                for key in ("ice","water"):
                    bands[key][col,b,:,0]=read_array(f,(nz,))
                    bands[key][col,b,:,1]=read_array(f,(nz,))
                    bands[key][col,b,:,2:]=read_array(f,(nz,4))
                if band>6:
                    bands["bf"][col,b]=read_array(f,(n1,))
                    bands["bs"][col,b]=read_array(f,())
                for ig in range(1,KG[b]+1):
                    assert tuple(read_array(f,(3,),np.int32)) == (nz,band,ig)
                    area,albedo,coszen,f0=read_array(f,(4,))
                    assert area == 1., "Default NFRACT=0 must select a single unit-area configuration"
                    emissivity=read_array(f,()) if band>6 else np.float32(0.)
                    bf=read_array(f,(n1,)) if band>6 else np.zeros(n1,np.float32)
                    bs=read_array(f,()) if band>6 else np.float32(0.)
                    optics=read_array(f,(nz,6))  # individual writes wc1..tt are column-major
                    flux=read_array(f,(n1,2))
                    if ig in (1,KG[b]):
                        transfer.append(dict(col=col,band=band,gpoint=ig,albedo=albedo,coszen=coszen,f0=f0,
                                             emissivity=emissivity,bf=bf,bs=bs,optics=optics,flux=flux))
        if f.read(1):
            raise ValueError("Trailing capture bytes")
    arrays = {"pre_"+k:np.asarray(v,np.float32) for k,v in pre.items()}
    arrays.update({"band_"+k:v for k,v in bands.items()})
    arrays.update({"in_"+k:v for k,v in inputs.items()})
    arrays.update({"transfer_"+k:np.asarray([row[k] for row in transfer]) for k in transfer[0]})
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,**arrays)
    receipt={"native_final_byte_equal_to_FG01":True,"columns":nc,"levels":nz,
             "active_band_rows":int(bands["active"].sum()),"transport_rows":len(transfer),
             "fixture_sha256":hashlib.sha256(out.read_bytes()).hexdigest(),
             "native_final_sha256":hashlib.sha256(final_path.read_bytes()).hexdigest(),
             "scope":"Native full calls unchanged; exact routine intermediates, first/last g-point per active band"}
    out.with_suffix(".json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    for name in ("input","final","reference","out"):ap.add_argument(name,type=Path)
    a=ap.parse_args()
    main(a.input,a.final,a.reference,a.out)
