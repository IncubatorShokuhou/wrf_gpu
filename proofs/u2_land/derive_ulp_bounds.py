"""Independently derive one-input-ULP envelopes using the pristine Fortran binaries.

No JAX import. These bounds supplement the pre-registered nominal parity
tolerances; they never replace them. Original compiler/math bodies are retained.
Usage: python derive_ulp_bounds.py ROOT_OF_LAND_ARTIFACTS OUTPUT_JSON
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def adjacent(value, positive):
    return float(np.nextafter(np.float32(value),np.float32(np.inf if positive else -np.inf)))


def invoke(binary,lines):
    result=subprocess.run([str(binary)],input="\n".join(lines)+"\n",text=True,capture_output=True,check=True,timeout=60)
    return [np.asarray([float(x) for x in line.split()],np.float32) for line in result.stdout.splitlines()]


def snow(root):
    fx=json.loads((root/"LAND01/snow_oracle.json").read_text())
    binary=root/"LAND01/build/oracle.exe"
    keys=("BATS_COSZ","BATS_VIS_NEW","BATS_NIR_NEW","BATS_VIS_AGE","BATS_NIR_AGE","BATS_VIS_DIR","BATS_NIR_DIR")
    globals=("SWEMX","TAU0","GRAIN_GROWTH","EXTRA_GROWTH","DIRT_SOOT")
    rows=[];owner=[]
    for index,case in enumerate(fx["cases"]):
        row=case["inputs"]
        rows.append(row);owner.append(index)
        for field in range(1,len(row)):
            for positive in (False,True):
                variant=row.copy();variant[field]=adjacent(variant[field],positive)
                rows.append(variant);owner.append(index)
    lines=[" ".join(str(fx["parameters"][key]) for key in keys),
           " ".join(str(fx["parameters"][key]) for key in globals),str(len(rows))]
    lines += [" ".join(map(str,row)) for row in rows]
    raw=invoke(binary,lines)
    results=np.asarray([np.concatenate((raw[2*i],raw[2*i+1])) for i in range(len(rows))])
    expected=np.asarray([case["output"]+case["class_output"] for case in fx["cases"]],np.float32)
    assert np.array_equal(results[::31],expected),"Nominal binary binding changed"
    low,high=expected.copy(),expected.copy()
    for index,result in zip(owner,results):
        assert np.isfinite(result).all()
        low[index]=np.minimum(low[index],result);high[index]=np.maximum(high[index],result)
    return dict(nominal=expected.tolist(),low=low.tolist(),high=high.tolist(),variants=len(rows),
                variables=fx["variables_out"]+["class_alb","class_ad1","class_ad2","class_ai1","class_ai2"]),binary


def irrigation(root):
    fx=json.loads((root/"LAND05/irrigation_oracle.json").read_text())
    binary=root/"LAND05/build/oracle.exe"
    rows=[];owner=[];nominals=[]
    for index,case in enumerate(fx["cases"]):
        variants=[case.copy()]
        for key in ("pltday","hsday","fveg","julian","irrfra","lai","t2","u","v","eair","dt"):
            for positive in (False,True):
                variant=case.copy();variant[key]=adjacent(variant[key],positive);variants.append(variant)
        for key in ("sh2o","smc","fractions","amounts"):
            for field in range(len(case[key])):
                for positive in (False,True):
                    variant=case.copy();variant[key]=case[key].copy()
                    variant[key][field]=adjacent(variant[key][field],positive);variants.append(variant)
        nominals.append(len(rows))
        for c in variants:
            rows.append([c["opt_irr"],c["opt_irrm"],c["nroot"],c["pltday"],c["hsday"],*c["sh2o"],*c["smc"],
              c["fveg"],c["julian"],c["irrfra"],c["lai"],*c["fractions"],c["t2"],c["u"],c["v"],c["eair"],
              *c["amounts"],*c["counts"],c["dt"]]);owner.append(index)
    keys=("irr_frac","irr_har","irr_lai","irr_mad","filoss","sprir_rate","micir_rate","firtfac","ir_rain")
    lines=[" ".join(str(fx["parameters"][key]) for key in keys)]
    lines += [" ".join(map(str,fx["parameters"][key])) for key in ("smcwlt","smcref","smcmax","bexp","dksat","dwsat")]
    lines += [" ".join(map(str,fx["zsoil"])),str(len(rows))]
    lines += [" ".join(map(str,row)) for row in rows]
    results=np.asarray(invoke(binary,lines))
    expected=np.asarray([case["expected"] for case in fx["cases"]],np.float32)
    assert np.array_equal(results[nominals],expected),"Nominal binary binding changed"
    low,high=expected.copy(),expected.copy()
    for index,result in zip(owner,results):
        # Negative adjacent zero amounts can enter unused Fortran paths; the
        # input-ULP envelope may use only finite reference outputs (never pass NaN).
        if not np.isfinite(result).all():
            raise ValueError(f"Nonfinite pristine irrigation ULP variant for column {index}")
        low[index]=np.minimum(low[index],result);high[index]=np.maximum(high[index],result)
    return dict(nominal=expected.tolist(),low=low.tolist(),high=high.tolist(),variants=len(rows),variables=fx["variables_out"]),binary


def skin(root):
    fx=json.loads((root/"LAND03/skin_oracle.json").read_text())
    binary=root/"LAND03/build/oracle.exe"
    base=np.asarray(fx["inputs"],np.float32)
    expected=np.asarray(fx["output"],np.float32)
    low,high=expected.copy(),expected.copy()
    for field in range(base.shape[1]):
        for positive in (False,True):
            rows=base.copy()
            # DT is one caller scalar; perturb it identically in every record.
            rows[:,field]=np.nextafter(rows[:,field],np.float32(np.inf if positive else -np.inf))
            lines=[" ".join(map(str,row)) for row in rows]
            result=np.asarray(invoke(binary,lines)).reshape(expected.shape)
            assert np.isfinite(result).all()
            low=np.minimum(low,result);high=np.maximum(high,result)
    nominal=np.asarray(invoke(binary,[" ".join(map(str,row)) for row in base])).reshape(expected.shape)
    assert np.array_equal(nominal,expected),"Nominal binary binding changed"
    return dict(nominal=expected.tolist(),low=low.tolist(),high=high.tolist(),variants=22,variables=fx["variables_out"]),binary


def main():
    root,out=map(Path,sys.argv[1:3])
    results={}
    for name,fn in (("snow",snow),("irrigation",irrigation),("skin",skin)):
        result,binary=fn(root)
        result["binary_sha256"]=hashlib.sha256(binary.read_bytes()).hexdigest()
        result["source_sha256"]=hashlib.sha256(binary.with_name("reference.f90").read_bytes()).hexdigest()
        result["driver_sha256"]=hashlib.sha256(binary.with_name("driver.f90").read_bytes()).hexdigest()
        results[name]=result
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(method="independent pristine Fortran +/- one REAL4 input ULP; nominal binding BYTE",bounds=results),indent=1)+"\n")
    print({key:value["variants"] for key,value in results.items()})


if __name__=="__main__":main()
