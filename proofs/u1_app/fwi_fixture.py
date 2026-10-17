"""Reproduce full-precision fixtures from a pinned external CFFDRS checkout.

Download cffdrs/{fwi,constants}.py at commit7d8931e into a reference directory,
then run python proofs/u1_app/fwi_fixture.py REFERENCE_DIR OUTPUT_JSON.
This script imports the independently maintained reference, never gpuwrf's FWI.
The tracked input census supplies 108 cases; test runs need no network/download.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

reference=Path(sys.argv[1])
source=reference/"fwi.py"
assert hashlib.sha256(source.read_bytes()).hexdigest()=="e572ac19b5f0d1ba76aa78b3b0d41119a71a1b44e98bb334696351d0ccfc7aaa"

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

constants=load("cffdrs.constants",reference/"constants.py")
m=load("cffdrs_reference",source)
fixture=Path(__file__).resolve().parents[2]/"tests/fixtures/post_apps/fwi_reference.json"
data=json.loads(fixture.read_text())
for row in data["cases"]:
    a=row["inputs"]
    t,r,w,rain,lat,mon=[a[k] for k in ("temperature_c","rh_percent","wind_kmh","rain_mm","latitude","month")]
    f=m.fine_fuel_moisture_code(a["ffmc_previous"],t,r,w,rain)
    d=m.duff_moisture_code(a["dmc_previous"],t,r,rain,lat,mon)
    c=m.drought_code(a["dc_previous"],t,r,rain,lat,mon)
    isi=m.initial_spread_index(f,w);b=m.buildup_index(d,c);fw=m.fire_weather_index(isi,b)
    row["expected"]=dict(ffmc=f,dmc=d,dc=c,isi=isi,bui=b,fwi=fw,dsr=.0272*fw**1.77)
data["ffmc_coefficient"]=constants.FFMC_COEFFICIENT
data["constants_source_sha256"]=hashlib.sha256((reference/"constants.py").read_bytes()).hexdigest()
Path(sys.argv[2]).write_text(json.dumps(data,indent=2)+"\n")
