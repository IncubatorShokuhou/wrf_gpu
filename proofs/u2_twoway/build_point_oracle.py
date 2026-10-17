"""Additional verbatim WRF point-feedback REAL/INTEGER savepoints."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from build_oracle import build,call

work,fixture=Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve()
lib,manifest=build(work)
data,cases={},[]
rng=np.random.default_rng(20261013)
for ratio in (3,5):
    for zone in (1,2):
        for method,dtype in (("copy_fcnm",np.float32),("copy_fcni",np.int32)):
            for layers in (1,4):
                for stagger in ("","U","V"):
                    p=rng.integers(0,20,size=(layers,20+(stagger=="V"),22+(stagger=="U"))).astype(dtype)
                    c=rng.integers(20,40,size=(layers,10*ratio+(stagger=="V"),12*ratio+(stagger=="U"))).astype(dtype)
                    if layers==1:p,c=p[0],c[0]
                    key=str(len(cases))
                    data["p"+key],data["c"+key]=p,c
                    data["o"+key]=call(lib,p,c,ratio=ratio,zone=zone,method=method,stagger=stagger)
                    cases.append(dict(ratio=ratio,zone=zone,method=method,stagger=stagger))
np.savez_compressed(fixture,**data)
manifest.update(cases=cases,fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest())
fixture.with_suffix(".json").write_text(json.dumps(manifest,indent=2)+"\n")
print(len(cases),"point operator cases",manifest["fixture_sha256"])
