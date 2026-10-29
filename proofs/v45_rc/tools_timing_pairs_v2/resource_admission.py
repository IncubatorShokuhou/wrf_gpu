"""Explicit requested shared-resource reservation. Stdlib, no GPU queries."""
from pathlib import Path
import json,os,sys
G=2**30
def free(p):s=os.statvfs(p);return s.f_bavail*s.f_frsize
r,d=free('/'),free('<DATA_ROOT>');ok=r>=24.25*G and d>=412.5*G
print(json.dumps(dict(scope='RESOURCE_ADMISSION_ONLY',ROOT_free_GiB=r/G,ROOT_required_GiB=24.25,ROOT_growth_GiB=.25,ROOT_floor_GiB=22,ROOT_extra_margin_GiB=2,DATA_free_GiB=d/G,DATA_required_GiB=412.5,DATA_peak_GiB=8,production_admitted=ok,CPU_smoke_only='--smoke'in sys.argv)))
raise SystemExit(0 if ok or '--smoke'in sys.argv else 95)
