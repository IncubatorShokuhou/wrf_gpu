"""Unmodified pristine errob + nudob oracle, canonical WRF constants, one rank.

No shared build/run directory is touched. module_dm's one-rank gather asserts
every observation belongs to this complete tile, then performs the identity;
the scalar allreduce is the identity. Original TILE_MASK is retained unchanged.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"oracle"))
from build import extract, FC, PRISTINE  # noqa: E402

HERE=Path(__file__).resolve().parent


def main(out):
    out.mkdir(parents=True,exist_ok=True)
    src=PRISTINE/"phys/module_fddaobs_rtfdda.F"
    text=src.read_text()
    routines="".join(extract(text,n) for n in ("errob","upoint","vpoint","print_obs_info","get_height_column","nudob","calc_rcouple_scales"))
    routines+=extract(text,"TILE_MASK","FUNCTION","LOGICAL ")
    routines+=extract(text,"ht_to_p","FUNCTION","REAL ")
    routines+=extract(text,"get_timewt","FUNCTION","REAL ")
    (out/"extracted.F90").write_text("module obs_full_oracle\ncontains\n"+routines+"end module\n")
    constants=PRISTINE/"share/module_model_constants.F"
    (out/"module_model_constants.f90").write_bytes(subprocess.check_output(["/lib/cpp","-P","-nostdinc","-DEM_CORE=1",str(constants)]))
    (out/"one_rank.f90").write_text('''module module_domain
end module
module module_dm
contains
real function wrf_dm_sum_real(x)
real x
wrf_dm_sum_real=x
end function
subroutine get_full_obs_vector(nsta,nerrf,niobf,um,vm,tm,errf)
integer nsta,nerrf,niobf
logical um(niobf),vm(niobf),tm(niobf)
real errf(nerrf,niobf)
if(.not.all(um(:nsta)).or..not.all(vm(:nsta)).or..not.all(tm(:nsta))) stop 2
end subroutine
end module
subroutine wrf_message(msg)
character(*) msg
print *,msg
end subroutine
subroutine wrf_error_fatal(msg)
character(*) msg
print *,msg
stop 1
end subroutine
''')
    flags=["-O2","-ftree-vectorize","-funroll-loops","-ffree-form","-ffree-line-length-none","-cpp","-DEM_CORE=1","-DDM_PARALLEL","-fcheck=bounds"]
    for kind in ("r4","r8"):
        work=out/kind
        (work/"tmp").mkdir(parents=True,exist_ok=True)
        env={**os.environ,"TMPDIR":str(work/"tmp")}
        extra=[] if kind=="r4" else ["-fdefault-real-8","-fdefault-double-8"]
        subprocess.run([FC,*flags,*extra,"-c",str(out/"module_model_constants.f90"),str(out/"one_rank.f90"),str(out/"extracted.F90")],cwd=work,env=env,check=True,timeout=180)
        for driver,binary in (("error_driver.F90","errors"),("spread_driver.F90","spread")):
            subprocess.run([FC,*flags,*extra,"module_model_constants.o","one_rank.o","extracted.o",str(HERE/driver),"-o",binary],cwd=work,env=env,check=True,timeout=60)
    provenance={"sources":{str(p.relative_to(PRISTINE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (src,constants)},
                "extract_sha256":hashlib.sha256((out/"extracted.F90").read_bytes()).hexdigest(),"flags":flags,
                "compiler":subprocess.check_output([FC,"--version"],text=True).splitlines()[0],
                "driver_sha256":hashlib.sha256((HERE/"error_driver.F90").read_bytes()).hexdigest(),
                "one_rank":"full domain; masks must all be true; identity gather/reduce"}
    (out/"build.json").write_text(json.dumps(provenance,indent=2)+"\n")


if __name__=="__main__":main(Path(sys.argv[1]).resolve())
