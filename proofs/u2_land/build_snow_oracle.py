"""Generate small REAL4 savepoints from verbatim pristine WRF subroutines.

Run under the lane CPU/QUIET/timeout launcher. Only declarations and the
driver are supplied here; each tested WRF subroutine is copied byte for byte.
No station/truth files or live WRF build are modified.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def extract(text, name):
    pattern = rf"^  SUBROUTINE {name}\b.*?^  END SUBROUTINE {name}\s*$"
    match = re.search(pattern, text, re.M | re.S | re.I)
    if not match:
        raise ValueError(f"Missing pristine routine {name}")
    return match[0]


def main():
    source, build, target = map(Path, sys.argv[1:4])
    build.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = source.read_text()
    routines = {name: extract(text, name) for name in ("ATM", "SNOW_AGE", "SNOWALB_BATS", "SNOWALB_CLASS")}
    constants = text[text.index("  REAL, PARAMETER :: GRAV"):text.index("  INTEGER, PRIVATE, PARAMETER :: MBAND")]
    # The unused ATM parameters argument needs a type, and the other routines
    # read ONLY these scalar MPTABLE slots. No replacement physics statements.
    names = ("BATS_COSZ", "BATS_VIS_NEW", "BATS_NIR_NEW", "BATS_VIS_AGE", "BATS_NIR_AGE",
             "BATS_VIS_DIR", "BATS_NIR_DIR", "SWEMX", "TAU0", "GRAIN_GROWTH", "EXTRA_GROWTH", "DIRT_SOOT")
    module = "module snow_reference\nimplicit none\ninteger :: OPT_SNF\n" + constants
    module += "type noahmp_parameters\nreal :: " + ",".join(names) + "\nend type\ncontains\n"
    module += "\n".join(routines.values()) + "\nend module\n"
    (build / "reference.f90").write_text(module)
    driver = '''program oracle
use snow_reference
implicit none
type(noahmp_parameters) :: p
integer :: n, i, s
real :: t,prs,qa,pc,pn,ps,pg,ph,cosz,tg,sneqvo,sneqv,tauss,albold,dt
real :: thair,qair,eair,rhoair,qprecc,qprecl,solad(2),solai(2),swdown,bd,rain,snow,fp,fpice,prcp,fage,ad(2),ai(2),alb
read(*,*) p%BATS_COSZ,p%BATS_VIS_NEW,p%BATS_NIR_NEW,p%BATS_VIS_AGE,p%BATS_NIR_AGE,p%BATS_VIS_DIR,p%BATS_NIR_DIR
read(*,*) p%SWEMX,p%TAU0,p%GRAIN_GROWTH,p%EXTRA_GROWTH,p%DIRT_SOOT
read(*,*) n
do i=1,n
read(*,*) s,t,prs,qa,pc,pn,ps,pg,ph,cosz,tg,sneqvo,sneqv,tauss,albold,dt
opt_snf=s
call ATM(p,prs,t,qa,pc,pn,0.,ps,pg,ph,500.,cosz,thair,qair,eair,rhoair,qprecc,qprecl,solad,solai,swdown,bd,rain,snow,fp,fpice,prcp)
call SNOW_AGE(p,dt,tg,sneqvo,sneqv,tauss,fage)
call SNOWALB_BATS(p,2,0.5,cosz,fage,ad,ai)
alb=albold
write(*,'(14(ES25.16E3,1X))') prcp,rain,snow,fp,fpice,bd,tauss,fage,ad,ai
call SNOWALB_CLASS(p,2,snow,dt,alb,albold,ad,ai,1,1)
write(*,'(5(ES25.16E3,1X))') alb,ad,ai
end do
end program
'''
    (build / "driver.f90").write_text(driver)
    fc = shutil.which("gfortran")
    if not fc:
        fc = "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"
    subprocess.run([fc, "-O2", "-ffree-form", "-ffree-line-length-none", "-ffp-contract=off",
                    "reference.f90", "driver.f90", "-o", "oracle.exe"], cwd=build, check=True, timeout=120)
    table = source.parent.parent / "run/MPTABLE.TBL"
    clean = "\n".join(line.split("!", 1)[0] for line in table.read_text().splitlines())
    table_names = names[:7] + ("SWEMX", "TAU0", "GRAIN_GROWTH", "EXTRA_GROWTH", "DIRT_SOOT")
    params = {key: float(re.search(rf"\b{key}\s*=\s*([-+\d.eEdD]+)", clean, re.I)[1].lower().replace("d", "e"))
              for key in table_names}
    # Censused constructed columns around EVERY WRF threshold, dry/saturated
    # air, no snow/fresh/aged snow, pure/mixed solid precipitation and no rain.
    cases = []
    for snf in range(1, 6):
        for index, temperature in enumerate((258., 270., 273.15, 273.16, 273.66, 275.16, 275.36, 275.66, 281.)):
            solid = ((0., 0., 0.), (0.0004, 0., 0.), (0., 0.0004, 0.), (0., 0., 0.0004),
                     (0.0001, 0.0002, 0.0001))[index % 5]
            vals = [snf, temperature, 80000., (0.001, 0.003, 0.005)[index % 3],
                    0.0002 if index % 2 else 0., 0. if index == 0 else 0.0004, *solid,
                    (0.01, 0.2, 0.8)[index % 3], temperature, 0. if index == 0 else 20.,
                    0. if index == 0 else 20. + index % 3, 0.1 * index, 0.65, 18.]
            cases.append(vals)
    lines = [" ".join(str(params[key]) for key in names[:7]),
             " ".join(str(params[key]) for key in names[7:]), str(len(cases))]
    lines += [" ".join(str(value) for value in row) for row in cases]
    result = subprocess.run([str((build / "oracle.exe").resolve())], input="\n".join(lines) + "\n",
                            text=True, capture_output=True, check=True, timeout=30)
    rows = [[float(x) for x in row.split()] for row in result.stdout.splitlines()]
    assert len(rows) == 2 * len(cases)
    assert all(np.isfinite(row).all() for row in rows)
    receipt = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   routine_sha256={name: hashlib.sha256(body.encode()).hexdigest() for name, body in routines.items()},
                   table_sha256=hashlib.sha256(table.read_bytes()).hexdigest(), parameters=params,
                   compiler=subprocess.check_output([fc, "--version"], text=True).splitlines()[0],
                   flags="-O2 -ffp-contract=off (WRF REAL kind 4)", created_utc=datetime.now(timezone.utc).isoformat(),
                   variables_in=["opt_snf", "sfctmp", "sfcprs", "qair", "prcpconv", "prcpnonc", "prcpsnow",
                                 "prcpgrpl", "prcphail", "cosz", "tg", "sneqvo", "sneqv", "tauss", "albold", "dt"],
                   variables_out=["prcp", "rain", "snow", "fp", "fpice", "bdfall", "tauss", "fage", "ad1", "ad2", "ai1", "ai2"],
                   tolerance=dict(atol=1.0e-6, rtol=3.0e-5),
                   census=dict(columns=len(cases), options=[1,2,3,4,5], constructed=True,
                               regimes="temperature thresholds; rain/snow/graupel/hail; dry/aged/fresh snow; wet-bulb humidity"),
                   cases=[dict(inputs=row, output=rows[2*i], class_output=rows[2*i+1]) for i,row in enumerate(cases)])
    target.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
