"""Compile the UNMODIFIED WRF physics_init has_reqc/i/s block for WSM/WDM selections.

Same method as proofs/s1_mpfix/build_radius_oracles.py (block extracted verbatim from
phys/module_physics_init.F, Registry scheme ids from frame/module_state_description.F,
EM_CORE=1, RRTMG LW/SW 4/4), extended to the mp ids that fixture does not cover.
Usage: python build_radius_flags.py <work_dir> <out.json>
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

WRF = Path("<USER_HOME>/src/wrf_pristine/WRF")
FC = "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"
CASES = (2, 3, 4, 6, 8, 10, 14, 16, 24, 26)


def main():
    work, out = Path(sys.argv[1]), Path(sys.argv[2])
    work.mkdir(parents=True, exist_ok=False)
    init = (WRF / "phys/module_physics_init.F").read_text()
    registry = (WRF / "frame/module_state_description.F").read_text()
    start = init.index("   has_reqc = 0")
    stop = init.index("! for P3,", start)
    block = init[start:stop] + "   ENDIF\n"
    constants = dict(re.findall(r"INTEGER, PARAMETER :: (\w+) = (\d+)", registry, re.I))
    needed = {w.lower() for w in re.findall(r"\b\w+\b", block)}
    params = "\n".join(f"integer,parameter::{k}={v}" for k, v in constants.items() if k.lower() in needed)
    program = f"""program flags
implicit none
{params}
type config_type
integer::use_mp_re=1,ra_lw_physics=4,ra_sw_physics=4,mp_physics,nssl_2moment_on=1
end type
type(config_type)::config_flags
integer::has_reqc,has_reqi,has_reqs,mp,use_re
integer,parameter::cases({len(CASES)})=[{','.join(map(str, CASES))}]
do mp=1,{len(CASES)}
do use_re=0,1
config_flags%mp_physics=cases(mp)
config_flags%use_mp_re=use_re
{block}
write(*,*)cases(mp),use_re,has_reqc,has_reqi,has_reqs
enddo
enddo
end program
"""
    (work / "flags.F90").write_text(program)
    cmd = [FC, "-O2", "-cpp", "-DEM_CORE=1", "-ffree-line-length-none", "flags.F90", "-o", "flags"]
    subprocess.run(cmd, cwd=work, check=True)
    rows = [list(map(int, line.split())) for line in subprocess.check_output([work / "flags"], text=True).splitlines()]
    result = dict(
        scope="literal pristine physics_init has_reqc/i/s selection (RRTMG 4/4); caller flags, not a forecast",
        sources={n: hashlib.sha256((WRF / n).read_bytes()).hexdigest()
                 for n in ("phys/module_physics_init.F", "frame/module_state_description.F")},
        block_sha256=hashlib.sha256(block.encode()).hexdigest(),
        program_sha256=hashlib.sha256(program.encode()).hexdigest(), compile_command=cmd,
        scheme_ids={k: int(v) for k, v in constants.items() if k.lower() in needed}, flags=rows)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(rows)


if __name__ == "__main__":
    main()
