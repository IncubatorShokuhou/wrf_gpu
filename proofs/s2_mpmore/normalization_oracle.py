"""Compile the verbatim WRF 4.7.1 NSSL namelist normalization (integer-only).

No pristine body is edited. The enclosing model_config_rec and wrf_debug are
minimal caller stubs; the entire original DO loop is retained, including its
default/legacy precedence. Build artifacts belong in the lane directory.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess


def build(source: Path, work: Path, output: Path, fc: str):
    text = source.read_text()
    start = text.index("      DO i = 1, model_config_rec % max_dom", text.index("Check for deprecated options with NSSL-MP"))
    end = text.index("      ENDDO", start) + len("      ENDDO")
    body = text[start:end]
    prefix = """program oracle
  implicit none
  integer, parameter :: NSSL_2MOM=18
  type config_t
    integer :: max_dom=1, mp_physics(1)
    logical :: grid_allowed(1)=.true.
    integer :: nssl_2moment_on=-1, nssl_hail_on=-1
    integer :: nssl_ccn_on=-1, nssl_density_on=-1, nssl_3moment=0
  end type
  type(config_t) :: model_config_rec
  integer :: i, mp, overrides
  character(1024) :: wrf_err_message
  character(32) :: arg
  call get_command_argument(1,arg)
  read(arg,*) mp
  call get_command_argument(2,arg)
  read(arg,*) overrides
  model_config_rec%mp_physics=mp
  if (overrides == 1) then
    model_config_rec%nssl_2moment_on=0
    model_config_rec%nssl_hail_on=0
    model_config_rec%nssl_ccn_on=0
    model_config_rec%nssl_density_on=0
  endif
"""
    suffix = """
  print '(6(I4,1X))', model_config_rec%mp_physics(1), &
    model_config_rec%nssl_2moment_on, model_config_rec%nssl_hail_on, &
    model_config_rec%nssl_ccn_on, model_config_rec%nssl_density_on, model_config_rec%nssl_3moment
contains
  subroutine wrf_debug(level,msg)
    integer, intent(in) :: level
    character(*), intent(in) :: msg
  end subroutine
end program
"""
    work.mkdir(parents=True, exist_ok=True)
    driver = work / "normalization.f90"
    driver.write_text(prefix + body + suffix)
    executable = work / "normalization"
    flags = ["-O2", "-ffree-form", "-ffree-line-length-none", "-finit-real=zero", "-finit-integer=0", "-finit-logical=false"]
    subprocess.run([fc, *flags, str(driver), "-o", str(executable)], check=True, timeout=60)
    rows = []
    for mp, overrides in ((17, 0), (18, 0), (17, 1), (19, 0), (21, 0), (22, 0)):
        run = subprocess.run([str(executable), str(mp), str(overrides)], check=True, capture_output=True, text=True, timeout=10)
        rows.append(dict(input_mp=mp, conflicting_flags=bool(overrides), resolved=list(map(int, run.stdout.split()))))
    assert rows[0]["resolved"] == rows[1]["resolved"] == rows[2]["resolved"] == [18, 1, 1, 1, 2, 0], rows
    result = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  first_line=text[:start].count("\n") + 1, body=body,
                  body_sha256=hashlib.sha256(body.encode()).hexdigest(), compiler=fc, flags=flags, rows=rows,
                  scope="WRF original caller normalization; existing default18 pristine column qualification reused")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "rows": rows}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("work", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--fc", default="<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran")
    args = p.parse_args()
    build(args.source, args.work, args.output, args.fc)
