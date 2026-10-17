"""Compile the UNTOUCHED WRF post-RK microphysics-caller statements into a REAL4 fixture.

Statements are copied verbatim from pristine WRF (no edits):
  calc_p_rho_phi hypsometric_opt=2   module_big_step_utilities_em.F:1045-1048 (al)
  moist_physics_prep_em              :5480 z_at_w, :5488 dz8w, :5542 rho = 1./(al+alb)
Operands are a REAL-structured column (lowest 4 faces) of the ORIGINAL CPU-WRF c05
history at 2023-01-15_01 (j35,i37, the h1 U-divergence hotspot): PH, PHB, MU+MUB,
C3F/C4F/C3H/C4H, P_TOP. ALB is a finite base-state-scale operand (not in history).
Usage: python build_moist_prep_oracle.py <cpu_wrfout> <out.json> <build_dir>
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import netCDF4 as nc
import numpy as np

FC = "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"
BASE = Path("<USER_HOME>/src/wrf_pristine/WRF")
J, I, NZ = 35, 37, 3


def main():
    wrfout, out, folder = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    source = BASE / "dyn_em/module_big_step_utilities_em.F"
    constants = BASE / "share/module_model_constants.F"
    lines = source.read_text().splitlines()
    const = constants.read_text().splitlines()

    def original(prefix, after):
        return next((i + 1, s.strip()) for i, s in enumerate(lines)
                    if i + 1 >= after and s.strip().startswith(prefix))

    hyp = [original(p, 1040) for p in ("pfu = c3f(k+1)*MUTS(i,j)", "pfd = c3f(k  )*MUTS(i,j)",
                                        "phm = c3h(k  )*MUTS(i,j)", "al(i,k,j) = (ph(i,k+1,j)-ph(i,k,j)+phb")]
    zaw = original("z_at_w(i,k,j) = (ph(i,k,j)+phb(i,k,j))/g", 5470)
    dz = original("dz8w(i,k,j) = z_at_w(i,k+1,j)-z_at_w(i,k,j)", 5480)
    rho = original("rho(i,k,j)  = 1./(al(i,k,j)+alb(i,k,j))", 5530)
    assert [h[0] for h in hyp] == [1045, 1046, 1047, 1048] and (zaw[0], dz[0], rho[0]) == (5480, 5488, 5542)
    gravity = next((i + 1, s.strip()) for i, s in enumerate(const) if "PARAMETER :: g = 9.81" in s)

    d = nc.Dataset(wrfout)
    f4 = lambda a: np.asarray(a, np.float32)  # noqa: E731
    arrays = dict(
        ph=f4(d.variables["PH"][0, :NZ + 1, J, I]), phb=f4(d.variables["PHB"][0, :NZ + 1, J, I]),
        muts=f4([d.variables["MU"][0, J, I] + d.variables["MUB"][0, J, I]]),
        c3f=f4(d.variables["C3F"][0, :NZ + 1]), c4f=f4(d.variables["C4F"][0, :NZ + 1]),
        c3h=f4(d.variables["C3H"][0, :NZ]), c4h=f4(d.variables["C4H"][0, :NZ]),
        ptop=f4([d.variables["P_TOP"][0]]), alb=f4([0.83125, 0.84375, 0.859375]))
    d.close()

    def lit(name):
        return ",".join(f"transfer(int(z'{int(n):08X}',int32),0.0_real32)" for n in arrays[name].view(np.uint32))

    program = f"""program oracle
use iso_fortran_env, only: real32,int32
implicit none
integer :: i,j,k,its,itf,jts,jtf,kts,ktf,kte,i_start,i_end,j_start,j_end,k_start
{gravity[1]}
real(real32) :: ph(1,{NZ + 1},1),phb(1,{NZ + 1},1),al(1,{NZ},1),alb(1,{NZ},1),MUTS(1,1)
real(real32) :: c3f({NZ + 1}),c4f({NZ + 1}),c3h({NZ}),c4h({NZ}),ptop,pfu,pfd,phm,ptopa(1),mua(1)
real(real32) :: z_at_w(1,{NZ + 1},1),dz8w(1,{NZ},1),rho(1,{NZ},1)
ph(1,:,1)=[{lit('ph')}]
phb(1,:,1)=[{lit('phb')}]
mua=[{lit('muts')}]
MUTS(1,1)=mua(1)
c3f=[{lit('c3f')}]
c4f=[{lit('c4f')}]
c3h=[{lit('c3h')}]
c4h=[{lit('c4h')}]
ptopa=[{lit('ptop')}]
ptop=ptopa(1)
alb(1,:,1)=[{lit('alb')}]
its=1;itf=1;jts=1;jtf=1;kts=1;ktf={NZ};kte={NZ + 1};i_start=1;i_end=1;j_start=1;j_end=1;k_start=1
DO j=jts,jtf
DO k=kts,ktf
DO i=its,itf
{hyp[0][1]}
{hyp[1][1]}
{hyp[2][1]}
{hyp[3][1]}
END DO
END DO
END DO
DO j = j_start, j_end
DO k = k_start, kte
DO i = i_start, i_end
{zaw[1]}
ENDDO
ENDDO
ENDDO
do j = j_start,j_end
do k = k_start, kte-1
do i = i_start, i_end
{dz[1]}
enddo
enddo
enddo
DO j = j_start, j_end
DO k = k_start, ktf
DO i = i_start, i_end
{rho[1]}
ENDDO
ENDDO
ENDDO
write(*,'(6(Z8.8,1X))') (transfer(rho(1,k,1),0_int32),k=1,{NZ}),(transfer(dz8w(1,k,1),0_int32),k=1,{NZ})
end program
"""
    folder.mkdir(parents=True, exist_ok=False)
    f = folder / "oracle.f90"
    f.write_text(program)
    exe = folder / "oracle"
    command = [FC, "-O0", "-ffp-contract=off", "-ffree-line-length-none", str(f), "-o", str(exe)]
    start = time.monotonic()
    subprocess.run(command, check=True, timeout=30)
    output = subprocess.check_output([str(exe)], text=True, timeout=10)
    values = np.asarray([int(s, 16) for s in output.split()], np.uint32).view(np.float32)
    assert values.size == 2 * NZ and np.isfinite(values).all()
    result = dict(
        reference="Untouched pristine WRF post-RK microphysics-caller REAL statements (calc_p_rho_phi hypso=2 al; "
                  "moist_physics_prep_em z_at_w/dz8w/rho); REAL-structured CPU-WRF c05 column operands, not forecast truth",
        source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        statements=hyp + [zaw, dz, rho], gravity=gravity,
        constants_sha256=hashlib.sha256(constants.read_bytes()).hexdigest(),
        operand_file=str(wrfout), operand_column=dict(j=J, i=I, faces=NZ + 1),
        inputs={k: v.tolist() for k, v in arrays.items()},
        rho=values[:NZ].tolist(), dz8w=values[NZ:].tolist(), output_hex=output.strip(),
        program_sha256=hashlib.sha256(program.encode()).hexdigest(),
        executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(), compile_command=command,
        elapsed_s=time.monotonic() - start,
        compiler=subprocess.check_output([FC, "--version"], text=True).splitlines()[0])
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("rho", "dz8w", "elapsed_s")}))


if __name__ == "__main__":
    main()
