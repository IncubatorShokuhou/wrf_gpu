"""Extract literal moist_physics_prep/finish assignments from pristine WRF.

No pristine body is edited. This small operator oracle supplies prescribed dry
MP outputs, isolating the adapter thermodynamic boundary from column physics.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

assert os.environ.get("JAX_PLATFORMS") == "cpu"
assert not Path("/tmp/wrf_gpu2_quiet").exists()
root, build = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
build.mkdir(parents=True, exist_ok=True)
wrf = root / "data/wrf_pristine/WRF/dyn_em/module_big_step_utilities_em.F"
lines = wrf.read_text().splitlines()
prep = next(l for l in lines if "th_phy(i,k,j) = (t_new(i,k,j) + t0) / (1. + (R_v/R_d) * qv(i,k,j))" in l)
idx = next(i for i,l in enumerate(lines) if "t_new(i,k,j) =  h_diabatic(i,k,j)*(1." in l)
finish = "\n".join(lines[idx:idx+3])
assert finish.endswith("- T0")
driver = f"""program thermo
implicit none
integer,parameter :: n=6
integer :: i,j,k
real :: r_v=461.6,r_d=287.,t0=300.,mpten,qvten
real :: t_new(1,n,1),qv(1,n,1),th_phy(1,n,1),h_diabatic(1,n,1),qv_diabatic(1,n,1)
t_new(1,:,1)=[-30.,-10.,0.,10.,20.,40.]
qv(1,:,1)=[0.,.001,.005,.01,.02,.025]
do k=1,n
i=1
j=1
write(*,*) t_new(i,k,j)+t0,qv(i,k,j)
{prep}
write(*,*) th_phy(i,k,j)
h_diabatic(i,k,j)=th_phy(i,k,j)
qv_diabatic(i,k,j)=qv(i,k,j)
th_phy(i,k,j)=th_phy(i,k,j)+real(k)*.25
qv(i,k,j)=qv(i,k,j)-real(k-1)*.0001
mpten=th_phy(i,k,j)-h_diabatic(i,k,j)
qvten=qv(i,k,j)-qv_diabatic(i,k,j)
write(*,*) th_phy(i,k,j),qv(i,k,j)
{finish}
write(*,*) t_new(i,k,j)+t0
enddo
end program
"""
src = build / "thermo.f90"
src.write_text(driver)
compiler = "<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/x86_64-conda-linux-gnu-gfortran"
flags = ["-O0", "-ffp-contract=off", "-finit-real=zero", "-finit-integer=0", "-finit-logical=false"]
subprocess.run([compiler, *flags, str(src), "-o", str(build/"thermo")], check=True, timeout=60)
output = subprocess.check_output([str(build/"thermo")], text=True, timeout=30)
values = [float(x) for x in output.split()]
assert len(values) == 36
keys = ["theta_m_in", "qv_in", "dry_theta_in", "dry_theta_out", "qv_out", "theta_m_out"]
result = {key: values[i::6] for i,key in enumerate(keys)}
result["provenance"] = dict(source=str(wrf.resolve()), source_sha256=hashlib.sha256(wrf.read_bytes()).hexdigest(),
                             prep=prep, finish=finish, compiler=compiler, flags=flags,
                             driver_sha256=hashlib.sha256(driver.encode()).hexdigest(),
                             scope="literal REAL4 WRF thermodynamic operator; prescribed MP output; no coupled trajectory")
(root / "proofs/t_mprain/thermo_oracle.json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result, indent=2))
