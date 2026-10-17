"""CAM caller cloud operands from retained c28 h4 vs unchanged original WRF.

The history State is reconstructed only for a same-input caller check, not a
trajectory/restart. Pristine cal_cldfra1 sees the exact T/P/moisture supplied to
CAM. Its independently computed one-input-ULP envelope is frozen by this script.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

root, case, output = (Path(s).resolve() for s in sys.argv[1:4])
sys.path.insert(0, str(root / "scripts/v025"))
import cpu_guard  # noqa: E402,F401
sys.path.insert(0, str(root / "src"))
import gpuwrf  # noqa: E402,F401
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from gpuwrf.coupling import cam_radiation as cam  # noqa: E402
from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains  # noqa: E402
from gpuwrf.io.netcdf_lock import Dataset  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402

assert jax.devices()[0].platform == "cpu" and cam.pc._rrtmg_real_enabled()
cfg = NestedPipelineConfig(case / "input", output.parent / "real-output", output.parent / "real-proof", hours=1, max_dom=1)
_, bundles, _, _, _, carries = _load_domains(cfg, ("d01",))
nml, state = bundles["d01"].namelist, carries["d01"].state
history = case / "cpu/wrfout_d01_2023-01-15_04:00:00"
with Dataset(history) as ds:
    read = lambda name: jnp.asarray(np.asarray(ds[name][0]), jnp.float32)
    qv = read("QVAPOR")
    p, ph, mu = read("P") + read("PB"), read("PH") + read("PHB"), read("MU") + read("MUB")
    state = state.replace(theta=(read("T")+jnp.float32(300.))*(1.+cam.pc._wrf_phy_rv_over_rd(jnp.float32)*qv),
                          p=p, p_total=p, ph=ph, ph_total=ph, mu=mu, mu_total=mu, qv=qv,
                          qc=read("QCLOUD"), qi=read("QICE"), qs=read("QSNOW"), qg=read("QGRAUP"),
                          t_skin=read("TSK"), cldfra_bl=jnp.zeros_like(qv), qc_bl=jnp.zeros_like(qv), qi_bl=jnp.zeros_like(qv))
kw = dict(time_utc=nml.time_utc, lead_seconds=jnp.float32(14400.), radiation_static=nml.radiation_static,
          land_state=carries["d01"].noahmp_land)
old = jax.jit(lambda s: cam.cam_operands(s, nml.grid, **kw))(state)
new = jax.jit(lambda s: cam.cam_operands(s, nml.grid, **kw, **om._rrtmg_pbl_cloud_kwargs(nml)))(state)
jax.block_until_ready(new)
arrays = {n:np.asarray(getattr(new,n)) for n in ("t_phy", "p_phy", "qv", "qc", "qi", "qs", "cldfra")}
score = (arrays["qc"] + arrays["qi"] + arrays["qs"]).sum(axis=1)
ids = np.unique(np.r_[np.argsort(score)[-8:], np.argmin(score),
                        np.argmax(arrays["qc"].sum(axis=1)), np.argmax(arrays["qi"].sum(axis=1)), np.argmax(arrays["qs"].sum(axis=1))])
inputs = np.stack([arrays[n][ids] for n in ("t_phy", "p_phy", "qv", "qc", "qi", "qs")], axis=-1).reshape(-1,6)
expanded = np.repeat(inputs[:,None,:], 13, axis=1)
for k in range(6):
    expanded[:,2*k+1,k] = np.nextafter(inputs[:,k], np.float32(-np.inf))
    expanded[:,2*k+2,k] = np.nextafter(inputs[:,k], np.float32(np.inf))
original = Path("<USER_HOME>/src/wrf_pristine/WRF/phys/module_radiation_driver.F").read_bytes()
start = original.index(b"   SUBROUTINE cal_cldfra1(")
end = original.index(b"   END SUBROUTINE cal_cldfra1",start)+len(b"   END SUBROUTINE cal_cldfra1")
body = original[start:end]+b"\n"
build = output.parent / "real-cloud-build"
build.mkdir(exist_ok=False)
(build/"cal_cldfra1.f90").write_bytes(body)
compiler = "<USER_HOME>/miniconda3/envs/wrfbuild/bin/x86_64-conda-linux-gnu-gfortran"
flags = ["-O2", "-ffp-contract=off", "-ffree-line-length-none", "-finit-real=zero", "-finit-integer=0", "-finit-logical=false",
         "-I<USER_HOME>/src/wrf_pristine/WRF/frame"]
subprocess.run([compiler,*flags,str(build/"cal_cldfra1.f90"),str(root/"proofs/t_camuw/cloud_oracle_driver.f90"),
                "-o",str(build/"oracle")],check=True,capture_output=True,timeout=60)
text = "\n".join("8 "+" ".join(map(str,row)) for row in expanded.reshape(-1,6))+"\n"
run = subprocess.run([str(build/"oracle")],input=text,text=True,capture_output=True,check=True,timeout=60)
values = np.array([[float(v) for v in line.split()] for line in run.stdout.splitlines()]).reshape(len(inputs),13,2)
assert np.isfinite(values).all()
expected = values[:,0,0]
bound = sum(np.maximum(abs(values[:,2*k+1,0]-expected),abs(values[:,2*k+2,0]-expected)) for k in range(6))
actual = arrays["cldfra"][ids].reshape(-1).astype(np.float64)
delta = abs(actual-expected)
outside = delta > bound
receipt = dict(source=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip(),
    reference="verbatim original WRF cal_cldfra1; caller operands from c28 original CPU-WRF h4 reconstruction",
    history=str(history), history_sha256=hashlib.sha256(history.read_bytes()).hexdigest(),
    source_sha256=hashlib.sha256(original).hexdigest(), subroutine_sha256=hashlib.sha256(body).hexdigest(),
    compiler=compiler, flags=flags, shape=list(state.theta.shape), column_indices=ids.tolist(),
    rows=len(inputs), native_calls=int(values.shape[0]*13), branches=np.unique(values[:,:,1]).tolist(),
    outside_input_ulp_envelope=int(np.count_nonzero(outside)), max_abs=float(delta.max()),
    full_old_cloud_nonzero=int(np.count_nonzero(np.asarray(old.cldfra))), full_new_cloud_nonzero=int(np.count_nonzero(arrays["cldfra"])),
    sampled_original_cloud_nonzero=int(np.count_nonzero(expected)), cache="NONE", gpu_executed=False,
    scope="Caller and original resolved-cloud gate; unchanged CAM spectral gates reused, final GPU D6 pending")
np.savez_compressed(output.with_suffix(".npz"), inputs=inputs, expected=expected, bound=bound, actual=actual, column_indices=ids)
output.write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
shutil.rmtree(build)
print(json.dumps(receipt,indent=2))
assert not outside.any(), receipt
assert receipt["full_old_cloud_nonzero"] == 0 < receipt["full_new_cloud_nonzero"]
