"""SPP MYNN surface-column gate against the unmodified original WRF module."""
from pathlib import Path
import inspect
import os
import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROOFS = ROOT / "proofs/v090"
THRESHOLDS = {"ust": .01, "hfx": .02, "lh": .03, "qsfc": .01, "br": .05,
              "zol": .05, "mol": .05, "psim": .03, "psih": .03, "rmol": .05,
              "u10": .01, "v10": .01, "t2": .001, "th2": .001, "regime": 0., "znt": .001}


@pytest.fixture(scope="module")
def surface_oracle(tmp_path_factory):
    dest = tmp_path_factory.mktemp("stochastic_surface_oracle")
    fc = os.environ.get("FC", "<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/gfortran")
    original = Path("<USER_HOME>/src/wrf_pristine/WRF/phys/module_sf_mynn.F")
    assert original.read_bytes() == (PROOFS/"module_sf_mynn_pristine.f90").read_bytes()
    (dest/"module_sf_mynn.f90").write_bytes(original.read_bytes())
    (dest/"module_model_constants.f90").write_bytes((PROOFS/"module_model_constants.f90").read_bytes())
    driver = (PROOFS/"mynn_oracle_driver.f90").read_text()
    assert driver.count("ZNT(i),UST(i),MOL(i),QSFC(i),HFX(i),QFX(i)") == 1
    driver = driver.replace("ZNT(i),UST(i),MOL(i),QSFC(i),HFX(i),QFX(i)",
                            "ZNT(i),UST(i),MOL(i),QSFC(i),HFX(i),QFX(i),rstoch1D(i)")
    driver = driver.replace("rstoch1D=0.0; ", "")
    (dest/"driver.f90").write_text(driver)
    subprocess.run([fc,"-O2","-ffp-contract=off","-ffree-line-length-none","-cpp",
                    "module_model_constants.f90","module_sf_mynn.f90","driver.f90","-o","oracle"],
                   cwd=dest, env={**os.environ,"TMPDIR":str(dest)},check=True,timeout=120,
                   capture_output=True)
    return dest/"oracle"


@pytest.mark.parametrize("pattern",[-.3, .27])
def test_spp_surface_original_wrf(surface_oracle, pattern):
    # Reuse the established exact-input adapter and its frozen per-field gates.
    # Only the input driver is extended; all Fortran physics stays byte-identical.
    sys.path.insert(0,str(PROOFS))
    try:
        import mynnsl_parity as reference
    finally:
        sys.path.pop(0)
    cases, _ = reference.make_cases()
    cases += [dict(cases[0],snowh=.2,tsk=270.,t1d=273.,qv=.002)]
    lines = [f"{len(cases)} 2 1 0 0 1 3000"]
    for case in cases:
        lines.append(" ".join(str(float(np.float32(case[k]))) for k in reference.IN_COLS)+f" {pattern}")
    result = subprocess.run([str(surface_oracle)],input="\n".join(lines)+"\n",
                            text=True,capture_output=True,check=True,timeout=60)
    rows = [line.split() for line in result.stdout.splitlines() if line and not line.startswith("#")]
    oracle = {key:np.asarray([float(row[i+1]) for row in rows])
              for i,key in enumerate(reference.OUT_COLS)}
    # The baseline helper omitted snowh because its cases have no snow. Extend
    # that input mapping and pass the pattern to the real production entry point.
    source = inspect.getsource(reference.run_production)
    source = source.replace("d = surface_layer_with_diagnostics(st)",
        "st.snowh = jnp.asarray(g('snowh').reshape(shape))\n"
        "    st.rho = jnp.asarray(g('rho').reshape(shape))\n"
        "    d = surface_layer_with_diagnostics(st, spp_pattern=jnp.full(shape,pattern))")
    namespace = {**vars(reference),"pattern":pattern}
    exec(compile(source,"<SPP surface exact-input adapter>","exec"),namespace)
    production = namespace["run_production"](cases)
    for key,tolerance in THRESHOLDS.items():
        a,b = production[key],oracle[key]
        absolute = np.max(np.abs(a-b))
        relative = np.max(np.abs(a-b)/np.maximum(np.abs(b),1.e-12))
        assert relative <= tolerance or absolute < 1.e-6, (key,absolute,relative,tolerance)
    # WRF does not persist its local perturbed roughness into prognostic ZNT.
    np.testing.assert_allclose(production["znt"][:8],.1,rtol=1.e-6)


@pytest.mark.parametrize("amplitude",[-.27,.27])
def test_spp_entrainment_pristine_dmp(tmp_path,amplitude):
    import importlib.util
    import json
    import jax
    helper = ROOT/"tests/v025/b_phys/plume_wrf_oracle.py"
    spec = importlib.util.spec_from_file_location("stochastic_dmp_reference",helper)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    namespace = {**vars(reference),"FC":os.environ.get("FC",
        "<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/gfortran")}
    source = inspect.getsource(reference.build).replace("def build(","def build_spp(")
    needle = "    (output / 'oracle.f90').write_text(source)"
    assert source.count(needle) == 1
    source = source.replace(needle,
        "    source = source.replace('pattern_spp_pbl(k) = 0.0_kind_phys',"
        f"'pattern_spp_pbl(k) = {amplitude}_kind_phys*cos(.31_kind_phys*real(k-1,kind_phys))')\n"
        "    source = source.replace('spp_pbl = 0','spp_pbl = 1')\n"+needle)
    exec(compile(source,"<original DMP input driver>","exec"),namespace)
    exe = namespace["build_spp"](tmp_path/"build")
    fixture = json.loads((ROOT/"tests/v025/b_phys/fixtures/limiter_active_wrf.json").read_text())
    inputs = {key:np.asarray(value,np.float32) for key,value in fixture["inputs"].items()}
    expected,_ = reference.run(exe,tmp_path/"run",inputs)
    pattern = np.float32(amplitude)*np.cos(np.float32(.31)*np.arange(inputs["thl"].shape[-1],dtype=np.float32))[None]
    from gpuwrf.kernels.phys_mynn_plume import dmp_mf_columns_native
    result = dmp_mf_columns_native(**inputs,spp_pattern=jnp.asarray(pattern))
    assert float(result["active"][0]) == 1
    count = 0
    for name,value in result.items():
        key = "edmf_"+name
        if key not in expected:
            continue
        actual,truth = np.asarray(value[0]).reshape(-1),expected[key].reshape(-1)
        assert np.isfinite(actual).all()
        if actual.shape != truth.shape:
            actual = actual[:truth.size]
        error = np.max(np.abs(actual-truth))/max(1.e-30,np.max(np.abs(truth)))
        assert error <= fixture["tolerance"], (name,error)
        count += 1
    assert count >= 12
    assert "float64" not in str(jax.make_jaxpr(lambda p:dmp_mf_columns_native(**inputs,spp_pattern=p))(
        jnp.asarray(pattern)))


@pytest.fixture(scope="module")
def pbl_oracle(tmp_path_factory):
    import ctypes
    import hashlib
    import json
    import re
    dest = tmp_path_factory.mktemp("original_spp_pbl")
    wrf = Path("<USER_HOME>/src/wrf_pristine/WRF")
    phys = wrf/"phys"
    text = (phys/"module_bl_mynnedmf.F").read_text()
    start = text.index("    if (spp_pbl==1) then",text.index("SUBROUTINE  mym_turbulence"))
    body = re.search(r"    if \(spp_pbl==1\) then.*?    endif",text[start:],re.S).group(0)
    wrapper = '''subroutine spp_cloud(n,b,th,p,ex,dz,qw,qc,qi,qs,qsq,pat,co,io,cf) bind(C)
use iso_c_binding
use module_bl_mynnedmf,only:mym_condensation
implicit none
integer(c_int),value::n,b
real(c_float),intent(in)::th(n,b),p(n,b),ex(n,b),dz(n,b),qw(n,b),qc(n,b),qi(n,b),qs(n,b),qsq(n,b),pat(n,b)
real(c_float),intent(out)::co(n,b),io(n,b),cf(n,b)
real(c_float)::zw(n+1),z(n),vt(n),vq(n),sgm(n)
integer::j,k
z=0.
do j=1,b
zw(1)=0.
do k=1,n
zw(k+1)=zw(k)+dz(k,j)
enddo
vt=0.;vq=0.;sgm=0.
call mym_condensation(1,n,1000.,dz(:,j),zw,1.,th(:,j),qw(:,j),qw(:,j)-qc(:,j)-qi(:,j)-qs(:,j), &
 qc(:,j),qi(:,j),qs(:,j),p(:,j),ex(:,j),z,qsq(:,j),z,z,z,2,co(:,j),io(:,j),cf(:,j), &
 800.,0.,vt,vq,th(:,j),sgm,1,pat(:,j))
enddo
end subroutine
subroutine spp_diff(n,dfm,dfh,pattern_spp_pbl1,zw) bind(C)
use iso_c_binding
implicit none
integer(c_int),value::n
real(c_float),intent(inout)::dfm(n),dfh(n)
real(c_float),intent(in)::pattern_spp_pbl1(n),zw(n)
real(c_float),parameter::zero=0.
integer::k,kts,kte,spp_pbl
kts=1;kte=n;spp_pbl=1
''' + body + '\nend subroutine\n'
    (dest/"oracle.f90").write_text(wrapper)
    fc = os.environ.get("FC","<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/gfortran")
    objects = [phys/"module_bl_mynnedmf.o",phys/"module_bl_mynnedmf_common.o",
               wrf/"share/module_model_constants.o",phys/"ccpp_kind_types.o"]
    command = [fc,"-O2","-ffp-contract=off","-fPIC","-shared","-ffree-line-length-none",
               "-I"+str(phys),"oracle.f90",*[str(p) for p in objects],"-o","oracle.so"]
    subprocess.run(command,cwd=dest,env={**os.environ,"TMPDIR":str(dest)},capture_output=True,check=True,timeout=120)
    (dest/"manifest.json").write_text(json.dumps(dict(source_sha256=hashlib.sha256(text.encode()).hexdigest(),
        objects={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in objects},command=command),indent=2))
    lib = ctypes.CDLL(str(dest/"oracle.so"))
    ptr = np.ctypeslib.ndpointer(dtype=np.float32,flags="C_CONTIGUOUS")
    lib.spp_cloud.argtypes = [ctypes.c_int,ctypes.c_int]+[ptr]*13
    lib.spp_cloud.restype = None
    lib.spp_diff.argtypes = [ctypes.c_int]+[ptr]*4
    lib.spp_diff.restype = None
    return lib


@pytest.mark.parametrize("amplitude",[-.27,.27])
def test_spp_cloudpdf_pristine_real_drizzle(pbl_oracle,amplitude):
    import json
    from gpuwrf.physics.mynn_sgs_cloud import mym_condensation_cloudpdf2
    data = json.loads((ROOT/"tests/v025/b_phys/fixtures/mynn_sgs_drizzle_wrf.json").read_text())
    a = np.asarray(data["input_values"],np.float32)
    values = {name:np.ascontiguousarray(a[:,:,index]) for index,name in enumerate(data["input_fields"])}
    th,p,dz = values["theta"],values["p"],values["dz"]
    ex = np.ascontiguousarray((p/np.float32(100000.))**np.float32(2./7.))
    denominator = np.float32(1)+values["qv"]
    qc,qi,qs = [np.ascontiguousarray(values[k]/denominator) for k in ("qc","qi","qs")]
    qw = np.ascontiguousarray(values["qv"]/denominator+qc+qi+qs)
    b,n = th.shape
    pattern = np.ascontiguousarray(np.broadcast_to(np.float32(amplitude)*np.cos(np.float32(.31)*np.arange(n,dtype=np.float32)),th.shape))
    expected = [np.empty_like(th) for _ in range(3)]
    pbl_oracle.spp_cloud(n,b,th,p,ex,dz,qw,qc,qi,qs,values["qsq"],pattern,*expected)
    actual = mym_condensation_cloudpdf2(theta=jnp.asarray(th),p=jnp.asarray(p),exner=jnp.asarray(ex),
        dz=jnp.asarray(dz),qw=jnp.asarray(qw),qc=jnp.asarray(qc),qi=jnp.asarray(qi),qs=jnp.asarray(qs),
        qsq=jnp.asarray(values["qsq"]),pblh=jnp.full((b,),800.,jnp.float32),spp_pattern=jnp.asarray(pattern))
    assert np.count_nonzero(expected[2]) > 0
    for candidate,truth in zip(actual,expected):
        np.testing.assert_allclose(candidate,truth,rtol=8.e-6,atol=64*np.finfo(np.float32).eps*np.max(np.abs(truth)))


def test_spp_diffusion_original_source_height_decay(pbl_oracle):
    from gpuwrf.physics.stochastic_spp import diffusion
    z = np.asarray([0.,1000.,7999.,8000.,9000.,14000.,24000.,35000.],np.float32)
    pattern = np.asarray([-.3,.27,.15,-.2,.3,-.25,.25,-.1],np.float32)
    dfm,dfh = np.linspace(.2,5,len(z),dtype=np.float32),np.linspace(.4,7,len(z),dtype=np.float32)
    expected = [dfm.copy(),dfh.copy()]
    pbl_oracle.spp_diff(len(z),*expected,pattern,z)
    actual = diffusion(jnp.asarray(dfm),jnp.asarray(dfh),jnp.asarray(pattern),jnp.asarray(z))
    for candidate,truth in zip(actual,expected):
        np.testing.assert_allclose(candidate,truth,rtol=4.e-6,atol=1.e-6)
