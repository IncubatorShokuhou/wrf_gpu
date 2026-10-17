"""Active missing regimes from rv-s2-mpmisc: pristine REAL operator witnesses.

These isolate original statements with actual production tendency/velocity
operands; low-vapor cases are constructed, not original full-column forecasts.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics.thompson_hail import column, native


ROOT = Path(column.__file__).resolve().parents[4]
PROOF = ROOT / "proofs/v040/mpmisc/thompson38"
sys.path.insert(0, str(PROOF))
from literal_reference import executor, LiteralState
from check_columns import parse


@pytest.fixture(scope="module")
def original_operators(tmp_path_factory):
    """No-FMA REAL statements from pristine module_mp_thompson.F."""
    source = Path("<USER_HOME>/src/wrf_pristine/WRF/phys/module_mp_thompson.F")
    expected = json.loads((PROOF / "original_build.json").read_text())["source_sha256"][source.name]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
    build = tmp_path_factory.mktemp("mp38_original_operators")
    program = build / "operators.f90"
    program.write_text('''program operators
implicit none
integer :: mode,n,k,ksed
real :: raw_qv,qvten,dt,qv,hgt_agl
real,allocatable :: dzq(:),rc(:),nc(:),vtck(:),vtnck(:),rho(:)
real,allocatable :: sed_c(:),sed_n(:),qcten(:),ncten(:)
read(*,*) mode,n
if (mode.eq.1) then
  do k=1,n
    read(*,*) raw_qv,qvten,dt
    ! Original :3192 and :3974; raw qv1d retained across cumulative tendency.
    qv = MAX(1.E-10, raw_qv + DT*qvten)
    write(*,'(ES24.16)') qv
  enddo
else
  allocate(dzq(n),rc(n),nc(n),vtck(n),vtnck(n),rho(n))
  allocate(sed_c(n+1),sed_n(n+1),qcten(n),ncten(n))
  do k=1,n
    read(*,*) dzq(k),rc(k),nc(k),vtck(k),vtnck(k),rho(k)
  enddo
  ! Original :3646-3655, kts=1 and kte=n, R2=1.E-6.
  ksed=1
  hgt_agl=0.
  do k=1,n-1
    if (rc(k).gt.1.E-6) ksed=k
    hgt_agl=hgt_agl+dzq(k)
    if (hgt_agl.gt.500.) exit
  enddo
  sed_c=0.;sed_n=0.;qcten=0.;ncten=0.
  do k=ksed,1,-1
    sed_c(k)=vtck(k)*rc(k)
    sed_n(k)=vtnck(k)*nc(k)
  enddo
  ! Original :3830-3833, zero initial cloud sedimentation tendency.
  do k=ksed,1,-1
    qcten(k)=qcten(k)+(sed_c(k+1)-sed_c(k))*(1./dzq(k))*(1./rho(k))
    ncten(k)=ncten(k)+(sed_n(k+1)-sed_n(k))*(1./dzq(k))*(1./rho(k))
  enddo
  do k=1,n
    write(*,'(2(ES24.16,1X))') qcten(k),ncten(k)
  enddo
endif
end program
''')
    exe = build / "operators"
    compiler = "<USER_HOME>/miniconda3/envs/wrfbuild/bin/x86_64-conda-linux-gnu-gfortran"
    subprocess.run([compiler, "-O2", "-ffp-contract=off", str(program), "-o", str(exe)],
                   check=True, timeout=30, env={**os.environ, "TMPDIR": str(build)})

    def run(mode, rows):
        text = f"{mode} {len(rows)}\n" + "\n".join(" ".join(str(float(x)) for x in row) for row in rows) + "\n"
        return np.asarray(subprocess.check_output([str(exe)], input=text, text=True, timeout=10).split(), np.float32)

    return run


def test_cloud_sedimentation_raw_submetre_dz_and_band(original_operators):
    original = json.loads((PROOF / "original_cloud_sed_kind.json").read_text())["original"]
    coeff = SimpleNamespace(**{name: jnp.asarray(value) for name, value in native._cloud_coefficients().items()})
    # Second column crosses 500 m only if the inherited one-metre floor is used:
    # raw .25+499.6=499.85 visits level3, floored 1+499.6=500.6 stops at level2.
    state = SimpleNamespace(qc=jnp.asarray([[original["rc(k)"], 0., 0., 0.],
                                          [original["rc(k)"]] * 4], jnp.float32),
        Nc=jnp.asarray([[original["nc(k)"], 2., 2., 2.], [original["nc(k)"]] * 4], jnp.float32),
        rho=jnp.ones((2, 4), jnp.float32), w=jnp.zeros((2, 4), jnp.float32),
        dz=jnp.asarray([[.25, 10., 10., 10.], [.25, 499.6, .25, 10.]], jnp.float32))
    with jax.disable_jit():
        vt, vtn = native.AERO._cloud_water_fall_speeds_aero(state, coeff, wrf_hail=True)
        mass, number, _ = native.AERO._sed_cloud_water_aero(state, .01, coeff,
                                                          cloud_sed_on=True, return_rates=True)
    for col in range(2):
        rows = np.stack([np.asarray(value[col]) for value in (state.dz, state.qc, state.Nc, vt, vtn, state.rho)], axis=1)
        expected = original_operators(2, rows).reshape(4, 2)
        np.testing.assert_array_equal(np.asarray(mass[col]), expected[:, 0])
        np.testing.assert_array_equal(np.asarray(number[col]), expected[:, 1])
    assert mass[0, 0] < 0 and number[0, 0] < 0  # Isolated quarter-metre flux.
    assert mass[1, 2] < 0 and number[1, 2] < 0  # Active sub-metre flux at band edge.


@pytest.mark.parametrize("raw_qv", [-1.e-7, 5.e-11])
def test_raw_vapor_every_cumulative_reconstruction(original_operators, raw_qv):
    fields = dict(qv="QV", qc="QC", qr="QR", qi="QI", qs="QS", qg="QG", Ni="NI", Nr="NR",
                  Nc="NC", Ng="NG", qvolg="QB", nwfa="NWFA", nifa="NIFA", T="T")
    active = 0
    for case in (2, 4):
        ref = parse(PROOF / f"original_case{case}.txt")
        cols = {name: np.asarray(value, np.float32)[None, :] for name, value in ref["columns"].items()}
        values = {name: cols[code + "_IN"].copy() for name, code in fields.items()}
        values["qv"][:] = np.float32(raw_qv)
        raw = values["qv"].copy()
        rho = np.float32(.622) * cols["P"] / (np.float32(287.04) * values["T"] * (raw + np.float32(.622)))
        state = LiteralState(**values, p=cols["P"], rho=rho, dz=cols["DELZ"], w=cols.get("W_IN", np.zeros_like(rho)))
        captured = []

        def capture(frame, event, result):
            if event != "return":
                return
            if frame.f_code.co_name == "_wrf_mixed_phase_sources":
                ctx = frame.f_locals["hail_context"]
                captured.append((np.asarray(ctx.baseline.qv).copy(), np.asarray(frame.f_locals["qvten"]).copy(), np.asarray(result[0].qv).copy()))
            elif frame.f_code.co_name == "_render":
                captured.append((np.asarray(frame.f_locals["baseline"].qv).copy(),
                    np.asarray(frame.f_locals["tendencies"]["qv"]).copy(), np.asarray(result.qv).copy()))

        dt = ref["scalars"]["DT"]
        with executor() as body:
            previous_profile = sys.getprofile()
            sys.setprofile(capture)
            try:
                body(state, dt)
            finally:
                sys.setprofile(previous_profile)
        assert len(captured) == 5  # Mixed source, saturation, evaporation, sedimentation, final writeback.
        active += np.count_nonzero(captured[0][1] > 0)
        for baseline, tendency, actual in captured:
            np.testing.assert_array_equal(baseline, raw)
            rows = np.stack([raw.ravel(), tendency.ravel(), np.full(raw.size, dt, np.float32)], axis=1)
            expected = original_operators(1, rows).reshape(raw.shape)
            np.testing.assert_array_equal(actual, expected)
    assert active == 64  # Both reviewer witnesses exercise active source thermodynamics.
