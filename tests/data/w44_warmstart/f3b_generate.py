"""Generate frozen ORIGINAL-WRF witnesses for tests/test_w44_cycling_witnesses.py.

Critic F3 (cr-w44-warmstart VERDICT.md): value witnesses that kill
  M5 (U/V coupled with MAPFAC_M) and M8 (input W kept when use_input_w=.false.).

Independence (E98): this script imports NO gpuwrf code. References are produced
only by unedited pristine WRF/WRFDA Fortran compiled with gfortran and called
through ctypes:
  * WRFDA da_calc_mu_uv.inc / da_couple.inc / da_couple_uv.inc (same compile
    command and call mapping as proofs/w44_warmstart/wrfda_oracle.py);
  * dyn_em/module_bc_em.F SUBROUTINE set_w_surface (verbatim text) and the
    dyn_em/start_em.F W gate block (verbatim text, "IF ( ( start_of_simulation
    .OR. config_flags%cycling ) ..." up to the following restart block), wrapped
    in a stub config/grid type harness (iso_c_binding).
Inputs are real WN3 0227 cold-start crops (U/V/MU/MUB/map factors/terrain/
thermodynamics); only W is synthetic (cold wrfinput W is identically 0, a
cycled W is nonzero), seeded and stored.

Usage (CPU only; caller pins cores/guards QUIET):
  python f3b_generate.py PRISTINE_WRF_ROOT CASE_DIR OUT_DIR WORK_DIR
"""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import netCDF4

GFORTRAN = "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"
FLAGS = ["-shared", "-fPIC", "-O0", "-fcheck=all", "-ffp-contract=off"]
SEED = 20261024

STUB = """module module_configure
  implicit none
  type grid_config_rec_type
    logical :: periodic_x = .false.
    logical :: periodic_y = .false.
    logical :: use_input_w = .false.
    logical :: cycling = .true.
    logical :: restart = .false.
  end type grid_config_rec_type
end module module_configure
"""

HARNESS_HEAD = """module wsfc_harness
  use iso_c_binding
  use module_configure
  use module_bc_oracle
  implicit none
  type domain_stub
    integer :: id = 1
    real, allocatable :: w_1(:,:,:), w_2(:,:,:), u_1(:,:,:), u_2(:,:,:), v_1(:,:,:), v_2(:,:,:)
    real, allocatable :: ht(:,:), msftx(:,:), msfty(:,:), znw(:)
    real :: cf1, cf2, cf3, rdx, rdy
  end type domain_stub
contains
  subroutine start_w(nx, ny, nzw, use_input_w, w_in, u_in, v_in, ht_in, msftx_in, msfty_in, &
                     znw_in, cf_in, dx_in, dy_in, w_out, w_set) bind(c, name="start_w")
    integer(c_int), value :: nx, ny, nzw, use_input_w
    real(c_float), intent(in) :: w_in(nx, ny, nzw), u_in(nx+1, ny, nzw-1), v_in(nx, ny+1, nzw-1)
    real(c_float), intent(in) :: ht_in(nx, ny), msftx_in(nx, ny), msfty_in(nx, ny), znw_in(nzw), cf_in(3)
    real(c_float), value :: dx_in, dy_in
    real(c_float), intent(out) :: w_out(nx, ny, nzw)
    integer(c_int), intent(out) :: w_set
    type(domain_stub) :: grid
    type(grid_config_rec_type) :: config_flags
    integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte
    integer :: i, j, k
    logical :: start_of_simulation, fill_w_flag, w_needs_to_be_set
    real :: w_max, w_min
    ! start_em.F:147-149 tile = entire patch; one patch = whole (staggered) domain.
    ids = 1; ide = nx + 1; jds = 1; jde = ny + 1; kds = 1; kde = nzw
    ims = ids; ime = ide; jms = jds; jme = jde; kms = kds; kme = kde
    its = ids; ite = ide; jts = jds; jte = jde; kts = kds; kte = kde
    allocate(grid%w_1(ims:ime,kms:kme,jms:jme), grid%w_2(ims:ime,kms:kme,jms:jme))
    allocate(grid%u_1(ims:ime,kms:kme,jms:jme), grid%u_2(ims:ime,kms:kme,jms:jme))
    allocate(grid%v_1(ims:ime,kms:kme,jms:jme), grid%v_2(ims:ime,kms:kme,jms:jme))
    allocate(grid%ht(ims:ime,jms:jme), grid%msftx(ims:ime,jms:jme), grid%msfty(ims:ime,jms:jme))
    allocate(grid%znw(kms:kme))
    grid%w_2 = 0.; grid%u_2 = 0.; grid%v_2 = 0.; grid%ht = 0.; grid%msftx = 0.; grid%msfty = 0.
    do j = 1, ny
      do k = 1, nzw
        do i = 1, nx
          grid%w_2(i,k,j) = w_in(i,j,k)
        end do
      end do
    end do
    do j = 1, ny
      do k = 1, nzw-1
        do i = 1, nx+1
          grid%u_2(i,k,j) = u_in(i,j,k)
        end do
      end do
    end do
    do j = 1, ny+1
      do k = 1, nzw-1
        do i = 1, nx
          grid%v_2(i,k,j) = v_in(i,j,k)
        end do
      end do
    end do
    do j = 1, ny
      do i = 1, nx
        grid%ht(i,j) = ht_in(i,j)
        grid%msftx(i,j) = msftx_in(i,j)
        grid%msfty(i,j) = msfty_in(i,j)
      end do
    end do
    grid%w_1 = grid%w_2; grid%u_1 = grid%u_2; grid%v_1 = grid%v_2
    grid%znw(kms:kme) = znw_in(1:nzw)
    grid%cf1 = cf_in(1); grid%cf2 = cf_in(2); grid%cf3 = cf_in(3)
    ! module_initialize_real.F:3762 grid%rdx = 1./config_flags%dx (REAL)
    grid%rdx = 1./dx_in; grid%rdy = 1./dy_in
    config_flags%use_input_w = (use_input_w /= 0)
    config_flags%cycling = .true.
    config_flags%restart = .false.
    start_of_simulation = .true.
    w_needs_to_be_set = .false.
    ! ---- verbatim dyn_em/start_em.F W gate block ----
"""

HARNESS_TAIL = """    ! ---- end verbatim ----
    do j = 1, ny
      do k = 1, nzw
        do i = 1, nx
          w_out(i,j,k) = grid%w_2(i,k,j)
        end do
      end do
    end do
    w_set = merge(1, 0, w_needs_to_be_set)
  end subroutine start_w
end module wsfc_harness
"""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract(path, start_text, stop_text, include_stop):
    lines = Path(path).read_text().splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.strip().startswith(start_text)]
    assert len(starts) == 1, (path, start_text, starts)
    s = starts[0]
    stop = next(i for i in range(s + 1, len(lines)) if lines[i].strip().startswith(stop_text))
    end = stop + 1 if include_stop else stop
    while not include_stop and not lines[end - 1].strip():
        end -= 1
    return "".join(lines[s:end]), (s + 1, end)


def crop(ds, name, js, is_):
    var = ds[name]
    data = np.asarray(var[0], np.float32)
    if data.ndim == 1:
        return np.ascontiguousarray(data)
    return np.ascontiguousarray(data[..., js, is_])


def coupling_crop(ds, j0, j1, i0, i1, full):
    """Mass rows j0:j1, cols i0:i1, with their staggered companions."""
    m, us, vs = (slice(j0, j1), slice(i0, i1)), (slice(j0, j1), slice(i0, i1 + 1)), (slice(j0, j1 + 1), slice(i0, i1))
    out = {"MU": crop(ds, "MU", *m), "MUB": crop(ds, "MUB", *m), "MAPFAC_M": crop(ds, "MAPFAC_M", *m),
           "MAPFAC_U": crop(ds, "MAPFAC_U", *us), "MAPFAC_V": crop(ds, "MAPFAC_V", *vs),
           "U": crop(ds, "U", *us), "V": crop(ds, "V", *vs), "C1H": crop(ds, "C1H", *m), "C2H": crop(ds, "C2H", *m)}
    if full:
        out.update(THM=crop(ds, "THM", *m), PH=crop(ds, "PH", *m), QVAPOR=crop(ds, "QVAPOR", *m),
                   C1F=crop(ds, "C1F", *m), C2F=crop(ds, "C2F", *m))
    return out


def couple(native, c, fields):
    pointer = lambda a: a.ctypes.data_as(ct.POINTER(ct.c_float))
    # Fresh Fortran-order COPIES: da_couple works in place and np.asfortranarray of a
    # transposed C array would alias (and silently couple) the stored inputs.
    f3 = lambda a: np.array(np.asarray(a).transpose(2, 1, 0), dtype=np.float32, order="F", copy=True)
    f2 = lambda a: np.array(np.asarray(a).T, dtype=np.float32, order="F", copy=True)
    nz, ny, nx = c["U"].shape[0], c["MU"].shape[0], c["MU"].shape[1]
    bounds = [ct.c_int(n) for n in (1, nx, 1, ny, 1, nz)]
    u, v = f3(c["U"]), f3(c["V"])
    mu, mub = f2(c["MU"]), f2(c["MUB"])
    msfu, msfv = f2(c["MAPFAC_U"]), f2(c["MAPFAC_V"])
    c1, c2 = (np.ascontiguousarray(c[n], np.float32) for n in ("C1H", "C2H"))
    native.da_couple_uv_(*[pointer(a) for a in (u, v, mu, mub, msfu, msfv, c1, c2)], *map(ct.byref, bounds))
    ref = {"U": np.ascontiguousarray(u.transpose(2, 1, 0)), "V": np.ascontiguousarray(v.transpose(2, 1, 0))}
    mass = f2(np.asarray(c["MU"]) + np.asarray(c["MUB"]))
    ones = np.ones((nx, ny), np.float32, order="F")
    for name in fields:
        origin = "THM" if name == "T" else name
        value = f3(c[origin])
        level = "F" if name in ("W", "PH") else "H"
        c1, c2 = (np.ascontiguousarray(c[f"C{n}{level}"], np.float32) for n in (1, 2))
        dims = [ct.c_int(n) for n in (1, nx, 1, ny, 1, value.shape[-1])]
        msf = f2(c["MAPFAC_M"]) if name == "W" else ones
        native.da_couple_(*[pointer(a) for a in (mass, value, msf, c1, c2)], *map(ct.byref, dims))
        ref[name] = np.ascontiguousarray(value.transpose(2, 1, 0))
    return ref


def main():
    pristine, case, out, work = map(Path, sys.argv[1:])
    work.mkdir(parents=True, exist_ok=False)
    out.mkdir(parents=True, exist_ok=True)
    da = pristine / "WRF/var/da/da_update_bc"
    da_sources = [da / n for n in ("da_calc_mu_uv.inc", "da_couple.inc", "da_couple_uv.inc")]
    coupling_cmd = [GFORTRAN, *FLAGS, "-x", "f95", *map(str, da_sources), "-o", str(work / "coupling.so")]
    subprocess.run(coupling_cmd, check=True, capture_output=True, text=True, timeout=120)
    coupling = ct.CDLL(str(work / "coupling.so"))

    bc_path, start_path = pristine / "WRF/dyn_em/module_bc_em.F", pristine / "WRF/dyn_em/start_em.F"
    setw_text, setw_lines = extract(bc_path, "SUBROUTINE set_w_surface(", "END SUBROUTINE set_w_surface", True)
    gate_text, gate_lines = extract(start_path, "IF ( ( start_of_simulation .OR. config_flags%cycling )",
                                    "IF ( .NOT. config_flags%restart ) THEN", False)
    (work / "stub_configure.F90").write_text(STUB)
    (work / "module_bc_oracle.F90").write_text(
        "module module_bc_oracle\n  use module_configure\n  implicit none\ncontains\n" + setw_text
        + "end module module_bc_oracle\n")
    (work / "wsfc_harness.F90").write_text(HARNESS_HEAD + gate_text + HARNESS_TAIL)
    setw_sources = [work / n for n in ("stub_configure.F90", "module_bc_oracle.F90", "wsfc_harness.F90")]
    setw_cmd = [GFORTRAN, *FLAGS, "-cpp", "-ffree-line-length-none", "-J", str(work),
                *map(str, setw_sources), "-o", str(work / "setw.so")]
    subprocess.run(setw_cmd, check=True, capture_output=True, text=True, timeout=120)
    setw = ct.CDLL(str(work / "setw.so"))

    d01, d03 = case / "wrfinput_d01", case / "wrfinput_d03"
    data, refs, checks = {}, {}, {}
    with netCDF4.Dataset(d01) as ds:
        ds.set_auto_mask(False)
        # Real Lambert d01 (9 km): a y-transect (one mass column: U broadcasts
        # against MAPFAC_M under M5) and an x-transect (one mass row: V does).
        crops = {"d01_ytransect": coupling_crop(ds, 20, 32, 3, 4, False),
                 "d01_xtransect": coupling_crop(ds, 50, 51, 100, 112, False)}
    with netCDF4.Dataset(d03) as ds:
        ds.set_auto_mask(False)
        assert int(ds.USE_THETA_M) == 1
        js, is_ = slice(41, 49), slice(40, 49)    # 8x9 columns on the Teide slopes, 1 km
        full = coupling_crop(ds, 41, 49, 40, 49, True)
        nz = full["THM"].shape[0]
        rng = np.random.default_rng(SEED)
        w = (np.float32(0.05) * rng.standard_normal((nz + 1, 8, 9))).astype(np.float32)
        assert np.abs(w[0]).min() > 1e-6
        full["W"] = w
        crops["d03_2d"] = full
        m = (js, is_)
        start = {n: crop(ds, n, *m) for n in (
            "T", "THM", "ALB", "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "PB", "PHB", "MUB", "MU",
            "HGT", "MAPFAC_MX", "MAPFAC_MY", "C1F", "C2F", "C3F", "C4F", "C3H", "C4H", "RDNW", "RDN", "ZNW")}
        start["P_TOP"] = np.asarray(ds["P_TOP"][0], np.float32).reshape(())
        start["U"] = crop(ds, "U", js, slice(40, 50))
        start["V"] = crop(ds, "V", slice(41, 50), is_)
        start["CF"] = np.array([np.asarray(ds[n][0], np.float32).item() for n in ("CF1", "CF2", "CF3")], np.float32)
        start["DXDY"] = np.array([ds.DX, ds.DY], np.float32)
        start["W_NONZERO_SFC"] = w
        w0 = w.copy()
        w0[0] = 0.
        start["W_ZERO_SFC"] = w0
    for name, c in crops.items():
        fields = ("T", "PH", "QVAPOR", "W") if name == "d03_2d" else ()
        before = {k: v.copy() for k, v in c.items()}
        ref = couple(coupling, c, fields)
        assert all(np.array_equal(before[k], c[k]) for k in c), "oracle call mutated stored inputs"
        for key, value in c.items():
            data[f"{name}__{key}"] = value
        for key, value in ref.items():
            refs[f"{name}__ref_{key}"] = value
        # Fixture discrimination: the equal-shape/broadcast M5 quotient differs.
        for f, stag in (("U", "MAPFAC_U"), ("V", "MAPFAC_V")):
            factor = c[stag]
            if f == "U":
                diff = factor[:, :-1] != c["MAPFAC_M"]
            else:
                diff = factor[:-1, :] != c["MAPFAC_M"]
            checks[f"{name}:{stag}!=MAPFAC_M (same-index cells)"] = int(diff.sum())
    nzw = start["ZNW"].shape[0]
    ny, nx = start["HGT"].shape
    pointer = lambda a: a.ctypes.data_as(ct.POINTER(ct.c_float))
    for label, w_in, use in (("nonzero_sfc_use0", "W_NONZERO_SFC", 0), ("nonzero_sfc_use1", "W_NONZERO_SFC", 1),
                             ("zero_sfc_use0", "W_ZERO_SFC", 0), ("zero_sfc_use1", "W_ZERO_SFC", 1)):
        arrays = [np.array(start[n], np.float32, order="C", copy=True) for n in (w_in, "U", "V", "HGT", "MAPFAC_MX", "MAPFAC_MY", "ZNW", "CF")]
        w_out = np.zeros((nzw, ny, nx), np.float32)
        flag = ct.c_int(-1)
        setw.start_w(ct.c_int(nx), ct.c_int(ny), ct.c_int(nzw), ct.c_int(use), *[pointer(a) for a in arrays],
                     ct.c_float(start["DXDY"][0]), ct.c_float(start["DXDY"][1]), pointer(w_out), ct.byref(flag))
        refs[f"setw__ref_W_{label}"] = w_out
        refs[f"setw__ref_set_{label}"] = np.array(flag.value, np.int32)
        checks[f"setw:{label}:w_needs_to_be_set"] = int(flag.value)
        checks[f"setw:{label}:max_abs_W_change_vs_input"] = float(np.abs(w_out - start[w_in]).max())
    for key, value in start.items():
        data[f"setw__{key}"] = value
    npz = out / "f3b_witness.npz"
    np.savez_compressed(npz, **data, **refs)
    provenance = {
        "purpose": "critic F3 deletion witnesses M5 (coupling map factor) and M8 (use_input_w W gate)",
        "generator": {"path": "tests/data/w44_warmstart/f3b_generate.py", "sha256": sha(__file__)},
        "imports_gpuwrf": "gpuwrf" in sys.modules,
        "compiler": {"path": GFORTRAN, "version": subprocess.run([GFORTRAN, "--version"], capture_output=True,
                                                                text=True).stdout.splitlines()[0],
                     "coupling_command": coupling_cmd, "set_w_surface_command": setw_cmd},
        "original_sources": {str(p): sha(p) for p in (*da_sources, bc_path, start_path)},
        "verbatim_extracts": {"module_bc_em.F set_w_surface lines": setw_lines,
                              "start_em.F W gate lines": gate_lines,
                              "set_w_surface_sha256": hashlib.sha256(setw_text.encode()).hexdigest(),
                              "gate_sha256": hashlib.sha256(gate_text.encode()).hexdigest()},
        "harness_sha256": {p.name: sha(p) for p in setw_sources},
        "inputs": {str(p): sha(p) for p in (d01, d03)},
        "crops": {"d01_ytransect": "d01 mass rows 20:32, col 3:4 (+stagger)",
                  "d01_xtransect": "d01 mass row 50:51, cols 100:112 (+stagger)",
                  "d03_2d": "d03 mass rows 41:49, cols 40:49 (+stagger), all 44 levels",
                  "setw": "same d03 crop treated as a whole domain (crop edges = domain edges in BOTH port and original)"},
        "synthetic": {"W": f"0.05*standard_normal, numpy default_rng({SEED}), float32; zero-surface variant sets level 0 to 0"},
        "comparison_rule": "byte equality of float32 arrays (wrfda_oracle.py rule); W gate flag equality",
        "checks": checks,
        "npz_sha256": sha(npz),
        "host": platform.platform(),
        "numpy": np.__version__, "netCDF4": netCDF4.__version__,
    }
    (out / "f3b_provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")
    for so in (work / "coupling.so", work / "setw.so"):
        so.unlink()
    print(json.dumps({"npz": str(npz), "bytes": npz.stat().st_size, "checks": checks}, indent=1))


if __name__ == "__main__":
    main()
