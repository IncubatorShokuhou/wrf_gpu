"""CK01: pristine WRF km_opt=4 horizontal eddy viscosity chain vs the product diff_opt=1 path on REAL convective frames.
Pristine: module_diffusion_em.F compute_diff_metrics + cal_deform_and_div + smag2d_km copied VERBATIM (regex, hashed)
into a stub module (grid_config_rec_type with every referenced flag: nested .true. for the #148 leaves (d02..d09),
specified/periodic/open/polar .false., sfs_opt 0, mix_full_fields .false., diff_opt 1, c_s 0.25; constants g 9.81,
prandtl 1/3 from module_model_constants), REAL4, caller args as module_first_rk_step_part2.F:434/517/619 (ph_2 + phb,
u_2 v_2 w_2, grid rdx/rdy/dn/dnw/fnm/fnp/cf1-3, smag2d_km(dx, dy, msftx, msfty, zx, zy)), halo 3 edge-padded.
Product: runtime/operational_mode._diffopt1 specified/nested path (wrf_nonperiodic_diffusion_metrics -> horizontal_deformation_2d
with real metrics -> smag2d_horizontal_km with map factors -> owned ring zero), release defaults.
Gate (PRE-REGISTERED here, no frozen Kh bound exists; = the frozen dynamics-operator form of bench_acoustic.compare
without the max(1,.) floor): on owned cells (ring >= 1) max|d| <= 2e-4 max|ref| and rms <= 2e-5 rms_ref for D11 D22 D12
xkmh xkhh. Census: cap cells (xkmh == 10*mlen_h) and max |deformation|.
usage: kh_oracle.py <frame.npz> [...] --out out.json"""
import ctypes, hashlib, json, os, re, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src")]
from gpuwrf._fast_defaults import apply_fast_path_defaults
apply_fast_path_defaults()
import numpy as np
import jax, jax.numpy as jnp
from gpuwrf.dynamics import explicit_diffusion as ED
from gpuwrf.runtime import operational_mode as OM
assert jax.devices()[0].platform == os.environ.get("W43_PLATFORM", "cpu")
WRF = Path("<USER_HOME>/src/wrf_pristine/WRF")
LANE = Path("<USER_HOME>/wrf_gpu2_lanes/w43-conv/kh_pristine")
FC = os.environ.get("FC", "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran")
H = 3
MUT = os.environ.get("W43_MUTANT", "")  # CK02 deletion mutant: nozx (no terrain-slope deformation correction)


def build():
    LANE.mkdir(parents=True, exist_ok=True)
    src = (WRF / "dyn_em/module_diffusion_em.F").read_text()
    names = ("compute_diff_metrics", "cal_deform_and_div", "smag2d_km")
    bodies = {n: re.search(r"(?ims)^\s*SUBROUTINE " + n + r"\b.*?^\s*END SUBROUTINE " + n + r"\b", src).group(0) for n in names}
    flags = sorted({f.lower() for b in bodies.values() for f in re.findall(r"config_flags%(\w+)", b, re.I)})
    vals = {"nested": "logical :: nested=.true.", "c_s": "real :: c_s=0.25", "diff_opt": "integer :: diff_opt=1",
            "sfs_opt": "integer :: sfs_opt=0"}
    cfg = "\n".join(vals.get(f, f"logical :: {f}=.false.") for f in flags)
    hdr = ("module kh_oracle\nuse iso_c_binding\nimplicit none\nreal, parameter :: g=9.81, prandtl=1./3.0\n"
           "integer, parameter :: P_r12=1, P_r13=2, P_r23=3, P_m11=4, P_m22=5, P_m33=6, P_m12=7, P_m13=8, P_m23=9\n"
           "type grid_config_rec_type\n" + cfg + "\nend type\ncontains\n")
    wrap = """
subroutine kh(nx,ny,nz,u,v,w,ph,phb,mux,muy,mvx,mvy,mtx,mty,dn,dnw,fnm,fnp,cf,rdx,rdy,dx,dy,d11,d22,d12,xkmh,xkhh) bind(c)
integer(c_int),value :: nx,ny,nz
real(c_float),value :: rdx,rdy,dx,dy
integer,parameter :: h=3
real(c_float),dimension(1-h:nx+1+h,1:nz+1,1-h:ny+1+h) :: u,v,w,ph,phb,d11,d22,d12,xkmh,xkhh
real(c_float),dimension(1-h:nx+1+h,1-h:ny+1+h) :: mux,muy,mvx,mvy,mtx,mty
real(c_float) :: dn(nz+1),dnw(nz+1),fnm(nz+1),fnp(nz+1),cf(3)
real,dimension(1-h:nx+1+h,1:nz+1,1-h:ny+1+h) :: z,rdz,rdzw,zx,zy,div,d33,d13,d23,xkmv,xkhv
real :: nba(1-h:nx+1+h,1:nz+1,1-h:ny+1+h,1), ub(nz+1), vb(nz+1)
type(grid_config_rec_type) :: cfg
z=0.;rdz=0.;rdzw=0.;zx=0.;zy=0.;div=0.;d33=0.;d13=0.;d23=0.;xkmv=0.;xkhv=0.;nba=0.;ub=0.;vb=0.
d11=0.;d22=0.;d12=0.;xkmh=0.;xkhh=0.
call compute_diff_metrics(cfg,ph,phb,z,rdz,rdzw,zx,zy,rdx,rdy, 1,nx+1,1,ny+1,1,nz+1, 1-h,nx+1+h,1-h,ny+1+h,1,nz+1, &
     1,nx+1,1,ny+1,1,nz+1)
call cal_deform_and_div(cfg,u,v,w,div,d11,d22,d33,d12,d13,d23,nba,1,ub,vb,mux,muy,mvx,mvy,mtx,mty, &
     rdx,rdy,dn,dnw,rdz,rdzw,fnm,fnp,cf(1),cf(2),cf(3),zx,zy, 1,nx+1,1,ny+1,1,nz+1, 1-h,nx+1+h,1-h,ny+1+h,1,nz+1, &
     1,nx+1,1,ny+1,1,nz+1)
call smag2d_km(cfg,xkmh,xkmv,xkhh,xkhv,d11,d22,d12,rdzw,dx,dy,mtx,mty,zx,zy, 1,nx+1,1,ny+1,1,nz+1, &
     1-h,nx+1+h,1-h,ny+1+h,1,nz+1, 1,nx+1,1,ny+1,1,nz+1)
end subroutine
end module
"""
    text = hdr + "\n".join(bodies[n] for n in names) + wrap
    (LANE / "kh_oracle.F90").write_text(text)
    cmd = [FC, "-O2", "-cpp", "-shared", "-fPIC", "-ffree-line-length-none", "-ffp-contract=off", "-fcheck=bounds",
           "-J", str(LANE), str(LANE / "kh_oracle.F90"), "-o", str(LANE / "kh.so")]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode:
        raise RuntimeError(cp.stderr[-3000:])
    (LANE / "manifest.json").write_text(json.dumps({n: hashlib.sha256(b.encode()).hexdigest() for n, b in bodies.items()} | {"flags": flags, "cmd": cmd}, indent=1))
    return LANE / "kh.so"


def pristine(lib, z):
    g = lambda v: np.asarray(z[v], np.float32)  # noqa: E731
    nz, ny, nx = g("T").shape
    def vol(a):
        a = np.pad(a, ((0, nz + 1 - a.shape[0]), (H, ny + 1 + H - a.shape[1]), (H, nx + 1 + H - a.shape[2])), mode="edge")
        return np.asfortranarray(a.transpose(2, 0, 1).astype(np.float32))
    def pl(a):
        a = np.pad(a, ((H, ny + 1 + H - a.shape[0]), (H, nx + 1 + H - a.shape[1])), mode="edge")
        return np.asfortranarray(a.T.astype(np.float32))
    vec = lambda a: np.pad(np.asarray(a, np.float32).reshape(-1), (0, nz + 1 - np.size(a)), mode="edge")  # noqa: E731
    ins = [vol(g(n)) for n in ("U", "V", "W", "PH", "PHB")] + [pl(g(n)) for n in ("MAPFAC_UX", "MAPFAC_UY", "MAPFAC_VX", "MAPFAC_VY", "MAPFAC_MX", "MAPFAC_MY")]
    ins += [vec(g(n)) for n in ("DN", "DNW", "FNM", "FNP")] + [np.asarray([float(z["CF1"]), float(z["CF2"]), float(z["CF3"])], np.float32)]
    outs = [np.zeros((nx + 1 + 2 * H, nz + 1, ny + 1 + 2 * H), np.float32, order="F") for _ in range(5)]
    fn = ctypes.CDLL(str(lib)).kh
    fn.argtypes = [ctypes.c_int] * 3 + [ctypes.c_void_p] * 16 + [ctypes.c_float] * 4 + [ctypes.c_void_p] * 5
    dx, dy = float(z["attr_DX"]), float(z["attr_DY"])
    fn(nx, ny, nz, *[a.ctypes.data for a in ins], 1.0 / dx, 1.0 / dy, dx, dy, *[a.ctypes.data for a in outs])
    cut = lambda a: a[H:H + nx, :nz, H:H + ny].transpose(1, 2, 0).copy()  # mass/corner (i,j) = WRF (i,j), i=1..nx  # noqa: E731
    return dict(zip(("d11", "d22", "d12", "xkmh", "xkhh"), [cut(a) for a in outs]))


def product(z):
    g = lambda v: jnp.asarray(np.asarray(z[v], np.float32))  # noqa: E731
    dx, dy = float(z["attr_DX"]), float(z["attr_DY"])
    zx, zy, rdzw = ED.wrf_nonperiodic_diffusion_metrics(g("PH") + g("PHB"), dx_m=dx, dy_m=dy)
    d11, d22, d12 = ED.horizontal_deformation_2d(g("U"), g("V"), dx_m=dx, dy_m=dy, msftx=g("MAPFAC_MX"), msfty=g("MAPFAC_MY"),
                                                 msfux=g("MAPFAC_UX"), msfuy=g("MAPFAC_UY"), msfvx=g("MAPFAC_VX"), msfvy=g("MAPFAC_VY"),
                                                 zx=None if MUT == "nozx" else zx, zy=None if MUT == "nozx" else zy, rdzw=rdzw, fnm=g("FNM"), fnp=g("FNP"), cf1=g("CF1"), cf2=g("CF2"),
                                                 cf3=g("CF3"), dn=g("DN"), dnw=g("DNW"))
    xkmh, xkhh = ED.smag2d_horizontal_km(d11, d22, d12, dx_m=dx, dy_m=dy, c_s=0.25, msftx=g("MAPFAC_MX"), msfty=g("MAPFAC_MY"))
    return {k: np.asarray(v, np.float64) for k, v in dict(d11=d11, d22=d22, d12=d12, xkmh=xkmh, xkhh=xkhh).items()}


def main():
    paths = [a for a in sys.argv[1:] if a.endswith(".npz")]
    dst = Path(sys.argv[sys.argv.index("--out") + 1])
    lib = build()
    res = {}
    for p in paths:
        z = np.load(p)
        w, q = pristine(lib, z), product(z)
        mlen = np.sqrt(float(z["attr_DX"]) / np.asarray(z["MAPFAC_MX"], np.float64) * float(z["attr_DY"]) / np.asarray(z["MAPFAC_MY"], np.float64))
        r = {"cap_cells_wrf": int(np.count_nonzero(np.abs(w["xkmh"][:, 1:-1, 1:-1] - np.float32(10.0) * mlen[None, 1:-1, 1:-1]) < 1e-3)),
             "max_xkmh_wrf": float(w["xkmh"].max())}
        for k in ("d11", "d22", "d12", "xkmh", "xkhh"):
            a, b = q[k][:, 1:-1, 1:-1], w[k][:, 1:-1, 1:-1].astype(np.float64)
            e = a - b
            rr, mr = float(np.sqrt(np.mean(b * b))), float(np.abs(b).max())
            rec = dict(max_abs=float(np.abs(e).max()), rms=float(np.sqrt(np.mean(e * e))), max_ref=mr, rms_ref=rr,
                       finite=bool(np.isfinite(a).all()))
            rec["passed"] = bool(rec["finite"] and rec["max_abs"] <= 2e-4 * mr and rec["rms"] <= 2e-5 * rr)
            i = np.unravel_index(np.argmax(np.abs(e)), e.shape)
            rec["worst_kji_interior"] = [int(x) for x in i]
            r[k] = rec
        res[os.path.basename(p)] = r
        print(os.path.basename(p), {k: (r[k]["passed"], "%.2e/%.2e" % (r[k]["max_abs"], r[k]["max_ref"])) for k in ("d11", "d22", "d12", "xkmh", "xkhh")},
              "cap", r["cap_cells_wrf"], flush=True)
    res["ALL_PASS"] = all(r[k]["passed"] for f, r in res.items() for k in ("d11", "d22", "d12", "xkmh", "xkhh"))
    dst.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
