"""Stage original WRF wrapper savepoints for all four P3 Registry variants.

These are constructed adversarial columns, not a coupled forecast. The full
pristine module is copied byte-for-byte; keyword calls reproduce WRF's caller
mapping, including optional NC/QZI presence, old TH/QV, and precipitation.
No completed port is compared here. Fixtures enable that subsequent work.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


def instrument_stages(text):
    fields = ('th','qv','qc','nc','qr','nr','t','rho','qvs','qvi','sup','supi',
              'qitot','nitot','qirim','birim','diag_effc','diag_effi','diag_di','diag_vmi','diag_rhoi')
    cats = {'qitot','nitot','qirim','birim','diag_effi','diag_di','diag_vmi','diag_rhoi'}
    args = ','.join(f'{name}(i,:,:)' if name in cats else f'{name}(i,:)' for name in fields)
    markers = ('enddo k_loop_1', 'enddo k_loop_main', 'enddo iice_loop_sedi_ice',
               'enddo k_loop_fz', 'enddo k_loop_final_diagnostics')
    for tag, marker in enumerate(markers, 1):
        import re
        pattern = rf'(?im)^(\s*{marker}[^\n]*)$'
        text, count = re.subn(pattern, lambda m: m[0] + f'\n call s2_p3_dump({tag},it,i,nCat,kts,kte,{args})', text)
        assert count == 1, marker
    decl = '\n'.join(f'real,intent(in) :: {name}(nk,ncat_stage)' if name in cats else f'real,intent(in) :: {name}(nk)' for name in fields)
    procedure = f'''\nsubroutine s2_p3_dump(tag,it,i,ncat_stage,kts,kte,{','.join(fields)})
integer,intent(in) :: tag,it,i,ncat_stage,kts,kte
integer :: nk,unit
character(len=32) :: file
{decl.replace('nk','kte-kts+1')}
nk=kte-kts+1
write(file,'("stage_",i1,".bin")') tag
open(newunit=unit,file=trim(file),access='stream',form='unformatted',status='unknown',position='append')
write(unit) tag,it,i,ncat_stage,nk
write(unit) {','.join(fields)}
close(unit)
end subroutine s2_p3_dump
'''
    import re
    text, count = re.subn(r'(?im)^(\s*end\s+module\b.*)$', lambda m: procedure + m[0], text)
    assert count == 1
    return text

FIELDS = ("th", "qv", "qc", "qr", "nr", "qi", "ni", "qir", "bir", "nc",
          "qi2", "ni2", "qir2", "bir2", "qzi", "th_old", "qv_old", "p", "exner", "dz", "w",
          "ze", "re_cloud", "re_ice", "vm", "di", "rhoi", "vm2", "di2", "rhoi2")
UNITS = dict.fromkeys(FIELDS, "1")
UNITS.update({k: "kg kg-1" for k in ("qv", "qc", "qr", "qi", "qir", "qi2", "qir2", "qv_old")})
UNITS.update({k: "kg-1" for k in ("nr", "ni", "nc", "ni2")})
UNITS.update({k: "m" for k in ("dz", "re_cloud", "re_ice", "di", "di2")})
UNITS.update({k: "m s-1" for k in ("w", "vm", "vm2")})
UNITS.update({"th": "K", "th_old": "K", "p": "Pa", "bir": "m3 kg-1", "bir2": "m3 kg-1",
              "qzi": "m3 kg-1 (advected sqrt(N*Z))", "ze": "dBZ", "rhoi": "kg m-3", "rhoi2": "kg m-3"})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def driver():
    def a(name):
        return f"x(:,:,:,{FIELDS.index(name)+1})"
    common = {"th_3d": "th", "qv_3d": "qv", "qc_3d": "qc", "qr_3d": "qr", "qnr_3d": "nr",
              "qi1_3d": "qi", "qni1_3d": "ni", "qir1_3d": "qir", "qib1_3d": "bir",
              "th_old_3d": "th_old", "qv_old_3d": "qv_old", "pii": "exner", "p": "p", "dz": "dz", "w": "w",
              "diag_zdbz_3d": "ze", "diag_effc_3d": "re_cloud", "diag_effi_3d": "re_ice",
              "diag_vmi_3d": "vm", "diag_di_3d": "di", "diag_rhopo_3d": "rhoi"}
    calls = []
    for scheme in (50, 51, 52, 53):
        names = dict(common)
        if scheme != 50:
            names["nc_3d"] = "nc"
        if scheme == 53:
            names["qzi1_3d"] = "qzi"
        if scheme == 52:
            names.update({"qi2_3d": "qi2", "qni2_3d": "ni2", "qir2_3d": "qir2", "qib2_3d": "bir2",
                          "diag_vmi2_3d": "vm2", "diag_di2_3d": "di2", "diag_rhopo2_3d": "rhoi2"})
        args = [f"{key}={a(name)}" for key, name in names.items()]
        args += ["dt=18.", "itimestep=step", f"n_iceCat={2 if scheme == 52 else 1}",
                 "rainnc=precip(:,:,1)", "rainncv=precip(:,:,2)", "sr=precip(:,:,3)",
                 "snownc=precip(:,:,4)", "snowncv=precip(:,:,5)",
                 "ids=1", "ide=nx+1", "jds=1", "jde=2", "kds=1", "kde=nz+1",
                 "ims=1", "ime=nx", "jms=1", "jme=1", "kms=1", "kme=nz",
                 "its=1", "ite=nx", "jts=1", "jte=1", "kts=1", "kte=nz"]
        name = "mp_p3_wrapper_wrf_2cat" if scheme == 52 else "mp_p3_wrapper_wrf"
        calls.append(f"case({scheme})\ncall {name}( &\n" + ", &\n".join(args) + ")")
    setfield = lambda field, expr: f"x(i,k,1,{FIELDS.index(field)+1})={expr}"
    init = [setfield("p", "85000.-3500.*real(k-1)"),
            setfield("exner", "(x(i,k,1,18)/100000.)**(287.15/1005.)"),
            setfield("th", "temp/x(i,k,1,19)"), setfield("th_old", "x(i,k,1,1)"),
            setfield("qv", "0.622*polysvp1(temp,1)/(x(i,k,1,18)-polysvp1(temp,1))*rh(i)"),
            setfield("qv_old", "x(i,k,1,2)"), setfield("dz", "250."), setfield("w", "up(i)"),
            setfield("qc", "cloud(i)"), setfield("nc", "200.e6"),
            setfield("qr", "rain(i)"), setfield("nr", "1.e5"),
            setfield("qi", "ice(i)"), setfield("ni", "1.e4"),
            setfield("qir", "ice(i)*frac(i)"), setfield("bir", "ice(i)*frac(i)/dens(i)"),
            setfield("qi2", "ice(i)*0.35"), setfield("ni2", "2.e3"),
            setfield("qir2", "ice(i)*0.35*0.7"), setfield("bir2", "ice(i)*0.35*0.7/700."),
            setfield("qzi", "sqrt(1.e-18*1.e4)")]
    return f"""program p3_columns
use microphy_p3
implicit none
integer,parameter :: nx=8,nz=12,nf={len(FIELDS)}
integer :: i,k,step,u,v,scheme,stat
real :: x(nx,nz,1,nf),precip(nx,1,5),temp
real :: base(nx)=[284.,276.,265.,255.,245.,235.,225.,215.]
real :: rh(nx)=[0.8,1.05,1.1,1.15,1.05,0.9,0.8,1.2]
real :: cloud(nx)=[0.,1.e-3,1.e-4,2.e-4,5.e-4,1.e-5,0.,1.e-4]
real :: rain(nx)=[0.,2.e-3,1.e-4,2.e-4,1.e-3,2.e-5,1.e-4,0.]
real :: ice(nx)=[0.,0.,1.e-6,1.e-5,1.e-4,5.e-4,1.e-3,2.e-3]
real :: frac(nx)=[0.,0.,0.,0.1,0.5,0.9,1.,0.3]
real :: dens(nx)=[50.,50.,50.,100.,300.,650.,900.,700.]
real :: up(nx)=[0.,1.,2.,3.,0.5,-0.5,0.,5.]
character(len=1024) :: tables,arg
call get_command_argument(1,tables)
call get_command_argument(2,arg); read(arg,*) scheme
call p3_init(trim(tables),merge(2,1,scheme==52),scheme==53,model='WRF',stat=stat,abort_on_err=.true.)
if (stat/=STATUS_OK) stop 2
x=0.; precip=0.
do k=1,nz
do i=1,nx
temp=base(i)-real(k-1)*1.2
{chr(10).join(init)}
enddo
enddo
if (scheme/=52) x(:,:,:,11:14)=0.
if (scheme/=53) x(:,:,:,15)=0.
if (scheme==50) x(:,:,:,10)=0.
open(newunit=u,file='frames.bin',access='stream',form='unformatted',status='replace')
open(newunit=v,file='precip.bin',access='stream',form='unformatted',status='replace')
write(u) x; write(v) precip
do step=1,3
select case(scheme)
{chr(10).join(calls)}
end select
write(u) x; write(v) precip
enddo
close(u);close(v)
end program
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrf-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--stages", action="store_true")
    parser.add_argument("--rates", action="store_true")
    parser.add_argument("--schemes", type=int, nargs="+", default=(50,51,52,53))
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.wrf_root = args.wrf_root.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.wrf_root / "phys/module_mp_p3.F"
    staged = args.out / source.name
    staged.write_text(instrument_stages(source.read_text())) if args.stages else staged.write_bytes(source.read_bytes())
    if args.rates:
        from rate_support import instrument_rates
        staged.write_text(instrument_rates(staged.read_text()))
    if not args.stages and not args.rates:
        assert sha(staged) == sha(source)
    code = args.out / "columns.f90"
    code.write_text(driver())
    flags = ["-cpp", "-ffree-form", "-ffree-line-length-none", "-O2", "-ffp-contract=off", "-fcheck=all"]
    subprocess.run([args.compiler, *flags, str(staged), str(code), "-o", str(args.out / "columns")],
                   cwd=args.out, check=True, timeout=120)
    manifest = {"fixture_id": "p3-original-wrf-wrappers-real4-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "source": str(source), "source_sha256": sha(source), "staged_sha256": sha(staged),
                "caller_source_sha256": sha(args.wrf_root / "phys/module_microphysics_driver.F"),
                "registry_sha256": sha(args.wrf_root / "Registry/Registry.EM_COMMON"),
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.wrf_root).decode().strip(),
                "flags": flags, "driver_sha256": sha(code), "fields": FIELDS, "units": UNITS,
                "compiler": subprocess.check_output([args.compiler, "--version"]).decode().splitlines()[0],
                "raw_tables": {p.name: sha(p) for p in (args.wrf_root / "run").glob("p3_lookupTable*.dat*")},
                "scenario": "8 constructed columns x12 levels; clear,warm rain,cold unrimed/rimed,mixed phase;3 calls dt18s",
                "qualification": "original WRF savepoint enabler ONLY; no JAX p3_main/coupled/GPU qualification",
                "stage_instrumentation": bool(args.stages),
                "rate_instrumentation": bool(args.rates),
                "files": {}, "census": {}}
    for scheme in args.schemes:
        work = args.out / str(scheme)
        work.mkdir(exist_ok=True)
        with (work / "stdout.log").open("w") as log:
            subprocess.run([str(args.out / "columns"), str(args.wrf_root / "run"), str(scheme)],
                           cwd=work, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        x = np.fromfile(work / "frames.bin", dtype=np.float32).reshape((8, 12, 1, len(FIELDS), 4), order="F")
        p = np.fromfile(work / "precip.bin", dtype=np.float32).reshape((8, 1, 5, 4), order="F")
        assert np.isfinite(x).all() and np.isfinite(p).all(), f"nonfinite original WRF mp{scheme}"
        data = {name: x[:, :, 0, i, :] for i, name in enumerate(FIELDS)}
        data.update({name: p[:, 0, i, :] for i, name in enumerate(("rainnc", "rainncv", "sr", "snownc", "snowncv"))})
        path = args.out / f"columns_mp{scheme}.npz"
        np.savez_compressed(path, **data)
        manifest["files"][path.name] = {"sha256": sha(path), "shape": [8, 12, 4], "dtype": "float32"}
        census = {name: int((data[name][:, :, 0] > 0).sum()) for name in ("qi", "qir", "qi2", "qir2", "qc", "qr")}
        census["changed_qi"] = int((data["qi"][:, :, 1:] != data["qi"][:, :, :1]).sum())
        census["changed_nc"] = int((data["nc"][:, :, 1:] != data["nc"][:, :, :1]).sum())
        census["surface_total_mm"] = float(data["rainnc"][:, -1].sum())
        manifest["census"][str(scheme)] = census
    (args.out / "columns_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["census"], indent=2))


if __name__ == "__main__":
    main()
