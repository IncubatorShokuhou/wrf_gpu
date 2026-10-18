"""Prepare the w41-stoch GPU qualification cases (CPU only, no JAX, no GPU).

Base = the v0.4.0 default Swiss composite c00 (mp8, cu0, MYNN bl5/sf5, Noah-MP, RRTMG; 43x43x45, dt 18 s,
24 h) with its inputs UNCHANGED (symlinks); only an &stoch group is appended to the namelist:
  st1  skebs=1                                   (Registry defaults)
  st2  sppt=1,  lengthscale_sppt=15 km
  st3  spp=1 (master -> conv/pbl/lsm), all three SPP lengthscales 15 km
  st4  skebs+sppt+spp+rand_perturb, nens=7, every vertstruc=1, all five Gaussian lengthscales 15 km
       (= the u2-stoch ST16 original-WRF qualification profile)
Gaussian lengthscales: the Registry defaults (150/500/150/700/50 km) on this 126 km domain gave NaN
SPP-PBL spectral coefficients (SPFORCC3/SPFORCS3) in ORIGINAL CPU-WRF (v-prep c41), so every Gaussian
pattern uses the resolved 15 km scale (as v-prep c46 / u2-stoch). Seeds = Registry defaults in every arm,
identical for CPU and GPU (same namelist file). The CPU dir gets a private copy of the pristine serial
WRF 4.7.1 executable with only the two E104 debug OPEN literals redirected (as v-prep / w41-topowind).
usage: python prepare_cases.py <out_root> [case ...]
"""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(sys.argv[1])
PRISTINE = Path('<USER_HOME>/src/wrf_pristine/WRF')
BASE = Path('<USER_HOME>/wrf_gpu2_lanes/v-prep/c00')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
L15 = 15000.
CASES = {
    'st0': {},   # matched OFF control on the same snapshot (GPU only; CPU reference = v-prep c00, same exe/inputs)
    'st1': dict(skebs=1),
    'st2': dict(sppt=1, lengthscale_sppt=L15),
    'st3': dict(spp=1, lengthscale_spp_conv=L15, lengthscale_spp_pbl=L15, lengthscale_spp_lsm=L15),
    'st4': dict(skebs=1, sppt=1, spp=1, rand_perturb=1, nens=7,
                skebs_vertstruc=1, sppt_vertstruc=1, rand_pert_vertstruc=1,
                vertstruc_spp_conv=1, vertstruc_spp_pbl=1, vertstruc_spp_lsm=1,
                lengthscale_sppt=L15, lengthscale_rand_pert=L15, lengthscale_spp_conv=L15,
                lengthscale_spp_pbl=L15, lengthscale_spp_lsm=L15),
}


def fortran(value):
    return f'{value:.1f}' if isinstance(value, float) else str(value)


def private_executable(case):
    raw = (PRISTINE / 'main/wrf.exe').read_bytes()
    result, patches = raw, []
    for index, old in enumerate((b'<DATA_ROOT>/wrf_gpu2/wrf_truth/em_quarter_ss_center_dump.txt',
                                 b'<DATA_ROOT>/wrf_gpu2/wrf_truth/em_grav2d_x_touchdown_dump.txt')):
        new = str(case / f'd{index}').encode()
        assert len(new) <= len(old) and result.count(old) == 1
        patches.append(dict(offset=result.index(old), old=old.decode(), new=new.decode()))
        result = result.replace(old, new.ljust(len(old), b' '))
    assert len(result) == len(raw)
    path = case / 'cpu/wrf.exe'
    path.write_bytes(result)
    path.chmod(0o755)
    return dict(original=str(PRISTINE / 'main/wrf.exe'), original_sha256=hashlib.sha256(raw).hexdigest(),
                private_sha256=sha(path), patches=patches, arithmetic_bytes_unchanged=True, ranks=1)


def prepare(name):
    case = ROOT / name
    inp, cpu = case / 'input', case / 'cpu'
    assert not case.exists(), f'{case} exists: never overwrite a prepared case'
    inp.mkdir(parents=True)
    cpu.mkdir()
    text = (BASE / 'input/namelist.input').read_text()
    assert '&stoch' not in text.lower()
    block = ('\n&stoch\n' + ''.join(f' {k} = {fortran(v)},\n' for k, v in CASES[name].items()) + '/\n'
             if CASES[name] else '')
    (inp / 'namelist.input').write_text(text.rstrip('\n') + '\n' + block)
    inputs = {}
    for f in ('wrfinput_d01', 'wrfbdy_d01'):
        src = (BASE / 'input' / f).resolve()
        (inp / f).symlink_to(src)
        inputs[f] = dict(source=str(src), sha256=sha(src))
    if not CASES[name]:
        assert sha(inp / 'namelist.input') == sha(BASE / 'input/namelist.input')
        (cpu.rmdir(), case.joinpath('cpu').symlink_to(BASE / 'cpu'))
        record = dict(schema='w41-stoch-prepare-v1', case=name, base=str(BASE), stoch=None,
                      namelist_sha256=sha(inp / 'namelist.input'), inputs=inputs,
                      cpu_reference='v-prep c00 original CPU-WRF (same pristine exe 52f914cb, same inputs/namelist)',
                      launcher_sha256=sha(__file__))
        (case / 'PREPARE.json').write_text(json.dumps(record, indent=1) + '\n')
        print(name, record['namelist_sha256'][:12], 'OFF control')
        return
    for item in (PRISTINE / 'run').iterdir():
        if item.is_file() and item.name not in ('namelist.input', 'wrf.exe', 'real.exe', 'ndown.exe', 'tc.exe',
                                                'ideal.exe') and not item.name.startswith('wrfout'):
            (cpu / item.name).symlink_to(item.resolve())
    for table in ('freezeH2O.dat', 'qr_acr_qg_V4.dat', 'qr_acr_qsV2.dat'):
        target = cpu / table
        if target.exists() or target.is_symlink():
            target.unlink()
        src = (BASE / 'cpu' / table).resolve()
        assert src.is_file(), src
        target.symlink_to(src)
    if not (cpu / 'MPTABLE.TBL').exists():
        (cpu / 'MPTABLE.TBL').symlink_to(PRISTINE / 'phys/noahmp/parameters/MPTABLE.TBL')
    for item in inp.iterdir():
        target = cpu / item.name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(item.resolve() if item.is_symlink() else item)
    record = dict(schema='w41-stoch-prepare-v1', case=name, base=str(BASE), stoch=CASES[name],
                  namelist_sha256=sha(inp / 'namelist.input'), base_namelist_sha256=sha(BASE / 'input/namelist.input'),
                  inputs=inputs, executable=private_executable(case), launcher_sha256=sha(__file__))
    (case / 'PREPARE.json').write_text(json.dumps(record, indent=1) + '\n')
    print(name, record['namelist_sha256'][:12], CASES[name])


if __name__ == '__main__':
    for n in sys.argv[2:] or list(CASES):
        prepare(n)
