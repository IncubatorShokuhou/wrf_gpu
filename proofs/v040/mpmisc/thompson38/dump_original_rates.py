"""Read original per-column rates through debugger savepoints, no body edits."""
import argparse
import json
from pathlib import Path
import subprocess


def main(build,out,case):
    out.mkdir(parents=True,exist_ok=True)
    rates=('pna_rca pna_sca pna_gca pnd_rcd pnd_scd pnd_gcd pnc_wau pnc_rcw pnc_scw pnc_gcw pni_wfz pni_inu pni_iha prr_wau prr_rcw prg_gcw prs_scw').split()
    real=('rho temp nwfa nifa rr nr rc nc rs rg ng idx_bg mvd_r mvd_c visco rhof smoe').split()
    double=('ilamg n0_g ilamr n0_r').split()
    commands=['set pagination off','set confirm off','break module_mp_thompson.F:2960',f'run {case}','print av_g','print bv_g','set language c']
    for name in rates+real+double:
        size=320 if name in rates+double else 160
        commands.append(f'dump binary memory {out}/{name}.bin (char*)&{name} (char*)&{name}+{size}')
    commands.extend(['disable 1','continue','quit'])
    (out/'commands.gdb').write_text('\n'.join(commands)+'\n')
    with (out/'gdb.log').open('w') as stream:
        subprocess.run(['gdb','-q','-batch','-x',str(out/'commands.gdb'),str(build/'oracle_debug')],cwd=build,stdout=stream,stderr=subprocess.STDOUT,check=True)
    assert all((out/(name+'.bin')).stat().st_size==(320 if name in rates+double else 160) for name in rates+real+double)
    (out/'receipt.json').write_text(json.dumps(dict(scope='Read-only original GDB rate savepoint before tendencies, source bodies unchanged',case=case,line=2960,rates=rates,real=real,double=double),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('build',type=Path);p.add_argument('out',type=Path);p.add_argument('--case',type=int,default=4)
    a=p.parse_args();main(a.build,a.out,a.case)
