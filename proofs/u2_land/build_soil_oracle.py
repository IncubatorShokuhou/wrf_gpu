"""Unchanged original Saxton–Rawls pedotransfer and actual table coefficients."""
from datetime import datetime,timezone
import hashlib,json,re,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from build_crop_oracle import extract


def main():
    source,build,target=map(Path,sys.argv[1:4]);build.mkdir(parents=True,exist_ok=True)
    text=source.read_text();body=extract(text,'PEDOTRANSFER_SR2006')
    from gpuwrf.physics.noahmp.tables import _parse_mptable_block
    from gpuwrf.physics.noahmp.soil_options import OUTPUTS
    wrf_root=next(p for p in source.parents if (p/'run/MPTABLE.TBL').exists());table=wrf_root/'run/MPTABLE.TBL'
    values=_parse_mptable_block(table.read_text(),'noahmp_optional_parameters')
    keys=sorted(set(k.lower() for k in re.findall(r'\bsr2006_\w+',body,re.I)))
    coefficients={k:values[k.upper()][0] for k in keys}
    definitions='\n'.join('real :: '+k+'(4)' for k in OUTPUTS)
    constants='\n'.join('real :: '+k+' = '+str(v) for k,v in coefficients.items())
    (build/'reference.f90').write_text('module module_sf_noahmplsm\nimplicit none\ntype noahmp_parameters\n'+definitions+
        '\nend type\nend module\nmodule noahmp_tables\nimplicit none\n'+constants+'\nend module\n'+body+'\n')
    (build/'driver.f90').write_text('''program oracle
use module_sf_noahmplsm
implicit none
integer :: n,i
real :: sand(4),clay(4),orgm(4)
type(noahmp_parameters) :: p
read(*,*)n
do i=1,n
read(*,*)sand,clay,orgm
call PEDOTRANSFER_SR2006(4,sand,clay,orgm,p)
write(*,'(36(ES25.16E3,1X))')'''+','.join('p%'+k for k in OUTPUTS)+'''
enddo
end program
''')
    fc=shutil.which('gfortran') or '<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
    flags=['-O0','-ffree-form','-ffree-line-length-none','-ffp-contract=off','-finit-real=zero','-finit-integer=0','-finit-logical=false']
    subprocess.run([fc,*flags,'reference.f90','driver.f90','-o','oracle.exe'],cwd=build,check=True,timeout=120)
    regimes=((.41,.18,0.),(.8,.05,0.),(.05,.8,0.),(.5,.3,0.),(0.,.3,0.),(.3,0.,0.),(-.1,.1,-1.),(.95,.05,0.),
        (.05,.95,0.),(.3,.3,1.),(.4,.2,5.),(.1,.5,10.),(.7,.1,0.),(.2,.7,0.),(.6,.2,0.),(.2,.2,0.))
    cases=[dict(sand=[regimes[(i+k)%16][0] for k in range(4)],clay=[regimes[(i+k)%16][1] for k in range(4)],
        orgm=[regimes[(i+k)%16][2] for k in range(4)]) for i in range(16)]
    def run(rows):
        lines=[str(len(rows))]+[' '.join(map(str,[*c['sand'],*c['clay'],*c['orgm']])) for c in rows]
        r=subprocess.run([str((build/'oracle.exe').resolve())],input='\n'.join(lines)+'\n',text=True,capture_output=True,check=True,timeout=30)
        out=np.fromstring(r.stdout,sep=' ',dtype=np.float32).reshape(len(rows),9,4);assert np.isfinite(out).all();return out
    expected=run(cases);variants=[];owner=[];indices=[]
    for i,c in enumerate(cases):
        indices.append(len(variants));variants.append(c.copy());owner.append(i)
        for key in ('sand','clay','orgm'):
            for k in range(4):
                for direction in (-np.inf,np.inf):
                    v=c.copy();v[key]=c[key].copy();v[key][k]=float(np.nextafter(np.float32(c[key][k]),np.float32(direction)))
                    variants.append(v);owner.append(i)
    results=run(variants);assert np.array_equal(results[indices],expected)
    low,high=expected.copy(),expected.copy()
    for i,row in zip(owner,results):low[i]=np.minimum(low[i],row);high[i]=np.maximum(high[i],row)
    for i,c in enumerate(cases):c.update(expected=expected[i].tolist(),low=low[i].tolist(),high=high[i].tolist())
    target.write_text(json.dumps(dict(source=str(source),source_commit='f52c197ed39d12e087d02c50f412d90d418f6186',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),routine_sha256=hashlib.sha256(body.encode()).hexdigest(),
        table_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),created_utc=datetime.now(timezone.utc).isoformat(),flags=flags,
        coefficients=coefficients,outputs=OUTPUTS,cases=cases,ulp_variants=len(variants),nominal_tolerance=dict(atol=1.e-6,rtol=3.e-6),
        scope='16 constructed four-layer columns incl default replacement/nonzeroorganic/wetdryextremes; actualSaxtonRawls table'),indent=1)+'\n')


if __name__=='__main__':main()
