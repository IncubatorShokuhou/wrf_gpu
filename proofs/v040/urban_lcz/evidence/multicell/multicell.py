"""Independent E11 probe: two identical cells in a single pristine caller tile."""
from pathlib import Path
import sys

root = Path('<USER_HOME>/wrf_gpu2_lanes/rv-u2-lcz/snapshot')
probe = root / 'proofs/v040/urban_lcz/build_oracle.py'
source = probe.read_text()
source = source.replace("Path('<USER_HOME>/wrf_gpu2_lanes/u2-lcz/oracle')",
                        "Path('<USER_HOME>/wrf_gpu2_lanes/rv-u2-lcz/multicell_build')")
source = source.replace("for prec in ('fp32','fp64'):", "for prec in ('fp32',):")
source = source.replace("for table in ('default','heat'):", "for table in ('default',):")
inject = '''
import re
driver=driver.replace('ime=1,','ime=2,').replace('ide=2,','ide=3,').replace('ite=1,','ite=2,')
declarations=driver[:driver.index('  CALL get_command_argument')].replace('&\\n',' ')
copies=['IVGTYP(2,:)=IVGTYP(1,:)','UTYPE_URB2D(2,:)=UTYPE_URB2D(1,:)']
for dims,names in re.findall(r'REAL, DIMENSION\\(([^\\n]+)\\) :: ([^\\n]+)', declarations):
    rank=dims.count(',')+1
    for name in names.replace('&','').split(','):
        name=name.strip()
        if name:
            suffix=',:'*(rank-1)
            copies.append(f'{name}(2{suffix})={name}(1{suffix})')
copy_block='\\n'.join('      '+s for s in copies)+'\\n'
driver=driver.replace('    CALL urban_var_init(',copy_block+'    CALL urban_var_init(')
driver=driver.replace('      CALL noahmp_urban(',copy_block+'      CALL noahmp_urban(')
extras=''
for field in ('HFX','TSK','U10_URB2D','V10_URB2D','UST'):
    extras+=f"      CALL d(r,s,'TILE2_{field}',{field}(2,1))\\n"
driver=driver.replace('      CALL dumpstate(r, s)',extras+'      CALL dumpstate(r, s)')
assert 'ite=2' in driver and len(copies)>100,len(copies)
print('Two-cell declaration copy count:',len(copies),flush=True)
'''
source = source.replace('# zero_all already', inject + '\n# zero_all already', 1)
exec(compile(source, str(probe), 'exec'), {'__file__': str(probe), '__name__': '__main__'})
