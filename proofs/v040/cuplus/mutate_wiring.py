"""Delete actual CuPlus source contributions in memory and require assertion kills."""
from pathlib import Path
import argparse,inspect,json,os,sys,textwrap
import xml.etree.ElementTree as ET


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--mutant',choices=('heat','u','v','number','mskf_cloud','cup_cloud'),required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    assert os.environ.get('JAX_PLATFORMS')=='cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    sys.path.insert(0,str(args.source/'scripts/v025'))
    import cpu_guard  # establish the repository guard before gpuwrf/JAX
    sys.path.insert(0,str(args.source/'src'))
    import pytest
    from gpuwrf.coupling import cuplus
    import gpuwrf.runtime.operational_mode as om
    mutant=args.mutant
    target='tests/v040/cuplus/test_rk_caller.py::test_pristine_rates_enter_rk_without_euler_increment[True-7-9]'
    module,name=om,'_source_leaf_dry_tendencies'
    if mutant=='heat':old,new='if kf_tendf is not None:','if False:'
    elif mutant=='u':
        old='ru_tendf_source=cu_u if ru_tendf_source is None else ru_tendf_source+cu_u'
        new='ru_tendf_source=jnp.zeros_like(cu_u)'
    elif mutant=='v':
        old='rv_tendf_source=cu_v if rv_tendf_source is None else rv_tendf_source+cu_v'
        new='rv_tendf_source=jnp.zeros_like(cu_v)'
    elif mutant=='number':
        module,name=cuplus,'rates'
        old="if 'Ni' in active:output['Ni']=back(carry.held['RQINCUTEN'])"
        new="if 'Ni' in active:output['Ni']=jnp.zeros_like(back(carry.held['RQINCUTEN']))"
    else:
        module,name=cuplus,'radiation_merge'
        if mutant=='mskf_cloud':
            old="if before_pbl and 'CLDFRA_DP_KF' in held:";new='if False:'
            target='tests/v040/cuplus/test_radiation_caller.py::test_original_merge_order_reaches_both_radiation_columns_without_state_mutation[11-0.0]'
        else:
            old="if not before_pbl and 'CLDFRA_CUP' in held:";new='if False:'
            target='tests/v040/cuplus/test_radiation_caller.py::test_original_merge_order_reaches_both_radiation_columns_without_state_mutation[10-60.0]'
    source=textwrap.dedent(inspect.getsource(getattr(module,name)))
    assert source.count(old)==1,(name,old)
    namespace=dict(vars(module))
    exec(compile(source.replace(old,new),f'<{mutant}-actual-source-deletion>','exec'),namespace)
    setattr(module,name,namespace[name])
    xml=args.output.with_suffix('.xml')
    os.chdir(args.source)
    status=pytest.main([target,'-q','--basetemp','<USER_HOME>/wrf_gpu2_lanes/s2-cuplus/pytest',f'--junitxml={xml}'])
    root=ET.parse(xml).getroot()
    failures=list(root.iter('failure'));errors=list(root.iter('error'))
    assertion_only=bool(failures) and all('AssertionError' in (f.text or '') or 'assert ' in (f.text or '') for f in failures)
    record=dict(mutant=mutant,function=f'{module.__name__}.{name}',removed=old,replacement=new,
        target=target,pytest_exit=int(status),failures=len(failures),errors=len(errors),assertion_only=assertion_only,
        files_changed=False,scope='actual in-memory source deletion; original fixtures unchanged')
    args.output.write_text(json.dumps(record,indent=2)+'\n')
    assert status==1 and assertion_only and not errors,record


if __name__=='__main__':main()
