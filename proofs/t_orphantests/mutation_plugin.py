"""External in-memory deletion controls; source checkout is never modified."""
import os
import pytest
from _pytest.monkeypatch import MonkeyPatch


def pytest_collection_modifyitems(items):
    family=os.environ['ORPHAN_MUTANT']
    original_attr=MonkeyPatch.setattr
    original_env=MonkeyPatch.setenv
    attributes={'history':('gpuwrf.runtime.operational_mode','_HISTORY_SWDOWN_HORIZONTAL'),
                'writer':('gpuwrf.io.wrfout_writer','_WRITER_RVOVRD_REAL'),
                'mynn':('gpuwrf.physics.mynn_constants','_MYNN_REAL_CONSTANTS'),
                'phy':('gpuwrf.coupling.physics_couplers','_PHY_PREP_REAL_CONST'),
                'pblh':('gpuwrf.physics.surface_layer','_MYNN_SFC_PBLH_WRF')}
    flags={'kf':'GPUWRF_KF_REAL_CONSTANTS','thompson':'GPUWRF_THOMPSON_WRF_CONSTANTS',
           'noah_qsfc':'GPUWRF_NOAH_QSFC_WRF','canwater':'GPUWRF_NOAH_CANWATER_ELAI',
           'land_q2':'GPUWRF_LAND_Q2_CAP_POSTLSM','condensation':'GPUWRF_MYNN_CONDENSATION_WRF',
           'predict':'GPUWRF_MYNN_PREDICT_SAW_FLOOR','source_parity':'GPUWRF_MYNN_SOURCE_PARITY2',
           'sfc_psi':'GPUWRF_MYNN_SFC_PSI_REAL','shsm':'GPUWRF_MYNN_SHSM_FLOORS',
           'rhosfc':'GPUWRF_MYNN_RHOSFC_WRF','tkeprod':'GPUWRF_MYNN_TKEPROD_UP'}
    if family in attributes:
        module,name=attributes[family]
        def force_attr(self,*args,**kwargs):
            if len(args)>=3 and getattr(args[0],'__name__',None)==module and args[1]==name:
                args=(*args[:2],False,*args[3:])
            return original_attr(self,*args,**kwargs)
        MonkeyPatch.setattr=force_attr
    if family in flags:
        flag=flags[family]
        original_setitem=type(os.environ).__setitem__
        def force_env(self,name,value):
            return original_setitem(self,name,'0' if name==flag else value)
        type(os.environ).__setitem__=force_env
        os.environ[flag]='0'
    if family=='noah':
        from gpuwrf.physics.noahmp import noahmp_driver as drv
        drv.URBAN_SOIL['csoil']=2.0e6
    if family=='radiation':
        from gpuwrf.physics import rrtmg_sw as sw
        sw._sw_reftra_wrf_enabled=lambda:False
    if family=='fence':
        from gpuwrf.runtime import operational_mode as op
        op._real_theta_sum_fence=lambda x:x
    print('ORPHAN_MUTANT_APPLIED='+family)
