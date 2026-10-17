"""Generated offline from pristine WRF CAM-MG; see proofs/s2_cammg/port.py.
Qualified against original WRF public/staged calls; native GPU qualification pending.
"""
import jax.numpy as jnp
from ._support import _r,_d,_int,_get,_put,_slice,_loop,_cond,_cast,_dtype,_pow,_powi,_retry,_zeros,_literal_array,_static,_neg

from .cloud import cal_cldfra_mp_1d
from .cloud import polysvp
def physics_ptend_reset(ptend_name,ptend_q,ptend_s,ptend_lq,ptend_ls,pcnst_in,*,_globals=None,xp=jnp):
    v=dict(_globals or {})
    if '__modules' in v:v.update(v['__modules'].get('module_mp_cammgmp_driver',{}))
    v['ptend_name']=ptend_name
    v['ptend_q']=ptend_q
    v['ptend_s']=ptend_s
    v['ptend_lq']=ptend_lq
    v['ptend_ls']=ptend_ls
    v['pcnst_in']=pcnst_in
    v['_jump']=_int(0)
    v['m']=_zeros((),'i32',xp)
    _outer1=dict(v)
    def _yes1(_carry):
        v=dict(_outer1);v.update(zip(('ptend_s',),_carry))
        _put(v,'ptend_s',None,_d(0.0))  # WRF:module_mp_cammgmp_driver.F:1572
        return tuple(v[_n] for _n in ('ptend_s',))
    def _no1(_carry):
        v=dict(_outer1);v.update(zip(('ptend_s',),_carry))
        return tuple(v[_n] for _n in ('ptend_s',))
    _result=_cond(v['ptend_ls'],_yes1,_no1,tuple(v[_n] for _n in ('ptend_s',)),xp)  # WRF:module_mp_cammgmp_driver.F:1571
    v.update(zip(('ptend_s',),_result))
    _outer2=dict(v)
    def _body2(_k,_carry):
        v=dict(_outer2);v.update(zip(('_jump', 'm', 'ptend_q'),_carry));v['m']=_int(_k)
        _outer3=dict(v)
        def _yes3(_carry):
            v=dict(_outer3);v.update(zip(('ptend_q',),_carry))
            _put(v,'ptend_q',(_slice(None,None,None),_slice(None,None,None),v['m'],),_d(0.0))  # WRF:module_mp_cammgmp_driver.F:1576
            return tuple(v[_n] for _n in ('ptend_q',))
        def _no3(_carry):
            v=dict(_outer3);v.update(zip(('ptend_q',),_carry))
            return tuple(v[_n] for _n in ('ptend_q',))
        _result=_cond(_get(v['ptend_lq'], (v['m'],)),_yes3,_no3,tuple(v[_n] for _n in ('ptend_q',)),xp)  # WRF:module_mp_cammgmp_driver.F:1575
        v.update(zip(('ptend_q',),_result))
        return tuple(v[_n] for _n in ('_jump', 'm', 'ptend_q'))
    _result=_loop(1,v['pcnst_in'],1,_body2,tuple(v[_n] for _n in ('_jump', 'm', 'ptend_q')),xp,jump_index=0)
    v.update(zip(('_jump', 'm', 'ptend_q'),_result))
    _put(v,'ptend_lq',(_slice(None,None,None),),False)  # WRF:module_mp_cammgmp_driver.F:1581
    _put(v,'ptend_ls',None,False)  # WRF:module_mp_cammgmp_driver.F:1582
    _put(v,'ptend_top_level',None,1)  # WRF:module_mp_cammgmp_driver.F:1584
    _put(v,'ptend_bot_level',None,v['pver'])  # WRF:module_mp_cammgmp_driver.F:1585
    return (v['ptend_name'],v['ptend_q'],v['ptend_s'],v['ptend_lq'],v['ptend_ls'],)

def physics_ptend_init(ptend_name,ptend_q,ptend_s,ptend_lq,ptend_ls,pcnst_in,*,_globals=None,xp=jnp):
    v=dict(_globals or {})
    if '__modules' in v:v.update(v['__modules'].get('module_mp_cammgmp_driver',{}))
    v['ptend_name']=ptend_name
    v['ptend_q']=ptend_q
    v['ptend_s']=ptend_s
    v['ptend_lq']=ptend_lq
    v['ptend_ls']=ptend_ls
    v['pcnst_in']=pcnst_in
    v['_jump']=_int(0)
    _put(v,'ptend_lq',(_slice(None,None,None),),True)  # WRF:module_mp_cammgmp_driver.F:1547
    _put(v,'ptend_ls',None,True)  # WRF:module_mp_cammgmp_driver.F:1548
    _call=physics_ptend_reset(v['ptend_name'],v['ptend_q'],v['ptend_s'],v['ptend_lq'],v['ptend_ls'],v['pcnst_in'],_globals=_globals,xp=xp)
    _put(v,'ptend_name',None,_call[0])  # WRF:module_mp_cammgmp_driver.F:1549
    _put(v,'ptend_q',None,_call[1])  # WRF:module_mp_cammgmp_driver.F:1549
    _put(v,'ptend_s',None,_call[2])  # WRF:module_mp_cammgmp_driver.F:1549
    _put(v,'ptend_lq',None,_call[3])  # WRF:module_mp_cammgmp_driver.F:1549
    _put(v,'ptend_ls',None,_call[4])  # WRF:module_mp_cammgmp_driver.F:1549
    return (v['ptend_name'],v['ptend_q'],v['ptend_s'],v['ptend_lq'],v['ptend_ls'],)

def physics_ptend_sum(ptend_ls,ptend_lq,ptend_s,ptend_q,ptend_sum_ls,ptend_sum_lq,ptend_sum_s,ptend_sum_q,pcnst_in,*,_globals=None,xp=jnp):
    v=dict(_globals or {})
    if '__modules' in v:v.update(v['__modules'].get('module_mp_cammgmp_driver',{}))
    v['ptend_ls']=ptend_ls
    v['ptend_lq']=ptend_lq
    v['ptend_s']=ptend_s
    v['ptend_q']=ptend_q
    v['ptend_sum_ls']=ptend_sum_ls
    v['ptend_sum_lq']=ptend_sum_lq
    v['ptend_sum_s']=ptend_sum_s
    v['ptend_sum_q']=ptend_sum_q
    v['pcnst_in']=pcnst_in
    v['_jump']=_int(0)
    v['k']=_zeros((),'i32',xp)
    v['m']=_zeros((),'i32',xp)
    v['ncol']=_zeros((),'i32',xp)
    v['i']=_zeros((),'i32',xp)
    _put(v,'ptend_top_level',None,1)  # WRF:module_mp_cammgmp_driver.F:1621
    _put(v,'ptend_bot_level',None,v['pver'])  # WRF:module_mp_cammgmp_driver.F:1622
    _put(v,'ncol',None,v['pcols'])  # WRF:module_mp_cammgmp_driver.F:1624
    _outer1=dict(v)
    def _yes1(_carry):
        v=dict(_outer1);v.update(zip(('i', 'k', 'ptend_sum_ls', 'ptend_sum_s'),_carry))
        _put(v,'ptend_sum_ls',None,True)  # WRF:module_mp_cammgmp_driver.F:1627
        _outer2=dict(v)
        def _body2(_k,_carry):
            v=dict(_outer2);v.update(zip(('_jump', 'i', 'k', 'ptend_sum_s'),_carry));v['i']=_int(_k)
            _outer3=dict(v)
            def _body3(_k,_carry):
                v=dict(_outer3);v.update(zip(('_jump', 'k', 'ptend_sum_s'),_carry));v['k']=_int(_k)
                _put(v,'ptend_sum_s',(v['i'],v['k'],),_get(v['ptend_sum_s'], (v['i'], v['k'])) + _get(v['ptend_s'], (v['i'], v['k'])))  # WRF:module_mp_cammgmp_driver.F:1630
                return tuple(v[_n] for _n in ('_jump', 'k', 'ptend_sum_s'))
            _result=_loop(v['ptend_top_level'],v['ptend_bot_level'],1,_body3,tuple(v[_n] for _n in ('_jump', 'k', 'ptend_sum_s')),xp,jump_index=0)
            v.update(zip(('_jump', 'k', 'ptend_sum_s'),_result))
            return tuple(v[_n] for _n in ('_jump', 'i', 'k', 'ptend_sum_s'))
        _result=_loop(1,v['ncol'],1,_body2,tuple(v[_n] for _n in ('_jump', 'i', 'k', 'ptend_sum_s')),xp,jump_index=0)
        v.update(zip(('_jump', 'i', 'k', 'ptend_sum_s'),_result))
        return tuple(v[_n] for _n in ('i', 'k', 'ptend_sum_ls', 'ptend_sum_s'))
    def _no1(_carry):
        v=dict(_outer1);v.update(zip(('i', 'k', 'ptend_sum_ls', 'ptend_sum_s'),_carry))
        return tuple(v[_n] for _n in ('i', 'k', 'ptend_sum_ls', 'ptend_sum_s'))
    _result=_cond(v['ptend_ls'],_yes1,_no1,tuple(v[_n] for _n in ('i', 'k', 'ptend_sum_ls', 'ptend_sum_s')),xp)  # WRF:module_mp_cammgmp_driver.F:1626
    v.update(zip(('i', 'k', 'ptend_sum_ls', 'ptend_sum_s'),_result))
    _outer4=dict(v)
    def _body4(_k,_carry):
        v=dict(_outer4);v.update(zip(('_jump', 'i', 'k', 'm', 'ptend_sum_lq', 'ptend_sum_q'),_carry));v['m']=_int(_k)
        _outer5=dict(v)
        def _yes5(_carry):
            v=dict(_outer5);v.update(zip(('i', 'k', 'ptend_sum_lq', 'ptend_sum_q'),_carry))
            _put(v,'ptend_sum_lq',(v['m'],),True)  # WRF:module_mp_cammgmp_driver.F:1638
            _outer6=dict(v)
            def _body6(_k,_carry):
                v=dict(_outer6);v.update(zip(('_jump', 'i', 'k', 'ptend_sum_q'),_carry));v['i']=_int(_k)
                _outer7=dict(v)
                def _body7(_k,_carry):
                    v=dict(_outer7);v.update(zip(('_jump', 'k', 'ptend_sum_q'),_carry));v['k']=_int(_k)
                    _put(v,'ptend_sum_q',(v['i'],v['k'],v['m'],),_get(v['ptend_sum_q'], (v['i'], v['k'], v['m'])) + _get(v['ptend_q'], (v['i'], v['k'], v['m'])))  # WRF:module_mp_cammgmp_driver.F:1641
                    return tuple(v[_n] for _n in ('_jump', 'k', 'ptend_sum_q'))
                _result=_loop(v['ptend_top_level'],v['ptend_bot_level'],1,_body7,tuple(v[_n] for _n in ('_jump', 'k', 'ptend_sum_q')),xp,jump_index=0)
                v.update(zip(('_jump', 'k', 'ptend_sum_q'),_result))
                return tuple(v[_n] for _n in ('_jump', 'i', 'k', 'ptend_sum_q'))
            _result=_loop(1,v['ncol'],1,_body6,tuple(v[_n] for _n in ('_jump', 'i', 'k', 'ptend_sum_q')),xp,jump_index=0)
            v.update(zip(('_jump', 'i', 'k', 'ptend_sum_q'),_result))
            return tuple(v[_n] for _n in ('i', 'k', 'ptend_sum_lq', 'ptend_sum_q'))
        def _no5(_carry):
            v=dict(_outer5);v.update(zip(('i', 'k', 'ptend_sum_lq', 'ptend_sum_q'),_carry))
            return tuple(v[_n] for _n in ('i', 'k', 'ptend_sum_lq', 'ptend_sum_q'))
        _result=_cond(_get(v['ptend_lq'], (v['m'],)),_yes5,_no5,tuple(v[_n] for _n in ('i', 'k', 'ptend_sum_lq', 'ptend_sum_q')),xp)  # WRF:module_mp_cammgmp_driver.F:1637
        v.update(zip(('i', 'k', 'ptend_sum_lq', 'ptend_sum_q'),_result))
        return tuple(v[_n] for _n in ('_jump', 'i', 'k', 'm', 'ptend_sum_lq', 'ptend_sum_q'))
    _result=_loop(1,v['pcnst_in'],1,_body4,tuple(v[_n] for _n in ('_jump', 'i', 'k', 'm', 'ptend_sum_lq', 'ptend_sum_q')),xp,jump_index=0)
    v.update(zip(('_jump', 'i', 'k', 'm', 'ptend_sum_lq', 'ptend_sum_q'),_result))
    return (v['ptend_sum_ls'],v['ptend_sum_lq'],v['ptend_sum_s'],v['ptend_sum_q'],)

def physics_update(state_lchnk,dt,state_q,ptend_q,state_s,ptend_s,ptend_name,ptend_lq,ptend_ls,pcnst_in,*,_globals=None,xp=jnp):
    v=dict(_globals or {})
    if '__modules' in v:v.update(v['__modules'].get('module_mp_cammgmp_driver',{}))
    v['state_lchnk']=state_lchnk
    v['dt']=dt
    v['state_q']=state_q
    v['ptend_q']=ptend_q
    v['state_s']=state_s
    v['ptend_s']=ptend_s
    v['ptend_name']=ptend_name
    v['ptend_lq']=ptend_lq
    v['ptend_ls']=ptend_ls
    v['pcnst_in']=pcnst_in
    v['_jump']=_int(0)
    v['name']=''
    v['microp_scheme']=''
    v['m']=_zeros((),'i32',xp)
    v['i']=_zeros((),'i32',xp)
    v['k']=_zeros((),'i32',xp)
    v['ncol']=_zeros((),'i32',xp)
    v['ixcldice']=_zeros((),'i32',xp)
    v['ixcldliq']=_zeros((),'i32',xp)
    v['ixnumice']=_zeros((),'i32',xp)
    v['ixnumliq']=_zeros((),'i32',xp)
    v['microp_scheme']='mg'
    _put(v,'ptend_top_level',None,1)  # WRF:module_mp_cammgmp_driver.F:1431
    _put(v,'ptend_bot_level',None,v['pver'])  # WRF:module_mp_cammgmp_driver.F:1432
    _put(v,'ncol',None,v['pcols'])  # WRF:module_mp_cammgmp_driver.F:1433
    _outer1=dict(v)
    def _yes1(_carry):
        v=dict(_outer1);v.update(zip(('i', 'k', 'state_s'),_carry))
        _outer2=dict(v)
        def _body2(_k,_carry):
            v=dict(_outer2);v.update(zip(('_jump', 'i', 'k', 'state_s'),_carry));v['k']=_int(_k)
            _outer3=dict(v)
            def _body3(_k,_carry):
                v=dict(_outer3);v.update(zip(('_jump', 'i', 'state_s'),_carry));v['i']=_int(_k)
                _put(v,'state_s',(v['i'],v['k'],),_get(v['state_s'], (v['i'], v['k'])) + _get(v['ptend_s'], (v['i'], v['k'])) * v['dt'])  # WRF:module_mp_cammgmp_driver.F:1439
                return tuple(v[_n] for _n in ('_jump', 'i', 'state_s'))
            _result=_loop(1,v['ncol'],1,_body3,tuple(v[_n] for _n in ('_jump', 'i', 'state_s')),xp,jump_index=0)
            v.update(zip(('_jump', 'i', 'state_s'),_result))
            return tuple(v[_n] for _n in ('_jump', 'i', 'k', 'state_s'))
        _result=_loop(v['ptend_top_level'],v['ptend_bot_level'],1,_body2,tuple(v[_n] for _n in ('_jump', 'i', 'k', 'state_s')),xp,jump_index=0)
        v.update(zip(('_jump', 'i', 'k', 'state_s'),_result))
        return tuple(v[_n] for _n in ('i', 'k', 'state_s'))
    def _no1(_carry):
        v=dict(_outer1);v.update(zip(('i', 'k', 'state_s'),_carry))
        return tuple(v[_n] for _n in ('i', 'k', 'state_s'))
    _result=_cond(v['ptend_ls'],_yes1,_no1,tuple(v[_n] for _n in ('i', 'k', 'state_s')),xp)  # WRF:module_mp_cammgmp_driver.F:1436
    v.update(zip(('i', 'k', 'state_s'),_result))
    _put(v,'ixcldice',None,_int(3))  # WRF:module_mp_cammgmp_driver.F:1445
    _put(v,'ixcldliq',None,_int(2))  # WRF:module_mp_cammgmp_driver.F:1446
    _put(v,'ixnumice',None,_int(5))  # WRF:module_mp_cammgmp_driver.F:1449
    _put(v,'ixnumliq',None,_int(4))  # WRF:module_mp_cammgmp_driver.F:1450
    _outer4=dict(v)
    def _body4(_k,_carry):
        v=dict(_outer4);v.update(zip(('_jump', 'i', 'k', 'm', 'state_q'),_carry));v['m']=_int(_k)
        _outer5=dict(v)
        def _yes5(_carry):
            v=dict(_outer5);v.update(zip(('i', 'k', 'state_q'),_carry))
            _outer6=dict(v)
            def _body6(_k,_carry):
                v=dict(_outer6);v.update(zip(('_jump', 'i', 'k', 'state_q'),_carry));v['k']=_int(_k)
                _outer7=dict(v)
                def _body7(_k,_carry):
                    v=dict(_outer7);v.update(zip(('_jump', 'i', 'state_q'),_carry));v['i']=_int(_k)
                    _put(v,'state_q',(v['i'],v['k'],v['m'],),_get(v['state_q'], (v['i'], v['k'], v['m'])) + _get(v['ptend_q'], (v['i'], v['k'], v['m'])) * v['dt'])  # WRF:module_mp_cammgmp_driver.F:1456
                    return tuple(v[_n] for _n in ('_jump', 'i', 'state_q'))
                _result=_loop(1,v['ncol'],1,_body7,tuple(v[_n] for _n in ('_jump', 'i', 'state_q')),xp,jump_index=0)
                v.update(zip(('_jump', 'i', 'state_q'),_result))
                return tuple(v[_n] for _n in ('_jump', 'i', 'k', 'state_q'))
            _result=_loop(v['ptend_top_level'],v['ptend_bot_level'],1,_body6,tuple(v[_n] for _n in ('_jump', 'i', 'k', 'state_q')),xp,jump_index=0)
            v.update(zip(('_jump', 'i', 'k', 'state_q'),_result))
            _outer8=dict(v)
            def _yes8(_carry):
                v=dict(_outer8);v.update(zip(('i', 'k', 'state_q'),_carry))
                _put(v,'state_q',(_slice(None,None,None),_slice(None,None,None),v['m'],),xp.maximum(_get(v['state_q'], (_slice(1, None, None), _slice(1, None, None), v['m'])),_get(v['qmin'], (v['m'],))))  # WRF:module_mp_cammgmp_driver.F:1464
                return tuple(v[_n] for _n in ('i', 'k', 'state_q'))
            def _no8(_carry):
                v=dict(_outer8);v.update(zip(('i', 'k', 'state_q'),_carry))
                _outer9=dict(v)
                def _body9(_k,_carry):
                    v=dict(_outer9);v.update(zip(('_jump', 'i', 'k', 'state_q'),_carry));v['k']=_int(_k)
                    _outer10=dict(v)
                    def _body10(_k,_carry):
                        v=dict(_outer10);v.update(zip(('_jump', 'i', 'state_q'),_carry));v['i']=_int(_k)
                        _put(v,'state_q',(v['i'],v['k'],v['m'],),xp.maximum(_d(1e-12), _get(v['state_q'], (v['i'], v['k'], v['m']))))  # WRF:module_mp_cammgmp_driver.F:1469
                        _put(v,'state_q',(v['i'],v['k'],v['m'],),xp.minimum(_d(10000000000.0), _get(v['state_q'], (v['i'], v['k'], v['m']))))  # WRF:module_mp_cammgmp_driver.F:1470
                        return tuple(v[_n] for _n in ('_jump', 'i', 'state_q'))
                    _result=_loop(1,v['ncol'],1,_body10,tuple(v[_n] for _n in ('_jump', 'i', 'state_q')),xp,jump_index=0)
                    v.update(zip(('_jump', 'i', 'state_q'),_result))
                    return tuple(v[_n] for _n in ('_jump', 'i', 'k', 'state_q'))
                _result=_loop(v['ptend_top_level'],v['ptend_bot_level'],1,_body9,tuple(v[_n] for _n in ('_jump', 'i', 'k', 'state_q')),xp,jump_index=0)
                v.update(zip(('_jump', 'i', 'k', 'state_q'),_result))
                return tuple(v[_n] for _n in ('i', 'k', 'state_q'))
            _result=_cond(xp.logical_and(v['m'] != v['ixnumice'], v['m'] != v['ixnumliq']),_yes8,_no8,tuple(v[_n] for _n in ('i', 'k', 'state_q')),xp)  # WRF:module_mp_cammgmp_driver.F:1462
            v.update(zip(('i', 'k', 'state_q'),_result))
            return tuple(v[_n] for _n in ('i', 'k', 'state_q'))
        def _no5(_carry):
            v=dict(_outer5);v.update(zip(('i', 'k', 'state_q'),_carry))
            return tuple(v[_n] for _n in ('i', 'k', 'state_q'))
        _result=_cond(_get(v['ptend_lq'], (v['m'],)),_yes5,_no5,tuple(v[_n] for _n in ('i', 'k', 'state_q')),xp)  # WRF:module_mp_cammgmp_driver.F:1453
        v.update(zip(('i', 'k', 'state_q'),_result))
        return tuple(v[_n] for _n in ('_jump', 'i', 'k', 'm', 'state_q'))
    _result=_loop(1,v['pcnst_in'],1,_body4,tuple(v[_n] for _n in ('_jump', 'i', 'k', 'm', 'state_q')),xp,jump_index=0)
    v.update(zip(('_jump', 'i', 'k', 'm', 'state_q'),_result))
    _outer11=dict(v)
    def _yes11(_carry):
        v=dict(_outer11);v.update(zip(('state_q',),_carry))
        _outer12=dict(v)
        def _yes12(_carry):
            v=dict(_outer12);v.update(zip(('state_q',),_carry))
            _outer13=dict(v)
            def _yes13(_carry):
                v=dict(_outer13);v.update(zip(('state_q',),_carry))
                return tuple(v[_n] for _n in ('state_q',))
            def _no13(_carry):
                v=dict(_outer13);v.update(zip(('state_q',),_carry))
                _outer14=dict(v)
                def _yes14(_carry):
                    v=dict(_outer14);v.update(zip(('state_q',),_carry))
                    _where15=_get(v['state_q'], (_slice(1, v['ncol'], None), _slice(1, v['pver'], None), v['ixcldliq'])) < _d(1e-36)
                    _put(v,'state_q',(_slice(None,v['ncol'],None),_slice(None,v['pver'],None),v['ixcldliq'],),xp.where(_where15,_d(0.0),_get(v['state_q'], (_slice(1, v['ncol'], None), _slice(1, v['pver'], None), v['ixcldliq']))))  # WRF:module_mp_cammgmp_driver.F:1485
                    _outer16=dict(v)
                    def _yes16(_carry):
                        v=dict(_outer16);v.update(zip(('state_q',),_carry))
                        _where17=_get(v['state_q'], (_slice(1, v['ncol'], None), _slice(1, v['pver'], None), v['ixcldliq'])) < _d(1e-36)
                        _put(v,'state_q',(_slice(None,v['ncol'],None),_slice(None,v['pver'],None),v['ixnumliq'],),xp.where(_where17,_d(0.0),_get(v['state_q'], (_slice(1, v['ncol'], None), _slice(1, v['pver'], None), v['ixnumliq']))))  # WRF:module_mp_cammgmp_driver.F:1490
                        return tuple(v[_n] for _n in ('state_q',))
                    def _no16(_carry):
                        v=dict(_outer16);v.update(zip(('state_q',),_carry))
                        return tuple(v[_n] for _n in ('state_q',))
                    _result=_cond(v['microp_scheme'] == 'mg',_yes16,_no16,tuple(v[_n] for _n in ('state_q',)),xp)  # WRF:module_mp_cammgmp_driver.F:1488
                    v.update(zip(('state_q',),_result))
                    return tuple(v[_n] for _n in ('state_q',))
                def _no14(_carry):
                    v=dict(_outer14);v.update(zip(('state_q',),_carry))
                    return tuple(v[_n] for _n in ('state_q',))
                _result=_cond(v['ptend_name'] == 'convect_deep',_yes14,_no14,tuple(v[_n] for _n in ('state_q',)),xp)  # WRF:module_mp_cammgmp_driver.F:1483
                v.update(zip(('state_q',),_result))
                return tuple(v[_n] for _n in ('state_q',))
            _result=_cond(xp.logical_or(v['ptend_name'] == 'stratiform', v['ptend_name'] == 'cldwat'),_yes13,_no13,tuple(v[_n] for _n in ('state_q',)),xp)  # WRF:module_mp_cammgmp_driver.F:1481
            v.update(zip(('state_q',),_result))
            return tuple(v[_n] for _n in ('state_q',))
        def _no12(_carry):
            v=dict(_outer12);v.update(zip(('state_q',),_carry))
            return tuple(v[_n] for _n in ('state_q',))
        _result=_cond(_get(v['ptend_lq'], (v['ixcldliq'],)),_yes12,_no12,tuple(v[_n] for _n in ('state_q',)),xp)  # WRF:module_mp_cammgmp_driver.F:1480
        v.update(zip(('state_q',),_result))
        return tuple(v[_n] for _n in ('state_q',))
    def _no11(_carry):
        v=dict(_outer11);v.update(zip(('state_q',),_carry))
        return tuple(v[_n] for _n in ('state_q',))
    _result=_cond(v['ixcldliq'] > 1,_yes11,_no11,tuple(v[_n] for _n in ('state_q',)),xp)  # WRF:module_mp_cammgmp_driver.F:1479
    v.update(zip(('state_q',),_result))
    _outer18=dict(v)
    def _yes18(_carry):
        v=dict(_outer18);v.update(zip(('state_q',),_carry))
        _outer19=dict(v)
        def _yes19(_carry):
            v=dict(_outer19);v.update(zip(('state_q',),_carry))
            _outer20=dict(v)
            def _yes20(_carry):
                v=dict(_outer20);v.update(zip(('state_q',),_carry))
                return tuple(v[_n] for _n in ('state_q',))
            def _no20(_carry):
                v=dict(_outer20);v.update(zip(('state_q',),_carry))
                _outer21=dict(v)
                def _yes21(_carry):
                    v=dict(_outer21);v.update(zip(('state_q',),_carry))
                    _where22=_get(v['state_q'], (_slice(1, v['ncol'], None), _slice(1, v['pver'], None), v['ixcldice'])) < _d(1e-36)
                    _put(v,'state_q',(_slice(None,v['ncol'],None),_slice(None,v['pver'],None),v['ixcldice'],),xp.where(_where22,_d(0.0),_get(v['state_q'], (_slice(1, v['ncol'], None), _slice(1, v['pver'], None), v['ixcldice']))))  # WRF:module_mp_cammgmp_driver.F:1504
                    _outer23=dict(v)
                    def _yes23(_carry):
                        v=dict(_outer23);v.update(zip(('state_q',),_carry))
                        _where24=_get(v['state_q'], (_slice(1, v['ncol'], None), _slice(1, v['pver'], None), v['ixcldice'])) < _d(1e-36)
                        _put(v,'state_q',(_slice(None,v['ncol'],None),_slice(None,v['pver'],None),v['ixnumice'],),xp.where(_where24,_d(0.0),_get(v['state_q'], (_slice(1, v['ncol'], None), _slice(1, v['pver'], None), v['ixnumice']))))  # WRF:module_mp_cammgmp_driver.F:1509
                        return tuple(v[_n] for _n in ('state_q',))
                    def _no23(_carry):
                        v=dict(_outer23);v.update(zip(('state_q',),_carry))
                        return tuple(v[_n] for _n in ('state_q',))
                    _result=_cond(v['microp_scheme'] == 'mg',_yes23,_no23,tuple(v[_n] for _n in ('state_q',)),xp)  # WRF:module_mp_cammgmp_driver.F:1507
                    v.update(zip(('state_q',),_result))
                    return tuple(v[_n] for _n in ('state_q',))
                def _no21(_carry):
                    v=dict(_outer21);v.update(zip(('state_q',),_carry))
                    return tuple(v[_n] for _n in ('state_q',))
                _result=_cond(v['ptend_name'] == 'convect_deep',_yes21,_no21,tuple(v[_n] for _n in ('state_q',)),xp)  # WRF:module_mp_cammgmp_driver.F:1502
                v.update(zip(('state_q',),_result))
                return tuple(v[_n] for _n in ('state_q',))
            _result=_cond(xp.logical_or(v['ptend_name'] == 'stratiform', v['ptend_name'] == 'cldwat'),_yes20,_no20,tuple(v[_n] for _n in ('state_q',)),xp)  # WRF:module_mp_cammgmp_driver.F:1500
            v.update(zip(('state_q',),_result))
            return tuple(v[_n] for _n in ('state_q',))
        def _no19(_carry):
            v=dict(_outer19);v.update(zip(('state_q',),_carry))
            return tuple(v[_n] for _n in ('state_q',))
        _result=_cond(_get(v['ptend_lq'], (v['ixcldice'],)),_yes19,_no19,tuple(v[_n] for _n in ('state_q',)),xp)  # WRF:module_mp_cammgmp_driver.F:1499
        v.update(zip(('state_q',),_result))
        return tuple(v[_n] for _n in ('state_q',))
    def _no18(_carry):
        v=dict(_outer18);v.update(zip(('state_q',),_carry))
        return tuple(v[_n] for _n in ('state_q',))
    _result=_cond(v['ixcldice'] > 1,_yes18,_no18,tuple(v[_n] for _n in ('state_q',)),xp)  # WRF:module_mp_cammgmp_driver.F:1498
    v.update(zip(('state_q',),_result))
    _call=physics_ptend_reset(v['ptend_name'],v['ptend_q'],v['ptend_s'],v['ptend_lq'],v['ptend_ls'],v['pcnst_in'],_globals=_globals,xp=xp)
    _put(v,'ptend_name',None,_call[0])  # WRF:module_mp_cammgmp_driver.F:1525
    _put(v,'ptend_q',None,_call[1])  # WRF:module_mp_cammgmp_driver.F:1525
    _put(v,'ptend_s',None,_call[2])  # WRF:module_mp_cammgmp_driver.F:1525
    _put(v,'ptend_lq',None,_call[3])  # WRF:module_mp_cammgmp_driver.F:1525
    _put(v,'ptend_ls',None,_call[4])  # WRF:module_mp_cammgmp_driver.F:1525
    return (v['state_q'],v['ptend_q'],v['state_s'],v['ptend_s'],v['ptend_name'],v['ptend_lq'],v['ptend_ls'],)

def findsp_water(lchnk,ncol,q,t,p,tsp,qsp,*,_globals=None,xp=jnp):
    v=dict(_globals or {})
    if '__modules' in v:v.update(v['__modules'].get('module_mp_cammgmp_driver',{}))
    v['lchnk']=lchnk
    v['ncol']=ncol
    v['q']=q
    v['t']=t
    v['p']=p
    v['tsp']=tsp
    v['qsp']=qsp
    v['_jump']=_int(0)
    v['_fatal']=_int(0)
    v['iulog']=''
    v['i']=_zeros((),'i32',xp)
    v['iter']=_zeros((),'i32',xp)
    v['l']=_zeros((),'i32',xp)
    v['error_found']=_zeros((),'bool',xp)
    v['omeps']=_zeros((),'f64',xp)
    v['es']=_zeros((),'f64',xp)
    v['desdt']=_zeros((),'f64',xp)
    v['dqsdt']=_zeros((),'f64',xp)
    v['dgdt']=_zeros((),'f64',xp)
    v['g']=_zeros((),'f64',xp)
    v['hlatsb']=_zeros((),'f64',xp)
    v['hlatvp']=_zeros((),'f64',xp)
    v['hltalt']=_zeros((v['pcols'],),'f64',xp)
    v['qs']=_zeros((),'f64',xp)
    v['tt0']=_zeros((),'f64',xp)
    v['t1']=_zeros((),'f64',xp)
    v['q1']=_zeros((),'f64',xp)
    v['dt']=_zeros((),'f64',xp)
    v['dq']=_zeros((),'f64',xp)
    v['dtm']=_zeros((),'f64',xp)
    v['dqm']=_zeros((),'f64',xp)
    v['qvd']=_zeros((),'f64',xp)
    v['a1']=_zeros((),'f64',xp)
    v['tmp']=_zeros((),'f64',xp)
    v['rair']=_zeros((),'f64',xp)
    v['r1b']=_zeros((),'f64',xp)
    v['c1']=_zeros((),'f64',xp)
    v['c2']=_zeros((),'f64',xp)
    v['c3']=_zeros((),'f64',xp)
    v['denom']=_zeros((),'f64',xp)
    v['dttol']=_zeros((),'f64',xp)
    v['dqtol']=_zeros((),'f64',xp)
    v['doit']=_zeros((v['pcols'],),'i32',xp)
    v['enin']=_zeros((v['pcols'],),'f64',xp)
    v['enout']=_zeros((v['pcols'],),'f64',xp)
    v['tlim']=_zeros((v['pcols'],),'f64',xp)
    _outer1=dict(v)
    def _yes1(_carry):
        v=dict(_outer1);v.update(zip(('omeps',),_carry))
        _put(v,'omeps',None,_d(1.0) - v['epsqs'])  # WRF:module_mp_cammgmp_driver.F:2251
        return tuple(v[_n] for _n in ('omeps',))
    def _no1(_carry):
        v=dict(_outer1);v.update(zip(('omeps',),_carry))
        return tuple(v[_n] for _n in ('omeps',))
    _result=_cond(v['_jump'] == 0,_yes1,_no1,tuple(v[_n] for _n in ('omeps',)),xp)  # WRF:module_mp_cammgmp_driver.F:2251
    v.update(zip(('omeps',),_result))
    _outer2=dict(v)
    def _yes2(_carry):
        v=dict(_outer2);v.update(zip(('a1',),_carry))
        _put(v,'a1',None,_d(7.5) * xp.log(_d(10.0)))  # WRF:module_mp_cammgmp_driver.F:2253
        return tuple(v[_n] for _n in ('a1',))
    def _no2(_carry):
        v=dict(_outer2);v.update(zip(('a1',),_carry))
        return tuple(v[_n] for _n in ('a1',))
    _result=_cond(v['_jump'] == 0,_yes2,_no2,tuple(v[_n] for _n in ('a1',)),xp)  # WRF:module_mp_cammgmp_driver.F:2253
    v.update(zip(('a1',),_result))
    _outer3=dict(v)
    def _yes3(_carry):
        v=dict(_outer3);v.update(zip(('rair',),_carry))
        _put(v,'rair',None,_d(287.04))  # WRF:module_mp_cammgmp_driver.F:2254
        return tuple(v[_n] for _n in ('rair',))
    def _no3(_carry):
        v=dict(_outer3);v.update(zip(('rair',),_carry))
        return tuple(v[_n] for _n in ('rair',))
    _result=_cond(v['_jump'] == 0,_yes3,_no3,tuple(v[_n] for _n in ('rair',)),xp)  # WRF:module_mp_cammgmp_driver.F:2254
    v.update(zip(('rair',),_result))
    _outer4=dict(v)
    def _yes4(_carry):
        v=dict(_outer4);v.update(zip(('c3',),_carry))
        _put(v,'c3',None,v['rair'] * v['a1'] / v['cp'])  # WRF:module_mp_cammgmp_driver.F:2255
        return tuple(v[_n] for _n in ('c3',))
    def _no4(_carry):
        v=dict(_outer4);v.update(zip(('c3',),_carry))
        return tuple(v[_n] for _n in ('c3',))
    _result=_cond(v['_jump'] == 0,_yes4,_no4,tuple(v[_n] for _n in ('c3',)),xp)  # WRF:module_mp_cammgmp_driver.F:2255
    v.update(zip(('c3',),_result))
    _outer5=dict(v)
    def _yes5(_carry):
        v=dict(_outer5);v.update(zip(('dtm',),_carry))
        _put(v,'dtm',None,_d(0.0))  # WRF:module_mp_cammgmp_driver.F:2256
        return tuple(v[_n] for _n in ('dtm',))
    def _no5(_carry):
        v=dict(_outer5);v.update(zip(('dtm',),_carry))
        return tuple(v[_n] for _n in ('dtm',))
    _result=_cond(v['_jump'] == 0,_yes5,_no5,tuple(v[_n] for _n in ('dtm',)),xp)  # WRF:module_mp_cammgmp_driver.F:2256
    v.update(zip(('dtm',),_result))
    _outer6=dict(v)
    def _yes6(_carry):
        v=dict(_outer6);v.update(zip(('dqm',),_carry))
        _put(v,'dqm',None,_d(0.0))  # WRF:module_mp_cammgmp_driver.F:2257
        return tuple(v[_n] for _n in ('dqm',))
    def _no6(_carry):
        v=dict(_outer6);v.update(zip(('dqm',),_carry))
        return tuple(v[_n] for _n in ('dqm',))
    _result=_cond(v['_jump'] == 0,_yes6,_no6,tuple(v[_n] for _n in ('dqm',)),xp)  # WRF:module_mp_cammgmp_driver.F:2257
    v.update(zip(('dqm',),_result))
    _outer7=dict(v)
    def _yes7(_carry):
        v=dict(_outer7);v.update(zip(('dttol',),_carry))
        _put(v,'dttol',None,_d(0.0001))  # WRF:module_mp_cammgmp_driver.F:2258
        return tuple(v[_n] for _n in ('dttol',))
    def _no7(_carry):
        v=dict(_outer7);v.update(zip(('dttol',),_carry))
        return tuple(v[_n] for _n in ('dttol',))
    _result=_cond(v['_jump'] == 0,_yes7,_no7,tuple(v[_n] for _n in ('dttol',)),xp)  # WRF:module_mp_cammgmp_driver.F:2258
    v.update(zip(('dttol',),_result))
    _outer8=dict(v)
    def _yes8(_carry):
        v=dict(_outer8);v.update(zip(('dqtol',),_carry))
        _put(v,'dqtol',None,_d(0.0001))  # WRF:module_mp_cammgmp_driver.F:2259
        return tuple(v[_n] for _n in ('dqtol',))
    def _no8(_carry):
        v=dict(_outer8);v.update(zip(('dqtol',),_carry))
        return tuple(v[_n] for _n in ('dqtol',))
    _result=_cond(v['_jump'] == 0,_yes8,_no8,tuple(v[_n] for _n in ('dqtol',)),xp)  # WRF:module_mp_cammgmp_driver.F:2259
    v.update(zip(('dqtol',),_result))
    _outer9=dict(v)
    def _yes9(_carry):
        v=dict(_outer9);v.update(zip(('tt0',),_carry))
        _put(v,'tt0',None,_d(273.15))  # WRF:module_mp_cammgmp_driver.F:2261
        return tuple(v[_n] for _n in ('tt0',))
    def _no9(_carry):
        v=dict(_outer9);v.update(zip(('tt0',),_carry))
        return tuple(v[_n] for _n in ('tt0',))
    _result=_cond(v['_jump'] == 0,_yes9,_no9,tuple(v[_n] for _n in ('tt0',)),xp)  # WRF:module_mp_cammgmp_driver.F:2261
    v.update(zip(('tt0',),_result))
    _outer10=dict(v)
    def _yes10(_carry):
        v=dict(_outer10);v.update(zip(('iter',),_carry))
        _put(v,'iter',None,8)  # WRF:module_mp_cammgmp_driver.F:2264
        return tuple(v[_n] for _n in ('iter',))
    def _no10(_carry):
        v=dict(_outer10);v.update(zip(('iter',),_carry))
        return tuple(v[_n] for _n in ('iter',))
    _result=_cond(v['_jump'] == 0,_yes10,_no10,tuple(v[_n] for _n in ('iter',)),xp)  # WRF:module_mp_cammgmp_driver.F:2264
    v.update(zip(('iter',),_result))
    _outer11=dict(v)
    def _yes11(_carry):
        v=dict(_outer11);v.update(zip(('c1', 'c2', 'denom', 'doit', 'enin', 'enout', 'es', 'hlatsb', 'hlatvp', 'hltalt', 'i', 'qs', 'qsp', 'qvd', 'r1b', 'tlim', 'tmp', 'tsp'),_carry))
        _outer12=dict(v)
        def _body12(_k,_carry):
            v=dict(_outer12);v.update(zip(('_jump', 'c1', 'c2', 'denom', 'doit', 'enin', 'enout', 'es', 'hlatsb', 'hlatvp', 'hltalt', 'i', 'qs', 'qsp', 'qvd', 'r1b', 'tlim', 'tmp', 'tsp'),_carry));v['i']=_int(_k)
            _put(v,'tlim',(v['i'],),xp.minimum(xp.maximum(_get(v['t'], (v['i'],)), _d(128.0)), _d(373.0)))  # WRF:module_mp_cammgmp_driver.F:2285
            _put(v,'es',None,polysvp(_get(v['tlim'], (v['i'],)), 0, xp=xp, _globals=_globals))  # WRF:module_mp_cammgmp_driver.F:2288
            _put(v,'denom',None,_get(v['p'], (v['i'],)) - v['omeps'] * v['es'])  # WRF:module_mp_cammgmp_driver.F:2289
            _put(v,'qs',None,v['epsqs'] * v['es'] / v['denom'])  # WRF:module_mp_cammgmp_driver.F:2290
            _put(v,'doit',(v['i'],),0)  # WRF:module_mp_cammgmp_driver.F:2291
            _put(v,'enout',(v['i'],),_d(1.0))  # WRF:module_mp_cammgmp_driver.F:2292
            _outer13=dict(v)
            def _yes13(_carry):
                v=dict(_outer13);v.update(zip(('c1', 'c2', 'doit', 'enin', 'es', 'hlatsb', 'hlatvp', 'hltalt', 'qs', 'qsp', 'qvd', 'r1b', 'tmp', 'tsp'),_carry))
                _put(v,'qs',None,xp.minimum(v['epsqs'] * v['es'] / v['denom'], _d(1.0)))  # WRF:module_mp_cammgmp_driver.F:2298
                _put(v,'hlatvp',None,v['hlatv'] - _r(2369.0) * (_get(v['tlim'], (v['i'],)) - v['tt0']))  # WRF:module_mp_cammgmp_driver.F:2324
                _put(v,'hlatsb',None,v['hlatv'])  # WRF:module_mp_cammgmp_driver.F:2325
                _outer14=dict(v)
                def _yes14(_carry):
                    v=dict(_outer14);v.update(zip(('hltalt',),_carry))
                    _put(v,'hltalt',(v['i'],),v['hlatsb'])  # WRF:module_mp_cammgmp_driver.F:2327
                    return tuple(v[_n] for _n in ('hltalt',))
                def _no14(_carry):
                    v=dict(_outer14);v.update(zip(('hltalt',),_carry))
                    _put(v,'hltalt',(v['i'],),v['hlatvp'])  # WRF:module_mp_cammgmp_driver.F:2329
                    return tuple(v[_n] for _n in ('hltalt',))
                _result=_cond(_get(v['tlim'], (v['i'],)) < v['tt0'],_yes14,_no14,tuple(v[_n] for _n in ('hltalt',)),xp)  # WRF:module_mp_cammgmp_driver.F:2326
                v.update(zip(('hltalt',),_result))
                _put(v,'enin',(v['i'],),v['cp'] * _get(v['tlim'], (v['i'],)) + _get(v['hltalt'], (v['i'],)) * _get(v['q'], (v['i'],)))  # WRF:module_mp_cammgmp_driver.F:2332
                _put(v,'tmp',None,_get(v['q'], (v['i'],)) - v['qs'])  # WRF:module_mp_cammgmp_driver.F:2335
                _put(v,'c1',None,_get(v['hltalt'], (v['i'],)) * v['c3'])  # WRF:module_mp_cammgmp_driver.F:2336
                _put(v,'c2',None,_powi(_get(v['tlim'], (v['i'],)) + _d(36.0), 2))  # WRF:module_mp_cammgmp_driver.F:2337
                _put(v,'r1b',None,v['c2'] / (v['c2'] + v['c1'] * v['qs']))  # WRF:module_mp_cammgmp_driver.F:2338
                _put(v,'qvd',None,v['r1b'] * v['tmp'])  # WRF:module_mp_cammgmp_driver.F:2339
                _put(v,'tsp',(v['i'],),_get(v['tlim'], (v['i'],)) + _get(v['hltalt'], (v['i'],)) / v['cp'] * v['qvd'])  # WRF:module_mp_cammgmp_driver.F:2340
                _put(v,'es',None,polysvp(_get(v['tsp'], (v['i'],)), 0, xp=xp, _globals=_globals))  # WRF:module_mp_cammgmp_driver.F:2348
                _put(v,'qsp',(v['i'],),xp.minimum(v['epsqs'] * v['es'] / (_get(v['p'], (v['i'],)) - v['omeps'] * v['es']), _d(1.0)))  # WRF:module_mp_cammgmp_driver.F:2349
                return tuple(v[_n] for _n in ('c1', 'c2', 'doit', 'enin', 'es', 'hlatsb', 'hlatvp', 'hltalt', 'qs', 'qsp', 'qvd', 'r1b', 'tmp', 'tsp'))
            def _no13(_carry):
                v=dict(_outer13);v.update(zip(('c1', 'c2', 'doit', 'enin', 'es', 'hlatsb', 'hlatvp', 'hltalt', 'qs', 'qsp', 'qvd', 'r1b', 'tmp', 'tsp'),_carry))
                _put(v,'doit',(v['i'],),1)  # WRF:module_mp_cammgmp_driver.F:2351
                _put(v,'tsp',(v['i'],),_get(v['tlim'], (v['i'],)))  # WRF:module_mp_cammgmp_driver.F:2352
                _put(v,'qsp',(v['i'],),_get(v['q'], (v['i'],)))  # WRF:module_mp_cammgmp_driver.F:2353
                _put(v,'enin',(v['i'],),_d(1.0))  # WRF:module_mp_cammgmp_driver.F:2354
                return tuple(v[_n] for _n in ('c1', 'c2', 'doit', 'enin', 'es', 'hlatsb', 'hlatvp', 'hltalt', 'qs', 'qsp', 'qvd', 'r1b', 'tmp', 'tsp'))
            _result=_cond(xp.logical_and(xp.logical_and(_get(v['p'], (v['i'],)) > _d(5.0) * v['es'], v['qs'] > _d(0.0)), v['qs'] < _d(0.5)),_yes13,_no13,tuple(v[_n] for _n in ('c1', 'c2', 'doit', 'enin', 'es', 'hlatsb', 'hlatvp', 'hltalt', 'qs', 'qsp', 'qvd', 'r1b', 'tmp', 'tsp')),xp)  # WRF:module_mp_cammgmp_driver.F:2294
            v.update(zip(('c1', 'c2', 'doit', 'enin', 'es', 'hlatsb', 'hlatvp', 'hltalt', 'qs', 'qsp', 'qvd', 'r1b', 'tmp', 'tsp'),_result))
            return tuple(v[_n] for _n in ('_jump', 'c1', 'c2', 'denom', 'doit', 'enin', 'enout', 'es', 'hlatsb', 'hlatvp', 'hltalt', 'i', 'qs', 'qsp', 'qvd', 'r1b', 'tlim', 'tmp', 'tsp'))
        _result=_loop(1,v['ncol'],1,_body12,tuple(v[_n] for _n in ('_jump', 'c1', 'c2', 'denom', 'doit', 'enin', 'enout', 'es', 'hlatsb', 'hlatvp', 'hltalt', 'i', 'qs', 'qsp', 'qvd', 'r1b', 'tlim', 'tmp', 'tsp')),xp,jump_index=0)
        v.update(zip(('_jump', 'c1', 'c2', 'denom', 'doit', 'enin', 'enout', 'es', 'hlatsb', 'hlatvp', 'hltalt', 'i', 'qs', 'qsp', 'qvd', 'r1b', 'tlim', 'tmp', 'tsp'),_result))
        return tuple(v[_n] for _n in ('c1', 'c2', 'denom', 'doit', 'enin', 'enout', 'es', 'hlatsb', 'hlatvp', 'hltalt', 'i', 'qs', 'qsp', 'qvd', 'r1b', 'tlim', 'tmp', 'tsp'))
    def _no11(_carry):
        v=dict(_outer11);v.update(zip(('c1', 'c2', 'denom', 'doit', 'enin', 'enout', 'es', 'hlatsb', 'hlatvp', 'hltalt', 'i', 'qs', 'qsp', 'qvd', 'r1b', 'tlim', 'tmp', 'tsp'),_carry))
        return tuple(v[_n] for _n in ('c1', 'c2', 'denom', 'doit', 'enin', 'enout', 'es', 'hlatsb', 'hlatvp', 'hltalt', 'i', 'qs', 'qsp', 'qvd', 'r1b', 'tlim', 'tmp', 'tsp'))
    _result=_cond(v['_jump'] == 0,_yes11,_no11,tuple(v[_n] for _n in ('c1', 'c2', 'denom', 'doit', 'enin', 'enout', 'es', 'hlatsb', 'hlatvp', 'hltalt', 'i', 'qs', 'qsp', 'qvd', 'r1b', 'tlim', 'tmp', 'tsp')),xp)  # WRF:module_mp_cammgmp_driver.F:2273
    v.update(zip(('c1', 'c2', 'denom', 'doit', 'enin', 'enout', 'es', 'hlatsb', 'hlatvp', 'hltalt', 'i', 'qs', 'qsp', 'qvd', 'r1b', 'tlim', 'tmp', 'tsp'),_result))
    _outer15=dict(v)
    def _yes15(_carry):
        v=dict(_outer15);v.update(zip(('_jump', 'desdt', 'dgdt', 'doit', 'dq', 'dqm', 'dqsdt', 'dt', 'dtm', 'enout', 'es', 'g', 'hlatsb', 'hlatvp', 'hltalt', 'i', 'l', 'q1', 'qs', 'qsp', 't1', 'tsp'),_carry))
        _outer16=dict(v)
        def _body16(_k,_carry):
            v=dict(_outer16);v.update(zip(('_jump', 'desdt', 'dgdt', 'doit', 'dq', 'dqm', 'dqsdt', 'dt', 'dtm', 'enout', 'es', 'g', 'hlatsb', 'hlatvp', 'hltalt', 'i', 'l', 'q1', 'qs', 'qsp', 't1', 'tsp'),_carry));v['l']=_int(_k)
            _outer17=dict(v)
            def _yes17(_carry):
                v=dict(_outer17);v.update(zip(('dtm',),_carry))
                _put(v,'dtm',None,0)  # WRF:module_mp_cammgmp_driver.F:2361
                return tuple(v[_n] for _n in ('dtm',))
            def _no17(_carry):
                v=dict(_outer17);v.update(zip(('dtm',),_carry))
                return tuple(v[_n] for _n in ('dtm',))
            _result=_cond(v['_jump'] == 0,_yes17,_no17,tuple(v[_n] for _n in ('dtm',)),xp)  # WRF:module_mp_cammgmp_driver.F:2361
            v.update(zip(('dtm',),_result))
            _outer18=dict(v)
            def _yes18(_carry):
                v=dict(_outer18);v.update(zip(('dqm',),_carry))
                _put(v,'dqm',None,0)  # WRF:module_mp_cammgmp_driver.F:2362
                return tuple(v[_n] for _n in ('dqm',))
            def _no18(_carry):
                v=dict(_outer18);v.update(zip(('dqm',),_carry))
                return tuple(v[_n] for _n in ('dqm',))
            _result=_cond(v['_jump'] == 0,_yes18,_no18,tuple(v[_n] for _n in ('dqm',)),xp)  # WRF:module_mp_cammgmp_driver.F:2362
            v.update(zip(('dqm',),_result))
            _outer19=dict(v)
            def _yes19(_carry):
                v=dict(_outer19);v.update(zip(('desdt', 'dgdt', 'doit', 'dq', 'dqm', 'dqsdt', 'dt', 'dtm', 'enout', 'es', 'g', 'hlatsb', 'hlatvp', 'hltalt', 'i', 'q1', 'qs', 'qsp', 't1', 'tsp'),_carry))
                _outer20=dict(v)
                def _body20(_k,_carry):
                    v=dict(_outer20);v.update(zip(('_jump', 'desdt', 'dgdt', 'doit', 'dq', 'dqm', 'dqsdt', 'dt', 'dtm', 'enout', 'es', 'g', 'hlatsb', 'hlatvp', 'hltalt', 'i', 'q1', 'qs', 'qsp', 't1', 'tsp'),_carry));v['i']=_int(_k)
                    _outer21=dict(v)
                    def _yes21(_carry):
                        v=dict(_outer21);v.update(zip(('desdt', 'dgdt', 'doit', 'dq', 'dqm', 'dqsdt', 'dt', 'dtm', 'enout', 'es', 'g', 'hlatsb', 'hlatvp', 'hltalt', 'q1', 'qs', 'qsp', 't1', 'tsp'),_carry))
                        _put(v,'es',None,polysvp(_get(v['tsp'], (v['i'],)), 0, xp=xp, _globals=_globals))  # WRF:module_mp_cammgmp_driver.F:2366
                        _put(v,'qs',None,xp.minimum(v['epsqs'] * v['es'] / (_get(v['p'], (v['i'],)) - v['omeps'] * v['es']), _d(1.0)))  # WRF:module_mp_cammgmp_driver.F:2370
                        _put(v,'hlatvp',None,v['hlatv'] - _r(2369.0) * (_get(v['tsp'], (v['i'],)) - v['tt0']))  # WRF:module_mp_cammgmp_driver.F:2402
                        _put(v,'hlatsb',None,v['hlatv'])  # WRF:module_mp_cammgmp_driver.F:2403
                        _outer22=dict(v)
                        def _yes22(_carry):
                            v=dict(_outer22);v.update(zip(('hltalt',),_carry))
                            _put(v,'hltalt',(v['i'],),v['hlatsb'])  # WRF:module_mp_cammgmp_driver.F:2405
                            return tuple(v[_n] for _n in ('hltalt',))
                        def _no22(_carry):
                            v=dict(_outer22);v.update(zip(('hltalt',),_carry))
                            _put(v,'hltalt',(v['i'],),v['hlatvp'])  # WRF:module_mp_cammgmp_driver.F:2407
                            return tuple(v[_n] for _n in ('hltalt',))
                        _result=_cond(_get(v['tsp'], (v['i'],)) < v['tt0'],_yes22,_no22,tuple(v[_n] for _n in ('hltalt',)),xp)  # WRF:module_mp_cammgmp_driver.F:2404
                        v.update(zip(('hltalt',),_result))
                        _put(v,'desdt',None,_get(v['hltalt'], (v['i'],)) * v['es'] / (v['rgasv'] * _get(v['tsp'], (v['i'],)) * _get(v['tsp'], (v['i'],))))  # WRF:module_mp_cammgmp_driver.F:2409
                        _put(v,'dqsdt',None,(v['epsqs'] + v['omeps'] * v['qs']) / (_get(v['p'], (v['i'],)) - v['omeps'] * v['es']) * v['desdt'])  # WRF:module_mp_cammgmp_driver.F:2411
                        _put(v,'g',None,_get(v['enin'], (v['i'],)) - (v['cp'] * _get(v['tsp'], (v['i'],)) + _get(v['hltalt'], (v['i'],)) * _get(v['qsp'], (v['i'],))))  # WRF:module_mp_cammgmp_driver.F:2413
                        _put(v,'dgdt',None,_neg(v['cp'] + _get(v['hltalt'], (v['i'],)) * v['dqsdt']))  # WRF:module_mp_cammgmp_driver.F:2414
                        _put(v,'t1',None,_get(v['tsp'], (v['i'],)) - v['g'] / v['dgdt'])  # WRF:module_mp_cammgmp_driver.F:2415
                        _put(v,'dt',None,xp.abs(v['t1'] - _get(v['tsp'], (v['i'],))) / v['t1'])  # WRF:module_mp_cammgmp_driver.F:2416
                        _put(v,'tsp',(v['i'],),xp.maximum(v['t1'], v['tmin']))  # WRF:module_mp_cammgmp_driver.F:2417
                        _put(v,'es',None,polysvp(_get(v['tsp'], (v['i'],)), 0, xp=xp, _globals=_globals))  # WRF:module_mp_cammgmp_driver.F:2419
                        _put(v,'q1',None,xp.minimum(v['epsqs'] * v['es'] / (_get(v['p'], (v['i'],)) - v['omeps'] * v['es']), _d(1.0)))  # WRF:module_mp_cammgmp_driver.F:2420
                        _put(v,'dq',None,xp.abs(v['q1'] - _get(v['qsp'], (v['i'],))) / xp.maximum(v['q1'], _d(1e-12)))  # WRF:module_mp_cammgmp_driver.F:2421
                        _put(v,'qsp',(v['i'],),v['q1'])  # WRF:module_mp_cammgmp_driver.F:2422
                        _put(v,'dtm',None,xp.maximum(v['dtm'], v['dt']))  # WRF:module_mp_cammgmp_driver.F:2428
                        _put(v,'dqm',None,xp.maximum(v['dqm'], v['dq']))  # WRF:module_mp_cammgmp_driver.F:2429
                        _outer23=dict(v)
                        def _yes23(_carry):
                            v=dict(_outer23);v.update(zip(('doit',),_carry))
                            _put(v,'doit',(v['i'],),2)  # WRF:module_mp_cammgmp_driver.F:2432
                            return tuple(v[_n] for _n in ('doit',))
                        def _no23(_carry):
                            v=dict(_outer23);v.update(zip(('doit',),_carry))
                            return tuple(v[_n] for _n in ('doit',))
                        _result=_cond(xp.logical_and(v['dt'] < v['dttol'], v['dq'] < v['dqtol']),_yes23,_no23,tuple(v[_n] for _n in ('doit',)),xp)  # WRF:module_mp_cammgmp_driver.F:2431
                        v.update(zip(('doit',),_result))
                        _put(v,'enout',(v['i'],),v['cp'] * _get(v['tsp'], (v['i'],)) + _get(v['hltalt'], (v['i'],)) * _get(v['qsp'], (v['i'],)))  # WRF:module_mp_cammgmp_driver.F:2434
                        _outer24=dict(v)
                        def _yes24(_carry):
                            v=dict(_outer24);v.update(zip(('doit',),_carry))
                            _put(v,'doit',(v['i'],),4)  # WRF:module_mp_cammgmp_driver.F:2441
                            return tuple(v[_n] for _n in ('doit',))
                        def _no24(_carry):
                            v=dict(_outer24);v.update(zip(('doit',),_carry))
                            return tuple(v[_n] for _n in ('doit',))
                        _result=_cond(_get(v['tsp'], (v['i'],)) < _d(174.16),_yes24,_no24,tuple(v[_n] for _n in ('doit',)),xp)  # WRF:module_mp_cammgmp_driver.F:2437
                        v.update(zip(('doit',),_result))
                        return tuple(v[_n] for _n in ('desdt', 'dgdt', 'doit', 'dq', 'dqm', 'dqsdt', 'dt', 'dtm', 'enout', 'es', 'g', 'hlatsb', 'hlatvp', 'hltalt', 'q1', 'qs', 'qsp', 't1', 'tsp'))
                    def _no21(_carry):
                        v=dict(_outer21);v.update(zip(('desdt', 'dgdt', 'doit', 'dq', 'dqm', 'dqsdt', 'dt', 'dtm', 'enout', 'es', 'g', 'hlatsb', 'hlatvp', 'hltalt', 'q1', 'qs', 'qsp', 't1', 'tsp'),_carry))
                        return tuple(v[_n] for _n in ('desdt', 'dgdt', 'doit', 'dq', 'dqm', 'dqsdt', 'dt', 'dtm', 'enout', 'es', 'g', 'hlatsb', 'hlatvp', 'hltalt', 'q1', 'qs', 'qsp', 't1', 'tsp'))
                    _result=_cond(_get(v['doit'], (v['i'],)) == 0,_yes21,_no21,tuple(v[_n] for _n in ('desdt', 'dgdt', 'doit', 'dq', 'dqm', 'dqsdt', 'dt', 'dtm', 'enout', 'es', 'g', 'hlatsb', 'hlatvp', 'hltalt', 'q1', 'qs', 'qsp', 't1', 'tsp')),xp)  # WRF:module_mp_cammgmp_driver.F:2364
                    v.update(zip(('desdt', 'dgdt', 'doit', 'dq', 'dqm', 'dqsdt', 'dt', 'dtm', 'enout', 'es', 'g', 'hlatsb', 'hlatvp', 'hltalt', 'q1', 'qs', 'qsp', 't1', 'tsp'),_result))
                    return tuple(v[_n] for _n in ('_jump', 'desdt', 'dgdt', 'doit', 'dq', 'dqm', 'dqsdt', 'dt', 'dtm', 'enout', 'es', 'g', 'hlatsb', 'hlatvp', 'hltalt', 'i', 'q1', 'qs', 'qsp', 't1', 'tsp'))
                _result=_loop(1,v['ncol'],1,_body20,tuple(v[_n] for _n in ('_jump', 'desdt', 'dgdt', 'doit', 'dq', 'dqm', 'dqsdt', 'dt', 'dtm', 'enout', 'es', 'g', 'hlatsb', 'hlatvp', 'hltalt', 'i', 'q1', 'qs', 'qsp', 't1', 'tsp')),xp,jump_index=0)
                v.update(zip(('_jump', 'desdt', 'dgdt', 'doit', 'dq', 'dqm', 'dqsdt', 'dt', 'dtm', 'enout', 'es', 'g', 'hlatsb', 'hlatvp', 'hltalt', 'i', 'q1', 'qs', 'qsp', 't1', 'tsp'),_result))
                return tuple(v[_n] for _n in ('desdt', 'dgdt', 'doit', 'dq', 'dqm', 'dqsdt', 'dt', 'dtm', 'enout', 'es', 'g', 'hlatsb', 'hlatvp', 'hltalt', 'i', 'q1', 'qs', 'qsp', 't1', 'tsp'))
            def _no19(_carry):
                v=dict(_outer19);v.update(zip(('desdt', 'dgdt', 'doit', 'dq', 'dqm', 'dqsdt', 'dt', 'dtm', 'enout', 'es', 'g', 'hlatsb', 'hlatvp', 'hltalt', 'i', 'q1', 'qs', 'qsp', 't1', 'tsp'),_carry))
                return tuple(v[_n] for _n in ('desdt', 'dgdt', 'doit', 'dq', 'dqm', 'dqsdt', 'dt', 'dtm', 'enout', 'es', 'g', 'hlatsb', 'hlatvp', 'hltalt', 'i', 'q1', 'qs', 'qsp', 't1', 'tsp'))
            _result=_cond(v['_jump'] == 0,_yes19,_no19,tuple(v[_n] for _n in ('desdt', 'dgdt', 'doit', 'dq', 'dqm', 'dqsdt', 'dt', 'dtm', 'enout', 'es', 'g', 'hlatsb', 'hlatvp', 'hltalt', 'i', 'q1', 'qs', 'qsp', 't1', 'tsp')),xp)  # WRF:module_mp_cammgmp_driver.F:2363
            v.update(zip(('desdt', 'dgdt', 'doit', 'dq', 'dqm', 'dqsdt', 'dt', 'dtm', 'enout', 'es', 'g', 'hlatsb', 'hlatvp', 'hltalt', 'i', 'q1', 'qs', 'qsp', 't1', 'tsp'),_result))
            _outer25=dict(v)
            def _yes25(_carry):
                v=dict(_outer25);v.update(zip(('_jump',),_carry))
                _outer26=dict(v)
                def _yes26(_carry):
                    v=dict(_outer26);v.update(zip(('_jump',),_carry))
                    v['_jump']=_int(10)  # WRF:module_mp_cammgmp_driver.F:2448
                    return tuple(v[_n] for _n in ('_jump',))
                def _no26(_carry):
                    v=dict(_outer26);v.update(zip(('_jump',),_carry))
                    return tuple(v[_n] for _n in ('_jump',))
                _result=_cond(xp.logical_and(v['dtm'] < v['dttol'], v['dqm'] < v['dqtol']),_yes26,_no26,tuple(v[_n] for _n in ('_jump',)),xp)  # WRF:module_mp_cammgmp_driver.F:2447
                v.update(zip(('_jump',),_result))
                return tuple(v[_n] for _n in ('_jump',))
            def _no25(_carry):
                v=dict(_outer25);v.update(zip(('_jump',),_carry))
                return tuple(v[_n] for _n in ('_jump',))
            _result=_cond(v['_jump'] == 0,_yes25,_no25,tuple(v[_n] for _n in ('_jump',)),xp)  # WRF:module_mp_cammgmp_driver.F:2447
            v.update(zip(('_jump',),_result))
            return tuple(v[_n] for _n in ('_jump', 'desdt', 'dgdt', 'doit', 'dq', 'dqm', 'dqsdt', 'dt', 'dtm', 'enout', 'es', 'g', 'hlatsb', 'hlatvp', 'hltalt', 'i', 'l', 'q1', 'qs', 'qsp', 't1', 'tsp'))
        _result=_loop(1,v['iter'],1,_body16,tuple(v[_n] for _n in ('_jump', 'desdt', 'dgdt', 'doit', 'dq', 'dqm', 'dqsdt', 'dt', 'dtm', 'enout', 'es', 'g', 'hlatsb', 'hlatvp', 'hltalt', 'i', 'l', 'q1', 'qs', 'qsp', 't1', 'tsp')),xp,jump_index=0)
        v.update(zip(('_jump', 'desdt', 'dgdt', 'doit', 'dq', 'dqm', 'dqsdt', 'dt', 'dtm', 'enout', 'es', 'g', 'hlatsb', 'hlatvp', 'hltalt', 'i', 'l', 'q1', 'qs', 'qsp', 't1', 'tsp'),_result))
        return tuple(v[_n] for _n in ('_jump', 'desdt', 'dgdt', 'doit', 'dq', 'dqm', 'dqsdt', 'dt', 'dtm', 'enout', 'es', 'g', 'hlatsb', 'hlatvp', 'hltalt', 'i', 'l', 'q1', 'qs', 'qsp', 't1', 'tsp'))
    def _no15(_carry):
        v=dict(_outer15);v.update(zip(('_jump', 'desdt', 'dgdt', 'doit', 'dq', 'dqm', 'dqsdt', 'dt', 'dtm', 'enout', 'es', 'g', 'hlatsb', 'hlatvp', 'hltalt', 'i', 'l', 'q1', 'qs', 'qsp', 't1', 'tsp'),_carry))
        return tuple(v[_n] for _n in ('_jump', 'desdt', 'dgdt', 'doit', 'dq', 'dqm', 'dqsdt', 'dt', 'dtm', 'enout', 'es', 'g', 'hlatsb', 'hlatvp', 'hltalt', 'i', 'l', 'q1', 'qs', 'qsp', 't1', 'tsp'))
    _result=_cond(v['_jump'] == 0,_yes15,_no15,tuple(v[_n] for _n in ('_jump', 'desdt', 'dgdt', 'doit', 'dq', 'dqm', 'dqsdt', 'dt', 'dtm', 'enout', 'es', 'g', 'hlatsb', 'hlatvp', 'hltalt', 'i', 'l', 'q1', 'qs', 'qsp', 't1', 'tsp')),xp)  # WRF:module_mp_cammgmp_driver.F:2360
    v.update(zip(('_jump', 'desdt', 'dgdt', 'doit', 'dq', 'dqm', 'dqsdt', 'dt', 'dtm', 'enout', 'es', 'g', 'hlatsb', 'hlatvp', 'hltalt', 'i', 'l', 'q1', 'qs', 'qsp', 't1', 'tsp'),_result))
    v['_jump']=xp.where(v['_jump']==10,_int(0),v['_jump'])
    _outer27=dict(v)
    def _yes27(_carry):
        v=dict(_outer27);v.update(zip(('error_found',),_carry))
        _put(v,'error_found',None,False)  # WRF:module_mp_cammgmp_driver.F:2454
        return tuple(v[_n] for _n in ('error_found',))
    def _no27(_carry):
        v=dict(_outer27);v.update(zip(('error_found',),_carry))
        return tuple(v[_n] for _n in ('error_found',))
    _result=_cond(v['_jump'] == 0,_yes27,_no27,tuple(v[_n] for _n in ('error_found',)),xp)  # WRF:module_mp_cammgmp_driver.F:2454
    v.update(zip(('error_found',),_result))
    _outer28=dict(v)
    def _yes28(_carry):
        v=dict(_outer28);v.update(zip(('_fatal', 'error_found', 'i'),_carry))
        _outer29=dict(v)
        def _yes29(_carry):
            v=dict(_outer29);v.update(zip(('_fatal', 'error_found', 'i'),_carry))
            _outer30=dict(v)
            def _body30(_k,_carry):
                v=dict(_outer30);v.update(zip(('_jump', 'error_found', 'i'),_carry));v['i']=_int(_k)
                _outer31=dict(v)
                def _yes31(_carry):
                    v=dict(_outer31);v.update(zip(('error_found',),_carry))
                    _put(v,'error_found',None,True)  # WRF:module_mp_cammgmp_driver.F:2457
                    return tuple(v[_n] for _n in ('error_found',))
                def _no31(_carry):
                    v=dict(_outer31);v.update(zip(('error_found',),_carry))
                    return tuple(v[_n] for _n in ('error_found',))
                _result=_cond(_get(v['doit'], (v['i'],)) == 0,_yes31,_no31,tuple(v[_n] for _n in ('error_found',)),xp)  # WRF:module_mp_cammgmp_driver.F:2457
                v.update(zip(('error_found',),_result))
                return tuple(v[_n] for _n in ('_jump', 'error_found', 'i'))
            _result=_loop(1,v['ncol'],1,_body30,tuple(v[_n] for _n in ('_jump', 'error_found', 'i')),xp,jump_index=0)
            v.update(zip(('_jump', 'error_found', 'i'),_result))
            _outer32=dict(v)
            def _yes32(_carry):
                v=dict(_outer32);v.update(zip(('_fatal', 'i'),_carry))
                _outer33=dict(v)
                def _body33(_k,_carry):
                    v=dict(_outer33);v.update(zip(('_fatal', '_jump', 'i'),_carry));v['i']=_int(_k)
                    _outer34=dict(v)
                    def _yes34(_carry):
                        v=dict(_outer34);v.update(zip(('_fatal',),_carry))
                        v['_fatal']=v['_fatal']+_int(1)  # Original WRF fatal event:module_mp_cammgmp_driver.F:2465; caller must refuse
                        return tuple(v[_n] for _n in ('_fatal',))
                    def _no34(_carry):
                        v=dict(_outer34);v.update(zip(('_fatal',),_carry))
                        return tuple(v[_n] for _n in ('_fatal',))
                    _result=_cond(_get(v['doit'], (v['i'],)) == 0,_yes34,_no34,tuple(v[_n] for _n in ('_fatal',)),xp)  # WRF:module_mp_cammgmp_driver.F:2461
                    v.update(zip(('_fatal',),_result))
                    return tuple(v[_n] for _n in ('_fatal', '_jump', 'i'))
                _result=_loop(1,v['ncol'],1,_body33,tuple(v[_n] for _n in ('_fatal', '_jump', 'i')),xp,jump_index=1)
                v.update(zip(('_fatal', '_jump', 'i'),_result))
                return tuple(v[_n] for _n in ('_fatal', 'i'))
            def _no32(_carry):
                v=dict(_outer32);v.update(zip(('_fatal', 'i'),_carry))
                return tuple(v[_n] for _n in ('_fatal', 'i'))
            _result=_cond(v['error_found'],_yes32,_no32,tuple(v[_n] for _n in ('_fatal', 'i')),xp)  # WRF:module_mp_cammgmp_driver.F:2459
            v.update(zip(('_fatal', 'i'),_result))
            return tuple(v[_n] for _n in ('_fatal', 'error_found', 'i'))
        def _no29(_carry):
            v=dict(_outer29);v.update(zip(('_fatal', 'error_found', 'i'),_carry))
            return tuple(v[_n] for _n in ('_fatal', 'error_found', 'i'))
        _result=_cond(xp.logical_or(v['dtm'] > v['dttol'], v['dqm'] > v['dqtol']),_yes29,_no29,tuple(v[_n] for _n in ('_fatal', 'error_found', 'i')),xp)  # WRF:module_mp_cammgmp_driver.F:2455
        v.update(zip(('_fatal', 'error_found', 'i'),_result))
        return tuple(v[_n] for _n in ('_fatal', 'error_found', 'i'))
    def _no28(_carry):
        v=dict(_outer28);v.update(zip(('_fatal', 'error_found', 'i'),_carry))
        return tuple(v[_n] for _n in ('_fatal', 'error_found', 'i'))
    _result=_cond(v['_jump'] == 0,_yes28,_no28,tuple(v[_n] for _n in ('_fatal', 'error_found', 'i')),xp)  # WRF:module_mp_cammgmp_driver.F:2455
    v.update(zip(('_fatal', 'error_found', 'i'),_result))
    _outer35=dict(v)
    def _yes35(_carry):
        v=dict(_outer35);v.update(zip(('error_found', 'i'),_carry))
        _outer36=dict(v)
        def _body36(_k,_carry):
            v=dict(_outer36);v.update(zip(('_jump', 'error_found', 'i'),_carry));v['i']=_int(_k)
            _outer37=dict(v)
            def _yes37(_carry):
                v=dict(_outer37);v.update(zip(('error_found',),_carry))
                _put(v,'error_found',None,True)  # WRF:module_mp_cammgmp_driver.F:2472
                return tuple(v[_n] for _n in ('error_found',))
            def _no37(_carry):
                v=dict(_outer37);v.update(zip(('error_found',),_carry))
                return tuple(v[_n] for _n in ('error_found',))
            _result=_cond(xp.logical_and(_get(v['doit'], (v['i'],)) == 2, xp.abs((_get(v['enin'], (v['i'],)) - _get(v['enout'], (v['i'],))) / (_get(v['enin'], (v['i'],)) + _get(v['enout'], (v['i'],)))) > _d(0.0001)),_yes37,_no37,tuple(v[_n] for _n in ('error_found',)),xp)  # WRF:module_mp_cammgmp_driver.F:2471
            v.update(zip(('error_found',),_result))
            return tuple(v[_n] for _n in ('_jump', 'error_found', 'i'))
        _result=_loop(1,v['ncol'],1,_body36,tuple(v[_n] for _n in ('_jump', 'error_found', 'i')),xp,jump_index=0)
        v.update(zip(('_jump', 'error_found', 'i'),_result))
        return tuple(v[_n] for _n in ('error_found', 'i'))
    def _no35(_carry):
        v=dict(_outer35);v.update(zip(('error_found', 'i'),_carry))
        return tuple(v[_n] for _n in ('error_found', 'i'))
    _result=_cond(v['_jump'] == 0,_yes35,_no35,tuple(v[_n] for _n in ('error_found', 'i')),xp)  # WRF:module_mp_cammgmp_driver.F:2470
    v.update(zip(('error_found', 'i'),_result))
    _outer38=dict(v)
    def _yes38(_carry):
        v=dict(_outer38);v.update(zip(('_fatal', 'i'),_carry))
        _outer39=dict(v)
        def _yes39(_carry):
            v=dict(_outer39);v.update(zip(('_fatal', 'i'),_carry))
            _outer40=dict(v)
            def _body40(_k,_carry):
                v=dict(_outer40);v.update(zip(('_fatal', '_jump', 'i'),_carry));v['i']=_int(_k)
                _outer41=dict(v)
                def _yes41(_carry):
                    v=dict(_outer41);v.update(zip(('_fatal',),_carry))
                    v['_fatal']=v['_fatal']+_int(1)  # Original WRF fatal event:module_mp_cammgmp_driver.F:2482; caller must refuse
                    return tuple(v[_n] for _n in ('_fatal',))
                def _no41(_carry):
                    v=dict(_outer41);v.update(zip(('_fatal',),_carry))
                    return tuple(v[_n] for _n in ('_fatal',))
                _result=_cond(xp.logical_and(_get(v['doit'], (v['i'],)) == 2, xp.abs((_get(v['enin'], (v['i'],)) - _get(v['enout'], (v['i'],))) / (_get(v['enin'], (v['i'],)) + _get(v['enout'], (v['i'],)))) > _d(0.0001)),_yes41,_no41,tuple(v[_n] for _n in ('_fatal',)),xp)  # WRF:module_mp_cammgmp_driver.F:2477
                v.update(zip(('_fatal',),_result))
                return tuple(v[_n] for _n in ('_fatal', '_jump', 'i'))
            _result=_loop(1,v['ncol'],1,_body40,tuple(v[_n] for _n in ('_fatal', '_jump', 'i')),xp,jump_index=1)
            v.update(zip(('_fatal', '_jump', 'i'),_result))
            return tuple(v[_n] for _n in ('_fatal', 'i'))
        def _no39(_carry):
            v=dict(_outer39);v.update(zip(('_fatal', 'i'),_carry))
            return tuple(v[_n] for _n in ('_fatal', 'i'))
        _result=_cond(v['error_found'],_yes39,_no39,tuple(v[_n] for _n in ('_fatal', 'i')),xp)  # WRF:module_mp_cammgmp_driver.F:2475
        v.update(zip(('_fatal', 'i'),_result))
        return tuple(v[_n] for _n in ('_fatal', 'i'))
    def _no38(_carry):
        v=dict(_outer38);v.update(zip(('_fatal', 'i'),_carry))
        return tuple(v[_n] for _n in ('_fatal', 'i'))
    _result=_cond(v['_jump'] == 0,_yes38,_no38,tuple(v[_n] for _n in ('_fatal', 'i')),xp)  # WRF:module_mp_cammgmp_driver.F:2475
    v.update(zip(('_fatal', 'i'),_result))
    return (v['tsp'],v['qsp'],v['_fatal'],)

def macrophysics_simple(is_first_step,pcnst,pcols,kte,ixcldliq,ixnumliq,lchnk,dtime,state1_t,state_pmid,state_q,dp_icwmr,sh_icwmr,dp_frac,sh_frac,ast,aist,alst,qi_mac,xland_pt,snowh_pt,state1_s,state1_q,rh_old,cldo,esl,qvs,ptend_loc_name,ptend_loc_ls,ptend_loc_lq,ptend_loc_s,ptend_loc_q,ptend_all_name,ptend_all_ls,ptend_all_lq,ptend_all_s,ptend_all_q,*,_globals=None,xp=jnp):
    v=dict(_globals or {})
    if '__modules' in v:v.update(v['__modules'].get('module_mp_cammgmp_driver',{}))
    v['is_first_step']=is_first_step
    v['pcnst']=pcnst
    v['pcols']=pcols
    v['kte']=kte
    v['ixcldliq']=ixcldliq
    v['ixnumliq']=ixnumliq
    v['lchnk']=lchnk
    v['dtime']=dtime
    v['state1_t']=state1_t
    v['state_pmid']=state_pmid
    v['state_q']=state_q
    v['dp_icwmr']=dp_icwmr
    v['sh_icwmr']=sh_icwmr
    v['dp_frac']=dp_frac
    v['sh_frac']=sh_frac
    v['ast']=ast
    v['aist']=aist
    v['alst']=alst
    v['qi_mac']=qi_mac
    v['xland_pt']=xland_pt
    v['snowh_pt']=snowh_pt
    v['state1_s']=state1_s
    v['state1_q']=state1_q
    v['rh_old']=rh_old
    v['cldo']=cldo
    v['esl']=esl
    v['qvs']=qvs
    v['ptend_loc_name']=ptend_loc_name
    v['ptend_loc_ls']=ptend_loc_ls
    v['ptend_loc_lq']=ptend_loc_lq
    v['ptend_loc_s']=ptend_loc_s
    v['ptend_loc_q']=ptend_loc_q
    v['ptend_all_name']=ptend_all_name
    v['ptend_all_ls']=ptend_all_ls
    v['ptend_all_lq']=ptend_all_lq
    v['ptend_all_s']=ptend_all_s
    v['ptend_all_q']=ptend_all_q
    v['_jump']=_int(0)
    v['_fatal']=_int(0)
    v['k']=_zeros((),'i32',xp)
    v['cv_frac']=_zeros((v['pcols'],v['kte'],),'f64',xp)
    v['st_frac']=_zeros((v['pcols'],v['kte'],),'f64',xp)
    v['rh_curr']=_zeros((v['pcols'],v['kte'],),'f64',xp)
    v['qc_nc']=_zeros((v['pcols'],v['kte'],),'f64',xp)
    v['del_qq']=_zeros((v['pcols'],v['kte'],),'f64',xp)
    v['del_cf']=_zeros((v['pcols'],v['kte'],),'f64',xp)
    v['rh_temp']=_zeros((v['pcols'],v['kte'],),'f64',xp)
    v['qsll']=_zeros((v['pcols'],),'f64',xp)
    v['tsp']=_zeros((v['pcols'],),'f64',xp)
    v['cldcr']=_zeros((),'f64',xp)
    v['factcc']=_zeros((),'f64',xp)
    v['fact_adj']=_zeros((),'f64',xp)
    _outer2=dict(v)
    def _yes2(_carry):
        v=dict(_outer2);v.update(zip(('ptend_loc_ls',),_carry))
        _put(v,'ptend_loc_ls',None,True)  # WRF:module_mp_cammgmp_driver.F:1846
        return tuple(v[_n] for _n in ('ptend_loc_ls',))
    def _no2(_carry):
        v=dict(_outer2);v.update(zip(('ptend_loc_ls',),_carry))
        return tuple(v[_n] for _n in ('ptend_loc_ls',))
    _result=_cond(v['_jump'] == 0,_yes2,_no2,tuple(v[_n] for _n in ('ptend_loc_ls',)),xp)  # WRF:module_mp_cammgmp_driver.F:1846
    v.update(zip(('ptend_loc_ls',),_result))
    _outer3=dict(v)
    def _yes3(_carry):
        v=dict(_outer3);v.update(zip(('ptend_loc_lq',),_carry))
        _put(v,'ptend_loc_lq',(1,),True)  # WRF:module_mp_cammgmp_driver.F:1847
        return tuple(v[_n] for _n in ('ptend_loc_lq',))
    def _no3(_carry):
        v=dict(_outer3);v.update(zip(('ptend_loc_lq',),_carry))
        return tuple(v[_n] for _n in ('ptend_loc_lq',))
    _result=_cond(v['_jump'] == 0,_yes3,_no3,tuple(v[_n] for _n in ('ptend_loc_lq',)),xp)  # WRF:module_mp_cammgmp_driver.F:1847
    v.update(zip(('ptend_loc_lq',),_result))
    _outer4=dict(v)
    def _yes4(_carry):
        v=dict(_outer4);v.update(zip(('ptend_loc_lq',),_carry))
        _put(v,'ptend_loc_lq',(v['ixcldliq'],),True)  # WRF:module_mp_cammgmp_driver.F:1848
        return tuple(v[_n] for _n in ('ptend_loc_lq',))
    def _no4(_carry):
        v=dict(_outer4);v.update(zip(('ptend_loc_lq',),_carry))
        return tuple(v[_n] for _n in ('ptend_loc_lq',))
    _result=_cond(v['_jump'] == 0,_yes4,_no4,tuple(v[_n] for _n in ('ptend_loc_lq',)),xp)  # WRF:module_mp_cammgmp_driver.F:1848
    v.update(zip(('ptend_loc_lq',),_result))
    _outer5=dict(v)
    def _yes5(_carry):
        v=dict(_outer5);v.update(zip(('ptend_loc_lq',),_carry))
        _put(v,'ptend_loc_lq',(v['ixnumliq'],),True)  # WRF:module_mp_cammgmp_driver.F:1849
        return tuple(v[_n] for _n in ('ptend_loc_lq',))
    def _no5(_carry):
        v=dict(_outer5);v.update(zip(('ptend_loc_lq',),_carry))
        return tuple(v[_n] for _n in ('ptend_loc_lq',))
    _result=_cond(v['_jump'] == 0,_yes5,_no5,tuple(v[_n] for _n in ('ptend_loc_lq',)),xp)  # WRF:module_mp_cammgmp_driver.F:1849
    v.update(zip(('ptend_loc_lq',),_result))
    _outer6=dict(v)
    def _yes6(_carry):
        v=dict(_outer6);v.update(zip(('factcc',),_carry))
        _put(v,'factcc',None,_d(0.1))  # WRF:module_mp_cammgmp_driver.F:1851
        return tuple(v[_n] for _n in ('factcc',))
    def _no6(_carry):
        v=dict(_outer6);v.update(zip(('factcc',),_carry))
        return tuple(v[_n] for _n in ('factcc',))
    _result=_cond(v['_jump'] == 0,_yes6,_no6,tuple(v[_n] for _n in ('factcc',)),xp)  # WRF:module_mp_cammgmp_driver.F:1851
    v.update(zip(('factcc',),_result))
    _outer7=dict(v)
    def _yes7(_carry):
        v=dict(_outer7);v.update(zip(('fact_adj',),_carry))
        _put(v,'fact_adj',None,_d(0.35))  # WRF:module_mp_cammgmp_driver.F:1852
        return tuple(v[_n] for _n in ('fact_adj',))
    def _no7(_carry):
        v=dict(_outer7);v.update(zip(('fact_adj',),_carry))
        return tuple(v[_n] for _n in ('fact_adj',))
    _result=_cond(v['_jump'] == 0,_yes7,_no7,tuple(v[_n] for _n in ('fact_adj',)),xp)  # WRF:module_mp_cammgmp_driver.F:1852
    v.update(zip(('fact_adj',),_result))
    _outer8=dict(v)
    def _yes8(_carry):
        v=dict(_outer8);v.update(zip(('cldcr',),_carry))
        _put(v,'cldcr',None,_d(0.8875))  # WRF:module_mp_cammgmp_driver.F:1853
        return tuple(v[_n] for _n in ('cldcr',))
    def _no8(_carry):
        v=dict(_outer8);v.update(zip(('cldcr',),_carry))
        return tuple(v[_n] for _n in ('cldcr',))
    _result=_cond(v['_jump'] == 0,_yes8,_no8,tuple(v[_n] for _n in ('cldcr',)),xp)  # WRF:module_mp_cammgmp_driver.F:1853
    v.update(zip(('cldcr',),_result))
    _outer9=dict(v)
    def _yes9(_carry):
        v=dict(_outer9);v.update(zip(('_fatal', 'aist', 'alst', 'ast', 'cldcr', 'cldo', 'cv_frac', 'del_cf', 'del_qq', 'esl', 'k', 'ptend_loc_q', 'ptend_loc_s', 'qc_nc', 'qsll', 'qvs', 'rh_curr', 'rh_old', 'rh_temp', 'st_frac', 'state1_t', 'tsp'),_carry))
        _outer10=dict(v)
        def _body10(_k,_carry):
            v=dict(_outer10);v.update(zip(('_fatal', '_jump', 'aist', 'alst', 'ast', 'cldcr', 'cldo', 'cv_frac', 'del_cf', 'del_qq', 'esl', 'k', 'ptend_loc_q', 'ptend_loc_s', 'qc_nc', 'qsll', 'qvs', 'rh_curr', 'rh_old', 'rh_temp', 'st_frac', 'state1_t', 'tsp'),_carry));v['k']=_int(_k)
            _outer11=dict(v)
            def _yes11(_carry):
                v=dict(_outer11);v.update(zip(('cldcr',),_carry))
                _outer12=dict(v)
                def _yes12(_carry):
                    v=dict(_outer12);v.update(zip(('cldcr',),_carry))
                    _put(v,'cldcr',None,_d(0.7875))  # WRF:module_mp_cammgmp_driver.F:1860
                    return tuple(v[_n] for _n in ('cldcr',))
                def _no12(_carry):
                    v=dict(_outer12);v.update(zip(('cldcr',),_carry))
                    _put(v,'cldcr',None,_d(0.8875))  # WRF:module_mp_cammgmp_driver.F:1862
                    return tuple(v[_n] for _n in ('cldcr',))
                _result=_cond(xp.logical_and(v['xland_pt'] < _d(1.1), v['snowh_pt'] < _d(0.001)),_yes12,_no12,tuple(v[_n] for _n in ('cldcr',)),xp)  # WRF:module_mp_cammgmp_driver.F:1859
                v.update(zip(('cldcr',),_result))
                return tuple(v[_n] for _n in ('cldcr',))
            def _no11(_carry):
                v=dict(_outer11);v.update(zip(('cldcr',),_carry))
                _outer13=dict(v)
                def _yes13(_carry):
                    v=dict(_outer13);v.update(zip(('cldcr',),_carry))
                    _put(v,'cldcr',None,_d(0.8))  # WRF:module_mp_cammgmp_driver.F:1865
                    return tuple(v[_n] for _n in ('cldcr',))
                def _no13(_carry):
                    v=dict(_outer13);v.update(zip(('cldcr',),_carry))
                    _put(v,'cldcr',None,(_d(0.8875) - _d(0.8)) / _d(30000.0) * (_get(v['state_pmid'], (1, v['k'])) - _d(40000.0)) + _d(0.8))  # WRF:module_mp_cammgmp_driver.F:1867
                    return tuple(v[_n] for _n in ('cldcr',))
                _result=_cond(_get(v['state_pmid'], (1, v['k'])) <= _d(40000.0),_yes13,_no13,tuple(v[_n] for _n in ('cldcr',)),xp)  # WRF:module_mp_cammgmp_driver.F:1864
                v.update(zip(('cldcr',),_result))
                return tuple(v[_n] for _n in ('cldcr',))
            _result=_cond(_get(v['state_pmid'], (1, v['k'])) >= _d(70000.0),_yes11,_no11,tuple(v[_n] for _n in ('cldcr',)),xp)  # WRF:module_mp_cammgmp_driver.F:1858
            v.update(zip(('cldcr',),_result))
            _put(v,'cldcr',None,xp.minimum(xp.maximum(_d(0.0), v['cldcr']), _d(1.0)))  # WRF:module_mp_cammgmp_driver.F:1870
            _call=cal_cldfra_mp_1d(_get(v['alst'], (1, v['k'])),_get(v['aist'], (1, v['k'])),_get(v['state1_q'], (1, v['k'], 1)),_get(v['state1_q'], (1, v['k'], v['ixcldliq'])),_get(v['qi_mac'], (1, v['k'])),_get(v['state1_t'], (1, v['k'])),_get(v['state_pmid'], (1, v['k'])),v['xland_pt'],v['snowh_pt'],_globals=_globals,xp=xp)
            _put(v,'alst',(1,v['k'],),_call[0])  # WRF:module_mp_cammgmp_driver.F:1872
            _put(v,'aist',(1,v['k'],),_call[1])  # WRF:module_mp_cammgmp_driver.F:1872
            _put(v,'qsll',None,_d(0.0))  # WRF:module_mp_cammgmp_driver.F:1873
            _put(v,'tsp',None,_d(0.0))  # WRF:module_mp_cammgmp_driver.F:1874
            _put(v,'del_qq',(1,v['k'],),_d(0.0))  # WRF:module_mp_cammgmp_driver.F:1875
            _put(v,'rh_temp',(1,v['k'],),_d(0.0))  # WRF:module_mp_cammgmp_driver.F:1876
            _put(v,'cv_frac',(1,v['k'],),xp.maximum(_d(0.0), xp.minimum(_d(0.8), _get(v['dp_frac'], (1, v['k'])) + _get(v['sh_frac'], (1, v['k'])))))  # WRF:module_mp_cammgmp_driver.F:1877
            _put(v,'alst',(1,v['k'],),(_d(1.0) - _get(v['cv_frac'], (1, v['k']))) * _get(v['alst'], (1, v['k'])))  # WRF:module_mp_cammgmp_driver.F:1878
            _put(v,'aist',(1,v['k'],),(_d(1.0) - _get(v['cv_frac'], (1, v['k']))) * _get(v['aist'], (1, v['k'])))  # WRF:module_mp_cammgmp_driver.F:1879
            _put(v,'ast',(1,v['k'],),xp.maximum(_get(v['alst'], (1, v['k'])), _get(v['aist'], (1, v['k']))))  # WRF:module_mp_cammgmp_driver.F:1880
            _put(v,'st_frac',(1,v['k'],),xp.minimum(_d(1.0), _get(v['ast'], (1, v['k'])) + _get(v['cv_frac'], (1, v['k']))))  # WRF:module_mp_cammgmp_driver.F:1881
            _put(v,'esl',(1,v['k'],),polysvp(_get(v['state1_t'], (1, v['k'])), 0, xp=xp, _globals=_globals))  # WRF:module_mp_cammgmp_driver.F:1882
            _put(v,'qvs',(1,v['k'],),xp.maximum(_d(1e-30), v['epsqs'] * _get(v['esl'], (1, v['k'])) / (_get(v['state_pmid'], (1, v['k'])) - (_d(1.0) - v['epsqs']) * _get(v['esl'], (1, v['k'])))))  # WRF:module_mp_cammgmp_driver.F:1883
            _put(v,'rh_curr',(1,v['k'],),xp.maximum(_d(0.0), _get(v['state1_q'], (1, v['k'], 1)) / _get(v['qvs'], (1, v['k']))))  # WRF:module_mp_cammgmp_driver.F:1884
            _put(v,'qc_nc',(1,v['k'],),_get(v['state1_q'], (1, v['k'], v['ixcldliq'])))  # WRF:module_mp_cammgmp_driver.F:1885
            _outer14=dict(v)
            def _yes14(_carry):
                v=dict(_outer14);v.update(zip(('cldo', 'rh_old'),_carry))
                _put(v,'rh_old',(1,v['k'],),_get(v['rh_curr'], (1, v['k'])))  # WRF:module_mp_cammgmp_driver.F:1888
                _put(v,'cldo',(1,v['k'],),xp.minimum(_d(1.0), _get(v['ast'], (1, v['k'])) + _get(v['cv_frac'], (1, v['k']))))  # WRF:module_mp_cammgmp_driver.F:1889
                return tuple(v[_n] for _n in ('cldo', 'rh_old'))
            def _no14(_carry):
                v=dict(_outer14);v.update(zip(('cldo', 'rh_old'),_carry))
                return tuple(v[_n] for _n in ('cldo', 'rh_old'))
            _result=_cond(v['is_first_step'],_yes14,_no14,tuple(v[_n] for _n in ('cldo', 'rh_old')),xp)  # WRF:module_mp_cammgmp_driver.F:1887
            v.update(zip(('cldo', 'rh_old'),_result))
            _outer15=dict(v)
            def _yes15(_carry):
                v=dict(_outer15);v.update(zip(('_fatal', 'del_cf', 'del_qq', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'tsp'),_carry))
                _call=findsp_water(v['lchnk'],v['pcols'],_get(v['state1_q'], (1, v['k'], 1)),_get(v['state1_t'], (1, v['k'])),_get(v['state_pmid'], (1, v['k'])),v['tsp'],v['qsll'],_globals=_globals,xp=xp)
                _put(v,'tsp',None,_call[0])  # WRF:module_mp_cammgmp_driver.F:1897
                _put(v,'qsll',None,_call[1])  # WRF:module_mp_cammgmp_driver.F:1897
                v['_fatal']=v['_fatal']+_call[2]
                _put(v,'del_qq',(1,v['k'],),xp.minimum(_d(0.003), xp.maximum(_d(0.0), (_get(v['state1_q'], (1, v['k'], 1)) - _get(v['qsll'], (1,))) * _d(0.999))))  # WRF:module_mp_cammgmp_driver.F:1898
                _put(v,'ptend_loc_q',(1,v['k'],v['ixcldliq'],),_get(v['del_qq'], (1, v['k'])) / v['dtime'])  # WRF:module_mp_cammgmp_driver.F:1899
                _put(v,'ptend_loc_s',(1,v['k'],),_get(v['del_qq'], (1, v['k'])) * v['latvap'] / v['dtime'])  # WRF:module_mp_cammgmp_driver.F:1900
                _put(v,'ptend_loc_q',(1,v['k'],1,),_neg(_get(v['del_qq'], (1, v['k']))) / v['dtime'])  # WRF:module_mp_cammgmp_driver.F:1901
                return tuple(v[_n] for _n in ('_fatal', 'del_cf', 'del_qq', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'tsp'))
            def _no15(_carry):
                v=dict(_outer15);v.update(zip(('_fatal', 'del_cf', 'del_qq', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'tsp'),_carry))
                _outer16=dict(v)
                def _yes16(_carry):
                    v=dict(_outer16);v.update(zip(('_fatal', 'del_cf', 'del_qq', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'tsp'),_carry))
                    _put(v,'del_cf',(1,v['k'],),_get(v['st_frac'], (1, v['k'])) - _get(v['cldo'], (1, v['k'])))  # WRF:module_mp_cammgmp_driver.F:1908
                    _outer17=dict(v)
                    def _yes17(_carry):
                        v=dict(_outer17);v.update(zip(('del_qq', 'ptend_loc_q', 'ptend_loc_s'),_carry))
                        _put(v,'del_qq',(1,v['k'],),_get(v['qc_nc'], (1, v['k'])) * _get(v['del_cf'], (1, v['k'])) / _get(v['cldo'], (1, v['k'])) * v['factcc'])  # WRF:module_mp_cammgmp_driver.F:1910
                        _put(v,'ptend_loc_q',(1,v['k'],v['ixcldliq'],),_get(v['del_qq'], (1, v['k'])) / v['dtime'])  # WRF:module_mp_cammgmp_driver.F:1911
                        _put(v,'ptend_loc_s',(1,v['k'],),_get(v['del_qq'], (1, v['k'])) * v['latvap'] / v['dtime'])  # WRF:module_mp_cammgmp_driver.F:1912
                        _put(v,'ptend_loc_q',(1,v['k'],1,),_neg(_get(v['del_qq'], (1, v['k']))) / v['dtime'])  # WRF:module_mp_cammgmp_driver.F:1913
                        _put(v,'ptend_loc_q',(1,v['k'],v['ixnumliq'],),_get(v['ptend_loc_q'], (1, v['k'], v['ixcldliq'])) * _get(v['state1_q'], (1, v['k'], v['ixnumliq'])) / xp.maximum(_d(1e-30), _get(v['state1_q'], (1, v['k'], v['ixcldliq']))))  # WRF:module_mp_cammgmp_driver.F:1914
                        _put(v,'ptend_loc_q',(1,v['k'],v['ixnumliq'],),xp.minimum(_d(0.0), xp.maximum(_get(v['ptend_loc_q'], (1, v['k'], v['ixnumliq'])), _neg(_get(v['state1_q'], (1, v['k'], v['ixnumliq']))) / v['dtime'])))  # WRF:module_mp_cammgmp_driver.F:1915
                        return tuple(v[_n] for _n in ('del_qq', 'ptend_loc_q', 'ptend_loc_s'))
                    def _no17(_carry):
                        v=dict(_outer17);v.update(zip(('del_qq', 'ptend_loc_q', 'ptend_loc_s'),_carry))
                        _outer18=dict(v)
                        def _yes18(_carry):
                            v=dict(_outer18);v.update(zip(('del_qq', 'ptend_loc_q', 'ptend_loc_s'),_carry))
                            _put(v,'del_qq',(1,v['k'],),xp.minimum(_get(v['state1_q'], (1, v['k'], 1)) * _d(0.1), xp.minimum(_d(0.003) * _get(v['st_frac'], (1, v['k'])), _get(v['qc_nc'], (1, v['k'])) * _get(v['del_cf'], (1, v['k'])) / _get(v['cldo'], (1, v['k'])))) * v['factcc'])  # WRF:module_mp_cammgmp_driver.F:1917
                            _put(v,'ptend_loc_q',(1,v['k'],v['ixcldliq'],),_get(v['del_qq'], (1, v['k'])) / v['dtime'])  # WRF:module_mp_cammgmp_driver.F:1918
                            _put(v,'ptend_loc_s',(1,v['k'],),_get(v['del_qq'], (1, v['k'])) * v['latvap'] / v['dtime'])  # WRF:module_mp_cammgmp_driver.F:1919
                            _put(v,'ptend_loc_q',(1,v['k'],1,),_neg(_get(v['del_qq'], (1, v['k']))) / v['dtime'])  # WRF:module_mp_cammgmp_driver.F:1920
                            _put(v,'ptend_loc_q',(1,v['k'],v['ixnumliq'],),_d(0.0))  # WRF:module_mp_cammgmp_driver.F:1921
                            return tuple(v[_n] for _n in ('del_qq', 'ptend_loc_q', 'ptend_loc_s'))
                        def _no18(_carry):
                            v=dict(_outer18);v.update(zip(('del_qq', 'ptend_loc_q', 'ptend_loc_s'),_carry))
                            _outer19=dict(v)
                            def _yes19(_carry):
                                v=dict(_outer19);v.update(zip(('del_qq', 'ptend_loc_q', 'ptend_loc_s'),_carry))
                                _put(v,'del_qq',(1,v['k'],),xp.minimum(_get(v['state1_q'], (1, v['k'], 1)) * _d(0.1), _d(2e-05) * _get(v['st_frac'], (1, v['k']))) * v['factcc'])  # WRF:module_mp_cammgmp_driver.F:1923
                                _put(v,'ptend_loc_q',(1,v['k'],v['ixcldliq'],),_get(v['del_qq'], (1, v['k'])) / v['dtime'])  # WRF:module_mp_cammgmp_driver.F:1924
                                _put(v,'ptend_loc_s',(1,v['k'],),_get(v['del_qq'], (1, v['k'])) * v['latvap'] / v['dtime'])  # WRF:module_mp_cammgmp_driver.F:1925
                                _put(v,'ptend_loc_q',(1,v['k'],1,),_neg(_get(v['del_qq'], (1, v['k']))) / v['dtime'])  # WRF:module_mp_cammgmp_driver.F:1926
                                _put(v,'ptend_loc_q',(1,v['k'],v['ixnumliq'],),_d(0.0))  # WRF:module_mp_cammgmp_driver.F:1927
                                return tuple(v[_n] for _n in ('del_qq', 'ptend_loc_q', 'ptend_loc_s'))
                            def _no19(_carry):
                                v=dict(_outer19);v.update(zip(('del_qq', 'ptend_loc_q', 'ptend_loc_s'),_carry))
                                _put(v,'del_qq',(1,v['k'],),_d(0.0))  # WRF:module_mp_cammgmp_driver.F:1929
                                _put(v,'ptend_loc_q',(1,v['k'],v['ixcldliq'],),_d(0.0))  # WRF:module_mp_cammgmp_driver.F:1930
                                _put(v,'ptend_loc_s',(1,v['k'],),_d(0.0))  # WRF:module_mp_cammgmp_driver.F:1931
                                _put(v,'ptend_loc_q',(1,v['k'],1,),_d(0.0))  # WRF:module_mp_cammgmp_driver.F:1932
                                _put(v,'ptend_loc_q',(1,v['k'],v['ixnumliq'],),_d(0.0))  # WRF:module_mp_cammgmp_driver.F:1933
                                return tuple(v[_n] for _n in ('del_qq', 'ptend_loc_q', 'ptend_loc_s'))
                            _result=_cond(xp.logical_and(_get(v['del_cf'], (1, v['k'])) > _d(0.0), _get(v['cldo'], (1, v['k'])) < _d(1e-08)),_yes19,_no19,tuple(v[_n] for _n in ('del_qq', 'ptend_loc_q', 'ptend_loc_s')),xp)  # WRF:module_mp_cammgmp_driver.F:1922
                            v.update(zip(('del_qq', 'ptend_loc_q', 'ptend_loc_s'),_result))
                            return tuple(v[_n] for _n in ('del_qq', 'ptend_loc_q', 'ptend_loc_s'))
                        _result=_cond(xp.logical_and(_get(v['del_cf'], (1, v['k'])) > _d(0.0), _get(v['cldo'], (1, v['k'])) > _d(1e-08)),_yes18,_no18,tuple(v[_n] for _n in ('del_qq', 'ptend_loc_q', 'ptend_loc_s')),xp)  # WRF:module_mp_cammgmp_driver.F:1916
                        v.update(zip(('del_qq', 'ptend_loc_q', 'ptend_loc_s'),_result))
                        return tuple(v[_n] for _n in ('del_qq', 'ptend_loc_q', 'ptend_loc_s'))
                    _result=_cond(_get(v['del_cf'], (1, v['k'])) < _d(0.0),_yes17,_no17,tuple(v[_n] for _n in ('del_qq', 'ptend_loc_q', 'ptend_loc_s')),xp)  # WRF:module_mp_cammgmp_driver.F:1909
                    v.update(zip(('del_qq', 'ptend_loc_q', 'ptend_loc_s'),_result))
                    return tuple(v[_n] for _n in ('_fatal', 'del_cf', 'del_qq', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'tsp'))
                def _no16(_carry):
                    v=dict(_outer16);v.update(zip(('_fatal', 'del_cf', 'del_qq', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'tsp'),_carry))
                    _outer20=dict(v)
                    def _yes20(_carry):
                        v=dict(_outer20);v.update(zip(('_fatal', 'del_qq', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'tsp'),_carry))
                        _call=findsp_water(v['lchnk'],v['pcols'],_get(v['state1_q'], (1, v['k'], 1)),_get(v['state1_t'], (1, v['k'])),_get(v['state_pmid'], (1, v['k'])),v['tsp'],v['qsll'],_globals=_globals,xp=xp)
                        _put(v,'tsp',None,_call[0])  # WRF:module_mp_cammgmp_driver.F:1936
                        _put(v,'qsll',None,_call[1])  # WRF:module_mp_cammgmp_driver.F:1936
                        v['_fatal']=v['_fatal']+_call[2]
                        _put(v,'del_qq',(1,v['k'],),xp.maximum(xp.minimum(_get(v['qc_nc'], (1, v['k'])), _get(v['qsll'], (1,)) - _get(v['state1_q'], (1, v['k'], 1))), _d(0.0)) * _d(0.99))  # WRF:module_mp_cammgmp_driver.F:1937
                        _put(v,'ptend_loc_q',(1,v['k'],v['ixcldliq'],),_neg(_get(v['del_qq'], (1, v['k']))) / v['dtime'])  # WRF:module_mp_cammgmp_driver.F:1938
                        _put(v,'ptend_loc_s',(1,v['k'],),_neg(_get(v['del_qq'], (1, v['k']))) * v['latvap'] / v['dtime'])  # WRF:module_mp_cammgmp_driver.F:1939
                        _put(v,'ptend_loc_q',(1,v['k'],1,),_get(v['del_qq'], (1, v['k'])) / v['dtime'])  # WRF:module_mp_cammgmp_driver.F:1940
                        _put(v,'ptend_loc_q',(1,v['k'],v['ixnumliq'],),_get(v['ptend_loc_q'], (1, v['k'], v['ixcldliq'])) * _get(v['state1_q'], (1, v['k'], v['ixnumliq'])) / xp.maximum(_d(1e-30), _get(v['state1_q'], (1, v['k'], v['ixcldliq']))))  # WRF:module_mp_cammgmp_driver.F:1941
                        _put(v,'ptend_loc_q',(1,v['k'],v['ixnumliq'],),xp.minimum(_d(0.0), xp.maximum(_get(v['ptend_loc_q'], (1, v['k'], v['ixnumliq'])), _neg(_get(v['state1_q'], (1, v['k'], v['ixnumliq']))) / v['dtime'])))  # WRF:module_mp_cammgmp_driver.F:1942
                        return tuple(v[_n] for _n in ('_fatal', 'del_qq', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'tsp'))
                    def _no20(_carry):
                        v=dict(_outer20);v.update(zip(('_fatal', 'del_qq', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'tsp'),_carry))
                        _outer21=dict(v)
                        def _yes21(_carry):
                            v=dict(_outer21);v.update(zip(('_fatal', 'del_qq', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'tsp'),_carry))
                            _call=findsp_water(v['lchnk'],v['pcols'],_get(v['state1_q'], (1, v['k'], 1)),_get(v['state1_t'], (1, v['k'])),_get(v['state_pmid'], (1, v['k'])),v['tsp'],v['qsll'],_globals=_globals,xp=xp)
                            _put(v,'tsp',None,_call[0])  # WRF:module_mp_cammgmp_driver.F:1944
                            _put(v,'qsll',None,_call[1])  # WRF:module_mp_cammgmp_driver.F:1944
                            v['_fatal']=v['_fatal']+_call[2]
                            _put(v,'del_qq',(1,v['k'],),xp.maximum(xp.minimum(_get(v['qc_nc'], (1, v['k'])), _get(v['qsll'], (1,)) - _get(v['state1_q'], (1, v['k'], 1))), _d(0.0)) * v['factcc'])  # WRF:module_mp_cammgmp_driver.F:1945
                            _put(v,'ptend_loc_q',(1,v['k'],v['ixcldliq'],),_neg(_get(v['del_qq'], (1, v['k']))) / v['dtime'])  # WRF:module_mp_cammgmp_driver.F:1946
                            _put(v,'ptend_loc_s',(1,v['k'],),_neg(_get(v['del_qq'], (1, v['k']))) * v['latvap'] / v['dtime'])  # WRF:module_mp_cammgmp_driver.F:1947
                            _put(v,'ptend_loc_q',(1,v['k'],1,),_get(v['del_qq'], (1, v['k'])) / v['dtime'])  # WRF:module_mp_cammgmp_driver.F:1948
                            _put(v,'ptend_loc_q',(1,v['k'],v['ixnumliq'],),_get(v['ptend_loc_q'], (1, v['k'], v['ixcldliq'])) * _get(v['state1_q'], (1, v['k'], v['ixnumliq'])) / xp.maximum(_d(1e-30), _get(v['state1_q'], (1, v['k'], v['ixcldliq']))))  # WRF:module_mp_cammgmp_driver.F:1949
                            _put(v,'ptend_loc_q',(1,v['k'],v['ixnumliq'],),xp.minimum(_d(0.0), xp.maximum(_get(v['ptend_loc_q'], (1, v['k'], v['ixnumliq'])), _neg(_get(v['state1_q'], (1, v['k'], v['ixnumliq']))) / v['dtime'])))  # WRF:module_mp_cammgmp_driver.F:1951
                            return tuple(v[_n] for _n in ('_fatal', 'del_qq', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'tsp'))
                        def _no21(_carry):
                            v=dict(_outer21);v.update(zip(('_fatal', 'del_qq', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'tsp'),_carry))
                            _put(v,'ptend_loc_q',(1,v['k'],v['ixcldliq'],),_d(0.0))  # WRF:module_mp_cammgmp_driver.F:1953
                            _put(v,'ptend_loc_s',(1,v['k'],),_d(0.0))  # WRF:module_mp_cammgmp_driver.F:1954
                            _put(v,'ptend_loc_q',(1,v['k'],1,),_d(0.0))  # WRF:module_mp_cammgmp_driver.F:1955
                            _put(v,'ptend_loc_q',(1,v['k'],v['ixnumliq'],),_d(0.0))  # WRF:module_mp_cammgmp_driver.F:1956
                            return tuple(v[_n] for _n in ('_fatal', 'del_qq', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'tsp'))
                        _result=_cond(xp.logical_and(xp.logical_and(_get(v['rh_curr'], (1, v['k'])) < v['cldcr'], _get(v['cldo'], (1, v['k'])) >= _d(0.01)), _get(v['qc_nc'], (1, v['k'])) > _d(1e-12)),_yes21,_no21,tuple(v[_n] for _n in ('_fatal', 'del_qq', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'tsp')),xp)  # WRF:module_mp_cammgmp_driver.F:1943
                        v.update(zip(('_fatal', 'del_qq', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'tsp'),_result))
                        return tuple(v[_n] for _n in ('_fatal', 'del_qq', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'tsp'))
                    _result=_cond(xp.logical_and(xp.logical_and(_get(v['rh_curr'], (1, v['k'])) < v['cldcr'], _get(v['cldo'], (1, v['k'])) < _d(0.01)), _get(v['qc_nc'], (1, v['k'])) > _d(1e-12)),_yes20,_no20,tuple(v[_n] for _n in ('_fatal', 'del_qq', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'tsp')),xp)  # WRF:module_mp_cammgmp_driver.F:1935
                    v.update(zip(('_fatal', 'del_qq', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'tsp'),_result))
                    return tuple(v[_n] for _n in ('_fatal', 'del_cf', 'del_qq', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'tsp'))
                _result=_cond(xp.logical_and(_get(v['rh_curr'], (1, v['k'])) > v['cldcr'], _get(v['rh_curr'], (1, v['k'])) <= _d(1.0)),_yes16,_no16,tuple(v[_n] for _n in ('_fatal', 'del_cf', 'del_qq', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'tsp')),xp)  # WRF:module_mp_cammgmp_driver.F:1907
                v.update(zip(('_fatal', 'del_cf', 'del_qq', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'tsp'),_result))
                return tuple(v[_n] for _n in ('_fatal', 'del_cf', 'del_qq', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'tsp'))
            _result=_cond(_get(v['rh_curr'], (1, v['k'])) > _d(1.0),_yes15,_no15,tuple(v[_n] for _n in ('_fatal', 'del_cf', 'del_qq', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'tsp')),xp)  # WRF:module_mp_cammgmp_driver.F:1896
            v.update(zip(('_fatal', 'del_cf', 'del_qq', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'tsp'),_result))
            _put(v,'state1_t',(1,v['k'],),_get(v['state1_t'], (1, v['k'])) + _get(v['ptend_loc_s'], (1, v['k'])) * v['dtime'] / v['cpair'])  # WRF:module_mp_cammgmp_driver.F:1958
            return tuple(v[_n] for _n in ('_fatal', '_jump', 'aist', 'alst', 'ast', 'cldcr', 'cldo', 'cv_frac', 'del_cf', 'del_qq', 'esl', 'k', 'ptend_loc_q', 'ptend_loc_s', 'qc_nc', 'qsll', 'qvs', 'rh_curr', 'rh_old', 'rh_temp', 'st_frac', 'state1_t', 'tsp'))
        _result=_loop(1,v['pver'],1,_body10,tuple(v[_n] for _n in ('_fatal', '_jump', 'aist', 'alst', 'ast', 'cldcr', 'cldo', 'cv_frac', 'del_cf', 'del_qq', 'esl', 'k', 'ptend_loc_q', 'ptend_loc_s', 'qc_nc', 'qsll', 'qvs', 'rh_curr', 'rh_old', 'rh_temp', 'st_frac', 'state1_t', 'tsp')),xp,jump_index=1)
        v.update(zip(('_fatal', '_jump', 'aist', 'alst', 'ast', 'cldcr', 'cldo', 'cv_frac', 'del_cf', 'del_qq', 'esl', 'k', 'ptend_loc_q', 'ptend_loc_s', 'qc_nc', 'qsll', 'qvs', 'rh_curr', 'rh_old', 'rh_temp', 'st_frac', 'state1_t', 'tsp'),_result))
        return tuple(v[_n] for _n in ('_fatal', 'aist', 'alst', 'ast', 'cldcr', 'cldo', 'cv_frac', 'del_cf', 'del_qq', 'esl', 'k', 'ptend_loc_q', 'ptend_loc_s', 'qc_nc', 'qsll', 'qvs', 'rh_curr', 'rh_old', 'rh_temp', 'st_frac', 'state1_t', 'tsp'))
    def _no9(_carry):
        v=dict(_outer9);v.update(zip(('_fatal', 'aist', 'alst', 'ast', 'cldcr', 'cldo', 'cv_frac', 'del_cf', 'del_qq', 'esl', 'k', 'ptend_loc_q', 'ptend_loc_s', 'qc_nc', 'qsll', 'qvs', 'rh_curr', 'rh_old', 'rh_temp', 'st_frac', 'state1_t', 'tsp'),_carry))
        return tuple(v[_n] for _n in ('_fatal', 'aist', 'alst', 'ast', 'cldcr', 'cldo', 'cv_frac', 'del_cf', 'del_qq', 'esl', 'k', 'ptend_loc_q', 'ptend_loc_s', 'qc_nc', 'qsll', 'qvs', 'rh_curr', 'rh_old', 'rh_temp', 'st_frac', 'state1_t', 'tsp'))
    _result=_cond(v['_jump'] == 0,_yes9,_no9,tuple(v[_n] for _n in ('_fatal', 'aist', 'alst', 'ast', 'cldcr', 'cldo', 'cv_frac', 'del_cf', 'del_qq', 'esl', 'k', 'ptend_loc_q', 'ptend_loc_s', 'qc_nc', 'qsll', 'qvs', 'rh_curr', 'rh_old', 'rh_temp', 'st_frac', 'state1_t', 'tsp')),xp)  # WRF:module_mp_cammgmp_driver.F:1857
    v.update(zip(('_fatal', 'aist', 'alst', 'ast', 'cldcr', 'cldo', 'cv_frac', 'del_cf', 'del_qq', 'esl', 'k', 'ptend_loc_q', 'ptend_loc_s', 'qc_nc', 'qsll', 'qvs', 'rh_curr', 'rh_old', 'rh_temp', 'st_frac', 'state1_t', 'tsp'),_result))
    _outer22=dict(v)
    def _yes22(_carry):
        v=dict(_outer22);v.update(zip(('ptend_all_lq', 'ptend_all_ls', 'ptend_all_q', 'ptend_all_s'),_carry))
        _call=physics_ptend_sum(v['ptend_loc_ls'],v['ptend_loc_lq'],v['ptend_loc_s'],v['ptend_loc_q'],v['ptend_all_ls'],v['ptend_all_lq'],v['ptend_all_s'],v['ptend_all_q'],v['pcnst'],_globals=_globals,xp=xp)
        _put(v,'ptend_all_ls',None,_call[0])  # WRF:module_mp_cammgmp_driver.F:1965
        _put(v,'ptend_all_lq',None,_call[1])  # WRF:module_mp_cammgmp_driver.F:1965
        _put(v,'ptend_all_s',None,_call[2])  # WRF:module_mp_cammgmp_driver.F:1965
        _put(v,'ptend_all_q',None,_call[3])  # WRF:module_mp_cammgmp_driver.F:1965
        return tuple(v[_n] for _n in ('ptend_all_lq', 'ptend_all_ls', 'ptend_all_q', 'ptend_all_s'))
    def _no22(_carry):
        v=dict(_outer22);v.update(zip(('ptend_all_lq', 'ptend_all_ls', 'ptend_all_q', 'ptend_all_s'),_carry))
        return tuple(v[_n] for _n in ('ptend_all_lq', 'ptend_all_ls', 'ptend_all_q', 'ptend_all_s'))
    _result=_cond(v['_jump'] == 0,_yes22,_no22,tuple(v[_n] for _n in ('ptend_all_lq', 'ptend_all_ls', 'ptend_all_q', 'ptend_all_s')),xp)  # WRF:module_mp_cammgmp_driver.F:1965
    v.update(zip(('ptend_all_lq', 'ptend_all_ls', 'ptend_all_q', 'ptend_all_s'),_result))
    _outer24=dict(v)
    def _yes24(_carry):
        v=dict(_outer24);v.update(zip(('ptend_loc_lq', 'ptend_loc_ls', 'ptend_loc_q', 'ptend_loc_s', 'state1_q', 'state1_s'),_carry))
        _call=physics_update(v['lchnk'],v['dtime'],v['state1_q'],v['ptend_loc_q'],v['state1_s'],v['ptend_loc_s'],v['ptend_loc_name'],v['ptend_loc_lq'],v['ptend_loc_ls'],v['pcnst'],_globals=_globals,xp=xp)
        _put(v,'state1_q',None,_call[0])  # WRF:module_mp_cammgmp_driver.F:1967
        _put(v,'ptend_loc_q',None,_call[1])  # WRF:module_mp_cammgmp_driver.F:1967
        _put(v,'state1_s',None,_call[2])  # WRF:module_mp_cammgmp_driver.F:1967
        _put(v,'ptend_loc_s',None,_call[3])  # WRF:module_mp_cammgmp_driver.F:1967
        _put(v,'ptend_loc_name',None,_call[4])  # WRF:module_mp_cammgmp_driver.F:1967
        _put(v,'ptend_loc_lq',None,_call[5])  # WRF:module_mp_cammgmp_driver.F:1967
        _put(v,'ptend_loc_ls',None,_call[6])  # WRF:module_mp_cammgmp_driver.F:1967
        return tuple(v[_n] for _n in ('ptend_loc_lq', 'ptend_loc_ls', 'ptend_loc_q', 'ptend_loc_s', 'state1_q', 'state1_s'))
    def _no24(_carry):
        v=dict(_outer24);v.update(zip(('ptend_loc_lq', 'ptend_loc_ls', 'ptend_loc_q', 'ptend_loc_s', 'state1_q', 'state1_s'),_carry))
        return tuple(v[_n] for _n in ('ptend_loc_lq', 'ptend_loc_ls', 'ptend_loc_q', 'ptend_loc_s', 'state1_q', 'state1_s'))
    _result=_cond(v['_jump'] == 0,_yes24,_no24,tuple(v[_n] for _n in ('ptend_loc_lq', 'ptend_loc_ls', 'ptend_loc_q', 'ptend_loc_s', 'state1_q', 'state1_s')),xp)  # WRF:module_mp_cammgmp_driver.F:1967
    v.update(zip(('ptend_loc_lq', 'ptend_loc_ls', 'ptend_loc_q', 'ptend_loc_s', 'state1_q', 'state1_s'),_result))
    _outer25=dict(v)
    def _yes25(_carry):
        v=dict(_outer25);v.update(zip(('ptend_loc_lq', 'ptend_loc_ls', 'ptend_loc_q', 'ptend_loc_s'),_carry))
        _call=physics_ptend_init(v['ptend_loc_name'],v['ptend_loc_q'],v['ptend_loc_s'],v['ptend_loc_lq'],v['ptend_loc_ls'],v['pcnst'],_globals=_globals,xp=xp)
        _put(v,'ptend_loc_name',None,_call[0])  # WRF:module_mp_cammgmp_driver.F:1968
        _put(v,'ptend_loc_q',None,_call[1])  # WRF:module_mp_cammgmp_driver.F:1968
        _put(v,'ptend_loc_s',None,_call[2])  # WRF:module_mp_cammgmp_driver.F:1968
        _put(v,'ptend_loc_lq',None,_call[3])  # WRF:module_mp_cammgmp_driver.F:1968
        _put(v,'ptend_loc_ls',None,_call[4])  # WRF:module_mp_cammgmp_driver.F:1968
        return tuple(v[_n] for _n in ('ptend_loc_lq', 'ptend_loc_ls', 'ptend_loc_q', 'ptend_loc_s'))
    def _no25(_carry):
        v=dict(_outer25);v.update(zip(('ptend_loc_lq', 'ptend_loc_ls', 'ptend_loc_q', 'ptend_loc_s'),_carry))
        return tuple(v[_n] for _n in ('ptend_loc_lq', 'ptend_loc_ls', 'ptend_loc_q', 'ptend_loc_s'))
    _result=_cond(v['_jump'] == 0,_yes25,_no25,tuple(v[_n] for _n in ('ptend_loc_lq', 'ptend_loc_ls', 'ptend_loc_q', 'ptend_loc_s')),xp)  # WRF:module_mp_cammgmp_driver.F:1968
    v.update(zip(('ptend_loc_lq', 'ptend_loc_ls', 'ptend_loc_q', 'ptend_loc_s'),_result))
    _outer27=dict(v)
    def _yes27(_carry):
        v=dict(_outer27);v.update(zip(('ptend_loc_ls',),_carry))
        _put(v,'ptend_loc_ls',None,True)  # WRF:module_mp_cammgmp_driver.F:1974
        return tuple(v[_n] for _n in ('ptend_loc_ls',))
    def _no27(_carry):
        v=dict(_outer27);v.update(zip(('ptend_loc_ls',),_carry))
        return tuple(v[_n] for _n in ('ptend_loc_ls',))
    _result=_cond(v['_jump'] == 0,_yes27,_no27,tuple(v[_n] for _n in ('ptend_loc_ls',)),xp)  # WRF:module_mp_cammgmp_driver.F:1974
    v.update(zip(('ptend_loc_ls',),_result))
    _outer28=dict(v)
    def _yes28(_carry):
        v=dict(_outer28);v.update(zip(('ptend_loc_lq',),_carry))
        _put(v,'ptend_loc_lq',(1,),True)  # WRF:module_mp_cammgmp_driver.F:1975
        return tuple(v[_n] for _n in ('ptend_loc_lq',))
    def _no28(_carry):
        v=dict(_outer28);v.update(zip(('ptend_loc_lq',),_carry))
        return tuple(v[_n] for _n in ('ptend_loc_lq',))
    _result=_cond(v['_jump'] == 0,_yes28,_no28,tuple(v[_n] for _n in ('ptend_loc_lq',)),xp)  # WRF:module_mp_cammgmp_driver.F:1975
    v.update(zip(('ptend_loc_lq',),_result))
    _outer29=dict(v)
    def _yes29(_carry):
        v=dict(_outer29);v.update(zip(('ptend_loc_lq',),_carry))
        _put(v,'ptend_loc_lq',(v['ixcldliq'],),True)  # WRF:module_mp_cammgmp_driver.F:1976
        return tuple(v[_n] for _n in ('ptend_loc_lq',))
    def _no29(_carry):
        v=dict(_outer29);v.update(zip(('ptend_loc_lq',),_carry))
        return tuple(v[_n] for _n in ('ptend_loc_lq',))
    _result=_cond(v['_jump'] == 0,_yes29,_no29,tuple(v[_n] for _n in ('ptend_loc_lq',)),xp)  # WRF:module_mp_cammgmp_driver.F:1976
    v.update(zip(('ptend_loc_lq',),_result))
    _outer30=dict(v)
    def _yes30(_carry):
        v=dict(_outer30);v.update(zip(('ptend_loc_lq',),_carry))
        _put(v,'ptend_loc_lq',(v['ixnumliq'],),True)  # WRF:module_mp_cammgmp_driver.F:1977
        return tuple(v[_n] for _n in ('ptend_loc_lq',))
    def _no30(_carry):
        v=dict(_outer30);v.update(zip(('ptend_loc_lq',),_carry))
        return tuple(v[_n] for _n in ('ptend_loc_lq',))
    _result=_cond(v['_jump'] == 0,_yes30,_no30,tuple(v[_n] for _n in ('ptend_loc_lq',)),xp)  # WRF:module_mp_cammgmp_driver.F:1977
    v.update(zip(('ptend_loc_lq',),_result))
    _outer31=dict(v)
    def _yes31(_carry):
        v=dict(_outer31);v.update(zip(('aist', 'alst', 'ast', 'del_qq', 'k', 'ptend_loc_q', 'ptend_loc_s', 'qc_nc', 'st_frac', 'state1_t'),_carry))
        _outer32=dict(v)
        def _body32(_k,_carry):
            v=dict(_outer32);v.update(zip(('_jump', 'aist', 'alst', 'ast', 'del_qq', 'k', 'ptend_loc_q', 'ptend_loc_s', 'qc_nc', 'st_frac', 'state1_t'),_carry));v['k']=_int(_k)
            _put(v,'del_qq',(1,v['k'],),_d(0.0))  # WRF:module_mp_cammgmp_driver.F:1984
            _call=cal_cldfra_mp_1d(_get(v['alst'], (1, v['k'])),_get(v['aist'], (1, v['k'])),_get(v['state1_q'], (1, v['k'], 1)),_get(v['state1_q'], (1, v['k'], v['ixcldliq'])),_get(v['qi_mac'], (1, v['k'])),_get(v['state1_t'], (1, v['k'])),_get(v['state_pmid'], (1, v['k'])),v['xland_pt'],v['snowh_pt'],_globals=_globals,xp=xp)
            _put(v,'alst',(1,v['k'],),_call[0])  # WRF:module_mp_cammgmp_driver.F:1985
            _put(v,'aist',(1,v['k'],),_call[1])  # WRF:module_mp_cammgmp_driver.F:1985
            _put(v,'alst',(1,v['k'],),(_d(1.0) - _get(v['cv_frac'], (1, v['k']))) * _get(v['alst'], (1, v['k'])))  # WRF:module_mp_cammgmp_driver.F:1986
            _put(v,'aist',(1,v['k'],),(_d(1.0) - _get(v['cv_frac'], (1, v['k']))) * _get(v['aist'], (1, v['k'])))  # WRF:module_mp_cammgmp_driver.F:1987
            _put(v,'ast',(1,v['k'],),xp.maximum(_get(v['alst'], (1, v['k'])), _get(v['aist'], (1, v['k']))))  # WRF:module_mp_cammgmp_driver.F:1988
            _put(v,'st_frac',(1,v['k'],),_get(v['alst'], (1, v['k'])))  # WRF:module_mp_cammgmp_driver.F:1989
            _put(v,'qc_nc',(1,v['k'],),xp.maximum(_d(0.0), _get(v['state1_q'], (1, v['k'], v['ixcldliq'])) - xp.maximum(_d(0.0), _get(v['dp_frac'], (1, v['k'])) * _get(v['dp_icwmr'], (1, v['k'])) + _get(v['sh_frac'], (1, v['k'])) * _get(v['sh_icwmr'], (1, v['k'])))))  # WRF:module_mp_cammgmp_driver.F:1991
            _outer33=dict(v)
            def _yes33(_carry):
                v=dict(_outer33);v.update(zip(('del_qq', 'ptend_loc_q', 'ptend_loc_s'),_carry))
                _outer34=dict(v)
                def _yes34(_carry):
                    v=dict(_outer34);v.update(zip(('del_qq', 'ptend_loc_q', 'ptend_loc_s'),_carry))
                    _put(v,'del_qq',(1,v['k'],),_get(v['qc_nc'], (1, v['k'])) - _d(0.003) * _get(v['st_frac'], (1, v['k'])) * v['fact_adj'])  # WRF:module_mp_cammgmp_driver.F:2013
                    _put(v,'ptend_loc_q',(1,v['k'],v['ixcldliq'],),_neg(_get(v['del_qq'], (1, v['k']))) / v['dtime'])  # WRF:module_mp_cammgmp_driver.F:2014
                    _put(v,'ptend_loc_q',(1,v['k'],1,),_get(v['del_qq'], (1, v['k'])) / v['dtime'])  # WRF:module_mp_cammgmp_driver.F:2015
                    _put(v,'ptend_loc_s',(1,v['k'],),_neg(_get(v['del_qq'], (1, v['k']))) * v['latvap'] / v['dtime'])  # WRF:module_mp_cammgmp_driver.F:2016
                    _put(v,'ptend_loc_q',(1,v['k'],v['ixnumliq'],),_get(v['ptend_loc_q'], (1, v['k'], v['ixcldliq'])) * _get(v['state1_q'], (1, v['k'], v['ixnumliq'])) / xp.maximum(_d(1e-20), _get(v['state1_q'], (1, v['k'], v['ixcldliq']))))  # WRF:module_mp_cammgmp_driver.F:2017
                    _put(v,'ptend_loc_q',(1,v['k'],v['ixnumliq'],),xp.minimum(_d(0.0), xp.maximum(_get(v['ptend_loc_q'], (1, v['k'], v['ixnumliq'])), _neg(_get(v['state1_q'], (1, v['k'], v['ixnumliq']))) / v['dtime'])))  # WRF:module_mp_cammgmp_driver.F:2018
                    return tuple(v[_n] for _n in ('del_qq', 'ptend_loc_q', 'ptend_loc_s'))
                def _no34(_carry):
                    v=dict(_outer34);v.update(zip(('del_qq', 'ptend_loc_q', 'ptend_loc_s'),_carry))
                    _outer35=dict(v)
                    def _yes35(_carry):
                        v=dict(_outer35);v.update(zip(('del_qq', 'ptend_loc_q', 'ptend_loc_s'),_carry))
                        _put(v,'del_qq',(1,v['k'],),_d(2e-05) * _get(v['st_frac'], (1, v['k'])) - _get(v['qc_nc'], (1, v['k'])))  # WRF:module_mp_cammgmp_driver.F:2024
                        _put(v,'del_qq',(1,v['k'],),xp.maximum(_d(0.0), xp.minimum(_get(v['del_qq'], (1, v['k'])), _get(v['state1_q'], (1, v['k'], 1)) * _d(0.1))) * v['fact_adj'])  # WRF:module_mp_cammgmp_driver.F:2025
                        _put(v,'ptend_loc_q',(1,v['k'],v['ixcldliq'],),_get(v['del_qq'], (1, v['k'])) / v['dtime'])  # WRF:module_mp_cammgmp_driver.F:2026
                        _put(v,'ptend_loc_q',(1,v['k'],1,),_neg(_get(v['del_qq'], (1, v['k']))) / v['dtime'])  # WRF:module_mp_cammgmp_driver.F:2027
                        _put(v,'ptend_loc_s',(1,v['k'],),_get(v['del_qq'], (1, v['k'])) * v['latvap'] / v['dtime'])  # WRF:module_mp_cammgmp_driver.F:2028
                        return tuple(v[_n] for _n in ('del_qq', 'ptend_loc_q', 'ptend_loc_s'))
                    def _no35(_carry):
                        v=dict(_outer35);v.update(zip(('del_qq', 'ptend_loc_q', 'ptend_loc_s'),_carry))
                        return tuple(v[_n] for _n in ('del_qq', 'ptend_loc_q', 'ptend_loc_s'))
                    _result=_cond(_get(v['qc_nc'], (1, v['k'])) < _d(2e-05) * _get(v['st_frac'], (1, v['k'])),_yes35,_no35,tuple(v[_n] for _n in ('del_qq', 'ptend_loc_q', 'ptend_loc_s')),xp)  # WRF:module_mp_cammgmp_driver.F:2023
                    v.update(zip(('del_qq', 'ptend_loc_q', 'ptend_loc_s'),_result))
                    return tuple(v[_n] for _n in ('del_qq', 'ptend_loc_q', 'ptend_loc_s'))
                _result=_cond(_get(v['qc_nc'], (1, v['k'])) > _d(0.003) * _get(v['st_frac'], (1, v['k'])),_yes34,_no34,tuple(v[_n] for _n in ('del_qq', 'ptend_loc_q', 'ptend_loc_s')),xp)  # WRF:module_mp_cammgmp_driver.F:2012
                v.update(zip(('del_qq', 'ptend_loc_q', 'ptend_loc_s'),_result))
                return tuple(v[_n] for _n in ('del_qq', 'ptend_loc_q', 'ptend_loc_s'))
            def _no33(_carry):
                v=dict(_outer33);v.update(zip(('del_qq', 'ptend_loc_q', 'ptend_loc_s'),_carry))
                return tuple(v[_n] for _n in ('del_qq', 'ptend_loc_q', 'ptend_loc_s'))
            _result=_cond(_get(v['st_frac'], (1, v['k'])) > _d(1e-05),_yes33,_no33,tuple(v[_n] for _n in ('del_qq', 'ptend_loc_q', 'ptend_loc_s')),xp)  # WRF:module_mp_cammgmp_driver.F:2007
            v.update(zip(('del_qq', 'ptend_loc_q', 'ptend_loc_s'),_result))
            _put(v,'state1_t',(1,v['k'],),_get(v['state1_t'], (1, v['k'])) + _get(v['ptend_loc_s'], (1, v['k'])) * v['dtime'] / v['cpair'])  # WRF:module_mp_cammgmp_driver.F:2031
            return tuple(v[_n] for _n in ('_jump', 'aist', 'alst', 'ast', 'del_qq', 'k', 'ptend_loc_q', 'ptend_loc_s', 'qc_nc', 'st_frac', 'state1_t'))
        _result=_loop(1,v['pver'],1,_body32,tuple(v[_n] for _n in ('_jump', 'aist', 'alst', 'ast', 'del_qq', 'k', 'ptend_loc_q', 'ptend_loc_s', 'qc_nc', 'st_frac', 'state1_t')),xp,jump_index=0)
        v.update(zip(('_jump', 'aist', 'alst', 'ast', 'del_qq', 'k', 'ptend_loc_q', 'ptend_loc_s', 'qc_nc', 'st_frac', 'state1_t'),_result))
        return tuple(v[_n] for _n in ('aist', 'alst', 'ast', 'del_qq', 'k', 'ptend_loc_q', 'ptend_loc_s', 'qc_nc', 'st_frac', 'state1_t'))
    def _no31(_carry):
        v=dict(_outer31);v.update(zip(('aist', 'alst', 'ast', 'del_qq', 'k', 'ptend_loc_q', 'ptend_loc_s', 'qc_nc', 'st_frac', 'state1_t'),_carry))
        return tuple(v[_n] for _n in ('aist', 'alst', 'ast', 'del_qq', 'k', 'ptend_loc_q', 'ptend_loc_s', 'qc_nc', 'st_frac', 'state1_t'))
    _result=_cond(v['_jump'] == 0,_yes31,_no31,tuple(v[_n] for _n in ('aist', 'alst', 'ast', 'del_qq', 'k', 'ptend_loc_q', 'ptend_loc_s', 'qc_nc', 'st_frac', 'state1_t')),xp)  # WRF:module_mp_cammgmp_driver.F:1983
    v.update(zip(('aist', 'alst', 'ast', 'del_qq', 'k', 'ptend_loc_q', 'ptend_loc_s', 'qc_nc', 'st_frac', 'state1_t'),_result))
    _outer36=dict(v)
    def _yes36(_carry):
        v=dict(_outer36);v.update(zip(('ptend_all_lq', 'ptend_all_ls', 'ptend_all_q', 'ptend_all_s'),_carry))
        _call=physics_ptend_sum(v['ptend_loc_ls'],v['ptend_loc_lq'],v['ptend_loc_s'],v['ptend_loc_q'],v['ptend_all_ls'],v['ptend_all_lq'],v['ptend_all_s'],v['ptend_all_q'],v['pcnst'],_globals=_globals,xp=xp)
        _put(v,'ptend_all_ls',None,_call[0])  # WRF:module_mp_cammgmp_driver.F:2036
        _put(v,'ptend_all_lq',None,_call[1])  # WRF:module_mp_cammgmp_driver.F:2036
        _put(v,'ptend_all_s',None,_call[2])  # WRF:module_mp_cammgmp_driver.F:2036
        _put(v,'ptend_all_q',None,_call[3])  # WRF:module_mp_cammgmp_driver.F:2036
        return tuple(v[_n] for _n in ('ptend_all_lq', 'ptend_all_ls', 'ptend_all_q', 'ptend_all_s'))
    def _no36(_carry):
        v=dict(_outer36);v.update(zip(('ptend_all_lq', 'ptend_all_ls', 'ptend_all_q', 'ptend_all_s'),_carry))
        return tuple(v[_n] for _n in ('ptend_all_lq', 'ptend_all_ls', 'ptend_all_q', 'ptend_all_s'))
    _result=_cond(v['_jump'] == 0,_yes36,_no36,tuple(v[_n] for _n in ('ptend_all_lq', 'ptend_all_ls', 'ptend_all_q', 'ptend_all_s')),xp)  # WRF:module_mp_cammgmp_driver.F:2036
    v.update(zip(('ptend_all_lq', 'ptend_all_ls', 'ptend_all_q', 'ptend_all_s'),_result))
    _outer38=dict(v)
    def _yes38(_carry):
        v=dict(_outer38);v.update(zip(('ptend_loc_lq', 'ptend_loc_ls', 'ptend_loc_q', 'ptend_loc_s', 'state1_q', 'state1_s'),_carry))
        _call=physics_update(v['lchnk'],v['dtime'],v['state1_q'],v['ptend_loc_q'],v['state1_s'],v['ptend_loc_s'],v['ptend_loc_name'],v['ptend_loc_lq'],v['ptend_loc_ls'],v['pcnst'],_globals=_globals,xp=xp)
        _put(v,'state1_q',None,_call[0])  # WRF:module_mp_cammgmp_driver.F:2038
        _put(v,'ptend_loc_q',None,_call[1])  # WRF:module_mp_cammgmp_driver.F:2038
        _put(v,'state1_s',None,_call[2])  # WRF:module_mp_cammgmp_driver.F:2038
        _put(v,'ptend_loc_s',None,_call[3])  # WRF:module_mp_cammgmp_driver.F:2038
        _put(v,'ptend_loc_name',None,_call[4])  # WRF:module_mp_cammgmp_driver.F:2038
        _put(v,'ptend_loc_lq',None,_call[5])  # WRF:module_mp_cammgmp_driver.F:2038
        _put(v,'ptend_loc_ls',None,_call[6])  # WRF:module_mp_cammgmp_driver.F:2038
        return tuple(v[_n] for _n in ('ptend_loc_lq', 'ptend_loc_ls', 'ptend_loc_q', 'ptend_loc_s', 'state1_q', 'state1_s'))
    def _no38(_carry):
        v=dict(_outer38);v.update(zip(('ptend_loc_lq', 'ptend_loc_ls', 'ptend_loc_q', 'ptend_loc_s', 'state1_q', 'state1_s'),_carry))
        return tuple(v[_n] for _n in ('ptend_loc_lq', 'ptend_loc_ls', 'ptend_loc_q', 'ptend_loc_s', 'state1_q', 'state1_s'))
    _result=_cond(v['_jump'] == 0,_yes38,_no38,tuple(v[_n] for _n in ('ptend_loc_lq', 'ptend_loc_ls', 'ptend_loc_q', 'ptend_loc_s', 'state1_q', 'state1_s')),xp)  # WRF:module_mp_cammgmp_driver.F:2038
    v.update(zip(('ptend_loc_lq', 'ptend_loc_ls', 'ptend_loc_q', 'ptend_loc_s', 'state1_q', 'state1_s'),_result))
    _outer39=dict(v)
    def _yes39(_carry):
        v=dict(_outer39);v.update(zip(('ptend_loc_lq', 'ptend_loc_ls', 'ptend_loc_q', 'ptend_loc_s'),_carry))
        _call=physics_ptend_init(v['ptend_loc_name'],v['ptend_loc_q'],v['ptend_loc_s'],v['ptend_loc_lq'],v['ptend_loc_ls'],v['pcnst'],_globals=_globals,xp=xp)
        _put(v,'ptend_loc_name',None,_call[0])  # WRF:module_mp_cammgmp_driver.F:2039
        _put(v,'ptend_loc_q',None,_call[1])  # WRF:module_mp_cammgmp_driver.F:2039
        _put(v,'ptend_loc_s',None,_call[2])  # WRF:module_mp_cammgmp_driver.F:2039
        _put(v,'ptend_loc_lq',None,_call[3])  # WRF:module_mp_cammgmp_driver.F:2039
        _put(v,'ptend_loc_ls',None,_call[4])  # WRF:module_mp_cammgmp_driver.F:2039
        return tuple(v[_n] for _n in ('ptend_loc_lq', 'ptend_loc_ls', 'ptend_loc_q', 'ptend_loc_s'))
    def _no39(_carry):
        v=dict(_outer39);v.update(zip(('ptend_loc_lq', 'ptend_loc_ls', 'ptend_loc_q', 'ptend_loc_s'),_carry))
        return tuple(v[_n] for _n in ('ptend_loc_lq', 'ptend_loc_ls', 'ptend_loc_q', 'ptend_loc_s'))
    _result=_cond(v['_jump'] == 0,_yes39,_no39,tuple(v[_n] for _n in ('ptend_loc_lq', 'ptend_loc_ls', 'ptend_loc_q', 'ptend_loc_s')),xp)  # WRF:module_mp_cammgmp_driver.F:2039
    v.update(zip(('ptend_loc_lq', 'ptend_loc_ls', 'ptend_loc_q', 'ptend_loc_s'),_result))
    _outer40=dict(v)
    def _yes40(_carry):
        v=dict(_outer40);v.update(zip(('_jump',),_carry))
        v['_jump']=_int(2000)  # WRF:module_mp_cammgmp_driver.F:2041
        return tuple(v[_n] for _n in ('_jump',))
    def _no40(_carry):
        v=dict(_outer40);v.update(zip(('_jump',),_carry))
        return tuple(v[_n] for _n in ('_jump',))
    _result=_cond(v['_jump'] == 0,_yes40,_no40,tuple(v[_n] for _n in ('_jump',)),xp)  # WRF:module_mp_cammgmp_driver.F:2041
    v.update(zip(('_jump',),_result))
    _outer42=dict(v)
    def _yes42(_carry):
        v=dict(_outer42);v.update(zip(('ptend_loc_ls',),_carry))
        _put(v,'ptend_loc_ls',None,True)  # WRF:module_mp_cammgmp_driver.F:2047
        return tuple(v[_n] for _n in ('ptend_loc_ls',))
    def _no42(_carry):
        v=dict(_outer42);v.update(zip(('ptend_loc_ls',),_carry))
        return tuple(v[_n] for _n in ('ptend_loc_ls',))
    _result=_cond(v['_jump'] == 0,_yes42,_no42,tuple(v[_n] for _n in ('ptend_loc_ls',)),xp)  # WRF:module_mp_cammgmp_driver.F:2047
    v.update(zip(('ptend_loc_ls',),_result))
    _outer43=dict(v)
    def _yes43(_carry):
        v=dict(_outer43);v.update(zip(('ptend_loc_lq',),_carry))
        _put(v,'ptend_loc_lq',(1,),True)  # WRF:module_mp_cammgmp_driver.F:2048
        return tuple(v[_n] for _n in ('ptend_loc_lq',))
    def _no43(_carry):
        v=dict(_outer43);v.update(zip(('ptend_loc_lq',),_carry))
        return tuple(v[_n] for _n in ('ptend_loc_lq',))
    _result=_cond(v['_jump'] == 0,_yes43,_no43,tuple(v[_n] for _n in ('ptend_loc_lq',)),xp)  # WRF:module_mp_cammgmp_driver.F:2048
    v.update(zip(('ptend_loc_lq',),_result))
    _outer44=dict(v)
    def _yes44(_carry):
        v=dict(_outer44);v.update(zip(('ptend_loc_lq',),_carry))
        _put(v,'ptend_loc_lq',(v['ixcldliq'],),True)  # WRF:module_mp_cammgmp_driver.F:2049
        return tuple(v[_n] for _n in ('ptend_loc_lq',))
    def _no44(_carry):
        v=dict(_outer44);v.update(zip(('ptend_loc_lq',),_carry))
        return tuple(v[_n] for _n in ('ptend_loc_lq',))
    _result=_cond(v['_jump'] == 0,_yes44,_no44,tuple(v[_n] for _n in ('ptend_loc_lq',)),xp)  # WRF:module_mp_cammgmp_driver.F:2049
    v.update(zip(('ptend_loc_lq',),_result))
    _outer45=dict(v)
    def _yes45(_carry):
        v=dict(_outer45);v.update(zip(('_fatal', 'del_qq', 'esl', 'k', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'qvs', 'rh_curr', 'state1_t', 'tsp'),_carry))
        _outer46=dict(v)
        def _body46(_k,_carry):
            v=dict(_outer46);v.update(zip(('_fatal', '_jump', 'del_qq', 'esl', 'k', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'qvs', 'rh_curr', 'state1_t', 'tsp'),_carry));v['k']=_int(_k)
            _put(v,'qsll',None,_d(0.0))  # WRF:module_mp_cammgmp_driver.F:2055
            _put(v,'tsp',None,_d(0.0))  # WRF:module_mp_cammgmp_driver.F:2056
            _put(v,'del_qq',(1,v['k'],),_d(0.0))  # WRF:module_mp_cammgmp_driver.F:2057
            _put(v,'esl',(1,v['k'],),polysvp(_get(v['state1_t'], (1, v['k'])), 0, xp=xp, _globals=_globals))  # WRF:module_mp_cammgmp_driver.F:2058
            _put(v,'qvs',(1,v['k'],),xp.maximum(_d(1e-30), v['epsqs'] * _get(v['esl'], (1, v['k'])) / (_get(v['state_pmid'], (1, v['k'])) - (_d(1.0) - v['epsqs']) * _get(v['esl'], (1, v['k'])))))  # WRF:module_mp_cammgmp_driver.F:2059
            _put(v,'rh_curr',(1,v['k'],),_get(v['state1_q'], (1, v['k'], 1)) / _get(v['qvs'], (1, v['k'])))  # WRF:module_mp_cammgmp_driver.F:2060
            _outer47=dict(v)
            def _yes47(_carry):
                v=dict(_outer47);v.update(zip(('_fatal', 'del_qq', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'tsp'),_carry))
                _call=findsp_water(v['lchnk'],v['pcols'],_get(v['state1_q'], (1, v['k'], 1)),_get(v['state1_t'], (1, v['k'])),_get(v['state_pmid'], (1, v['k'])),v['tsp'],v['qsll'],_globals=_globals,xp=xp)
                _put(v,'tsp',None,_call[0])  # WRF:module_mp_cammgmp_driver.F:2063
                _put(v,'qsll',None,_call[1])  # WRF:module_mp_cammgmp_driver.F:2063
                v['_fatal']=v['_fatal']+_call[2]
                _put(v,'del_qq',(1,v['k'],),xp.minimum(_d(0.02), xp.maximum(_d(0.0), (_get(v['state1_q'], (1, v['k'], 1)) - _get(v['qsll'], (1,))) * _d(0.999))))  # WRF:module_mp_cammgmp_driver.F:2064
                _put(v,'ptend_loc_q',(1,v['k'],v['ixcldliq'],),_get(v['del_qq'], (1, v['k'])) / v['dtime'])  # WRF:module_mp_cammgmp_driver.F:2065
                _put(v,'ptend_loc_s',(1,v['k'],),_get(v['del_qq'], (1, v['k'])) * v['latvap'] / v['dtime'])  # WRF:module_mp_cammgmp_driver.F:2066
                _put(v,'ptend_loc_q',(1,v['k'],1,),_neg(_get(v['del_qq'], (1, v['k']))) / v['dtime'])  # WRF:module_mp_cammgmp_driver.F:2067
                return tuple(v[_n] for _n in ('_fatal', 'del_qq', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'tsp'))
            def _no47(_carry):
                v=dict(_outer47);v.update(zip(('_fatal', 'del_qq', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'tsp'),_carry))
                return tuple(v[_n] for _n in ('_fatal', 'del_qq', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'tsp'))
            _result=_cond(_get(v['rh_curr'], (1, v['k'])) > _d(1.0),_yes47,_no47,tuple(v[_n] for _n in ('_fatal', 'del_qq', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'tsp')),xp)  # WRF:module_mp_cammgmp_driver.F:2062
            v.update(zip(('_fatal', 'del_qq', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'tsp'),_result))
            _put(v,'state1_t',(1,v['k'],),_get(v['state1_t'], (1, v['k'])) + _get(v['ptend_loc_s'], (1, v['k'])) * v['dtime'] / v['cpair'])  # WRF:module_mp_cammgmp_driver.F:2069
            return tuple(v[_n] for _n in ('_fatal', '_jump', 'del_qq', 'esl', 'k', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'qvs', 'rh_curr', 'state1_t', 'tsp'))
        _result=_loop(1,v['pver'],1,_body46,tuple(v[_n] for _n in ('_fatal', '_jump', 'del_qq', 'esl', 'k', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'qvs', 'rh_curr', 'state1_t', 'tsp')),xp,jump_index=1)
        v.update(zip(('_fatal', '_jump', 'del_qq', 'esl', 'k', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'qvs', 'rh_curr', 'state1_t', 'tsp'),_result))
        return tuple(v[_n] for _n in ('_fatal', 'del_qq', 'esl', 'k', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'qvs', 'rh_curr', 'state1_t', 'tsp'))
    def _no45(_carry):
        v=dict(_outer45);v.update(zip(('_fatal', 'del_qq', 'esl', 'k', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'qvs', 'rh_curr', 'state1_t', 'tsp'),_carry))
        return tuple(v[_n] for _n in ('_fatal', 'del_qq', 'esl', 'k', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'qvs', 'rh_curr', 'state1_t', 'tsp'))
    _result=_cond(v['_jump'] == 0,_yes45,_no45,tuple(v[_n] for _n in ('_fatal', 'del_qq', 'esl', 'k', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'qvs', 'rh_curr', 'state1_t', 'tsp')),xp)  # WRF:module_mp_cammgmp_driver.F:2054
    v.update(zip(('_fatal', 'del_qq', 'esl', 'k', 'ptend_loc_q', 'ptend_loc_s', 'qsll', 'qvs', 'rh_curr', 'state1_t', 'tsp'),_result))
    _outer48=dict(v)
    def _yes48(_carry):
        v=dict(_outer48);v.update(zip(('ptend_all_lq', 'ptend_all_ls', 'ptend_all_q', 'ptend_all_s'),_carry))
        _call=physics_ptend_sum(v['ptend_loc_ls'],v['ptend_loc_lq'],v['ptend_loc_s'],v['ptend_loc_q'],v['ptend_all_ls'],v['ptend_all_lq'],v['ptend_all_s'],v['ptend_all_q'],v['pcnst'],_globals=_globals,xp=xp)
        _put(v,'ptend_all_ls',None,_call[0])  # WRF:module_mp_cammgmp_driver.F:2074
        _put(v,'ptend_all_lq',None,_call[1])  # WRF:module_mp_cammgmp_driver.F:2074
        _put(v,'ptend_all_s',None,_call[2])  # WRF:module_mp_cammgmp_driver.F:2074
        _put(v,'ptend_all_q',None,_call[3])  # WRF:module_mp_cammgmp_driver.F:2074
        return tuple(v[_n] for _n in ('ptend_all_lq', 'ptend_all_ls', 'ptend_all_q', 'ptend_all_s'))
    def _no48(_carry):
        v=dict(_outer48);v.update(zip(('ptend_all_lq', 'ptend_all_ls', 'ptend_all_q', 'ptend_all_s'),_carry))
        return tuple(v[_n] for _n in ('ptend_all_lq', 'ptend_all_ls', 'ptend_all_q', 'ptend_all_s'))
    _result=_cond(v['_jump'] == 0,_yes48,_no48,tuple(v[_n] for _n in ('ptend_all_lq', 'ptend_all_ls', 'ptend_all_q', 'ptend_all_s')),xp)  # WRF:module_mp_cammgmp_driver.F:2074
    v.update(zip(('ptend_all_lq', 'ptend_all_ls', 'ptend_all_q', 'ptend_all_s'),_result))
    _outer50=dict(v)
    def _yes50(_carry):
        v=dict(_outer50);v.update(zip(('ptend_loc_lq', 'ptend_loc_ls', 'ptend_loc_q', 'ptend_loc_s', 'state1_q', 'state1_s'),_carry))
        _call=physics_update(v['lchnk'],v['dtime'],v['state1_q'],v['ptend_loc_q'],v['state1_s'],v['ptend_loc_s'],v['ptend_loc_name'],v['ptend_loc_lq'],v['ptend_loc_ls'],v['pcnst'],_globals=_globals,xp=xp)
        _put(v,'state1_q',None,_call[0])  # WRF:module_mp_cammgmp_driver.F:2076
        _put(v,'ptend_loc_q',None,_call[1])  # WRF:module_mp_cammgmp_driver.F:2076
        _put(v,'state1_s',None,_call[2])  # WRF:module_mp_cammgmp_driver.F:2076
        _put(v,'ptend_loc_s',None,_call[3])  # WRF:module_mp_cammgmp_driver.F:2076
        _put(v,'ptend_loc_name',None,_call[4])  # WRF:module_mp_cammgmp_driver.F:2076
        _put(v,'ptend_loc_lq',None,_call[5])  # WRF:module_mp_cammgmp_driver.F:2076
        _put(v,'ptend_loc_ls',None,_call[6])  # WRF:module_mp_cammgmp_driver.F:2076
        return tuple(v[_n] for _n in ('ptend_loc_lq', 'ptend_loc_ls', 'ptend_loc_q', 'ptend_loc_s', 'state1_q', 'state1_s'))
    def _no50(_carry):
        v=dict(_outer50);v.update(zip(('ptend_loc_lq', 'ptend_loc_ls', 'ptend_loc_q', 'ptend_loc_s', 'state1_q', 'state1_s'),_carry))
        return tuple(v[_n] for _n in ('ptend_loc_lq', 'ptend_loc_ls', 'ptend_loc_q', 'ptend_loc_s', 'state1_q', 'state1_s'))
    _result=_cond(v['_jump'] == 0,_yes50,_no50,tuple(v[_n] for _n in ('ptend_loc_lq', 'ptend_loc_ls', 'ptend_loc_q', 'ptend_loc_s', 'state1_q', 'state1_s')),xp)  # WRF:module_mp_cammgmp_driver.F:2076
    v.update(zip(('ptend_loc_lq', 'ptend_loc_ls', 'ptend_loc_q', 'ptend_loc_s', 'state1_q', 'state1_s'),_result))
    _outer51=dict(v)
    def _yes51(_carry):
        v=dict(_outer51);v.update(zip(('ptend_loc_lq', 'ptend_loc_ls', 'ptend_loc_q', 'ptend_loc_s'),_carry))
        _call=physics_ptend_init(v['ptend_loc_name'],v['ptend_loc_q'],v['ptend_loc_s'],v['ptend_loc_lq'],v['ptend_loc_ls'],v['pcnst'],_globals=_globals,xp=xp)
        _put(v,'ptend_loc_name',None,_call[0])  # WRF:module_mp_cammgmp_driver.F:2077
        _put(v,'ptend_loc_q',None,_call[1])  # WRF:module_mp_cammgmp_driver.F:2077
        _put(v,'ptend_loc_s',None,_call[2])  # WRF:module_mp_cammgmp_driver.F:2077
        _put(v,'ptend_loc_lq',None,_call[3])  # WRF:module_mp_cammgmp_driver.F:2077
        _put(v,'ptend_loc_ls',None,_call[4])  # WRF:module_mp_cammgmp_driver.F:2077
        return tuple(v[_n] for _n in ('ptend_loc_lq', 'ptend_loc_ls', 'ptend_loc_q', 'ptend_loc_s'))
    def _no51(_carry):
        v=dict(_outer51);v.update(zip(('ptend_loc_lq', 'ptend_loc_ls', 'ptend_loc_q', 'ptend_loc_s'),_carry))
        return tuple(v[_n] for _n in ('ptend_loc_lq', 'ptend_loc_ls', 'ptend_loc_q', 'ptend_loc_s'))
    _result=_cond(v['_jump'] == 0,_yes51,_no51,tuple(v[_n] for _n in ('ptend_loc_lq', 'ptend_loc_ls', 'ptend_loc_q', 'ptend_loc_s')),xp)  # WRF:module_mp_cammgmp_driver.F:2077
    v.update(zip(('ptend_loc_lq', 'ptend_loc_ls', 'ptend_loc_q', 'ptend_loc_s'),_result))
    v['_jump']=xp.where(v['_jump']==2000,_int(0),v['_jump'])
    _outer52=dict(v)
    def _yes52(_carry):
        v=dict(_outer52);v.update(zip(('aist', 'alst', 'ast', 'k', 'state1_q'),_carry))
        _outer53=dict(v)
        def _body53(_k,_carry):
            v=dict(_outer53);v.update(zip(('_jump', 'aist', 'alst', 'ast', 'k', 'state1_q'),_carry));v['k']=_int(_k)
            _put(v,'state1_q',(1,v['k'],1,),xp.maximum(_get(v['qmin'], (1,)), _get(v['state1_q'], (1, v['k'], 1))))  # WRF:module_mp_cammgmp_driver.F:2085
            _put(v,'state1_q',(1,v['k'],v['ixcldliq'],),xp.maximum(_get(v['qmin'], (v['ixcldliq'],)), _get(v['state1_q'], (1, v['k'], v['ixcldliq']))))  # WRF:module_mp_cammgmp_driver.F:2086
            _put(v,'state1_q',(1,v['k'],v['ixcldice'],),xp.maximum(_get(v['qmin'], (v['ixcldice'],)), _get(v['state1_q'], (1, v['k'], v['ixcldice']))))  # WRF:module_mp_cammgmp_driver.F:2087
            _put(v,'state1_q',(1,v['k'],v['ixnumliq'],),xp.maximum(_get(v['qmin'], (v['ixnumliq'],)), _get(v['state1_q'], (1, v['k'], v['ixnumliq']))))  # WRF:module_mp_cammgmp_driver.F:2088
            _put(v,'state1_q',(1,v['k'],v['ixnumice'],),xp.maximum(_get(v['qmin'], (v['ixnumice'],)), _get(v['state1_q'], (1, v['k'], v['ixnumice']))))  # WRF:module_mp_cammgmp_driver.F:2089
            _call=cal_cldfra_mp_1d(_get(v['alst'], (1, v['k'])),_get(v['aist'], (1, v['k'])),_get(v['state1_q'], (1, v['k'], 1)),_get(v['state1_q'], (1, v['k'], v['ixcldliq'])),_get(v['qi_mac'], (1, v['k'])),_get(v['state1_t'], (1, v['k'])),_get(v['state_pmid'], (1, v['k'])),v['xland_pt'],v['snowh_pt'],_globals=_globals,xp=xp)
            _put(v,'alst',(1,v['k'],),_call[0])  # WRF:module_mp_cammgmp_driver.F:2090
            _put(v,'aist',(1,v['k'],),_call[1])  # WRF:module_mp_cammgmp_driver.F:2090
            _put(v,'aist',(1,v['k'],),(_d(1.0) - _get(v['cv_frac'], (1, v['k']))) * _get(v['aist'], (1, v['k'])))  # WRF:module_mp_cammgmp_driver.F:2092
            _put(v,'alst',(1,v['k'],),(_d(1.0) - _get(v['cv_frac'], (1, v['k']))) * _get(v['alst'], (1, v['k'])))  # WRF:module_mp_cammgmp_driver.F:2093
            _put(v,'ast',(1,v['k'],),xp.maximum(_get(v['alst'], (1, v['k'])), _get(v['aist'], (1, v['k']))))  # WRF:module_mp_cammgmp_driver.F:2094
            return tuple(v[_n] for _n in ('_jump', 'aist', 'alst', 'ast', 'k', 'state1_q'))
        _result=_loop(1,v['pver'],1,_body53,tuple(v[_n] for _n in ('_jump', 'aist', 'alst', 'ast', 'k', 'state1_q')),xp,jump_index=0)
        v.update(zip(('_jump', 'aist', 'alst', 'ast', 'k', 'state1_q'),_result))
        return tuple(v[_n] for _n in ('aist', 'alst', 'ast', 'k', 'state1_q'))
    def _no52(_carry):
        v=dict(_outer52);v.update(zip(('aist', 'alst', 'ast', 'k', 'state1_q'),_carry))
        return tuple(v[_n] for _n in ('aist', 'alst', 'ast', 'k', 'state1_q'))
    _result=_cond(v['_jump'] == 0,_yes52,_no52,tuple(v[_n] for _n in ('aist', 'alst', 'ast', 'k', 'state1_q')),xp)  # WRF:module_mp_cammgmp_driver.F:2084
    v.update(zip(('aist', 'alst', 'ast', 'k', 'state1_q'),_result))
    return (v['state1_t'],v['ast'],v['aist'],v['alst'],v['state1_s'],v['state1_q'],v['rh_old'],v['cldo'],v['esl'],v['qvs'],v['ptend_loc_name'],v['ptend_loc_ls'],v['ptend_loc_lq'],v['ptend_loc_s'],v['ptend_loc_q'],v['ptend_all_name'],v['ptend_all_ls'],v['ptend_all_lq'],v['ptend_all_s'],v['ptend_all_q'],v['_fatal'],)

ARGUMENTS={'physics_ptend_reset': ['ptend_name', 'ptend_q', 'ptend_s', 'ptend_lq', 'ptend_ls', 'pcnst_in'], 'physics_ptend_init': ['ptend_name', 'ptend_q', 'ptend_s', 'ptend_lq', 'ptend_ls', 'pcnst_in'], 'physics_ptend_sum': ['ptend_ls', 'ptend_lq', 'ptend_s', 'ptend_q', 'ptend_sum_ls', 'ptend_sum_lq', 'ptend_sum_s', 'ptend_sum_q', 'pcnst_in'], 'physics_update': ['state_lchnk', 'dt', 'state_q', 'ptend_q', 'state_s', 'ptend_s', 'ptend_name', 'ptend_lq', 'ptend_ls', 'pcnst_in'], 'findsp_water': ['lchnk', 'ncol', 'q', 't', 'p', 'tsp', 'qsp'], 'macrophysics_simple': ['is_first_step', 'pcnst', 'pcols', 'kte', 'ixcldliq', 'ixnumliq', 'lchnk', 'dtime', 'state1_t', 'state_pmid', 'state_q', 'dp_icwmr', 'sh_icwmr', 'dp_frac', 'sh_frac', 'ast', 'aist', 'alst', 'qi_mac', 'xland_pt', 'snowh_pt', 'state1_s', 'state1_q', 'rh_old', 'cldo', 'esl', 'qvs', 'ptend_loc_name', 'ptend_loc_ls', 'ptend_loc_lq', 'ptend_loc_s', 'ptend_loc_q', 'ptend_all_name', 'ptend_all_ls', 'ptend_all_lq', 'ptend_all_s', 'ptend_all_q']}
OUTPUTS={'physics_ptend_reset': ['ptend_name', 'ptend_q', 'ptend_s', 'ptend_lq', 'ptend_ls'], 'physics_ptend_init': ['ptend_name', 'ptend_q', 'ptend_s', 'ptend_lq', 'ptend_ls'], 'physics_ptend_sum': ['ptend_sum_ls', 'ptend_sum_lq', 'ptend_sum_s', 'ptend_sum_q'], 'physics_update': ['state_q', 'ptend_q', 'state_s', 'ptend_s', 'ptend_name', 'ptend_lq', 'ptend_ls'], 'findsp_water': ['tsp', 'qsp', '_fatal'], 'macrophysics_simple': ['state1_t', 'ast', 'aist', 'alst', 'state1_s', 'state1_q', 'rh_old', 'cldo', 'esl', 'qvs', 'ptend_loc_name', 'ptend_loc_ls', 'ptend_loc_lq', 'ptend_loc_s', 'ptend_loc_q', 'ptend_all_name', 'ptend_all_ls', 'ptend_all_lq', 'ptend_all_s', 'ptend_all_q', '_fatal']}
VARIABLES={'physics_ptend_reset': {'pcnst_in': {'kind': 'i32', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'ptend_name': {'kind': 'str', 'dims': [], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'ptend_ls': {'kind': 'bool', 'dims': [], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'ptend_lq': {'kind': 'bool', 'dims': ['pcnst_in'], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'ptend_s': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'ptend_q': {'kind': 'f64', 'dims': ['pcols', 'pver', 'pcnst_in'], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'm': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}}, 'physics_ptend_init': {'pcnst_in': {'kind': 'i32', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'ptend_name': {'kind': 'str', 'dims': [], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'ptend_ls': {'kind': 'bool', 'dims': [], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'ptend_lq': {'kind': 'bool', 'dims': ['pcnst_in'], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'ptend_s': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'ptend_q': {'kind': 'f64', 'dims': ['pcols', 'pver', 'pcnst_in'], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}}, 'physics_ptend_sum': {'pcnst_in': {'kind': 'i32', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'ptend_ls': {'kind': 'bool', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'ptend_lq': {'kind': 'bool', 'dims': ['pcnst_in'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'ptend_s': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'ptend_q': {'kind': 'f64', 'dims': ['pcols', 'pver', 'pcnst_in'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'ptend_sum_ls': {'kind': 'bool', 'dims': [], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'ptend_sum_lq': {'kind': 'bool', 'dims': ['pcnst_in'], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'ptend_sum_s': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'ptend_sum_q': {'kind': 'f64', 'dims': ['pcols', 'pver', 'pcnst_in'], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'k': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'm': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'ncol': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'i': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}}, 'physics_update': {'state_lchnk': {'kind': 'i32', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'pcnst_in': {'kind': 'i32', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'dt': {'kind': 'f64', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'ptend_name': {'kind': 'str', 'dims': [], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'ptend_ls': {'kind': 'bool', 'dims': [], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'ptend_lq': {'kind': 'bool', 'dims': ['pcnst_in'], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'ptend_s': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'ptend_q': {'kind': 'f64', 'dims': ['pcols', 'pver', 'pcnst_in'], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'state_q': {'kind': 'f64', 'dims': ['pcols', 'pver', 'pcnst_in'], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'state_s': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'name': {'kind': 'str', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'microp_scheme': {'kind': 'str', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'm': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'i': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'k': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'ncol': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'ixcldice': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'ixcldliq': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'ixnumice': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'ixnumliq': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}}, 'findsp_water': {'lchnk': {'kind': 'i32', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'ncol': {'kind': 'i32', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'q': {'kind': 'f64', 'dims': ['pcols'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 't': {'kind': 'f64', 'dims': ['pcols'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'p': {'kind': 'f64', 'dims': ['pcols'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'tsp': {'kind': 'f64', 'dims': ['pcols'], 'intent': 'out', 'initial': None, 'parameter': False, 'allocatable': False}, 'qsp': {'kind': 'f64', 'dims': ['pcols'], 'intent': 'out', 'initial': None, 'parameter': False, 'allocatable': False}, 'iulog': {'kind': 'str', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'i': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'lflg': {'kind': 'bool', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'iter': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'l': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'error_found': {'kind': 'bool', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'omeps': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'trinv': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'es': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'desdt': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dqsdt': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dgdt': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'g': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'weight': {'kind': 'f64', 'dims': ['pcols'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'hlatsb': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'hlatvp': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'hltalt': {'kind': 'f64', 'dims': ['pcols'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'tterm': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'qs': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'tc': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'tt0': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 't1': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'q1': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dt': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dq': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dtm': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dqm': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'qvd': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'a1': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'tmp': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'rair': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'r1b': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'c1': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'c2': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'c3': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'denom': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dttol': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dqtol': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'doit': {'kind': 'i32', 'dims': ['pcols'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'enin': {'kind': 'f64', 'dims': ['pcols'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'enout': {'kind': 'f64', 'dims': ['pcols'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'tlim': {'kind': 'f64', 'dims': ['pcols'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}}, 'macrophysics_simple': {'is_first_step': {'kind': 'bool', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'pcnst': {'kind': 'i32', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'pcols': {'kind': 'i32', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'kte': {'kind': 'i32', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'ixcldliq': {'kind': 'i32', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'ixnumliq': {'kind': 'i32', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'lchnk': {'kind': 'i32', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'dtime': {'kind': 'f64', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'xland_pt': {'kind': 'f64', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'snowh_pt': {'kind': 'f64', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'state_pmid': {'kind': 'f64', 'dims': ['pcols', 'kte'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'dp_icwmr': {'kind': 'f64', 'dims': ['pcols', 'kte'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'sh_icwmr': {'kind': 'f64', 'dims': ['pcols', 'kte'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'dp_frac': {'kind': 'f64', 'dims': ['pcols', 'kte'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'sh_frac': {'kind': 'f64', 'dims': ['pcols', 'kte'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'qi_mac': {'kind': 'f64', 'dims': ['pcols', 'kte'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'state_q': {'kind': 'f64', 'dims': ['pcols', 'kte', 'pcnst'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'ptend_loc_name': {'kind': 'str', 'dims': [], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'ptend_all_name': {'kind': 'str', 'dims': [], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'ptend_loc_ls': {'kind': 'bool', 'dims': [], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'ptend_all_ls': {'kind': 'bool', 'dims': [], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'ptend_loc_lq': {'kind': 'bool', 'dims': ['pcnst'], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'ptend_all_lq': {'kind': 'bool', 'dims': ['pcnst'], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'ptend_loc_s': {'kind': 'f64', 'dims': ['pcols', 'kte'], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'ptend_all_s': {'kind': 'f64', 'dims': ['pcols', 'kte'], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'rh_old': {'kind': 'f64', 'dims': ['pcols', 'kte'], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'ast': {'kind': 'f64', 'dims': ['pcols', 'kte'], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'alst': {'kind': 'f64', 'dims': ['pcols', 'kte'], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'aist': {'kind': 'f64', 'dims': ['pcols', 'kte'], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'cldo': {'kind': 'f64', 'dims': ['pcols', 'kte'], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'esl': {'kind': 'f64', 'dims': ['pcols', 'kte'], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'qvs': {'kind': 'f64', 'dims': ['pcols', 'kte'], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'state1_s': {'kind': 'f64', 'dims': ['pcols', 'kte'], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'state1_t': {'kind': 'f64', 'dims': ['pcols', 'kte'], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'ptend_loc_q': {'kind': 'f64', 'dims': ['pcols', 'kte', 'pcnst'], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'ptend_all_q': {'kind': 'f64', 'dims': ['pcols', 'kte', 'pcnst'], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'state1_q': {'kind': 'f64', 'dims': ['pcols', 'kte', 'pcnst'], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'k': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'cv_frac': {'kind': 'f64', 'dims': ['pcols', 'kte'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'st_frac': {'kind': 'f64', 'dims': ['pcols', 'kte'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'ttt': {'kind': 'f64', 'dims': ['pcols', 'kte'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'rh_curr': {'kind': 'f64', 'dims': ['pcols', 'kte'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'del_rh': {'kind': 'f64', 'dims': ['pcols', 'kte'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'qc_nc': {'kind': 'f64', 'dims': ['pcols', 'kte'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'del_qq': {'kind': 'f64', 'dims': ['pcols', 'kte'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'del_cf': {'kind': 'f64', 'dims': ['pcols', 'kte'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'rh_temp': {'kind': 'f64', 'dims': ['pcols', 'kte'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'qsll': {'kind': 'f64', 'dims': ['pcols'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'tsp': {'kind': 'f64', 'dims': ['pcols'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'cldcr': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'factcc': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'fact_adj': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'wrfmessage': {'kind': 'str', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}}}
