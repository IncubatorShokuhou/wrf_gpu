"""Generated literal P3 level-local processes; see proofs/s2_p3/port_processes.py."""
from .process_support import *  # noqa: F403


@_with_double
def final(v, tables, *, ncat, log_predictnc, log_3momentice, observer=None):
    v = dict(v)
    iparam = 3
    scpf_on = log_predictssat = False
    niter_mui = 5
    log_hmossopon = ncat > 1
    alive = v['level_active']
    level_alive = alive
    v.setdefault('birim', _initial(v['qv'], 'birim', ncat))
    v.setdefault('cdist', _initial(v['qv'], 'cdist', ncat))
    v.setdefault('cdist1', _initial(v['qv'], 'cdist1', ncat))
    v.setdefault('d_new', _initial(v['qv'], 'd_new', ncat))
    v.setdefault('diag_di', _initial(v['qv'], 'diag_di', ncat))
    v.setdefault('diag_effc', _initial(v['qv'], 'diag_effc', ncat))
    v.setdefault('diag_effi', _initial(v['qv'], 'diag_effi', ncat))
    v.setdefault('diag_rhoi', _initial(v['qv'], 'diag_rhoi', ncat))
    v.setdefault('diag_vmi', _initial(v['qv'], 'diag_vmi', ncat))
    v.setdefault('diag_ze', _initial(v['qv'], 'diag_ze', ncat))
    v.setdefault('diam_ice', _initial(v['qv'], 'diam_ice', ncat))
    v.setdefault('dum1', _initial(v['qv'], 'dum1', ncat))
    v.setdefault('dum1z', _initial(v['qv'], 'dum1z', ncat))
    v.setdefault('dum2', _initial(v['qv'], 'dum2', ncat))
    v.setdefault('dum4', _initial(v['qv'], 'dum4', ncat))
    v.setdefault('dum5', _initial(v['qv'], 'dum5', ncat))
    v.setdefault('dum6', _initial(v['qv'], 'dum6', ncat))
    v.setdefault('dumi', _initial(v['qv'], 'dumi', ncat))
    v.setdefault('dumii', _initial(v['qv'], 'dumii', ncat))
    v.setdefault('dumjj', _initial(v['qv'], 'dumjj', ncat))
    v.setdefault('dumzz', _initial(v['qv'], 'dumzz', ncat))
    v.setdefault('f1pr02', _initial(v['qv'], 'f1pr02', ncat))
    v.setdefault('f1pr06', _initial(v['qv'], 'f1pr06', ncat))
    v.setdefault('f1pr09', _initial(v['qv'], 'f1pr09', ncat))
    v.setdefault('f1pr10', _initial(v['qv'], 'f1pr10', ncat))
    v.setdefault('f1pr13', _initial(v['qv'], 'f1pr13', ncat))
    v.setdefault('f1pr15', _initial(v['qv'], 'f1pr15', ncat))
    v.setdefault('f1pr16', _initial(v['qv'], 'f1pr16', ncat))
    v.setdefault('f1pr22', _initial(v['qv'], 'f1pr22', ncat))
    v.setdefault('f1pr23', _initial(v['qv'], 'f1pr23', ncat))
    v.setdefault('iice_dest', _initial(v['qv'], 'iice_dest', ncat))
    v.setdefault('lamc', _initial(v['qv'], 'lamc', ncat))
    v.setdefault('lammax', _initial(v['qv'], 'lammax', ncat))
    v.setdefault('lammin', _initial(v['qv'], 'lammin', ncat))
    v.setdefault('lamr', _initial(v['qv'], 'lamr', ncat))
    v.setdefault('mu_c', _initial(v['qv'], 'mu_c', ncat))
    v.setdefault('mu_i', _initial(v['qv'], 'mu_i', ncat))
    v.setdefault('mu_i_new', _initial(v['qv'], 'mu_i_new', ncat))
    v.setdefault('mu_r', _initial(v['qv'], 'mu_r', ncat))
    v.setdefault('n_nuc', _initial(v['qv'], 'n_nuc', ncat))
    v.setdefault('nc', _initial(v['qv'], 'nc', ncat))
    v.setdefault('nitot', _initial(v['qv'], 'nitot', ncat))
    v.setdefault('nr', _initial(v['qv'], 'nr', ncat))
    v.setdefault('nu', _initial(v['qv'], 'nu', ncat))
    v.setdefault('q_nuc', _initial(v['qv'], 'q_nuc', ncat))
    v.setdefault('qc', _initial(v['qv'], 'qc', ncat))
    v.setdefault('qirim', _initial(v['qv'], 'qirim', ncat))
    v.setdefault('qitot', _initial(v['qv'], 'qitot', ncat))
    v.setdefault('qr', _initial(v['qv'], 'qr', ncat))
    v.setdefault('qv', _initial(v['qv'], 'qv', ncat))
    v.setdefault('rhop', _initial(v['qv'], 'rhop', ncat))
    v.setdefault('th', _initial(v['qv'], 'th', ncat))
    v.setdefault('tmp1', _initial(v['qv'], 'tmp1', ncat))
    v.setdefault('tmp2', _initial(v['qv'], 'tmp2', ncat))
    v.setdefault('ze_ice', _initial(v['qv'], 'ze_ice', ncat))
    v.setdefault('ze_rain', _initial(v['qv'], 'ze_rain', ncat))
    v.setdefault('zitot', _initial(v['qv'], 'zitot', ncat))
    assert 'deltad_init' in v, 'P3 required WRF input/constant missing: deltad_init'
    assert 'densize' in v, 'P3 required WRF input/constant missing: densize'
    assert 'dnu' in v, 'P3 required WRF input/constant missing: dnu'
    assert 'inv_cp' in v, 'P3 required WRF input/constant missing: inv_cp'
    assert 'inv_rho' in v, 'P3 required WRF input/constant missing: inv_rho'
    assert 'inv_rho_rimemax' in v, 'P3 required WRF input/constant missing: inv_rho_rimemax'
    assert 'invexn' in v, 'P3 required WRF input/constant missing: invexn'
    assert 'iscf' in v, 'P3 required WRF input/constant missing: iscf'
    assert 'isize' in v, 'P3 required WRF input/constant missing: isize'
    assert 'max_total_ni' in v, 'P3 required WRF input/constant missing: max_total_ni'
    assert 'mu_i_max' in v, 'P3 required WRF input/constant missing: mu_i_max'
    assert 'nsmall' in v, 'P3 required WRF input/constant missing: nsmall'
    assert 'pi' in v, 'P3 required WRF input/constant missing: pi'
    assert 'qsmall' in v, 'P3 required WRF input/constant missing: qsmall'
    assert 'rho' in v, 'P3 required WRF input/constant missing: rho'
    assert 'rhofaci' in v, 'P3 required WRF input/constant missing: rhofaci'
    assert 'rimsize' in v, 'P3 required WRF input/constant missing: rimsize'
    assert 't' in v, 'P3 required WRF input/constant missing: t'
    assert 'thrd' in v, 'P3 required WRF input/constant missing: thrd'
    assert 'xlf' in v, 'P3 required WRF input/constant missing: xlf'
    assert 'xxls' in v, 'P3 required WRF input/constant missing: xxls'
    assert 'xxlv' in v, 'P3 required WRF input/constant missing: xxlv'
    assert 'zsize' in v, 'P3 required WRF input/constant missing: zsize'
    assert 'zsmall' in v, 'P3 required WRF input/constant missing: zsmall'
    _put(v, 'diam_ice', None, _r(0.0), alive)  # WRF:4531
    for iice in range(1, (ncat) + (1 if (1) > 0 else -1), 1):
        mask_1 = jnp.logical_and(alive, _get(v['qitot'], (iice - 1,)) >= v['qsmall'])
        taken_1 = mask_1
        _put(v, 'dum1', None, _max(_get(v['nitot'], (iice - 1,)), v['nsmall']), mask_1)  # WRF:4534
        _put(v, 'dum2', None, _r(500.0), mask_1)  # WRF:4535
        _put(v, 'diam_ice', ((iice) - 1,), (_get(v['qitot'], (iice - 1,)) * _r(6.0) / (v['dum1'] * v['dum2'] * v['pi'])) ** v['thrd'], mask_1)  # WRF:4536
    mask_2 = jnp.logical_and(alive, jnp.logical_and(v['qc'] >= v['qsmall'], v['t'] < _r(233.15)))
    taken_2 = mask_2
    _put(v, 'q_nuc', None, v['qc'], mask_2)  # WRF:4542
    _put(v, 'nc', None, _max(v['nc'], v['nsmall']), mask_2)  # WRF:4543
    _put(v, 'n_nuc', None, v['nc'], mask_2)  # WRF:4544
    if ncat > 1:
        _put(v, 'dum1', None, _r(900.0), mask_2)  # WRF:4548
        _put(v, 'd_new', None, (v['q_nuc'] * _r(6.0) / (v['pi'] * v['dum1'] * v['n_nuc'])) ** v['thrd'], mask_2)  # WRF:4549
        call_out = _call_icecat_destination(v['qitot'] * _expand(v['iscf'], 1), v['diam_ice'], v['d_new'], v['deltad_init'], v['iice_dest'], tables=tables, three=log_3momentice)
        _put(v, 'iice_dest', None, call_out[0], mask_2)  # WRF:4550
    else:
        _put(v, 'iice_dest', None, 1, mask_2)  # WRF:4554
    _put(v, 'qirim', ((v['iice_dest']) - 1,), _get(v['qirim'], (v['iice_dest'] - 1,)) + v['q_nuc'], mask_2)  # WRF:4557
    _put(v, 'qitot', ((v['iice_dest']) - 1,), _get(v['qitot'], (v['iice_dest'] - 1,)) + v['q_nuc'], mask_2)  # WRF:4558
    _put(v, 'birim', ((v['iice_dest']) - 1,), _get(v['birim'], (v['iice_dest'] - 1,)) + v['q_nuc'] * v['inv_rho_rimemax'], mask_2)  # WRF:4559
    _put(v, 'nitot', ((v['iice_dest']) - 1,), _get(v['nitot'], (v['iice_dest'] - 1,)) + v['n_nuc'], mask_2)  # WRF:4560
    mask_3 = jnp.logical_and(mask_2, jnp.logical_and(log_3momentice, v['n_nuc'] >= v['nsmall']))
    taken_3 = mask_3
    _put(v, 'tmp1', None, v['q_nuc'] * _r(6.0) / (_r(900.0) * v['pi']), mask_3)  # WRF:4565
    call_out = _call_get_cloud_dsd2(v['qc'], v['nc'], v['mu_c'], v['rho'], v['nu'], v['dnu'], v['lamc'], v['lammin'], v['lammax'], v['cdist'], v['cdist1'], v['iscf'], tables=tables, three=log_3momentice)
    _put(v, 'nc', None, call_out[0], mask_3)  # WRF:4566
    _put(v, 'mu_c', None, call_out[1], mask_3)  # WRF:4566
    _put(v, 'nu', None, call_out[2], mask_3)  # WRF:4566
    _put(v, 'lamc', None, call_out[3], mask_3)  # WRF:4566
    _put(v, 'lammin', None, call_out[4], mask_3)  # WRF:4566
    _put(v, 'lammax', None, call_out[5], mask_3)  # WRF:4566
    _put(v, 'cdist', None, call_out[6], mask_3)  # WRF:4566
    _put(v, 'cdist1', None, call_out[7], mask_3)  # WRF:4566
    _put(v, 'mu_i_new', None, v['mu_c'], mask_3)  # WRF:4568
    _put(v, 'zitot', ((v['iice_dest']) - 1,), _get(v['zitot'], (v['iice_dest'] - 1,)) + g_of_mu(v['mu_i_new']) * v['tmp1'] ** 2 / v['n_nuc'], mask_3)  # WRF:4569
    _put(v, 'th', None, v['th'] + v['invexn'] * v['q_nuc'] * v['xlf'] * v['inv_cp'], mask_2)  # WRF:4572
    _put(v, 'qc', None, _r(0.0), mask_2)  # WRF:4573
    _put(v, 'nc', None, _r(0.0), mask_2)  # WRF:4574
    mask_4 = jnp.logical_and(alive, jnp.logical_and(v['qr'] >= v['qsmall'], v['t'] < _r(233.15)))
    taken_4 = mask_4
    _put(v, 'q_nuc', None, v['qr'], mask_4)  # WRF:4580
    _put(v, 'nr', None, _max(v['nr'], v['nsmall']), mask_4)  # WRF:4581
    _put(v, 'n_nuc', None, v['nr'], mask_4)  # WRF:4582
    if ncat > 1:
        _put(v, 'dum1', None, _r(900.0), mask_4)  # WRF:4585
        _put(v, 'd_new', None, (v['q_nuc'] * _r(6.0) / (v['pi'] * v['dum1'] * v['n_nuc'])) ** v['thrd'], mask_4)  # WRF:4586
        call_out = _call_icecat_destination(v['qitot'] * _expand(v['iscf'], 1), v['diam_ice'], v['d_new'], v['deltad_init'], v['iice_dest'], tables=tables, three=log_3momentice)
        _put(v, 'iice_dest', None, call_out[0], mask_4)  # WRF:4587
    else:
        _put(v, 'iice_dest', None, 1, mask_4)  # WRF:4590
    _put(v, 'qirim', ((v['iice_dest']) - 1,), _get(v['qirim'], (v['iice_dest'] - 1,)) + v['q_nuc'], mask_4)  # WRF:4593
    _put(v, 'qitot', ((v['iice_dest']) - 1,), _get(v['qitot'], (v['iice_dest'] - 1,)) + v['q_nuc'], mask_4)  # WRF:4594
    _put(v, 'birim', ((v['iice_dest']) - 1,), _get(v['birim'], (v['iice_dest'] - 1,)) + v['q_nuc'] * v['inv_rho_rimemax'], mask_4)  # WRF:4595
    _put(v, 'nitot', ((v['iice_dest']) - 1,), _get(v['nitot'], (v['iice_dest'] - 1,)) + v['n_nuc'], mask_4)  # WRF:4596
    mask_5 = jnp.logical_and(mask_4, jnp.logical_and(log_3momentice, v['n_nuc'] >= v['qsmall']))
    taken_5 = mask_5
    _put(v, 'tmp1', None, v['q_nuc'] * _r(6.0) / (_r(900.0) * v['pi']), mask_5)  # WRF:4599
    _put(v, 'mu_i_new', None, v['mu_r'], mask_5)  # WRF:4600
    _put(v, 'zitot', ((v['iice_dest']) - 1,), _get(v['zitot'], (v['iice_dest'] - 1,)) + g_of_mu(v['mu_i_new']) * v['tmp1'] ** 2 / v['n_nuc'], mask_5)  # WRF:4601
    _put(v, 'th', None, v['th'] + v['invexn'] * v['q_nuc'] * v['xlf'] * v['inv_cp'], mask_4)  # WRF:4604
    _put(v, 'qr', None, _r(0.0), mask_4)  # WRF:4605
    _put(v, 'nr', None, _r(0.0), mask_4)  # WRF:4606
    if ncat > 1:
        for iice in range(1, (ncat) + (1 if (1) > 0 else -1), 1):
            mask_6 = jnp.logical_and(alive, _get(v['qitot'], (iice - 1,)) >= v['qsmall'])
            taken_6 = mask_6
            _put(v, 'nitot', ((iice) - 1,), _max(_get(v['nitot'], (iice - 1,)), v['nsmall']), mask_6)  # WRF:4623
            call_out = _call_calc_bulkrhorime(_get(v['qitot'], (iice - 1,)), _get(v['qirim'], (iice - 1,)), _get(v['birim'], (iice - 1,)), v['rhop'], tables=tables, three=log_3momentice)
            _put(v, 'qirim', ((iice) - 1,), call_out[0], mask_6)  # WRF:4624
            _put(v, 'birim', ((iice) - 1,), call_out[1], mask_6)  # WRF:4624
            _put(v, 'rhop', None, call_out[2], mask_6)  # WRF:4624
            if not log_3momentice:
                call_out = _call_find_lookuptable_indices_1a(v['dumi'], v['dumjj'], v['dumii'], v['dum1'], v['dum4'], v['dum5'], v['isize'], v['rimsize'], v['densize'], _get(v['qitot'], (iice - 1,)), _get(v['nitot'], (iice - 1,)), _get(v['qirim'], (iice - 1,)), v['rhop'], tables=tables, three=log_3momentice)
                _put(v, 'dumi', None, call_out[0], mask_6)  # WRF:4628
                _put(v, 'dumjj', None, call_out[1], mask_6)  # WRF:4628
                _put(v, 'dumii', None, call_out[2], mask_6)  # WRF:4628
                _put(v, 'dum1', None, call_out[3], mask_6)  # WRF:4628
                _put(v, 'dum4', None, call_out[4], mask_6)  # WRF:4628
                _put(v, 'dum5', None, call_out[5], mask_6)  # WRF:4628
                call_out = _call_access_lookup_table(v['dumjj'], v['dumii'], v['dumi'], 11, v['dum1'], v['dum4'], v['dum5'], v['f1pr15'], tables=tables, three=log_3momentice)
                _put(v, 'f1pr15', None, call_out[0], mask_6)  # WRF:4631
            else:
                call_out = _call_find_lookuptable_indices_1a(v['dumi'], v['dumjj'], v['dumii'], v['dum1'], v['dum4'], v['dum5'], v['isize'], v['rimsize'], v['densize'], _get(v['qitot'], (iice - 1,)), _get(v['nitot'], (iice - 1,)), _get(v['qirim'], (iice - 1,)), v['rhop'], tables=tables, three=log_3momentice)
                _put(v, 'dumi', None, call_out[0], mask_6)  # WRF:4635
                _put(v, 'dumjj', None, call_out[1], mask_6)  # WRF:4635
                _put(v, 'dumii', None, call_out[2], mask_6)  # WRF:4635
                _put(v, 'dum1', None, call_out[3], mask_6)  # WRF:4635
                _put(v, 'dum4', None, call_out[4], mask_6)  # WRF:4635
                _put(v, 'dum5', None, call_out[5], mask_6)  # WRF:4635
                _put(v, 'zitot', ((iice) - 1,), _max(_get(v['zitot'], (iice - 1,)), v['zsmall']), mask_6)  # WRF:4638
                _put(v, 'dum1z', None, _r(6.0) / (_r(200.0) * v['pi']) * _get(v['qitot'], (iice - 1,)), mask_6)  # WRF:4639
                for imu in range(1, (niter_mui) + (1 if (1) > 0 else -1), 1):
                    _put(v, 'mu_i', None, compute_mu_3moment(_get(v['nitot'], (iice - 1,)), v['dum1z'], _get(v['zitot'], (iice - 1,)), v['mu_i_max']), mask_6)  # WRF:4641
                    call_out = _call_find_lookuptable_indices_1c(v['dumzz'], v['dum6'], v['zsize'], _get(v['qitot'], (iice - 1,)), v['mu_i'], tables=tables, three=log_3momentice)
                    _put(v, 'dumzz', None, call_out[0], mask_6)  # WRF:4642
                    _put(v, 'dum6', None, call_out[1], mask_6)  # WRF:4642
                    call_out = _call_access_lookup_table_3mom(v['dumzz'], v['dumjj'], v['dumii'], v['dumi'], 12, v['dum1'], v['dum4'], v['dum5'], v['dum6'], v['f1pr16'], tables=tables, three=log_3momentice)
                    _put(v, 'f1pr16', None, call_out[0], mask_6)  # WRF:4643
                    _put(v, 'dum1z', None, _r(6.0) / (v['f1pr16'] * v['pi']) * _get(v['qitot'], (iice - 1,)), mask_6)  # WRF:4644
                call_out = _call_access_lookup_table_3mom(v['dumzz'], v['dumjj'], v['dumii'], v['dumi'], 11, v['dum1'], v['dum4'], v['dum5'], v['dum6'], v['f1pr15'], tables=tables, three=log_3momentice)
                _put(v, 'f1pr15', None, call_out[0], mask_6)  # WRF:4646
            if log_3momentice:
                _put(v, 'dum1', None, _r(6.0) / (v['f1pr16'] * v['pi']) * _get(v['qitot'], (iice - 1,)), mask_6)  # WRF:4652
                _put(v, 'tmp1', None, g_of_mu(_r(0.0)), mask_6)  # WRF:4653
                _put(v, 'tmp2', None, g_of_mu(_r(20.0)), mask_6)  # WRF:4654
                _put(v, 'zitot', ((iice) - 1,), _min(_get(v['zitot'], (iice - 1,)), v['tmp1'] * v['dum1'] ** 2 / _get(v['nitot'], (iice - 1,))), mask_6)  # WRF:4655
                _put(v, 'zitot', ((iice) - 1,), _max(_get(v['zitot'], (iice - 1,)), v['tmp2'] * v['dum1'] ** 2 / _get(v['nitot'], (iice - 1,))), mask_6)  # WRF:4656
            _put(v, 'diag_di', ((iice) - 1,), v['f1pr15'], mask_6)  # WRF:4659
            mask_7 = jnp.logical_and(alive, jnp.logical_not(taken_6))
            _put(v, 'qv', None, v['qv'] + _get(v['qitot'], (iice - 1,)), mask_7)  # WRF:4663
            _put(v, 'th', None, v['th'] - v['invexn'] * _get(v['qitot'], (iice - 1,)) * v['xxls'] * v['inv_cp'], mask_7)  # WRF:4664
            _put(v, 'qitot', ((iice) - 1,), _r(0.0), mask_7)  # WRF:4665
            _put(v, 'nitot', ((iice) - 1,), _r(0.0), mask_7)  # WRF:4666
            _put(v, 'qirim', ((iice) - 1,), _r(0.0), mask_7)  # WRF:4667
            _put(v, 'birim', ((iice) - 1,), _r(0.0), mask_7)  # WRF:4668
            mask_8 = jnp.logical_and(mask_7, log_3momentice)
            _put(v, 'zitot', ((iice) - 1,), 0, mask_8)  # WRF:4669
            _put(v, 'diag_di', ((iice) - 1,), _r(0.0), mask_7)  # WRF:4670
        for iice in range(ncat, (2) + (1 if (-1) > 0 else -1), -1):
            _put(v, 'tmp1', None, jnp.abs(_get(v['diag_di'], (iice - 1,)) - _get(v['diag_di'], (iice - 1 - 1,))), alive)  # WRF:4679
            mask_9 = jnp.logical_and(alive, jnp.logical_and(jnp.logical_and(v['tmp1'] <= v['deltad_init'], _get(v['qitot'], (iice - 1,)) > _r(0.0)), _get(v['qitot'], (iice - 1 - 1,)) > _r(0.0)))
            taken_9 = mask_9
            _put(v, 'qitot', ((iice - 1) - 1,), _get(v['qitot'], (iice - 1 - 1,)) + _get(v['qitot'], (iice - 1,)), mask_9)  # WRF:4681
            _put(v, 'nitot', ((iice - 1) - 1,), _get(v['nitot'], (iice - 1 - 1,)) + _get(v['nitot'], (iice - 1,)), mask_9)  # WRF:4682
            _put(v, 'qirim', ((iice - 1) - 1,), _get(v['qirim'], (iice - 1 - 1,)) + _get(v['qirim'], (iice - 1,)), mask_9)  # WRF:4683
            _put(v, 'birim', ((iice - 1) - 1,), _get(v['birim'], (iice - 1 - 1,)) + _get(v['birim'], (iice - 1,)), mask_9)  # WRF:4684
            _put(v, 'qitot', ((iice) - 1,), _r(0.0), mask_9)  # WRF:4685
            _put(v, 'nitot', ((iice) - 1,), _r(0.0), mask_9)  # WRF:4686
            _put(v, 'qirim', ((iice) - 1,), _r(0.0), mask_9)  # WRF:4687
            _put(v, 'birim', ((iice) - 1,), _r(0.0), mask_9)  # WRF:4688
            if log_3momentice:
                _put(v, 'zitot', ((iice - 1) - 1,), _get(v['zitot'], (iice - 1 - 1,)) + _get(v['zitot'], (iice - 1,)), mask_9)  # WRF:4690
                _put(v, 'zitot', ((iice) - 1,), _r(0.0), mask_9)  # WRF:4691
    mask_10 = jnp.logical_and(alive, v['qc'] * v['iscf'] >= v['qsmall'])
    taken_10 = mask_10
    call_out = _call_get_cloud_dsd2(v['qc'], v['nc'], v['mu_c'], v['rho'], v['nu'], v['dnu'], v['lamc'], v['lammin'], v['lammax'], v['tmp1'], v['tmp2'], v['iscf'], tables=tables, three=log_3momentice)
    _put(v, 'nc', None, call_out[0], mask_10)  # WRF:4726
    _put(v, 'mu_c', None, call_out[1], mask_10)  # WRF:4726
    _put(v, 'nu', None, call_out[2], mask_10)  # WRF:4726
    _put(v, 'lamc', None, call_out[3], mask_10)  # WRF:4726
    _put(v, 'lammin', None, call_out[4], mask_10)  # WRF:4726
    _put(v, 'lammax', None, call_out[5], mask_10)  # WRF:4726
    _put(v, 'tmp1', None, call_out[6], mask_10)  # WRF:4726
    _put(v, 'tmp2', None, call_out[7], mask_10)  # WRF:4726
    _put(v, 'diag_effc', None, _r(0.5) * (v['mu_c'] + _r(3.0)) / v['lamc'], mask_10)  # WRF:4728
    mask_11 = jnp.logical_and(alive, jnp.logical_not(taken_10))
    _put(v, 'qv', None, v['qv'] + v['qc'], mask_11)  # WRF:4730
    _put(v, 'th', None, v['th'] - v['invexn'] * v['qc'] * v['xxlv'] * v['inv_cp'], mask_11)  # WRF:4731
    _put(v, 'qc', None, _r(0.0), mask_11)  # WRF:4732
    _put(v, 'nc', None, _r(0.0), mask_11)  # WRF:4733
    mask_12 = jnp.logical_and(alive, v['qr'] >= v['qsmall'])
    taken_12 = mask_12
    call_out = _call_get_rain_dsd2(v['qr'], v['nr'], v['mu_r'], v['lamr'], v['tmp1'], v['tmp2'], _r(1.0), tables=tables, three=log_3momentice)
    _put(v, 'nr', None, call_out[0], mask_12)  # WRF:4739
    _put(v, 'mu_r', None, call_out[1], mask_12)  # WRF:4739
    _put(v, 'lamr', None, call_out[2], mask_12)  # WRF:4739
    _put(v, 'tmp1', None, call_out[3], mask_12)  # WRF:4739
    _put(v, 'tmp2', None, call_out[4], mask_12)  # WRF:4739
    _put(v, 'ze_rain', None, v['rho'] * v['nr'] * (v['mu_r'] + _r(6.0)) * (v['mu_r'] + _r(5.0)) * (v['mu_r'] + _r(4.0)) * (v['mu_r'] + _r(3.0)) * (v['mu_r'] + _r(2.0)) * (v['mu_r'] + _r(1.0)) / v['lamr'] ** 6, mask_12)  # WRF:4755
    _put(v, 'ze_rain', None, _max(v['ze_rain'], _r(1e-22)), mask_12)  # WRF:4757
    mask_13 = jnp.logical_and(alive, jnp.logical_not(taken_12))
    _put(v, 'qv', None, v['qv'] + v['qr'], mask_13)  # WRF:4759
    _put(v, 'th', None, v['th'] - v['invexn'] * v['qr'] * v['xxlv'] * v['inv_cp'], mask_13)  # WRF:4760
    _put(v, 'qr', None, _r(0.0), mask_13)  # WRF:4761
    _put(v, 'nr', None, _r(0.0), mask_13)  # WRF:4762
    call_out = _call_impose_max_total_ni(v['nitot'], v['max_total_ni'], v['inv_rho'], tables=tables, three=log_3momentice)
    _put(v, 'nitot', None, call_out[0], alive)  # WRF:4767
    for iice in range(1, (ncat) + (1 if (1) > 0 else -1), 1):
        mask_14 = jnp.logical_and(alive, _get(v['qitot'], (iice - 1,)) >= v['qsmall'])
        taken_14 = mask_14
        _put(v, 'nitot', ((iice) - 1,), _max(_get(v['nitot'], (iice - 1,)), v['nsmall']), mask_14)  # WRF:4774
        _put(v, 'nr', None, _max(v['nr'], v['nsmall']), mask_14)  # WRF:4775
        call_out = _call_calc_bulkrhorime(_get(v['qitot'], (iice - 1,)), _get(v['qirim'], (iice - 1,)), _get(v['birim'], (iice - 1,)), v['rhop'], tables=tables, three=log_3momentice)
        _put(v, 'qirim', ((iice) - 1,), call_out[0], mask_14)  # WRF:4777
        _put(v, 'birim', ((iice) - 1,), call_out[1], mask_14)  # WRF:4777
        _put(v, 'rhop', None, call_out[2], mask_14)  # WRF:4777
        if not log_3momentice:
            call_out = _call_find_lookuptable_indices_1a(v['dumi'], v['dumjj'], v['dumii'], v['dum1'], v['dum4'], v['dum5'], v['isize'], v['rimsize'], v['densize'], _get(v['qitot'], (iice - 1,)), _get(v['nitot'], (iice - 1,)), _get(v['qirim'], (iice - 1,)), v['rhop'], tables=tables, three=log_3momentice)
            _put(v, 'dumi', None, call_out[0], mask_14)  # WRF:4781
            _put(v, 'dumjj', None, call_out[1], mask_14)  # WRF:4781
            _put(v, 'dumii', None, call_out[2], mask_14)  # WRF:4781
            _put(v, 'dum1', None, call_out[3], mask_14)  # WRF:4781
            _put(v, 'dum4', None, call_out[4], mask_14)  # WRF:4781
            _put(v, 'dum5', None, call_out[5], mask_14)  # WRF:4781
            call_out = _call_access_lookup_table(v['dumjj'], v['dumii'], v['dumi'], 2, v['dum1'], v['dum4'], v['dum5'], v['f1pr02'], tables=tables, three=log_3momentice)
            _put(v, 'f1pr02', None, call_out[0], mask_14)  # WRF:4785
            call_out = _call_access_lookup_table(v['dumjj'], v['dumii'], v['dumi'], 6, v['dum1'], v['dum4'], v['dum5'], v['f1pr06'], tables=tables, three=log_3momentice)
            _put(v, 'f1pr06', None, call_out[0], mask_14)  # WRF:4786
            call_out = _call_access_lookup_table(v['dumjj'], v['dumii'], v['dumi'], 7, v['dum1'], v['dum4'], v['dum5'], v['f1pr09'], tables=tables, three=log_3momentice)
            _put(v, 'f1pr09', None, call_out[0], mask_14)  # WRF:4787
            call_out = _call_access_lookup_table(v['dumjj'], v['dumii'], v['dumi'], 8, v['dum1'], v['dum4'], v['dum5'], v['f1pr10'], tables=tables, three=log_3momentice)
            _put(v, 'f1pr10', None, call_out[0], mask_14)  # WRF:4788
            call_out = _call_access_lookup_table(v['dumjj'], v['dumii'], v['dumi'], 9, v['dum1'], v['dum4'], v['dum5'], v['f1pr13'], tables=tables, three=log_3momentice)
            _put(v, 'f1pr13', None, call_out[0], mask_14)  # WRF:4789
            call_out = _call_access_lookup_table(v['dumjj'], v['dumii'], v['dumi'], 11, v['dum1'], v['dum4'], v['dum5'], v['f1pr15'], tables=tables, three=log_3momentice)
            _put(v, 'f1pr15', None, call_out[0], mask_14)  # WRF:4790
            call_out = _call_access_lookup_table(v['dumjj'], v['dumii'], v['dumi'], 12, v['dum1'], v['dum4'], v['dum5'], v['f1pr16'], tables=tables, three=log_3momentice)
            _put(v, 'f1pr16', None, call_out[0], mask_14)  # WRF:4791
            call_out = _call_access_lookup_table(v['dumjj'], v['dumii'], v['dumi'], 13, v['dum1'], v['dum4'], v['dum5'], v['f1pr22'], tables=tables, three=log_3momentice)
            _put(v, 'f1pr22', None, call_out[0], mask_14)  # WRF:4792
            call_out = _call_access_lookup_table(v['dumjj'], v['dumii'], v['dumi'], 14, v['dum1'], v['dum4'], v['dum5'], v['f1pr23'], tables=tables, three=log_3momentice)
            _put(v, 'f1pr23', None, call_out[0], mask_14)  # WRF:4793
        else:
            call_out = _call_find_lookuptable_indices_1a(v['dumi'], v['dumjj'], v['dumii'], v['dum1'], v['dum4'], v['dum5'], v['isize'], v['rimsize'], v['densize'], _get(v['qitot'], (iice - 1,)), _get(v['nitot'], (iice - 1,)), _get(v['qirim'], (iice - 1,)), v['rhop'], tables=tables, three=log_3momentice)
            _put(v, 'dumi', None, call_out[0], mask_14)  # WRF:4797
            _put(v, 'dumjj', None, call_out[1], mask_14)  # WRF:4797
            _put(v, 'dumii', None, call_out[2], mask_14)  # WRF:4797
            _put(v, 'dum1', None, call_out[3], mask_14)  # WRF:4797
            _put(v, 'dum4', None, call_out[4], mask_14)  # WRF:4797
            _put(v, 'dum5', None, call_out[5], mask_14)  # WRF:4797
            _put(v, 'zitot', ((iice) - 1,), _max(_get(v['zitot'], (iice - 1,)), v['zsmall']), mask_14)  # WRF:4804
            _put(v, 'dum1z', None, _r(6.0) / (_r(200.0) * v['pi']) * _get(v['qitot'], (iice - 1,)), mask_14)  # WRF:4806
            for imu in range(1, (niter_mui) + (1 if (1) > 0 else -1), 1):
                _put(v, 'mu_i', None, compute_mu_3moment(_get(v['nitot'], (iice - 1,)), v['dum1z'], _get(v['zitot'], (iice - 1,)), v['mu_i_max']), mask_14)  # WRF:4809
                call_out = _call_find_lookuptable_indices_1c(v['dumzz'], v['dum6'], v['zsize'], _get(v['qitot'], (iice - 1,)), v['mu_i'], tables=tables, three=log_3momentice)
                _put(v, 'dumzz', None, call_out[0], mask_14)  # WRF:4810
                _put(v, 'dum6', None, call_out[1], mask_14)  # WRF:4810
                call_out = _call_access_lookup_table_3mom(v['dumzz'], v['dumjj'], v['dumii'], v['dumi'], 12, v['dum1'], v['dum4'], v['dum5'], v['dum6'], v['f1pr16'], tables=tables, three=log_3momentice)
                _put(v, 'f1pr16', None, call_out[0], mask_14)  # WRF:4811
                _put(v, 'dum1z', None, _r(6.0) / (v['f1pr16'] * v['pi']) * _get(v['qitot'], (iice - 1,)), mask_14)  # WRF:4812
            call_out = _call_access_lookup_table_3mom(v['dumzz'], v['dumjj'], v['dumii'], v['dumi'], 2, v['dum1'], v['dum4'], v['dum5'], v['dum6'], v['f1pr02'], tables=tables, three=log_3momentice)
            _put(v, 'f1pr02', None, call_out[0], mask_14)  # WRF:4817
            call_out = _call_access_lookup_table_3mom(v['dumzz'], v['dumjj'], v['dumii'], v['dumi'], 6, v['dum1'], v['dum4'], v['dum5'], v['dum6'], v['f1pr06'], tables=tables, three=log_3momentice)
            _put(v, 'f1pr06', None, call_out[0], mask_14)  # WRF:4818
            call_out = _call_access_lookup_table_3mom(v['dumzz'], v['dumjj'], v['dumii'], v['dumi'], 7, v['dum1'], v['dum4'], v['dum5'], v['dum6'], v['f1pr09'], tables=tables, three=log_3momentice)
            _put(v, 'f1pr09', None, call_out[0], mask_14)  # WRF:4819
            call_out = _call_access_lookup_table_3mom(v['dumzz'], v['dumjj'], v['dumii'], v['dumi'], 8, v['dum1'], v['dum4'], v['dum5'], v['dum6'], v['f1pr10'], tables=tables, three=log_3momentice)
            _put(v, 'f1pr10', None, call_out[0], mask_14)  # WRF:4820
            call_out = _call_access_lookup_table_3mom(v['dumzz'], v['dumjj'], v['dumii'], v['dumi'], 9, v['dum1'], v['dum4'], v['dum5'], v['dum6'], v['f1pr13'], tables=tables, three=log_3momentice)
            _put(v, 'f1pr13', None, call_out[0], mask_14)  # WRF:4821
            call_out = _call_access_lookup_table_3mom(v['dumzz'], v['dumjj'], v['dumii'], v['dumi'], 11, v['dum1'], v['dum4'], v['dum5'], v['dum6'], v['f1pr15'], tables=tables, three=log_3momentice)
            _put(v, 'f1pr15', None, call_out[0], mask_14)  # WRF:4822
            call_out = _call_access_lookup_table_3mom(v['dumzz'], v['dumjj'], v['dumii'], v['dumi'], 14, v['dum1'], v['dum4'], v['dum5'], v['dum6'], v['f1pr22'], tables=tables, three=log_3momentice)
            _put(v, 'f1pr22', None, call_out[0], mask_14)  # WRF:4826
            call_out = _call_access_lookup_table_3mom(v['dumzz'], v['dumjj'], v['dumii'], v['dumi'], 15, v['dum1'], v['dum4'], v['dum5'], v['dum6'], v['f1pr23'], tables=tables, three=log_3momentice)
            _put(v, 'f1pr23', None, call_out[0], mask_14)  # WRF:4827
        _put(v, 'nitot', ((iice) - 1,), _min(_get(v['nitot'], (iice - 1,)), v['f1pr09'] * _get(v['qitot'], (iice - 1,))), mask_14)  # WRF:4833
        _put(v, 'nitot', ((iice) - 1,), _max(_get(v['nitot'], (iice - 1,)), v['f1pr10'] * _get(v['qitot'], (iice - 1,))), mask_14)  # WRF:4834
        if log_3momentice:
            _put(v, 'dum1', None, _r(6.0) / (v['f1pr16'] * v['pi']) * _get(v['qitot'], (iice - 1,)), mask_14)  # WRF:4839
            _put(v, 'tmp1', None, g_of_mu(_r(0.0)), mask_14)  # WRF:4840
            _put(v, 'tmp2', None, g_of_mu(_r(20.0)), mask_14)  # WRF:4841
            _put(v, 'zitot', ((iice) - 1,), _min(_get(v['zitot'], (iice - 1,)), v['tmp1'] * v['dum1'] ** 2 / _get(v['nitot'], (iice - 1,))), mask_14)  # WRF:4842
            _put(v, 'zitot', ((iice) - 1,), _max(_get(v['zitot'], (iice - 1,)), v['tmp2'] * v['dum1'] ** 2 / _get(v['nitot'], (iice - 1,))), mask_14)  # WRF:4843
        mask_15 = jnp.logical_and(mask_14, _get(v['qirim'], (iice - 1,)) < v['qsmall'])
        taken_15 = mask_15
        _put(v, 'qirim', ((iice) - 1,), _r(0.0), mask_15)  # WRF:4850
        _put(v, 'birim', ((iice) - 1,), _r(0.0), mask_15)  # WRF:4851
        _put(v, 'diag_vmi', ((iice) - 1,), v['f1pr02'] * v['rhofaci'], mask_14)  # WRF:4856
        _put(v, 'diag_effi', ((iice) - 1,), v['f1pr06'], mask_14)  # WRF:4857
        _put(v, 'diag_di', ((iice) - 1,), v['f1pr15'], mask_14)  # WRF:4858
        _put(v, 'diag_rhoi', ((iice) - 1,), v['f1pr16'], mask_14)  # WRF:4859
        _put(v, 'ze_ice', None, v['ze_ice'] + _r(0.1892) * v['f1pr13'] * _get(v['nitot'], (iice - 1,)) * v['rho'], mask_14)  # WRF:4868
        _put(v, 'ze_ice', None, _max(v['ze_ice'], _r(1e-22)), mask_14)  # WRF:4869
        mask_16 = jnp.logical_and(alive, jnp.logical_not(taken_14))
        _put(v, 'qv', None, v['qv'] + _get(v['qitot'], (iice - 1,)), mask_16)  # WRF:4873
        _put(v, 'th', None, v['th'] - v['invexn'] * _get(v['qitot'], (iice - 1,)) * v['xxls'] * v['inv_cp'], mask_16)  # WRF:4874
        _put(v, 'qitot', ((iice) - 1,), _r(0.0), mask_16)  # WRF:4875
        _put(v, 'nitot', ((iice) - 1,), _r(0.0), mask_16)  # WRF:4876
        _put(v, 'qirim', ((iice) - 1,), _r(0.0), mask_16)  # WRF:4877
        _put(v, 'birim', ((iice) - 1,), _r(0.0), mask_16)  # WRF:4878
        mask_17 = jnp.logical_and(mask_16, log_3momentice)
        _put(v, 'zitot', ((iice) - 1,), 0, mask_17)  # WRF:4879
        _put(v, 'diag_di', ((iice) - 1,), _r(0.0), mask_16)  # WRF:4880
    _put(v, 'diag_ze', None, _r(10.0) * jnp.log10((v['ze_rain'] + v['ze_ice']) * _r(1e+18)), alive)  # WRF:4887
    mask_18 = jnp.logical_and(alive, v['qr'] < v['qsmall'])
    taken_18 = mask_18
    _put(v, 'nr', None, _r(0.0), mask_18)  # WRF:4892
    return v
