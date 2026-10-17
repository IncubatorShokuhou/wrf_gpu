"""Generated literal P3 level-local processes; see proofs/s2_p3/port_processes.py."""
from .process_support import *  # noqa: F403


@_with_double
def processes(v, tables, *, ncat, log_predictnc, log_3momentice, observer=None):
    v = dict(v)
    iparam = 3
    scpf_on = log_predictssat = False
    niter_mui = 5
    log_hmossopon = ncat > 1
    alive = v['level_active']
    level_alive = alive
    v.setdefault('aaa', _initial(v['qv'], 'aaa', ncat))
    v.setdefault('aact', _initial(v['qv'], 'aact', ncat))
    v.setdefault('ab', _initial(v['qv'], 'ab', ncat))
    v.setdefault('abi', _initial(v['qv'], 'abi', ncat))
    v.setdefault('birim', _initial(v['qv'], 'birim', ncat))
    v.setdefault('cdist', _initial(v['qv'], 'cdist', ncat))
    v.setdefault('cdist1', _initial(v['qv'], 'cdist1', ncat))
    v.setdefault('cdistr', _initial(v['qv'], 'cdistr', ncat))
    v.setdefault('d_c', _initial(v['qv'], 'd_c', ncat))
    v.setdefault('d_new', _initial(v['qv'], 'd_new', ncat))
    v.setdefault('diam_ice', _initial(v['qv'], 'diam_ice', ncat))
    v.setdefault('dqsdt', _initial(v['qv'], 'dqsdt', ncat))
    v.setdefault('dqsidt', _initial(v['qv'], 'dqsidt', ncat))
    v.setdefault('dum', _initial(v['qv'], 'dum', ncat))
    v.setdefault('dum1', _initial(v['qv'], 'dum1', ncat))
    v.setdefault('dum1c', _initial(v['qv'], 'dum1c', ncat))
    v.setdefault('dum1z', _initial(v['qv'], 'dum1z', ncat))
    v.setdefault('dum2', _initial(v['qv'], 'dum2', ncat))
    v.setdefault('dum3', _initial(v['qv'], 'dum3', ncat))
    v.setdefault('dum4', _initial(v['qv'], 'dum4', ncat))
    v.setdefault('dum4c', _initial(v['qv'], 'dum4c', ncat))
    v.setdefault('dum5', _initial(v['qv'], 'dum5', ncat))
    v.setdefault('dum5c', _initial(v['qv'], 'dum5c', ncat))
    v.setdefault('dum6', _initial(v['qv'], 'dum6', ncat))
    v.setdefault('dumi', _initial(v['qv'], 'dumi', ncat))
    v.setdefault('dumic', _initial(v['qv'], 'dumic', ncat))
    v.setdefault('dumii', _initial(v['qv'], 'dumii', ncat))
    v.setdefault('dumiic', _initial(v['qv'], 'dumiic', ncat))
    v.setdefault('dumj', _initial(v['qv'], 'dumj', ncat))
    v.setdefault('dumjj', _initial(v['qv'], 'dumjj', ncat))
    v.setdefault('dumjjc', _initial(v['qv'], 'dumjjc', ncat))
    v.setdefault('dumm0', _initial(v['qv'], 'dumm0', ncat))
    v.setdefault('dumm3', _initial(v['qv'], 'dumm3', ncat))
    v.setdefault('dumqv', _initial(v['qv'], 'dumqv', ncat))
    v.setdefault('dumqvi', _initial(v['qv'], 'dumqvi', ncat))
    v.setdefault('dumqvs', _initial(v['qv'], 'dumqvs', ncat))
    v.setdefault('dums', _initial(v['qv'], 'dums', ncat))
    v.setdefault('dumt', _initial(v['qv'], 'dumt', ncat))
    v.setdefault('dumzz', _initial(v['qv'], 'dumzz', ncat))
    v.setdefault('dv', _initial(v['qv'], 'dv', ncat))
    v.setdefault('eii', _initial(v['qv'], 'eii', ncat))
    v.setdefault('eii_fact', _initial(v['qv'], 'eii_fact', ncat))
    v.setdefault('epsc', _initial(v['qv'], 'epsc', ncat))
    v.setdefault('epsi', _initial(v['qv'], 'epsi', ncat))
    v.setdefault('epsi_tot', _initial(v['qv'], 'epsi_tot', ncat))
    v.setdefault('epsilon', _initial(v['qv'], 'epsilon', ncat))
    v.setdefault('epsr', _initial(v['qv'], 'epsr', ncat))
    v.setdefault('f1pr02', _initial(v['qv'], 'f1pr02', ncat))
    v.setdefault('f1pr03', _initial(v['qv'], 'f1pr03', ncat))
    v.setdefault('f1pr04', _initial(v['qv'], 'f1pr04', ncat))
    v.setdefault('f1pr05', _initial(v['qv'], 'f1pr05', ncat))
    v.setdefault('f1pr07', _initial(v['qv'], 'f1pr07', ncat))
    v.setdefault('f1pr08', _initial(v['qv'], 'f1pr08', ncat))
    v.setdefault('f1pr09', _initial(v['qv'], 'f1pr09', ncat))
    v.setdefault('f1pr10', _initial(v['qv'], 'f1pr10', ncat))
    v.setdefault('f1pr14', _initial(v['qv'], 'f1pr14', ncat))
    v.setdefault('f1pr16', _initial(v['qv'], 'f1pr16', ncat))
    v.setdefault('f1pr17', _initial(v['qv'], 'f1pr17', ncat))
    v.setdefault('f1pr18', _initial(v['qv'], 'f1pr18', ncat))
    v.setdefault('iice_dest', _initial(v['qv'], 'iice_dest', ncat))
    v.setdefault('inv_dum3', _initial(v['qv'], 'inv_dum3', ncat))
    v.setdefault('itc', _initial(v['qv'], 'itc', ncat))
    v.setdefault('kap', _initial(v['qv'], 'kap', ncat))
    v.setdefault('lamc', _initial(v['qv'], 'lamc', ncat))
    v.setdefault('lammax', _initial(v['qv'], 'lammax', ncat))
    v.setdefault('lammin', _initial(v['qv'], 'lammin', ncat))
    v.setdefault('lamr', _initial(v['qv'], 'lamr', ncat))
    v.setdefault('log_exitlevel', _initial(v['qv'], 'log_exitlevel', ncat))
    v.setdefault('log_hydrometeorspresent', _initial(v['qv'], 'log_hydrometeorspresent', ncat))
    v.setdefault('log_wetgrowth', _initial(v['qv'], 'log_wetgrowth', ncat))
    v.setdefault('logn0r', _initial(v['qv'], 'logn0r', ncat))
    v.setdefault('mu', _initial(v['qv'], 'mu', ncat))
    v.setdefault('mu_c', _initial(v['qv'], 'mu_c', ncat))
    v.setdefault('mu_i', _initial(v['qv'], 'mu_i', ncat))
    v.setdefault('mu_i_new', _initial(v['qv'], 'mu_i_new', ncat))
    v.setdefault('mu_r', _initial(v['qv'], 'mu_r', ncat))
    v.setdefault('n_nuc', _initial(v['qv'], 'n_nuc', ncat))
    v.setdefault('nc', _initial(v['qv'], 'nc', ncat))
    v.setdefault('ncacc', _initial(v['qv'], 'ncacc', ncat))
    v.setdefault('ncautc', _initial(v['qv'], 'ncautc', ncat))
    v.setdefault('ncautr', _initial(v['qv'], 'ncautr', ncat))
    v.setdefault('nccol', _initial(v['qv'], 'nccol', ncat))
    v.setdefault('nchetc', _initial(v['qv'], 'nchetc', ncat))
    v.setdefault('ncheti', _initial(v['qv'], 'ncheti', ncat))
    v.setdefault('ncnuc', _initial(v['qv'], 'ncnuc', ncat))
    v.setdefault('ncshdc', _initial(v['qv'], 'ncshdc', ncat))
    v.setdefault('ncslf', _initial(v['qv'], 'ncslf', ncat))
    v.setdefault('nicol', _initial(v['qv'], 'nicol', ncat))
    v.setdefault('nimlt', _initial(v['qv'], 'nimlt', ncat))
    v.setdefault('nimul', _initial(v['qv'], 'nimul', ncat))
    v.setdefault('ninuc', _initial(v['qv'], 'ninuc', ncat))
    v.setdefault('nislf', _initial(v['qv'], 'nislf', ncat))
    v.setdefault('nisub', _initial(v['qv'], 'nisub', ncat))
    v.setdefault('nitot', _initial(v['qv'], 'nitot', ncat))
    v.setdefault('nr', _initial(v['qv'], 'nr', ncat))
    v.setdefault('nrcol', _initial(v['qv'], 'nrcol', ncat))
    v.setdefault('nrevp', _initial(v['qv'], 'nrevp', ncat))
    v.setdefault('nrhetc', _initial(v['qv'], 'nrhetc', ncat))
    v.setdefault('nrheti', _initial(v['qv'], 'nrheti', ncat))
    v.setdefault('nrshdr', _initial(v['qv'], 'nrshdr', ncat))
    v.setdefault('nrslf', _initial(v['qv'], 'nrslf', ncat))
    v.setdefault('nu', _initial(v['qv'], 'nu', ncat))
    v.setdefault('oabi', _initial(v['qv'], 'oabi', ncat))
    v.setdefault('oxx', _initial(v['qv'], 'oxx', ncat))
    v.setdefault('q_nuc', _initial(v['qv'], 'q_nuc', ncat))
    v.setdefault('qc', _initial(v['qv'], 'qc', ncat))
    v.setdefault('qcacc', _initial(v['qv'], 'qcacc', ncat))
    v.setdefault('qcaut', _initial(v['qv'], 'qcaut', ncat))
    v.setdefault('qccol', _initial(v['qv'], 'qccol', ncat))
    v.setdefault('qccon', _initial(v['qv'], 'qccon', ncat))
    v.setdefault('qcevp', _initial(v['qv'], 'qcevp', ncat))
    v.setdefault('qchetc', _initial(v['qv'], 'qchetc', ncat))
    v.setdefault('qcheti', _initial(v['qv'], 'qcheti', ncat))
    v.setdefault('qcnuc', _initial(v['qv'], 'qcnuc', ncat))
    v.setdefault('qcon_satadj', _initial(v['qv'], 'qcon_satadj', ncat))
    v.setdefault('qcshd', _initial(v['qv'], 'qcshd', ncat))
    v.setdefault('qdep_satadj', _initial(v['qv'], 'qdep_satadj', ncat))
    v.setdefault('qicol', _initial(v['qv'], 'qicol', ncat))
    v.setdefault('qidep', _initial(v['qv'], 'qidep', ncat))
    v.setdefault('qimlt', _initial(v['qv'], 'qimlt', ncat))
    v.setdefault('qinuc', _initial(v['qv'], 'qinuc', ncat))
    v.setdefault('qirim', _initial(v['qv'], 'qirim', ncat))
    v.setdefault('qisub', _initial(v['qv'], 'qisub', ncat))
    v.setdefault('qitot', _initial(v['qv'], 'qitot', ncat))
    v.setdefault('qr', _initial(v['qv'], 'qr', ncat))
    v.setdefault('qrcol', _initial(v['qv'], 'qrcol', ncat))
    v.setdefault('qrcon', _initial(v['qv'], 'qrcon', ncat))
    v.setdefault('qrevp', _initial(v['qv'], 'qrevp', ncat))
    v.setdefault('qrhetc', _initial(v['qv'], 'qrhetc', ncat))
    v.setdefault('qrheti', _initial(v['qv'], 'qrheti', ncat))
    v.setdefault('qrmul', _initial(v['qv'], 'qrmul', ncat))
    v.setdefault('qsat0', _initial(v['qv'], 'qsat0', ncat))
    v.setdefault('qv', _initial(v['qv'], 'qv', ncat))
    v.setdefault('qv_tmp', _initial(v['qv'], 'qv_tmp', ncat))
    v.setdefault('qvi', _initial(v['qv'], 'qvi', ncat))
    v.setdefault('qvs', _initial(v['qv'], 'qvs', ncat))
    v.setdefault('qwgrth', _initial(v['qv'], 'qwgrth', ncat))
    v.setdefault('ratio', _initial(v['qv'], 'ratio', ncat))
    v.setdefault('rdumii', _initial(v['qv'], 'rdumii', ncat))
    v.setdefault('rdumjj', _initial(v['qv'], 'rdumjj', ncat))
    v.setdefault('rhop', _initial(v['qv'], 'rhop', ncat))
    v.setdefault('rhorime_c', _initial(v['qv'], 'rhorime_c', ncat))
    v.setdefault('ri', _initial(v['qv'], 'ri', ncat))
    v.setdefault('rimefraction', _initial(v['qv'], 'rimefraction', ncat))
    v.setdefault('rimevolume', _initial(v['qv'], 'rimevolume', ncat))
    v.setdefault('sc', _initial(v['qv'], 'sc', ncat))
    v.setdefault('sigvl', _initial(v['qv'], 'sigvl', ncat))
    v.setdefault('sinks', _initial(v['qv'], 'sinks', ncat))
    v.setdefault('sm1', _initial(v['qv'], 'sm1', ncat))
    v.setdefault('sm2', _initial(v['qv'], 'sm2', ncat))
    v.setdefault('sources', _initial(v['qv'], 'sources', ncat))
    v.setdefault('ssat', _initial(v['qv'], 'ssat', ncat))
    v.setdefault('ssat_cld', _initial(v['qv'], 'ssat_cld', ncat))
    v.setdefault('ssat_clr', _initial(v['qv'], 'ssat_clr', ncat))
    v.setdefault('ssat_r', _initial(v['qv'], 'ssat_r', ncat))
    v.setdefault('sup', _initial(v['qv'], 'sup', ncat))
    v.setdefault('sup_cld', _initial(v['qv'], 'sup_cld', ncat))
    v.setdefault('sup_r', _initial(v['qv'], 'sup_r', ncat))
    v.setdefault('supi', _initial(v['qv'], 'supi', ncat))
    v.setdefault('supi_cld', _initial(v['qv'], 'supi_cld', ncat))
    v.setdefault('t', _initial(v['qv'], 't', ncat))
    v.setdefault('t_tmp', _initial(v['qv'], 't_tmp', ncat))
    v.setdefault('th', _initial(v['qv'], 'th', ncat))
    v.setdefault('tmp1', _initial(v['qv'], 'tmp1', ncat))
    v.setdefault('tmp2', _initial(v['qv'], 'tmp2', ncat))
    v.setdefault('tmpdbl1', _initial(v['qv'], 'tmpdbl1', ncat))
    v.setdefault('tmpdbl2', _initial(v['qv'], 'tmpdbl2', ncat))
    v.setdefault('tmpdbl3', _initial(v['qv'], 'tmpdbl3', ncat))
    v.setdefault('uu1', _initial(v['qv'], 'uu1', ncat))
    v.setdefault('uu2', _initial(v['qv'], 'uu2', ncat))
    v.setdefault('v_impact', _initial(v['qv'], 'v_impact', ncat))
    v.setdefault('vt_qc', _initial(v['qv'], 'vt_qc', ncat))
    v.setdefault('vtrmi1', _initial(v['qv'], 'vtrmi1', ncat))
    v.setdefault('xx', _initial(v['qv'], 'xx', ncat))
    v.setdefault('zitot', _initial(v['qv'], 'zitot', ncat))
    assert 'acn' in v, 'P3 required WRF input/constant missing: acn'
    assert 'aimm' in v, 'P3 required WRF input/constant missing: aimm'
    assert 'bact' in v, 'P3 required WRF input/constant missing: bact'
    assert 'bcn' in v, 'P3 required WRF input/constant missing: bcn'
    assert 'bsmall' in v, 'P3 required WRF input/constant missing: bsmall'
    assert 'clbfact_dep' in v, 'P3 required WRF input/constant missing: clbfact_dep'
    assert 'clbfact_sub' in v, 'P3 required WRF input/constant missing: clbfact_sub'
    assert 'cons3' in v, 'P3 required WRF input/constant missing: cons3'
    assert 'cons5' in v, 'P3 required WRF input/constant missing: cons5'
    assert 'cons6' in v, 'P3 required WRF input/constant missing: cons6'
    assert 'cons7' in v, 'P3 required WRF input/constant missing: cons7'
    assert 'cons8' in v, 'P3 required WRF input/constant missing: cons8'
    assert 'cp' in v, 'P3 required WRF input/constant missing: cp'
    assert 'cpw' in v, 'P3 required WRF input/constant missing: cpw'
    assert 'deltad_init' in v, 'P3 required WRF input/constant missing: deltad_init'
    assert 'densize' in v, 'P3 required WRF input/constant missing: densize'
    assert 'dnu' in v, 'P3 required WRF input/constant missing: dnu'
    assert 'dt' in v, 'P3 required WRF input/constant missing: dt'
    assert 'e0' in v, 'P3 required WRF input/constant missing: e0'
    assert 'eci' in v, 'P3 required WRF input/constant missing: eci'
    assert 'eri' in v, 'P3 required WRF input/constant missing: eri'
    assert 'f1r' in v, 'P3 required WRF input/constant missing: f1r'
    assert 'f2r' in v, 'P3 required WRF input/constant missing: f2r'
    assert 'g' in v, 'P3 required WRF input/constant missing: g'
    assert 'iisize' in v, 'P3 required WRF input/constant missing: iisize'
    assert 'inv_cp' in v, 'P3 required WRF input/constant missing: inv_cp'
    assert 'inv_rho' in v, 'P3 required WRF input/constant missing: inv_rho'
    assert 'inv_rho_rimemax' in v, 'P3 required WRF input/constant missing: inv_rho_rimemax'
    assert 'inv_rm1' in v, 'P3 required WRF input/constant missing: inv_rm1'
    assert 'inv_rm2' in v, 'P3 required WRF input/constant missing: inv_rm2'
    assert 'invexn' in v, 'P3 required WRF input/constant missing: invexn'
    assert 'iscf' in v, 'P3 required WRF input/constant missing: iscf'
    assert 'isize' in v, 'P3 required WRF input/constant missing: isize'
    assert 'ispf' in v, 'P3 required WRF input/constant missing: ispf'
    assert 'it' in v, 'P3 required WRF input/constant missing: it'
    assert 'kc' in v, 'P3 required WRF input/constant missing: kc'
    assert 'kr' in v, 'P3 required WRF input/constant missing: kr'
    assert 'max_total_ni' in v, 'P3 required WRF input/constant missing: max_total_ni'
    assert 'mi0' in v, 'P3 required WRF input/constant missing: mi0'
    assert 'mu_i_initial' in v, 'P3 required WRF input/constant missing: mu_i_initial'
    assert 'mu_i_max' in v, 'P3 required WRF input/constant missing: mu_i_max'
    assert 'mw' in v, 'P3 required WRF input/constant missing: mw'
    assert 'nanew1' in v, 'P3 required WRF input/constant missing: nanew1'
    assert 'nanew2' in v, 'P3 required WRF input/constant missing: nanew2'
    assert 'nccnst' in v, 'P3 required WRF input/constant missing: nccnst'
    assert 'nmltratio' in v, 'P3 required WRF input/constant missing: nmltratio'
    assert 'nsmall' in v, 'P3 required WRF input/constant missing: nsmall'
    assert 'odt' in v, 'P3 required WRF input/constant missing: odt'
    assert 'pi' in v, 'P3 required WRF input/constant missing: pi'
    assert 'piov6' in v, 'P3 required WRF input/constant missing: piov6'
    assert 'pres' in v, 'P3 required WRF input/constant missing: pres'
    assert 'qsmall' in v, 'P3 required WRF input/constant missing: qsmall'
    assert 'qv_cld' in v, 'P3 required WRF input/constant missing: qv_cld'
    assert 'qv_clr' in v, 'P3 required WRF input/constant missing: qv_clr'
    assert 'qv_old' in v, 'P3 required WRF input/constant missing: qv_old'
    assert 'rcollsize' in v, 'P3 required WRF input/constant missing: rcollsize'
    assert 'rd' in v, 'P3 required WRF input/constant missing: rd'
    assert 'revap_table' in v, 'P3 required WRF input/constant missing: revap_table'
    assert 'rho' in v, 'P3 required WRF input/constant missing: rho'
    assert 'rhofaci' in v, 'P3 required WRF input/constant missing: rhofaci'
    assert 'rhow' in v, 'P3 required WRF input/constant missing: rhow'
    assert 'rimsize' in v, 'P3 required WRF input/constant missing: rimsize'
    assert 'rr' in v, 'P3 required WRF input/constant missing: rr'
    assert 'rv' in v, 'P3 required WRF input/constant missing: rv'
    assert 'scf' in v, 'P3 required WRF input/constant missing: scf'
    assert 'sig1' in v, 'P3 required WRF input/constant missing: sig1'
    assert 'sig2' in v, 'P3 required WRF input/constant missing: sig2'
    assert 'spf' in v, 'P3 required WRF input/constant missing: spf'
    assert 'spf_clr' in v, 'P3 required WRF input/constant missing: spf_clr'
    assert 't_old' in v, 'P3 required WRF input/constant missing: t_old'
    assert 'thrd' in v, 'P3 required WRF input/constant missing: thrd'
    assert 'xlf' in v, 'P3 required WRF input/constant missing: xlf'
    assert 'xxls' in v, 'P3 required WRF input/constant missing: xxls'
    assert 'xxlv' in v, 'P3 required WRF input/constant missing: xxlv'
    assert 'zsize' in v, 'P3 required WRF input/constant missing: zsize'
    assert 'zsmall' in v, 'P3 required WRF input/constant missing: zsmall'
    _put(v, 'log_exitlevel', None, True, alive)  # WRF:2453
    mask_1 = jnp.logical_and(alive, jnp.logical_or(v['qc'] >= v['qsmall'], v['qr'] >= v['qsmall']))
    _put(v, 'log_exitlevel', None, False, mask_1)  # WRF:2454
    for iice in range(1, (ncat) + (1 if (1) > 0 else -1), 1):
        mask_2 = jnp.logical_and(alive, _get(v['qitot'], (iice - 1,)) >= v['qsmall'])
        _put(v, 'log_exitlevel', None, False, mask_2)  # WRF:2456
    mask_3 = jnp.logical_and(alive, jnp.logical_and(jnp.logical_and(scpf_on, v['log_exitlevel']), v['scf'] < _r(0.01)))
    alive = jnp.logical_and(alive, jnp.logical_not(mask_3))
    level_alive = alive
    mask_4 = jnp.logical_and(alive, jnp.logical_and(jnp.logical_and(not scpf_on, v['log_exitlevel']), jnp.logical_or(jnp.logical_and(v['t'] < _r(273.15), v['supi'] < _r(-0.05)), jnp.logical_and(v['t'] >= _r(273.15), v['sup'] < _r(-0.05)))))
    alive = jnp.logical_and(alive, jnp.logical_not(mask_4))
    level_alive = alive
    _put(v, 'qcacc', None, _r(0.0), alive)  # WRF:2467
    _put(v, 'qrevp', None, _r(0.0), alive)  # WRF:2467
    _put(v, 'qccon', None, _r(0.0), alive)  # WRF:2467
    _put(v, 'qcaut', None, _r(0.0), alive)  # WRF:2468
    _put(v, 'qcevp', None, _r(0.0), alive)  # WRF:2468
    _put(v, 'qrcon', None, _r(0.0), alive)  # WRF:2468
    _put(v, 'ncacc', None, _r(0.0), alive)  # WRF:2469
    _put(v, 'ncnuc', None, _r(0.0), alive)  # WRF:2469
    _put(v, 'ncslf', None, _r(0.0), alive)  # WRF:2469
    _put(v, 'ncautc', None, _r(0.0), alive)  # WRF:2470
    _put(v, 'qcnuc', None, _r(0.0), alive)  # WRF:2470
    _put(v, 'nrslf', None, _r(0.0), alive)  # WRF:2470
    _put(v, 'nrevp', None, _r(0.0), alive)  # WRF:2471
    _put(v, 'ncautr', None, _r(0.0), alive)  # WRF:2471
    _put(v, 'qchetc', None, _r(0.0), alive)  # WRF:2474
    _put(v, 'qisub', None, _r(0.0), alive)  # WRF:2474
    _put(v, 'nrshdr', None, _r(0.0), alive)  # WRF:2474
    _put(v, 'qcheti', None, _r(0.0), alive)  # WRF:2475
    _put(v, 'qrcol', None, _r(0.0), alive)  # WRF:2475
    _put(v, 'qcshd', None, _r(0.0), alive)  # WRF:2475
    _put(v, 'qrhetc', None, _r(0.0), alive)  # WRF:2476
    _put(v, 'qimlt', None, _r(0.0), alive)  # WRF:2476
    _put(v, 'qccol', None, _r(0.0), alive)  # WRF:2476
    _put(v, 'qrheti', None, _r(0.0), alive)  # WRF:2477
    _put(v, 'qinuc', None, _r(0.0), alive)  # WRF:2477
    _put(v, 'nimlt', None, _r(0.0), alive)  # WRF:2477
    _put(v, 'nchetc', None, _r(0.0), alive)  # WRF:2478
    _put(v, 'nccol', None, _r(0.0), alive)  # WRF:2478
    _put(v, 'ncshdc', None, _r(0.0), alive)  # WRF:2478
    _put(v, 'ncheti', None, _r(0.0), alive)  # WRF:2479
    _put(v, 'nrcol', None, _r(0.0), alive)  # WRF:2479
    _put(v, 'nislf', None, _r(0.0), alive)  # WRF:2479
    _put(v, 'nrhetc', None, _r(0.0), alive)  # WRF:2480
    _put(v, 'ninuc', None, _r(0.0), alive)  # WRF:2480
    _put(v, 'qidep', None, _r(0.0), alive)  # WRF:2480
    _put(v, 'nrheti', None, _r(0.0), alive)  # WRF:2481
    _put(v, 'nisub', None, _r(0.0), alive)  # WRF:2481
    _put(v, 'qwgrth', None, _r(0.0), alive)  # WRF:2481
    _put(v, 'qrmul', None, _r(0.0), alive)  # WRF:2482
    _put(v, 'nimul', None, _r(0.0), alive)  # WRF:2482
    _put(v, 'qicol', None, _r(0.0), alive)  # WRF:2482
    _put(v, 'nicol', None, _r(0.0), alive)  # WRF:2483
    _put(v, 'log_wetgrowth', None, False, alive)  # WRF:2485
    if log_predictssat:
        _put(v, 'dqsdt', None, v['xxlv'] * v['qvs'] / (v['rv'] * v['t'] * v['t']), alive)  # WRF:2496
        _put(v, 'ab', None, _r(1.0) + v['dqsdt'] * v['xxlv'] * v['inv_cp'], alive)  # WRF:2497
        _put(v, 'epsilon', None, (v['qv'] - v['qvs'] - v['ssat']) / v['ab'], alive)  # WRF:2498
        _put(v, 'epsilon', None, _max(v['epsilon'], -v['qc']), alive)  # WRF:2499
        mask_5 = jnp.logical_and(alive, v['ssat'] < _r(0.0))
        _put(v, 'epsilon', None, _min(_r(0.0), v['epsilon']), mask_5)  # WRF:2503
        mask_6 = jnp.logical_and(alive, jnp.abs(v['epsilon']) >= _r(1e-15))
        taken_6 = mask_6
        _put(v, 'qc', None, v['qc'] + v['epsilon'], mask_6)  # WRF:2507
        _put(v, 'qv', None, v['qv'] - v['epsilon'], mask_6)  # WRF:2508
        _put(v, 'th', None, v['th'] + v['epsilon'] * v['invexn'] * v['xxlv'] * v['inv_cp'], mask_6)  # WRF:2509
        _put(v, 't', None, v['th'] * (_r(1e-05) * v['pres']) ** (v['rd'] * v['inv_cp']), mask_6)  # WRF:2511
        _put(v, 'qvs', None, qv_sat(v['t'], v['pres'], 0), mask_6)  # WRF:2512
        _put(v, 'qvi', None, qv_sat(v['t'], v['pres'], 1), mask_6)  # WRF:2513
        _put(v, 'sup', None, v['qv'] / v['qvs'] - _r(1.0), mask_6)  # WRF:2514
        _put(v, 'supi', None, v['qv'] / v['qvi'] - _r(1.0), mask_6)  # WRF:2515
        _put(v, 'ssat', None, v['qv'] - v['qvs'], mask_6)  # WRF:2516
    _put(v, 'log_exitlevel', None, True, alive)  # WRF:2523
    mask_7 = jnp.logical_and(alive, jnp.logical_or(v['qc'] >= v['qsmall'], v['qr'] >= v['qsmall']))
    _put(v, 'log_exitlevel', None, False, mask_7)  # WRF:2524
    for iice in range(1, (ncat) + (1 if (1) > 0 else -1), 1):
        mask_8 = jnp.logical_and(alive, _get(v['qitot'], (iice - 1,)) >= v['qsmall'])
        _put(v, 'log_exitlevel', None, False, mask_8)  # WRF:2526
    mask_9 = jnp.logical_and(alive, v['log_exitlevel'])
    alive = jnp.logical_and(alive, jnp.logical_not(mask_9))
    _put(v, 'mu', None, _r(1.496e-06) * v['t'] ** _r(1.5) / (v['t'] + _r(120.0)), alive)  # WRF:2531
    _put(v, 'dv', None, _r(8.794e-05) * v['t'] ** _r(1.81) / v['pres'], alive)  # WRF:2532
    _put(v, 'sc', None, v['mu'] / (v['rho'] * v['dv']), alive)  # WRF:2533
    _put(v, 'dum', None, _r(1.0) / (v['rv'] * v['t'] ** 2), alive)  # WRF:2534
    _put(v, 'dqsdt', None, v['xxlv'] * v['qvs'] * v['dum'], alive)  # WRF:2535
    _put(v, 'dqsidt', None, v['xxls'] * v['qvi'] * v['dum'], alive)  # WRF:2536
    _put(v, 'ab', None, _r(1.0) + v['dqsdt'] * v['xxlv'] * v['inv_cp'], alive)  # WRF:2537
    _put(v, 'abi', None, _r(1.0) + v['dqsidt'] * v['xxls'] * v['inv_cp'], alive)  # WRF:2538
    _put(v, 'kap', None, _r(1414.0) * v['mu'], alive)  # WRF:2539
    mask_10 = jnp.logical_and(alive, v['t'] < _r(253.15))
    taken_10 = mask_10
    _put(v, 'eii', None, _r(0.001), mask_10)  # WRF:2549
    mask_11 = jnp.logical_and(jnp.logical_and(alive, jnp.logical_not(taken_10)), jnp.logical_and(v['t'] >= _r(253.15), v['t'] < _r(273.15)))
    taken_10 = jnp.logical_or(taken_10, mask_11)
    _put(v, 'eii', None, _r(0.001) + (v['t'] - _r(253.15)) * (_r(0.3) - _r(0.001)) / _r(20.0), mask_11)  # WRF:2551
    mask_12 = jnp.logical_and(jnp.logical_and(alive, jnp.logical_not(taken_10)), v['t'] >= _r(273.15))
    taken_10 = jnp.logical_or(taken_10, mask_12)
    _put(v, 'eii', None, _r(0.3), mask_12)  # WRF:2553
    call_out = _call_get_cloud_dsd2(v['qc'], v['nc'], v['mu_c'], v['rho'], v['nu'], v['dnu'], v['lamc'], v['lammin'], v['lammax'], v['cdist'], v['cdist1'], v['iscf'], tables=tables, three=log_3momentice)
    _put(v, 'nc', None, call_out[0], alive)  # WRF:2556
    _put(v, 'mu_c', None, call_out[1], alive)  # WRF:2556
    _put(v, 'nu', None, call_out[2], alive)  # WRF:2556
    _put(v, 'lamc', None, call_out[3], alive)  # WRF:2556
    _put(v, 'lammin', None, call_out[4], alive)  # WRF:2556
    _put(v, 'lammax', None, call_out[5], alive)  # WRF:2556
    _put(v, 'cdist', None, call_out[6], alive)  # WRF:2556
    _put(v, 'cdist1', None, call_out[7], alive)  # WRF:2556
    call_out = _call_get_rain_dsd2(v['qr'], v['nr'], v['mu_r'], v['lamr'], v['cdistr'], v['logn0r'], v['ispf'], tables=tables, three=log_3momentice)
    _put(v, 'nr', None, call_out[0], alive)  # WRF:2560
    _put(v, 'mu_r', None, call_out[1], alive)  # WRF:2560
    _put(v, 'lamr', None, call_out[2], alive)  # WRF:2560
    _put(v, 'cdistr', None, call_out[3], alive)  # WRF:2560
    _put(v, 'logn0r', None, call_out[4], alive)  # WRF:2560
    _put(v, 'epsi_tot', None, _r(0.0), alive)  # WRF:2564
    call_out = _call_impose_max_total_ni(v['nitot'], v['max_total_ni'], v['inv_rho'], tables=tables, three=log_3momentice)
    _put(v, 'nitot', None, call_out[0], alive)  # WRF:2566
    for iice in range(1, (ncat) + (1 if (1) > 0 else -1), 1):
        mask_13 = jnp.logical_and(alive, _get(v['qitot'], (iice - 1,)) >= v['qsmall'])
        taken_13 = mask_13
        _put(v, 'nitot', ((iice) - 1,), _max(_get(v['nitot'], (iice - 1,)), v['nsmall']), mask_13)  # WRF:2573
        _put(v, 'nr', None, _max(v['nr'], v['nsmall']), mask_13)  # WRF:2574
        _put(v, 'dum2', None, _r(500.0), mask_13)  # WRF:2577
        _put(v, 'diam_ice', ((iice) - 1,), (_get(v['qitot'], (iice - 1,)) * _r(6.0) / (_get(v['nitot'], (iice - 1,)) * v['dum2'] * v['pi'])) ** v['thrd'], mask_13)  # WRF:2578
        call_out = _call_calc_bulkrhorime(_get(v['qitot'], (iice - 1,)), _get(v['qirim'], (iice - 1,)), _get(v['birim'], (iice - 1,)), v['rhop'], tables=tables, three=log_3momentice)
        _put(v, 'qirim', ((iice) - 1,), call_out[0], mask_13)  # WRF:2580
        _put(v, 'birim', ((iice) - 1,), call_out[1], mask_13)  # WRF:2580
        _put(v, 'rhop', None, call_out[2], mask_13)  # WRF:2580
        call_out = _call_find_lookuptable_indices_1a(v['dumi'], v['dumjj'], v['dumii'], v['dum1'], v['dum4'], v['dum5'], v['isize'], v['rimsize'], v['densize'], _get(v['qitot'], (iice - 1,)), _get(v['nitot'], (iice - 1,)), _get(v['qirim'], (iice - 1,)), v['rhop'], tables=tables, three=log_3momentice)
        _put(v, 'dumi', None, call_out[0], mask_13)  # WRF:2582
        _put(v, 'dumjj', None, call_out[1], mask_13)  # WRF:2582
        _put(v, 'dumii', None, call_out[2], mask_13)  # WRF:2582
        _put(v, 'dum1', None, call_out[3], mask_13)  # WRF:2582
        _put(v, 'dum4', None, call_out[4], mask_13)  # WRF:2582
        _put(v, 'dum5', None, call_out[5], mask_13)  # WRF:2582
        call_out = _call_find_lookuptable_indices_1b(v['dumj'], v['dum3'], v['rcollsize'], v['qr'], v['nr'], tables=tables, three=log_3momentice)
        _put(v, 'dumj', None, call_out[0], mask_13)  # WRF:2585
        _put(v, 'dum3', None, call_out[1], mask_13)  # WRF:2585
        if not log_3momentice:
            call_out = _call_access_lookup_table(v['dumjj'], v['dumii'], v['dumi'], 2, v['dum1'], v['dum4'], v['dum5'], v['f1pr02'], tables=tables, three=log_3momentice)
            _put(v, 'f1pr02', None, call_out[0], mask_13)  # WRF:2590
            call_out = _call_access_lookup_table(v['dumjj'], v['dumii'], v['dumi'], 3, v['dum1'], v['dum4'], v['dum5'], v['f1pr03'], tables=tables, three=log_3momentice)
            _put(v, 'f1pr03', None, call_out[0], mask_13)  # WRF:2591
            call_out = _call_access_lookup_table(v['dumjj'], v['dumii'], v['dumi'], 4, v['dum1'], v['dum4'], v['dum5'], v['f1pr04'], tables=tables, three=log_3momentice)
            _put(v, 'f1pr04', None, call_out[0], mask_13)  # WRF:2592
            call_out = _call_access_lookup_table(v['dumjj'], v['dumii'], v['dumi'], 5, v['dum1'], v['dum4'], v['dum5'], v['f1pr05'], tables=tables, three=log_3momentice)
            _put(v, 'f1pr05', None, call_out[0], mask_13)  # WRF:2593
            call_out = _call_access_lookup_table(v['dumjj'], v['dumii'], v['dumi'], 7, v['dum1'], v['dum4'], v['dum5'], v['f1pr09'], tables=tables, three=log_3momentice)
            _put(v, 'f1pr09', None, call_out[0], mask_13)  # WRF:2594
            call_out = _call_access_lookup_table(v['dumjj'], v['dumii'], v['dumi'], 8, v['dum1'], v['dum4'], v['dum5'], v['f1pr10'], tables=tables, three=log_3momentice)
            _put(v, 'f1pr10', None, call_out[0], mask_13)  # WRF:2595
            call_out = _call_access_lookup_table(v['dumjj'], v['dumii'], v['dumi'], 10, v['dum1'], v['dum4'], v['dum5'], v['f1pr14'], tables=tables, three=log_3momentice)
            _put(v, 'f1pr14', None, call_out[0], mask_13)  # WRF:2596
            mask_14 = jnp.logical_and(mask_13, v['qr'] >= v['qsmall'])
            taken_14 = mask_14
            call_out = _call_access_lookup_table_coll(v['dumjj'], v['dumii'], v['dumj'], v['dumi'], 1, v['dum1'], v['dum3'], v['dum4'], v['dum5'], v['f1pr07'], tables=tables, three=log_3momentice)
            _put(v, 'f1pr07', None, call_out[0], mask_14)  # WRF:2600
            call_out = _call_access_lookup_table_coll(v['dumjj'], v['dumii'], v['dumj'], v['dumi'], 2, v['dum1'], v['dum3'], v['dum4'], v['dum5'], v['f1pr08'], tables=tables, three=log_3momentice)
            _put(v, 'f1pr08', None, call_out[0], mask_14)  # WRF:2601
            mask_15 = jnp.logical_and(mask_13, jnp.logical_not(taken_14))
            _put(v, 'f1pr07', None, _r(-99.0), mask_15)  # WRF:2603
            _put(v, 'f1pr08', None, _r(-99.0), mask_15)  # WRF:2604
        else:
            _put(v, 'zitot', ((iice) - 1,), _max(_get(v['zitot'], (iice - 1,)), v['zsmall']), mask_13)  # WRF:2612
            _put(v, 'dum1z', None, _r(6.0) / (_r(200.0) * v['pi']) * _get(v['qitot'], (iice - 1,)), mask_13)  # WRF:2614
            for imu in range(1, (niter_mui) + (1 if (1) > 0 else -1), 1):
                _put(v, 'mu_i', None, compute_mu_3moment(_get(v['nitot'], (iice - 1,)), v['dum1z'], _get(v['zitot'], (iice - 1,)), v['mu_i_max']), mask_13)  # WRF:2617
                call_out = _call_find_lookuptable_indices_1c(v['dumzz'], v['dum6'], v['zsize'], _get(v['qitot'], (iice - 1,)), v['mu_i'], tables=tables, three=log_3momentice)
                _put(v, 'dumzz', None, call_out[0], mask_13)  # WRF:2618
                _put(v, 'dum6', None, call_out[1], mask_13)  # WRF:2618
                call_out = _call_access_lookup_table_3mom(v['dumzz'], v['dumjj'], v['dumii'], v['dumi'], 12, v['dum1'], v['dum4'], v['dum5'], v['dum6'], v['f1pr16'], tables=tables, three=log_3momentice)
                _put(v, 'f1pr16', None, call_out[0], mask_13)  # WRF:2619
                _put(v, 'dum1z', None, _r(6.0) / (v['f1pr16'] * v['pi']) * _get(v['qitot'], (iice - 1,)), mask_13)  # WRF:2620
            call_out = _call_access_lookup_table_3mom(v['dumzz'], v['dumjj'], v['dumii'], v['dumi'], 2, v['dum1'], v['dum4'], v['dum5'], v['dum6'], v['f1pr02'], tables=tables, three=log_3momentice)
            _put(v, 'f1pr02', None, call_out[0], mask_13)  # WRF:2626
            call_out = _call_access_lookup_table_3mom(v['dumzz'], v['dumjj'], v['dumii'], v['dumi'], 3, v['dum1'], v['dum4'], v['dum5'], v['dum6'], v['f1pr03'], tables=tables, three=log_3momentice)
            _put(v, 'f1pr03', None, call_out[0], mask_13)  # WRF:2627
            call_out = _call_access_lookup_table_3mom(v['dumzz'], v['dumjj'], v['dumii'], v['dumi'], 4, v['dum1'], v['dum4'], v['dum5'], v['dum6'], v['f1pr04'], tables=tables, three=log_3momentice)
            _put(v, 'f1pr04', None, call_out[0], mask_13)  # WRF:2628
            call_out = _call_access_lookup_table_3mom(v['dumzz'], v['dumjj'], v['dumii'], v['dumi'], 5, v['dum1'], v['dum4'], v['dum5'], v['dum6'], v['f1pr05'], tables=tables, three=log_3momentice)
            _put(v, 'f1pr05', None, call_out[0], mask_13)  # WRF:2629
            call_out = _call_access_lookup_table_3mom(v['dumzz'], v['dumjj'], v['dumii'], v['dumi'], 7, v['dum1'], v['dum4'], v['dum5'], v['dum6'], v['f1pr09'], tables=tables, three=log_3momentice)
            _put(v, 'f1pr09', None, call_out[0], mask_13)  # WRF:2630
            call_out = _call_access_lookup_table_3mom(v['dumzz'], v['dumjj'], v['dumii'], v['dumi'], 8, v['dum1'], v['dum4'], v['dum5'], v['dum6'], v['f1pr10'], tables=tables, three=log_3momentice)
            _put(v, 'f1pr10', None, call_out[0], mask_13)  # WRF:2631
            call_out = _call_access_lookup_table_3mom(v['dumzz'], v['dumjj'], v['dumii'], v['dumi'], 10, v['dum1'], v['dum4'], v['dum5'], v['dum6'], v['f1pr14'], tables=tables, three=log_3momentice)
            _put(v, 'f1pr14', None, call_out[0], mask_13)  # WRF:2632
            mask_16 = jnp.logical_and(mask_13, v['qr'] >= v['qsmall'])
            taken_16 = mask_16
            call_out = _call_access_lookup_table_coll_3mom(v['dumzz'], v['dumjj'], v['dumii'], v['dumj'], v['dumi'], 1, v['dum1'], v['dum3'], v['dum4'], v['dum5'], v['dum6'], v['f1pr07'], tables=tables, three=log_3momentice)
            _put(v, 'f1pr07', None, call_out[0], mask_16)  # WRF:2637
            call_out = _call_access_lookup_table_coll_3mom(v['dumzz'], v['dumjj'], v['dumii'], v['dumj'], v['dumi'], 2, v['dum1'], v['dum3'], v['dum4'], v['dum5'], v['dum6'], v['f1pr08'], tables=tables, three=log_3momentice)
            _put(v, 'f1pr08', None, call_out[0], mask_16)  # WRF:2638
            mask_17 = jnp.logical_and(mask_13, jnp.logical_not(taken_16))
            _put(v, 'f1pr07', None, _r(-99.0), mask_17)  # WRF:2640
            _put(v, 'f1pr08', None, _r(-99.0), mask_17)  # WRF:2641
        _put(v, 'nitot', ((iice) - 1,), _min(_get(v['nitot'], (iice - 1,)), v['f1pr09'] * _get(v['qitot'], (iice - 1,))), mask_13)  # WRF:2649
        _put(v, 'nitot', ((iice) - 1,), _max(_get(v['nitot'], (iice - 1,)), v['f1pr10'] * _get(v['qitot'], (iice - 1,))), mask_13)  # WRF:2650
        if log_3momentice:
            _put(v, 'dum1', None, _r(6.0) / (v['f1pr16'] * v['pi']) * _get(v['qitot'], (iice - 1,)), mask_13)  # WRF:2655
            _put(v, 'tmp1', None, g_of_mu(_r(0.0)), mask_13)  # WRF:2656
            _put(v, 'tmp2', None, g_of_mu(_r(20.0)), mask_13)  # WRF:2657
            _put(v, 'zitot', ((iice) - 1,), _min(_get(v['zitot'], (iice - 1,)), v['tmp1'] * v['dum1'] ** 2 / _get(v['nitot'], (iice - 1,))), mask_13)  # WRF:2658
            _put(v, 'zitot', ((iice) - 1,), _max(_get(v['zitot'], (iice - 1,)), v['tmp2'] * v['dum1'] ** 2 / _get(v['nitot'], (iice - 1,))), mask_13)  # WRF:2659
        mask_18 = jnp.logical_and(mask_13, _get(v['qirim'], (iice - 1,)) > _r(0.0))
        taken_18 = mask_18
        _put(v, 'tmp1', None, _get(v['qirim'], (iice - 1,)) / _get(v['qitot'], (iice - 1,)), mask_18)  # WRF:2666
        mask_19 = jnp.logical_and(mask_18, v['tmp1'] < _r(0.6))
        taken_19 = mask_19
        _put(v, 'eii_fact', ((iice) - 1,), _r(1.0), mask_19)  # WRF:2668
        mask_20 = jnp.logical_and(jnp.logical_and(mask_18, jnp.logical_not(taken_19)), jnp.logical_and(v['tmp1'] >= _r(0.6), v['tmp1'] < _r(0.9)))
        taken_19 = jnp.logical_or(taken_19, mask_20)
        _put(v, 'eii_fact', ((iice) - 1,), _r(1.0) - (v['tmp1'] - _r(0.6)) / _r(0.3), mask_20)  # WRF:2671
        mask_21 = jnp.logical_and(jnp.logical_and(mask_18, jnp.logical_not(taken_19)), v['tmp1'] >= _r(0.9))
        taken_19 = jnp.logical_or(taken_19, mask_21)
        _put(v, 'eii_fact', ((iice) - 1,), _r(0.0), mask_21)  # WRF:2673
        mask_22 = jnp.logical_and(mask_13, jnp.logical_not(taken_18))
        _put(v, 'eii_fact', ((iice) - 1,), _r(1.0), mask_22)  # WRF:2676
        mask_23 = jnp.logical_and(alive, jnp.logical_and(jnp.logical_and(_get(v['qitot'], (iice - 1,)) >= v['qsmall'], v['qc'] >= v['qsmall']), v['t'] <= _r(273.15)))
        taken_23 = mask_23
        _put(v, 'qccol', ((iice) - 1,), v['rhofaci'] * v['f1pr04'] * v['qc'] * v['eci'] * v['rho'] * _get(v['nitot'], (iice - 1,)) * v['iscf'], mask_23)  # WRF:2698
        _put(v, 'nccol', ((iice) - 1,), v['rhofaci'] * v['f1pr04'] * v['nc'] * v['eci'] * v['rho'] * _get(v['nitot'], (iice - 1,)) * v['iscf'], mask_23)  # WRF:2699
        mask_24 = jnp.logical_and(alive, jnp.logical_and(jnp.logical_and(_get(v['qitot'], (iice - 1,)) >= v['qsmall'], v['qc'] >= v['qsmall']), v['t'] > _r(273.15)))
        taken_24 = mask_24
        _put(v, 'qcshd', ((iice) - 1,), v['rhofaci'] * v['f1pr04'] * v['qc'] * v['eci'] * v['rho'] * _get(v['nitot'], (iice - 1,)) * v['iscf'], mask_24)  # WRF:2706
        _put(v, 'nccol', ((iice) - 1,), v['rhofaci'] * v['f1pr04'] * v['nc'] * v['eci'] * v['rho'] * _get(v['nitot'], (iice - 1,)) * v['iscf'], mask_24)  # WRF:2707
        _put(v, 'ncshdc', ((iice) - 1,), _get(v['qcshd'], (iice - 1,)) * _r(1923000.0), mask_24)  # WRF:2709
        mask_25 = jnp.logical_and(alive, jnp.logical_and(jnp.logical_and(_get(v['qitot'], (iice - 1,)) >= v['qsmall'], v['qr'] >= v['qsmall']), v['t'] <= _r(273.15)))
        taken_25 = mask_25
        _put(v, 'qrcol', ((iice) - 1,), _r(10.0) ** (v['f1pr08'] + v['logn0r']) * v['rho'] * v['rhofaci'] * v['eri'] * _get(v['nitot'], (iice - 1,)) * v['iscf'] * (v['spf'] - v['spf_clr']), mask_25)  # WRF:2729
        _put(v, 'nrcol', ((iice) - 1,), _r(10.0) ** (v['f1pr07'] + v['logn0r']) * v['rho'] * v['rhofaci'] * v['eri'] * _get(v['nitot'], (iice - 1,)) * v['iscf'] * (v['spf'] - v['spf_clr']), mask_25)  # WRF:2730
        mask_26 = jnp.logical_and(alive, jnp.logical_and(jnp.logical_and(_get(v['qitot'], (iice - 1,)) >= v['qsmall'], v['qr'] >= v['qsmall']), v['t'] > _r(273.15)))
        taken_26 = mask_26
        _put(v, 'nrcol', ((iice) - 1,), _r(10.0) ** (v['f1pr07'] + v['logn0r']) * v['rho'] * v['rhofaci'] * v['eri'] * _get(v['nitot'], (iice - 1,)) * v['iscf'] * (v['spf'] - v['spf_clr']), mask_26)  # WRF:2741
        _put(v, 'dum', None, _r(10.0) ** (v['f1pr08'] + v['logn0r']) * v['rho'] * v['rhofaci'] * v['eri'] * _get(v['nitot'], (iice - 1,)) * v['iscf'] * (v['spf'] - v['spf_clr']), mask_26)  # WRF:2743
        if iice >= 2:
            mask_27 = jnp.logical_and(alive, _get(v['qitot'], (iice - 1,)) >= v['qsmall'])
            taken_27 = mask_27
            for catcoll in range(1, (iice - 1) + (1 if (1) > 0 else -1), 1):
                mask_28 = jnp.logical_and(mask_27, _get(v['qitot'], (catcoll - 1,)) >= v['qsmall'])
                taken_28 = mask_28
                call_out = _call_find_lookuptable_indices_2(v['dumi'], v['dumii'], v['dumjj'], v['dumic'], v['dumiic'], v['dumjjc'], v['dum1'], v['dum4'], v['dum5'], v['dum1c'], v['dum4c'], v['dum5c'], v['iisize'], v['rimsize'], v['densize'], _get(v['qitot'], (iice - 1,)), _get(v['qitot'], (catcoll - 1,)), _get(v['nitot'], (iice - 1,)), _get(v['nitot'], (catcoll - 1,)), _get(v['qirim'], (iice - 1,)), _get(v['qirim'], (catcoll - 1,)), _get(v['birim'], (iice - 1,)), _get(v['birim'], (catcoll - 1,)), tables=tables, three=log_3momentice)
                _put(v, 'dumi', None, call_out[0], mask_28)  # WRF:2771
                _put(v, 'dumii', None, call_out[1], mask_28)  # WRF:2771
                _put(v, 'dumjj', None, call_out[2], mask_28)  # WRF:2771
                _put(v, 'dumic', None, call_out[3], mask_28)  # WRF:2771
                _put(v, 'dumiic', None, call_out[4], mask_28)  # WRF:2771
                _put(v, 'dumjjc', None, call_out[5], mask_28)  # WRF:2771
                _put(v, 'dum1', None, call_out[6], mask_28)  # WRF:2771
                _put(v, 'dum4', None, call_out[7], mask_28)  # WRF:2771
                _put(v, 'dum5', None, call_out[8], mask_28)  # WRF:2771
                _put(v, 'dum1c', None, call_out[9], mask_28)  # WRF:2771
                _put(v, 'dum4c', None, call_out[10], mask_28)  # WRF:2771
                _put(v, 'dum5c', None, call_out[11], mask_28)  # WRF:2771
                call_out = _call_access_lookup_table_colli(v['dumjjc'], v['dumiic'], v['dumic'], v['dumjj'], v['dumii'], v['dumi'], 1, v['dum1c'], v['dum4c'], v['dum5c'], v['dum1'], v['dum4'], v['dum5'], v['f1pr17'], tables=tables, three=log_3momentice)
                _put(v, 'f1pr17', None, call_out[0], mask_28)  # WRF:2778
                call_out = _call_access_lookup_table_colli(v['dumjjc'], v['dumiic'], v['dumic'], v['dumjj'], v['dumii'], v['dumi'], 2, v['dum1c'], v['dum4c'], v['dum5c'], v['dum1'], v['dum4'], v['dum5'], v['f1pr18'], tables=tables, three=log_3momentice)
                _put(v, 'f1pr18', None, call_out[0], mask_28)  # WRF:2780
                _put(v, 'nicol', ((catcoll) - 1, (iice) - 1,), v['f1pr17'] * v['rhofaci'] * v['rho'] * _get(v['nitot'], (catcoll - 1,)) * _get(v['nitot'], (iice - 1,)) * v['iscf'], mask_28)  # WRF:2786
                _put(v, 'qicol', ((catcoll) - 1, (iice) - 1,), v['f1pr18'] * v['rhofaci'] * v['rho'] * _get(v['nitot'], (catcoll - 1,)) * _get(v['nitot'], (iice - 1,)) * v['iscf'], mask_28)  # WRF:2788
                _put(v, 'nicol', ((catcoll) - 1, (iice) - 1,), v['eii'] * _get(v['eii_fact'], (iice - 1,)) * _get(v['nicol'], (catcoll - 1, iice - 1)), mask_28)  # WRF:2791
                _put(v, 'qicol', ((catcoll) - 1, (iice) - 1,), v['eii'] * _get(v['eii_fact'], (iice - 1,)) * _get(v['qicol'], (catcoll - 1, iice - 1)), mask_28)  # WRF:2792
                _put(v, 'nicol', ((catcoll) - 1, (iice) - 1,), _min(_get(v['nicol'], (catcoll - 1, iice - 1)), _get(v['nitot'], (catcoll - 1,)) * v['odt']), mask_28)  # WRF:2793
                _put(v, 'qicol', ((catcoll) - 1, (iice) - 1,), _min(_get(v['qicol'], (catcoll - 1, iice - 1)), _get(v['qitot'], (catcoll - 1,)) * v['odt']), mask_28)  # WRF:2794
                call_out = _call_calc_bulkrhorime(_get(v['qitot'], (catcoll - 1,)), _get(v['qirim'], (catcoll - 1,)), _get(v['birim'], (catcoll - 1,)), v['rhop'], tables=tables, three=log_3momentice)
                _put(v, 'qirim', ((catcoll) - 1,), call_out[0], mask_28)  # WRF:2799
                _put(v, 'birim', ((catcoll) - 1,), call_out[1], mask_28)  # WRF:2799
                _put(v, 'rhop', None, call_out[2], mask_28)  # WRF:2799
                call_out = _call_find_lookuptable_indices_2(v['dumi'], v['dumii'], v['dumjj'], v['dumic'], v['dumiic'], v['dumjjc'], v['dum1'], v['dum4'], v['dum5'], v['dum1c'], v['dum4c'], v['dum5c'], v['iisize'], v['rimsize'], v['densize'], _get(v['qitot'], (catcoll - 1,)), _get(v['qitot'], (iice - 1,)), _get(v['nitot'], (catcoll - 1,)), _get(v['nitot'], (iice - 1,)), _get(v['qirim'], (catcoll - 1,)), _get(v['qirim'], (iice - 1,)), _get(v['birim'], (catcoll - 1,)), _get(v['birim'], (iice - 1,)), tables=tables, three=log_3momentice)
                _put(v, 'dumi', None, call_out[0], mask_28)  # WRF:2801
                _put(v, 'dumii', None, call_out[1], mask_28)  # WRF:2801
                _put(v, 'dumjj', None, call_out[2], mask_28)  # WRF:2801
                _put(v, 'dumic', None, call_out[3], mask_28)  # WRF:2801
                _put(v, 'dumiic', None, call_out[4], mask_28)  # WRF:2801
                _put(v, 'dumjjc', None, call_out[5], mask_28)  # WRF:2801
                _put(v, 'dum1', None, call_out[6], mask_28)  # WRF:2801
                _put(v, 'dum4', None, call_out[7], mask_28)  # WRF:2801
                _put(v, 'dum5', None, call_out[8], mask_28)  # WRF:2801
                _put(v, 'dum1c', None, call_out[9], mask_28)  # WRF:2801
                _put(v, 'dum4c', None, call_out[10], mask_28)  # WRF:2801
                _put(v, 'dum5c', None, call_out[11], mask_28)  # WRF:2801
                call_out = _call_access_lookup_table_colli(v['dumjjc'], v['dumiic'], v['dumic'], v['dumjj'], v['dumii'], v['dumi'], 1, v['dum1c'], v['dum4c'], v['dum5c'], v['dum1'], v['dum4'], v['dum5'], v['f1pr17'], tables=tables, three=log_3momentice)
                _put(v, 'f1pr17', None, call_out[0], mask_28)  # WRF:2808
                call_out = _call_access_lookup_table_colli(v['dumjjc'], v['dumiic'], v['dumic'], v['dumjj'], v['dumii'], v['dumi'], 2, v['dum1c'], v['dum4c'], v['dum5c'], v['dum1'], v['dum4'], v['dum5'], v['f1pr18'], tables=tables, three=log_3momentice)
                _put(v, 'f1pr18', None, call_out[0], mask_28)  # WRF:2811
                _put(v, 'nicol', ((iice) - 1, (catcoll) - 1,), v['f1pr17'] * v['rhofaci'] * v['rho'] * _get(v['nitot'], (iice - 1,)) * _get(v['nitot'], (catcoll - 1,)) * v['iscf'], mask_28)  # WRF:2814
                _put(v, 'qicol', ((iice) - 1, (catcoll) - 1,), v['f1pr18'] * v['rhofaci'] * v['rho'] * _get(v['nitot'], (iice - 1,)) * _get(v['nitot'], (catcoll - 1,)) * v['iscf'], mask_28)  # WRF:2816
                _put(v, 'nicol', ((iice) - 1, (catcoll) - 1,), v['eii'] * _get(v['eii_fact'], (catcoll - 1,)) * _get(v['nicol'], (iice - 1, catcoll - 1)), mask_28)  # WRF:2820
                _put(v, 'qicol', ((iice) - 1, (catcoll) - 1,), v['eii'] * _get(v['eii_fact'], (catcoll - 1,)) * _get(v['qicol'], (iice - 1, catcoll - 1)), mask_28)  # WRF:2821
                _put(v, 'nicol', ((iice) - 1, (catcoll) - 1,), _min(_get(v['nicol'], (iice - 1, catcoll - 1)), _get(v['nitot'], (iice - 1,)) * v['odt']), mask_28)  # WRF:2822
                _put(v, 'qicol', ((iice) - 1, (catcoll) - 1,), _min(_get(v['qicol'], (iice - 1, catcoll - 1)), _get(v['qitot'], (iice - 1,)) * v['odt']), mask_28)  # WRF:2823
        mask_29 = jnp.logical_and(alive, _get(v['qitot'], (iice - 1,)) >= v['qsmall'])
        taken_29 = mask_29
        _put(v, 'nislf', ((iice) - 1,), v['f1pr03'] * v['rho'] * v['eii'] * _get(v['eii_fact'], (iice - 1,)) * v['rhofaci'] * _get(v['nitot'], (iice - 1,)) * _get(v['nitot'], (iice - 1,)) * v['iscf'], mask_29)  # WRF:2841
        mask_30 = jnp.logical_and(alive, jnp.logical_and(_get(v['qitot'], (iice - 1,)) >= v['qsmall'], v['t'] > _r(273.15)))
        taken_30 = mask_30
        _put(v, 'qsat0', None, _r(0.622) * v['e0'] / (v['pres'] - v['e0']), mask_30)  # WRF:2852
        _put(v, 'dum', None, _r(0.0), mask_30)  # WRF:2856
        _put(v, 'qimlt', ((iice) - 1,), ((v['f1pr05'] + v['f1pr14'] * v['sc'] ** v['thrd'] * (v['rhofaci'] * v['rho'] / v['mu']) ** _r(0.5)) * ((v['t'] - _r(273.15)) * v['kap'] - v['rho'] * v['xxlv'] * v['dv'] * (v['qsat0'] - v['qv_cld'])) * _r(2.0) * v['pi'] / v['xlf'] + v['dum']) * _get(v['nitot'], (iice - 1,)), mask_30)  # WRF:2860
        _put(v, 'qimlt', ((iice) - 1,), _max(_get(v['qimlt'], (iice - 1,)), _r(0.0)), mask_30)  # WRF:2863
        _put(v, 'nimlt', ((iice) - 1,), _get(v['qimlt'], (iice - 1,)) * (_get(v['nitot'], (iice - 1,)) / _get(v['qitot'], (iice - 1,))), mask_30)  # WRF:2864
        mask_31 = jnp.logical_and(alive, jnp.logical_and(jnp.logical_and(_get(v['qitot'], (iice - 1,)) >= v['qsmall'], v['qc'] + v['qr'] >= _r(1e-06)), v['t'] < _r(273.15)))
        taken_31 = mask_31
        _put(v, 'qsat0', None, _r(0.622) * v['e0'] / (v['pres'] - v['e0']), mask_31)  # WRF:2875
        _put(v, 'qwgrth', ((iice) - 1,), (v['f1pr05'] + v['f1pr14'] * v['sc'] ** v['thrd'] * (v['rhofaci'] * v['rho'] / v['mu']) ** _r(0.5)) * _r(2.0) * v['pi'] * (v['rho'] * v['xxlv'] * v['dv'] * (v['qsat0'] - v['qv_cld']) - (v['t'] - _r(273.15)) * v['kap']) / (v['xlf'] + v['cpw'] * (v['t'] - _r(273.15))) * _get(v['nitot'], (iice - 1,)), mask_31)  # WRF:2876
        _put(v, 'qwgrth', ((iice) - 1,), _max(_get(v['qwgrth'], (iice - 1,)), _r(0.0)), mask_31)  # WRF:2879
        _put(v, 'dum', None, _max(_r(0.0), _get(v['qccol'], (iice - 1,)) + _get(v['qrcol'], (iice - 1,)) - _get(v['qwgrth'], (iice - 1,))), mask_31)  # WRF:2881
        mask_32 = jnp.logical_and(mask_31, v['dum'] >= _r(1e-10))
        taken_32 = mask_32
        _put(v, 'nrshdr', ((iice) - 1,), _get(v['nrshdr'], (iice - 1,)) + v['dum'] * _r(1923000.0), mask_32)  # WRF:2883
        mask_33 = jnp.logical_and(mask_32, _get(v['qccol'], (iice - 1,)) + _get(v['qrcol'], (iice - 1,)) >= _r(1e-10))
        taken_33 = mask_33
        _put(v, 'dum1', None, _r(1.0) / (_get(v['qccol'], (iice - 1,)) + _get(v['qrcol'], (iice - 1,))), mask_33)  # WRF:2885
        _put(v, 'qcshd', ((iice) - 1,), _get(v['qcshd'], (iice - 1,)) + v['dum'] * _get(v['qccol'], (iice - 1,)) * v['dum1'], mask_33)  # WRF:2886
        _put(v, 'qccol', ((iice) - 1,), _get(v['qccol'], (iice - 1,)) - v['dum'] * _get(v['qccol'], (iice - 1,)) * v['dum1'], mask_33)  # WRF:2887
        _put(v, 'qrcol', ((iice) - 1,), _get(v['qrcol'], (iice - 1,)) - v['dum'] * _get(v['qrcol'], (iice - 1,)) * v['dum1'], mask_33)  # WRF:2888
        _put(v, 'log_wetgrowth', ((iice) - 1,), True, mask_32)  # WRF:2891
        mask_34 = jnp.logical_and(alive, jnp.logical_and(_get(v['qitot'], (iice - 1,)) >= v['qsmall'], v['t'] < _r(273.15)))
        taken_34 = mask_34
        _put(v, 'epsi', ((iice) - 1,), (v['f1pr05'] + v['f1pr14'] * v['sc'] ** v['thrd'] * (v['rhofaci'] * v['rho'] / v['mu']) ** _r(0.5)) * _r(2.0) * v['pi'] * v['rho'] * v['dv'] * _get(v['nitot'], (iice - 1,)), mask_34)  # WRF:2901
        _put(v, 'epsi_tot', None, v['epsi_tot'] + _get(v['epsi'], (iice - 1,)), mask_34)  # WRF:2903
        mask_35 = jnp.logical_and(alive, jnp.logical_not(taken_34))
        _put(v, 'epsi', ((iice) - 1,), _r(0.0), mask_35)  # WRF:2905
        mask_36 = jnp.logical_and(alive, jnp.logical_and(_get(v['qccol'], (iice - 1,)) >= v['qsmall'], v['t'] < _r(273.15)))
        taken_36 = mask_36
        _put(v, 'vtrmi1', None, v['f1pr02'] * v['rhofaci'], mask_36)  # WRF:2928
        _put(v, 'itc', None, _r(1.0) / _min(_r(-0.001), v['t'] - _r(273.15)), mask_36)  # WRF:2929
        mask_37 = jnp.logical_and(mask_36, v['qc'] >= v['qsmall'])
        taken_37 = mask_37
        _put(v, 'vt_qc', None, v['acn'] * gamma(_r(4.0) + v['bcn'] + v['mu_c']) / (v['lamc'] ** v['bcn'] * gamma(v['mu_c'] + _r(4.0))), mask_37)  # WRF:2935
        _put(v, 'd_c', None, (v['mu_c'] + _r(4.0)) / v['lamc'], mask_37)  # WRF:2937
        _put(v, 'v_impact', None, jnp.abs(v['vtrmi1'] - v['vt_qc']), mask_37)  # WRF:2938
        _put(v, 'ri', None, -(_r(500000.0) * v['d_c']) * v['v_impact'] * v['itc'], mask_37)  # WRF:2939
        _put(v, 'ri', None, _max(_r(1.0), _min(v['ri'], _r(12.0))), mask_37)  # WRF:2941
        mask_38 = jnp.logical_and(mask_37, v['ri'] <= _r(8.0))
        taken_38 = mask_38
        _put(v, 'rhorime_c', ((iice) - 1,), (_r(0.051) + _r(0.114) * v['ri'] - _r(0.0055) * v['ri'] ** 2) * _r(1000.0), mask_38)  # WRF:2943
        mask_39 = jnp.logical_and(mask_37, jnp.logical_not(taken_38))
        _put(v, 'rhorime_c', ((iice) - 1,), _r(611.0) + _r(72.25) * (v['ri'] - _r(8.0)), mask_39)  # WRF:2949
        mask_40 = jnp.logical_and(alive, jnp.logical_not(taken_36))
        _put(v, 'rhorime_c', ((iice) - 1,), _r(400.0), mask_40)  # WRF:2967
    mask_41 = jnp.logical_and(alive, jnp.logical_and(v['qc'] >= v['qsmall'], v['t'] <= _r(269.15)))
    taken_41 = mask_41
    _put(v, 'dum', None, (_r(1.0) / v['lamc']) ** 3, mask_41)  # WRF:2988
    _put(v, 'tmp1', None, v['cdist1'] * jnp.exp(v['aimm'] * (_r(273.15) - v['t'])), mask_41)  # WRF:2994
    _put(v, 'q_nuc', None, v['cons6'] * gamma(_r(7.0) + v['mu_c']) * v['tmp1'] * v['dum'] ** 2, mask_41)  # WRF:2995
    _put(v, 'n_nuc', None, v['cons5'] * gamma(v['mu_c'] + _r(4.0)) * v['tmp1'] * v['dum'], mask_41)  # WRF:2996
    if ncat > 1:
        _put(v, 'dum1', None, _r(900.0), mask_41)  # WRF:3005
        _put(v, 'd_new', None, (v['q_nuc'] * _r(6.0) / (v['pi'] * v['dum1'] * v['n_nuc'])) ** v['thrd'], mask_41)  # WRF:3006
        call_out = _call_icecat_destination(v['qitot'] * _expand(v['iscf'], 1), v['diam_ice'], v['d_new'], v['deltad_init'], v['iice_dest'], tables=tables, three=log_3momentice)
        _put(v, 'iice_dest', None, call_out[0], mask_41)  # WRF:3007
    else:
        _put(v, 'iice_dest', None, 1, mask_41)  # WRF:3011
    _put(v, 'qcheti', ((v['iice_dest']) - 1,), v['q_nuc'], mask_41)  # WRF:3013
    _put(v, 'ncheti', ((v['iice_dest']) - 1,), v['n_nuc'], mask_41)  # WRF:3014
    mask_42 = jnp.logical_and(alive, jnp.logical_and(v['qr'] * v['ispf'] >= v['qsmall'], v['t'] <= _r(269.15)))
    taken_42 = mask_42
    _put(v, 'tmpdbl1', None, jnp.exp(_d(jnp.log(v['cdistr']) + jnp.log(gamma(_r(7.0) + v['mu_r'])) - _r(6.0) * jnp.log(v['lamr']))), mask_42)  # WRF:3025
    _put(v, 'tmpdbl2', None, jnp.exp(_d(jnp.log(v['cdistr']) + jnp.log(gamma(v['mu_r'] + _r(4.0))) - _r(3.0) * jnp.log(v['lamr']))), mask_42)  # WRF:3026
    _put(v, 'tmpdbl3', None, jnp.exp(_d(v['aimm'] * (_r(273.15) - v['t']))), mask_42)  # WRF:3027
    _put(v, 'q_nuc', None, v['cons6'] * _r(v['tmpdbl1'] * v['tmpdbl3']) * v['spf'], mask_42)  # WRF:3028
    _put(v, 'n_nuc', None, v['cons5'] * _r(v['tmpdbl2'] * v['tmpdbl3']) * v['spf'], mask_42)  # WRF:3029
    if ncat > 1:
        _put(v, 'dum1', None, _r(900.0), mask_42)  # WRF:3033
        _put(v, 'd_new', None, (v['q_nuc'] * _r(6.0) / (v['pi'] * v['dum1'] * v['n_nuc'])) ** v['thrd'], mask_42)  # WRF:3034
        call_out = _call_icecat_destination(v['qitot'] * _expand(v['iscf'], 1), v['diam_ice'], v['d_new'], v['deltad_init'], v['iice_dest'], tables=tables, three=log_3momentice)
        _put(v, 'iice_dest', None, call_out[0], mask_42)  # WRF:3035
    else:
        _put(v, 'iice_dest', None, 1, mask_42)  # WRF:3039
    _put(v, 'qrheti', ((v['iice_dest']) - 1,), v['q_nuc'], mask_42)  # WRF:3041
    _put(v, 'nrheti', ((v['iice_dest']) - 1,), v['n_nuc'], mask_42)  # WRF:3042
    if log_hmossopon:
        if ncat > 1:
            _put(v, 'd_new', None, _r(1e-05), alive)  # WRF:3053
            call_out = _call_icecat_destination(v['qitot'] * _expand(v['iscf'], 1), v['diam_ice'], v['d_new'], v['deltad_init'], v['iice_dest'], tables=tables, three=log_3momentice)
            _put(v, 'iice_dest', None, call_out[0], alive)  # WRF:3054
        else:
            _put(v, 'iice_dest', None, 1, alive)  # WRF:3057
        for iice in range(1, (ncat) + (1 if (1) > 0 else -1), 1):
            mask_43 = jnp.logical_and(alive, jnp.logical_and(jnp.logical_and(_get(v['qitot'], (iice - 1,)) >= v['qsmall'], _get(v['diam_ice'], (iice - 1,)) >= _r(0.004)), jnp.logical_or(_get(v['qccol'], (iice - 1,)) > _r(0.0), _get(v['qrcol'], (iice - 1,)) > _r(0.0))))
            taken_43 = mask_43
            mask_44 = jnp.logical_and(mask_43, v['t'] > _r(270.15))
            taken_44 = mask_44
            _put(v, 'dum', None, _r(0.0), mask_44)  # WRF:3068
            mask_45 = jnp.logical_and(jnp.logical_and(mask_43, jnp.logical_not(taken_44)), jnp.logical_and(v['t'] <= _r(270.15), v['t'] > _r(268.15)))
            taken_44 = jnp.logical_or(taken_44, mask_45)
            _put(v, 'dum', None, (_r(270.15) - v['t']) * _r(0.5), mask_45)  # WRF:3070
            mask_46 = jnp.logical_and(jnp.logical_and(mask_43, jnp.logical_not(taken_44)), jnp.logical_and(v['t'] <= _r(268.15), v['t'] >= _r(265.15)))
            taken_44 = jnp.logical_or(taken_44, mask_46)
            _put(v, 'dum', None, (v['t'] - _r(265.15)) * v['thrd'], mask_46)  # WRF:3072
            mask_47 = jnp.logical_and(jnp.logical_and(mask_43, jnp.logical_not(taken_44)), v['t'] < _r(265.15))
            taken_44 = jnp.logical_or(taken_44, mask_47)
            _put(v, 'dum', None, _r(0.0), mask_47)  # WRF:3074
            _put(v, 'dum1', None, _r(350000.0) * _get(v['qrcol'], (iice - 1,)) * v['dum'] * _r(1000.0), mask_43)  # WRF:3091
            _put(v, 'dum2', None, v['dum1'] * v['piov6'] * _r(900.0) * _r(9.999998977483753e-16), mask_43)  # WRF:3092
            _put(v, 'qrcol', ((iice) - 1,), _get(v['qrcol'], (iice - 1,)) - v['dum2'], mask_43)  # WRF:3093
            mask_48 = jnp.logical_and(mask_43, _get(v['qrcol'], (iice - 1,)) < _r(0.0))
            taken_48 = mask_48
            _put(v, 'dum2', None, _get(v['qrcol'], (iice - 1,)) + v['dum2'], mask_48)  # WRF:3095
            _put(v, 'qrcol', ((iice) - 1,), _r(0.0), mask_48)  # WRF:3096
            _put(v, 'qrmul', ((v['iice_dest']) - 1,), _get(v['qrmul'], (v['iice_dest'] - 1,)) + v['dum2'], mask_43)  # WRF:3099
            _put(v, 'nimul', ((v['iice_dest']) - 1,), _get(v['nimul'], (v['iice_dest'] - 1,)) + v['dum2'] / (v['piov6'] * _r(900.0) * _r(9.999998977483753e-16)), mask_43)  # WRF:3100
    mask_49 = jnp.logical_and(alive, v['qr'] * v['ispf'] >= v['qsmall'])
    taken_49 = mask_49
    call_out = _call_find_lookuptable_indices_3(v['dumii'], v['dumjj'], v['dum1'], v['rdumii'], v['rdumjj'], v['inv_dum3'], v['mu_r'], v['lamr'], tables=tables, three=log_3momentice)
    _put(v, 'dumii', None, call_out[0], mask_49)  # WRF:3115
    _put(v, 'dumjj', None, call_out[1], mask_49)  # WRF:3115
    _put(v, 'dum1', None, call_out[2], mask_49)  # WRF:3115
    _put(v, 'rdumii', None, call_out[3], mask_49)  # WRF:3115
    _put(v, 'rdumjj', None, call_out[4], mask_49)  # WRF:3115
    _put(v, 'inv_dum3', None, call_out[5], mask_49)  # WRF:3115
    _put(v, 'dum1', None, v['revap_table'][_int(v['dumii'] - 1), _int(v['dumjj'] - 1)] + (v['rdumii'] - _r(v['dumii'])) * (v['revap_table'][_int(v['dumii'] + 1 - 1), _int(v['dumjj'] - 1)] - v['revap_table'][_int(v['dumii'] - 1), _int(v['dumjj'] - 1)]), mask_49)  # WRF:3117
    _put(v, 'dum2', None, v['revap_table'][_int(v['dumii'] - 1), _int(v['dumjj'] + 1 - 1)] + (v['rdumii'] - _r(v['dumii'])) * (v['revap_table'][_int(v['dumii'] + 1 - 1), _int(v['dumjj'] + 1 - 1)] - v['revap_table'][_int(v['dumii'] - 1), _int(v['dumjj'] + 1 - 1)]), mask_49)  # WRF:3120
    _put(v, 'dum', None, v['dum1'] + (v['rdumjj'] - _r(v['dumjj'])) * (v['dum2'] - v['dum1']), mask_49)  # WRF:3123
    _put(v, 'epsr', None, _r(2.0) * v['pi'] * v['cdistr'] * v['rho'] * v['dv'] * (v['f1r'] * gamma(v['mu_r'] + _r(2.0)) / v['lamr'] + v['f2r'] * (v['rho'] / v['mu']) ** _r(0.5) * v['sc'] ** v['thrd'] * v['dum']), mask_49)  # WRF:3125
    mask_50 = jnp.logical_and(alive, jnp.logical_not(taken_49))
    _put(v, 'epsr', None, _r(0.0), mask_50)  # WRF:3128
    mask_51 = jnp.logical_and(alive, v['qc'] >= v['qsmall'])
    taken_51 = mask_51
    _put(v, 'epsc', None, _r(2.0) * v['pi'] * v['rho'] * v['dv'] * v['cdist'], mask_51)  # WRF:3132
    mask_52 = jnp.logical_and(alive, jnp.logical_not(taken_51))
    _put(v, 'epsc', None, _r(0.0), mask_52)  # WRF:3134
    mask_53 = jnp.logical_and(alive, v['t'] < _r(273.15))
    taken_53 = mask_53
    _put(v, 'oabi', None, _r(1.0) / v['abi'], mask_53)  # WRF:3138
    _put(v, 'xx', None, v['epsc'] + v['epsr'] + v['epsi_tot'] * (_r(1.0) + v['xxls'] * v['inv_cp'] * v['dqsdt']) * v['oabi'], mask_53)  # WRF:3139
    mask_54 = jnp.logical_and(alive, jnp.logical_not(taken_53))
    _put(v, 'xx', None, v['epsc'] + v['epsr'], mask_54)  # WRF:3141
    _put(v, 'dumqvi', None, v['qvi'], alive)  # WRF:3144
    _put(v, 'dum', None, -v['cp'] / v['g'] * (v['t'] - v['t_old']) * v['odt'], alive)  # WRF:3171
    mask_55 = jnp.logical_and(alive, v['t'] < _r(273.15))
    taken_55 = mask_55
    _put(v, 'aaa', None, (v['qv'] - v['qv_old']) * v['odt'] - v['dqsdt'] * (-v['dum'] * v['g'] * v['inv_cp']) - (v['qvs'] - v['dumqvi']) * (_r(1.0) + v['xxls'] * v['inv_cp'] * v['dqsdt']) * v['oabi'] * v['epsi_tot'], mask_55)  # WRF:3176
    mask_56 = jnp.logical_and(alive, jnp.logical_not(taken_55))
    _put(v, 'aaa', None, (v['qv'] - v['qv_old']) * v['odt'] - v['dqsdt'] * (-v['dum'] * v['g'] * v['inv_cp']), mask_56)  # WRF:3179
    _put(v, 'xx', None, _max(_r(1e-20), v['xx']), alive)  # WRF:3182
    _put(v, 'oxx', None, _r(1.0) / v['xx'], alive)  # WRF:3183
    if not scpf_on:
        _put(v, 'ssat_cld', None, v['ssat'], alive)  # WRF:3186
        _put(v, 'ssat_r', None, v['ssat'], alive)  # WRF:3187
        _put(v, 'sup_cld', None, v['sup'], alive)  # WRF:3188
        _put(v, 'sup_r', None, v['sup'], alive)  # WRF:3189
        _put(v, 'supi_cld', None, v['supi'], alive)  # WRF:3190
    else:
        _put(v, 'ssat_cld', None, v['qv_cld'] - v['qvs'], alive)  # WRF:3192
        _put(v, 'ssat_clr', None, v['qv_clr'] - v['qvs'], alive)  # WRF:3193
        _put(v, 'ssat_r', None, v['ssat_cld'] * (v['spf'] - v['spf_clr']) + v['ssat_clr'] * v['spf_clr'], alive)  # WRF:3195
        _put(v, 'sup_r', None, v['ssat_r'] / v['qvs'], alive)  # WRF:3196
        _put(v, 'sup_cld', None, v['ssat_cld'] / v['qvs'], alive)  # WRF:3197
        _put(v, 'supi_cld', None, v['qv_cld'] / v['qvi'] - _r(1.0), alive)  # WRF:3198
    mask_57 = jnp.logical_and(alive, v['qc'] >= v['qsmall'])
    _put(v, 'qccon', None, (v['aaa'] * v['epsc'] * v['oxx'] + (v['ssat_cld'] * v['scf'] - v['aaa'] * v['oxx']) * v['odt'] * v['epsc'] * v['oxx'] * (_r(1.0) - _r(jnp.exp(-_d(v['xx'] * v['dt']))))) / v['ab'], mask_57)  # WRF:3201
    mask_58 = jnp.logical_and(alive, v['qr'] >= v['qsmall'])
    _put(v, 'qrcon', None, (v['aaa'] * v['epsr'] * v['oxx'] + (v['ssat_r'] * v['spf'] - v['aaa'] * v['oxx']) * v['odt'] * v['epsr'] * v['oxx'] * (_r(1.0) - _r(jnp.exp(-_d(v['xx'] * v['dt']))))) / v['ab'], mask_58)  # WRF:3203
    mask_59 = jnp.logical_and(alive, jnp.logical_and(v['sup_cld'] < _r(-0.001), v['qc'] < _r(1e-12)))
    _put(v, 'qccon', None, -v['qc'] * v['odt'], mask_59)  # WRF:3207
    mask_60 = jnp.logical_and(alive, jnp.logical_and(v['sup_r'] < _r(-0.001), v['qr'] < _r(1e-12)))
    _put(v, 'qrcon', None, -v['qr'] * v['odt'], mask_60)  # WRF:3208
    mask_61 = jnp.logical_and(alive, v['qccon'] < _r(0.0))
    taken_61 = mask_61
    _put(v, 'qcevp', None, -v['qccon'], mask_61)  # WRF:3211
    _put(v, 'qccon', None, _r(0.0), mask_61)  # WRF:3212
    mask_62 = jnp.logical_and(alive, jnp.logical_not(taken_61))
    _put(v, 'qccon', None, _min(v['qccon'], v['qv'] * v['odt']), mask_62)  # WRF:3214
    mask_63 = jnp.logical_and(alive, v['qrcon'] < _r(0.0))
    taken_63 = mask_63
    _put(v, 'qrevp', None, -v['qrcon'], mask_63)  # WRF:3218
    _put(v, 'nrevp', None, v['qrevp'] * (v['nr'] / v['qr']), mask_63)  # WRF:3219
    _put(v, 'qrcon', None, _r(0.0), mask_63)  # WRF:3221
    mask_64 = jnp.logical_and(alive, jnp.logical_not(taken_63))
    _put(v, 'qrcon', None, _min(v['qrcon'], v['qv'] * v['odt']), mask_64)  # WRF:3223
    for iice in range(1, (ncat) + (1 if (1) > 0 else -1), 1):
        mask_65 = jnp.logical_and(alive, jnp.logical_and(_get(v['qitot'], (iice - 1,)) >= v['qsmall'], v['t'] < _r(273.15)))
        taken_65 = mask_65
        _put(v, 'qidep', ((iice) - 1,), (v['aaa'] * _get(v['epsi'], (iice - 1,)) * v['oxx'] + (v['ssat_cld'] * v['scf'] - v['aaa'] * v['oxx']) * v['odt'] * _get(v['epsi'], (iice - 1,)) * v['oxx'] * (_r(1.0) - _r(jnp.exp(-_d(v['xx'] * v['dt']))))) * v['oabi'] + (v['qvs'] - v['dumqvi']) * _get(v['epsi'], (iice - 1,)) * v['oabi'], mask_65)  # WRF:3230
        mask_66 = jnp.logical_and(alive, jnp.logical_and(v['supi_cld'] < _r(-0.001), _get(v['qitot'], (iice - 1,)) < _r(1e-12)))
        _put(v, 'qidep', ((iice) - 1,), -_get(v['qitot'], (iice - 1,)) * v['odt'], mask_66)  # WRF:3235
        mask_67 = jnp.logical_and(alive, _get(v['qidep'], (iice - 1,)) < _r(0.0))
        taken_67 = mask_67
        _put(v, 'qisub', ((iice) - 1,), -_get(v['qidep'], (iice - 1,)), mask_67)  # WRF:3245
        _put(v, 'qisub', ((iice) - 1,), _get(v['qisub'], (iice - 1,)) * v['clbfact_sub'], mask_67)  # WRF:3246
        _put(v, 'qisub', ((iice) - 1,), _min(_get(v['qisub'], (iice - 1,)), _get(v['qitot'], (iice - 1,)) * v['odt']), mask_67)  # WRF:3247
        _put(v, 'nisub', ((iice) - 1,), _get(v['qisub'], (iice - 1,)) * (_get(v['nitot'], (iice - 1,)) / _get(v['qitot'], (iice - 1,))), mask_67)  # WRF:3248
        _put(v, 'qidep', ((iice) - 1,), _r(0.0), mask_67)  # WRF:3249
        mask_68 = jnp.logical_and(alive, jnp.logical_not(taken_67))
        _put(v, 'qidep', ((iice) - 1,), _get(v['qidep'], (iice - 1,)) * v['clbfact_dep'], mask_68)  # WRF:3251
        _put(v, 'qidep', ((iice) - 1,), _min(_get(v['qidep'], (iice - 1,)), v['qv'] * v['odt']), mask_68)  # WRF:3252
    alive = level_alive  # WRF: restore nucleation path at label444
    if not scpf_on:
        _put(v, 'sup_cld', None, v['sup'], alive)  # WRF:3265
        _put(v, 'supi_cld', None, v['supi'], alive)  # WRF:3266
    else:
        _put(v, 'supi_cld', None, v['qv_cld'] / v['qvi'] - _r(1.0), alive)  # WRF:3268
        _put(v, 'sup_cld', None, v['qv_cld'] / v['qvs'] - _r(1.0), alive)  # WRF:3269
    mask_69 = jnp.logical_and(alive, jnp.logical_and(v['t'] < _r(258.15), v['supi_cld'] >= _r(0.05)))
    taken_69 = mask_69
    _put(v, 'dum', None, _r(0.005) * jnp.exp(_r(0.304) * (_r(273.15) - v['t'])) * _r(1000.0) * v['inv_rho'], mask_69)  # WRF:3274
    _put(v, 'dum', None, _min(v['dum'], _r(100000.0) * v['inv_rho'] * v['scf']), mask_69)  # WRF:3276
    _put(v, 'n_nuc', None, _max(_r(0.0), (v['dum'] - _sum(v['nitot'])) * v['odt']), mask_69)  # WRF:3277
    mask_70 = jnp.logical_and(mask_69, v['n_nuc'] >= _r(1e-20))
    taken_70 = mask_70
    _put(v, 'q_nuc', None, _max(_r(0.0), (v['dum'] - _sum(v['nitot'])) * v['mi0'] * v['odt']), mask_70)  # WRF:3280
    if ncat > 1:
        _put(v, 'dum1', None, _r(900.0), mask_70)  # WRF:3283
        _put(v, 'd_new', None, (v['q_nuc'] * _r(6.0) / (v['pi'] * v['dum1'] * v['n_nuc'])) ** v['thrd'], mask_70)  # WRF:3284
        call_out = _call_icecat_destination(v['qitot'] * _expand(v['iscf'], 1), v['diam_ice'], v['d_new'], v['deltad_init'], v['iice_dest'], tables=tables, three=log_3momentice)
        _put(v, 'iice_dest', None, call_out[0], mask_70)  # WRF:3285
    else:
        _put(v, 'iice_dest', None, 1, mask_70)  # WRF:3288
    _put(v, 'qinuc', ((v['iice_dest']) - 1,), v['q_nuc'], mask_70)  # WRF:3290
    _put(v, 'ninuc', ((v['iice_dest']) - 1,), v['n_nuc'], mask_70)  # WRF:3291
    mask_71 = jnp.logical_and(alive, jnp.logical_and(jnp.logical_and(not log_predictnc, v['sup_cld'] > _r(1e-06)), v['it'] > 1))
    taken_71 = mask_71
    _put(v, 'dum', None, v['nccnst'] * v['inv_rho'] * v['cons7'] - v['qc'], mask_71)  # WRF:3304
    _put(v, 'dum', None, _max(_r(0.0), v['dum'] * v['iscf']), mask_71)  # WRF:3305
    _put(v, 'dumqvs', None, qv_sat(v['t'], v['pres'], 0), mask_71)  # WRF:3306
    _put(v, 'dqsdt', None, v['xxlv'] * v['dumqvs'] / (v['rv'] * v['t'] * v['t']), mask_71)  # WRF:3307
    _put(v, 'ab', None, _r(1.0) + v['dqsdt'] * v['xxlv'] * v['inv_cp'], mask_71)  # WRF:3308
    _put(v, 'dum', None, _max(_r(0.0), _min(v['dum'], (v['qv_cld'] - v['dumqvs']) / v['ab'])), mask_71)  # WRF:3309
    _put(v, 'qcnuc', None, v['dum'] * v['odt'] * v['scf'], mask_71)  # WRF:3310
    if log_predictnc:
        mask_72 = jnp.logical_and(alive, v['sup_cld'] > _r(1e-06))
        taken_72 = mask_72
        _put(v, 'dum1', None, _r(1.0) / v['bact'] ** _r(0.5), mask_72)  # WRF:3317
        _put(v, 'sigvl', None, _r(0.0761) - _r(0.000155) * (v['t'] - _r(273.15)), mask_72)  # WRF:3318
        _put(v, 'aact', None, _r(2.0) * v['mw'] / (v['rhow'] * v['rr'] * v['t']) * v['sigvl'], mask_72)  # WRF:3319
        _put(v, 'sm1', None, _r(2.0) * v['dum1'] * (v['aact'] * v['thrd'] * v['inv_rm1']) ** _r(1.5), mask_72)  # WRF:3320
        _put(v, 'sm2', None, _r(2.0) * v['dum1'] * (v['aact'] * v['thrd'] * v['inv_rm2']) ** _r(1.5), mask_72)  # WRF:3321
        _put(v, 'uu1', None, _r(2.0) * jnp.log(v['sm1'] / v['sup_cld']) / (_r(4.242) * jnp.log(v['sig1'])), mask_72)  # WRF:3322
        _put(v, 'uu2', None, _r(2.0) * jnp.log(v['sm2'] / v['sup_cld']) / (_r(4.242) * jnp.log(v['sig2'])), mask_72)  # WRF:3323
        _put(v, 'dum1', None, v['nanew1'] * _r(0.5) * (_r(1.0) - derf(v['uu1'])), mask_72)  # WRF:3324
        _put(v, 'dum2', None, v['nanew2'] * _r(0.5) * (_r(1.0) - derf(v['uu2'])), mask_72)  # WRF:3325
        _put(v, 'dum2', None, _min(v['nanew1'] + v['nanew2'], v['dum1'] + v['dum2']), mask_72)  # WRF:3327
        _put(v, 'dum2', None, (v['dum2'] - v['nc'] * v['iscf']) * v['odt'] * v['scf'], mask_72)  # WRF:3328
        _put(v, 'dum2', None, _max(_r(0.0), v['dum2']), mask_72)  # WRF:3329
        _put(v, 'ncnuc', None, v['dum2'], mask_72)  # WRF:3330
        mask_73 = jnp.logical_and(mask_72, v['it'] <= 1)
        taken_73 = mask_73
        _put(v, 'qcnuc', None, _r(0.0), mask_73)  # WRF:3334
        mask_74 = jnp.logical_and(mask_72, jnp.logical_not(taken_73))
        _put(v, 'qcnuc', None, v['ncnuc'] * v['cons7'], mask_74)  # WRF:3336
    mask_75 = jnp.logical_and(alive, v['it'] <= 1)
    taken_75 = mask_75
    _put(v, 'dumt', None, v['th'] * (v['pres'] * _r(1e-05)) ** (v['rd'] * v['inv_cp']), mask_75)  # WRF:3349
    _put(v, 'dumqv', None, v['qv_cld'], mask_75)  # WRF:3350
    _put(v, 'dumqvs', None, qv_sat(v['dumt'], v['pres'], 0), mask_75)  # WRF:3351
    _put(v, 'dums', None, v['dumqv'] - v['dumqvs'], mask_75)  # WRF:3352
    _put(v, 'qccon', None, v['dums'] / (_r(1.0) + v['xxlv'] ** 2 * v['dumqvs'] / (v['cp'] * v['rv'] * v['dumt'] ** 2)) * v['odt'] * v['scf'], mask_75)  # WRF:3353
    _put(v, 'qccon', None, _max(_r(0.0), v['qccon']), mask_75)  # WRF:3354
    mask_76 = jnp.logical_and(mask_75, v['qccon'] <= _r(1e-07))
    _put(v, 'qccon', None, _r(0.0), mask_76)  # WRF:3355
    mask_77 = jnp.logical_and(alive, v['qc'] * v['iscf'] >= _r(1e-08))
    taken_77 = mask_77
    if iparam == 1:
        _put(v, 'dum', None, _r(1.0) - v['qc'] * v['iscf'] / (v['qc'] * v['iscf'] + v['qr'] * v['ispf'] * (v['spf'] - v['spf_clr'])), mask_77)  # WRF:3367
        _put(v, 'dum1', None, _r(600.0) * v['dum'] ** _r(0.68) * (_r(1.0) - v['dum'] ** _r(0.68)) ** 3, mask_77)  # WRF:3368
        _put(v, 'qcaut', None, v['kc'] * _r(1.9230769e-05) * (v['nu'] + _r(2.0)) * (v['nu'] + _r(4.0)) / (v['nu'] + _r(1.0)) ** 2 * (v['rho'] * v['qc'] * v['iscf'] * _r(0.001)) ** 4 / (v['rho'] * v['nc'] * v['iscf'] * _r(1e-06)) ** 2 * (_r(1.0) + v['dum1'] / (_r(1.0) - v['dum']) ** 2) * _r(1000.0) * v['inv_rho'] * v['scf'], mask_77)  # WRF:3369
        _put(v, 'ncautc', None, v['qcaut'] * _r(7692307600.0), mask_77)  # WRF:3373
    elif iparam == 2:
        mask_78 = jnp.logical_and(mask_77, v['nc'] * v['iscf'] * v['rho'] * _r(1e-06) < _r(100.0))
        taken_78 = mask_78
        _put(v, 'qcaut', None, _r(6e+28) * v['inv_rho'] * v['mu_c'] ** _r(-1.7) * (_r(1e-06) * v['rho'] * v['nc'] * v['iscf']) ** _r(-3.3) * (_r(0.001) * v['rho'] * v['qc'] * v['iscf']) ** _r(4.7) * v['scf'], mask_78)  # WRF:3379
        mask_79 = jnp.logical_and(mask_77, jnp.logical_not(taken_78))
        _put(v, 'dum', None, _r(41.46) + (v['nc'] * v['iscf'] * _r(1e-06) * v['rho'] - _r(100.0)) * (_r(37.53) - _r(41.46)) * _r(0.005), mask_79)  # WRF:3384
        _put(v, 'dum1', None, _r(39.36) + (v['nc'] * v['iscf'] * _r(1e-06) * v['rho'] - _r(100.0)) * (_r(30.72) - _r(39.36)) * _r(0.005), mask_79)  # WRF:3385
        _put(v, 'qcaut', None, v['dum'] + (v['mu_c'] - _r(5.0)) * (v['dum1'] - v['dum']) * _r(0.1), mask_79)  # WRF:3386
        _put(v, 'qcaut', None, jnp.exp(v['qcaut']) * (_r(0.001) * v['rho'] * v['qc'] * v['iscf']) ** _r(4.7) * _r(1000.0) * v['inv_rho'] * v['scf'], mask_79)  # WRF:3388
        _put(v, 'ncautc', None, _r(7700000000.0) * v['qcaut'], mask_77)  # WRF:3392
    elif iparam == 3:
        _put(v, 'dum', None, v['qc'] * v['iscf'], mask_77)  # WRF:3397
        _put(v, 'qcaut', None, _r(1350.0) * v['dum'] ** _r(2.47) * (v['nc'] * v['iscf'] * _r(1e-06) * v['rho']) ** _r(-1.79) * v['scf'], mask_77)  # WRF:3398
        _put(v, 'ncautr', None, v['qcaut'] * v['cons3'], mask_77)  # WRF:3400
        _put(v, 'ncautc', None, v['qcaut'] * v['nc'] / v['qc'], mask_77)  # WRF:3401
    elif iparam == 4:
        _put(v, 'dum', None, v['qc'] * v['iscf'], mask_77)  # WRF:3406
        _put(v, 'qcaut', None, _r(79800000000.0) * v['dum'] ** _r(4.22) * (v['nc'] * v['iscf'] * _r(1e-06) * v['rho']) ** _r(-3.01) * v['scf'], mask_77)  # WRF:3407
        _put(v, 'ncautr', None, v['qcaut'] * v['cons8'], mask_77)  # WRF:3408
        _put(v, 'ncautc', None, v['qcaut'] * v['nc'] / v['qc'], mask_77)  # WRF:3409
    mask_80 = jnp.logical_and(mask_77, v['qcaut'] == _r(0.0))
    _put(v, 'ncautc', None, _r(0.0), mask_80)  # WRF:3413
    mask_81 = jnp.logical_and(mask_77, v['ncautc'] == _r(0.0))
    _put(v, 'qcaut', None, _r(0.0), mask_81)  # WRF:3414
    mask_82 = jnp.logical_and(alive, v['qc'] >= v['qsmall'])
    taken_82 = mask_82
    if iparam == 1:
        _put(v, 'ncslf', None, -v['kc'] * (_r(0.001) * v['rho'] * v['qc'] * v['iscf']) ** 2 * (v['nu'] + _r(2.0)) / (v['nu'] + _r(1.0)) * _r(1000000.0) * v['inv_rho'] * v['scf'] + v['ncautc'], mask_82)  # WRF:3425
    elif iparam == 2:
        _put(v, 'ncslf', None, _r(-5.5e+16) * v['inv_rho'] * v['mu_c'] ** _r(-0.63) * (_r(0.001) * v['rho'] * v['qc'] * v['iscf']) ** 2 * v['scf'], mask_82)  # WRF:3429
    elif iparam == 3 or iparam == 4:
        _put(v, 'ncslf', None, _r(0.0), mask_82)  # WRF:3432
    mask_83 = jnp.logical_and(alive, jnp.logical_and(v['qr'] >= v['qsmall'], v['qc'] >= v['qsmall']))
    taken_83 = mask_83
    if iparam == 1:
        _put(v, 'dum2', None, v['spf'] - v['spf_clr'], mask_83)  # WRF:3444
        _put(v, 'dum', None, _r(1.0) - v['qc'] * v['iscf'] / (v['qc'] * v['iscf'] + v['qr'] * v['ispf']), mask_83)  # WRF:3445
        _put(v, 'dum1', None, (v['dum'] / (v['dum'] + _r(0.0005))) ** 4, mask_83)  # WRF:3446
        _put(v, 'qcacc', None, v['kr'] * v['rho'] * _r(0.001) * v['qc'] * v['iscf'] * v['qr'] * v['ispf'] * v['dum1'] * v['dum2'], mask_83)  # WRF:3447
        _put(v, 'ncacc', None, v['qcacc'] * v['rho'] * _r(0.001) * (v['nc'] * v['rho'] * _r(1e-06)) / (v['qc'] * v['rho'] * _r(0.001)) * _r(1000000.0) * v['inv_rho'], mask_83)  # WRF:3448
    elif iparam == 2:
        _put(v, 'dum2', None, v['spf'] - v['spf_clr'], mask_83)  # WRF:3452
        _put(v, 'dum', None, v['qc'] * v['iscf'] * v['qr'] * v['ispf'], mask_83)  # WRF:3453
        _put(v, 'qcacc', None, _r(6.0) * v['rho'] * v['dum'] * v['dum2'], mask_83)  # WRF:3454
        _put(v, 'ncacc', None, v['qcacc'] * v['rho'] * _r(0.001) * (v['nc'] * v['rho'] * _r(1e-06)) / (v['qc'] * v['rho'] * _r(0.001)) * _r(1000000.0) * v['inv_rho'], mask_83)  # WRF:3455
    elif iparam == 3:
        _put(v, 'dum2', None, v['spf'] - v['spf_clr'], mask_83)  # WRF:3459
        _put(v, 'qcacc', None, _r(67.0) * (v['qc'] * v['iscf'] * v['qr'] * v['ispf']) ** _r(1.15) * v['dum2'], mask_83)  # WRF:3460
        _put(v, 'ncacc', None, v['qcacc'] * v['nc'] / v['qc'], mask_83)  # WRF:3461
    elif iparam == 4:
        _put(v, 'dum2', None, v['spf'] - v['spf_clr'], mask_83)  # WRF:3464
        _put(v, 'qcacc', None, _r(8.53) * (v['qc'] * v['iscf']) ** _r(1.05) * (v['qr'] * v['ispf']) ** _r(0.98) * v['dum2'], mask_83)  # WRF:3465
        _put(v, 'ncacc', None, v['qcacc'] * v['nc'] / v['qc'], mask_83)  # WRF:3466
    mask_84 = jnp.logical_and(mask_83, v['qcacc'] == _r(0.0))
    _put(v, 'ncacc', None, _r(0.0), mask_84)  # WRF:3469
    mask_85 = jnp.logical_and(mask_83, v['ncacc'] == _r(0.0))
    _put(v, 'qcacc', None, _r(0.0), mask_85)  # WRF:3470
    mask_86 = jnp.logical_and(alive, v['qr'] >= v['qsmall'])
    taken_86 = mask_86
    _put(v, 'dum1', None, _r(0.00028), mask_86)  # WRF:3481
    _put(v, 'nr', None, _max(v['nr'], v['nsmall']), mask_86)  # WRF:3482
    _put(v, 'dum2', None, (v['qr'] / (v['pi'] * v['rhow'] * v['nr'])) ** v['thrd'], mask_86)  # WRF:3488
    mask_87 = jnp.logical_and(mask_86, v['dum2'] < v['dum1'])
    taken_87 = mask_87
    _put(v, 'dum', None, _r(1.0), mask_87)  # WRF:3490
    mask_88 = jnp.logical_and(jnp.logical_and(mask_86, jnp.logical_not(taken_87)), v['dum2'] >= v['dum1'])
    taken_87 = jnp.logical_or(taken_87, mask_88)
    _put(v, 'dum', None, _r(2.0) - jnp.exp(_r(2300.0) * (v['dum2'] - v['dum1'])), mask_88)  # WRF:3492
    if iparam == 1.0:
        _put(v, 'nrslf', None, v['dum'] * v['kr'] * _r(0.001) * v['qr'] * v['ispf'] * v['nr'] * v['ispf'] * v['rho'] * v['spf'], mask_86)  # WRF:3497
    elif iparam == 2 or iparam == 3:
        _put(v, 'nrslf', None, v['dum'] * _r(5.78) * v['nr'] * v['ispf'] * v['qr'] * v['ispf'] * v['rho'] * v['spf'], mask_86)  # WRF:3499
    elif iparam == 4:
        _put(v, 'nrslf', None, v['dum'] * _r(205.0) * (v['qr'] * v['ispf']) ** _r(1.55) * (v['nr'] * _r(1e-06) * v['rho'] * v['ispf']) ** _r(0.6) * _r(1000000.0) / v['rho'] * v['spf'], mask_86)  # WRF:3501
    _put(v, 'dumqvs', None, qv_sat(v['t'], v['pres'], 0), alive)  # WRF:3515
    _put(v, 'qcon_satadj', None, (v['qv_cld'] - v['dumqvs']) / (_r(1.0) + v['xxlv'] ** 2 * v['dumqvs'] / (v['cp'] * v['rv'] * v['t'] ** 2)) * v['odt'] * v['scf'], alive)  # WRF:3516
    _put(v, 'tmp1', None, v['qccon'] + v['qrcon'] + v['qcnuc'], alive)  # WRF:3518
    mask_89 = jnp.logical_and(alive, jnp.logical_and(v['tmp1'] > _r(0.0), v['qcon_satadj'] < _r(0.0)))
    taken_89 = mask_89
    _put(v, 'qccon', None, _r(0.0), mask_89)  # WRF:3520
    _put(v, 'qrcon', None, _r(0.0), mask_89)  # WRF:3521
    _put(v, 'qcnuc', None, _r(0.0), mask_89)  # WRF:3522
    _put(v, 'ncnuc', None, _r(0.0), mask_89)  # WRF:3523
    mask_90 = jnp.logical_and(alive, jnp.logical_not(taken_89))
    mask_91 = jnp.logical_and(mask_90, jnp.logical_and(v['tmp1'] > _r(0.0), v['tmp1'] > v['qcon_satadj']))
    taken_91 = mask_91
    _put(v, 'ratio', None, _max(_r(0.0), v['qcon_satadj']) / v['tmp1'], mask_91)  # WRF:3526
    _put(v, 'ratio', None, _min(_r(1.0), v['ratio']), mask_91)  # WRF:3527
    _put(v, 'qccon', None, v['qccon'] * v['ratio'], mask_91)  # WRF:3528
    _put(v, 'qrcon', None, v['qrcon'] * v['ratio'], mask_91)  # WRF:3529
    _put(v, 'qcnuc', None, v['qcnuc'] * v['ratio'], mask_91)  # WRF:3530
    _put(v, 'ncnuc', None, v['ncnuc'] * v['ratio'], mask_91)  # WRF:3531
    mask_92 = jnp.logical_and(jnp.logical_and(mask_90, jnp.logical_not(taken_91)), v['qcevp'] + v['qrevp'] > _r(0.0))
    taken_91 = jnp.logical_or(taken_91, mask_92)
    _put(v, 'ratio', None, _max(_r(0.0), -v['qcon_satadj']) / (v['qcevp'] + v['qrevp']), mask_92)  # WRF:3533
    _put(v, 'ratio', None, _min(_r(1.0), v['ratio']), mask_92)  # WRF:3534
    _put(v, 'qcevp', None, v['qcevp'] * v['ratio'], mask_92)  # WRF:3535
    _put(v, 'qrevp', None, v['qrevp'] * v['ratio'], mask_92)  # WRF:3536
    _put(v, 'nrevp', None, v['nrevp'] * v['ratio'], mask_92)  # WRF:3537
    _put(v, 'qv_tmp', None, v['qv_cld'] + (-v['qcnuc'] - v['qccon'] - v['qrcon'] + v['qcevp'] + v['qrevp']) * v['dt'], alive)  # WRF:3542
    _put(v, 't_tmp', None, v['t'] + (v['qcnuc'] + v['qccon'] + v['qrcon'] - v['qcevp'] - v['qrevp']) * v['xxlv'] * v['inv_cp'] * v['dt'], alive)  # WRF:3543
    _put(v, 'dumqvi', None, qv_sat(v['t_tmp'], v['pres'], 1), alive)  # WRF:3544
    _put(v, 'qdep_satadj', None, (v['qv_tmp'] - v['dumqvi']) / (_r(1.0) + v['xxls'] ** 2 * v['dumqvi'] / (v['cp'] * v['rv'] * v['t_tmp'] ** 2)) * v['odt'] * v['scf'], alive)  # WRF:3545
    _put(v, 'tmp1', None, _sum(v['qidep']) + _sum(v['qinuc']), alive)  # WRF:3547
    mask_93 = jnp.logical_and(alive, jnp.logical_and(v['tmp1'] > _r(0.0), v['qdep_satadj'] < _r(0.0)))
    taken_93 = mask_93
    _put(v, 'qidep', None, _r(0.0), mask_93)  # WRF:3549
    _put(v, 'qinuc', None, _r(0.0), mask_93)  # WRF:3550
    _put(v, 'ninuc', None, _r(0.0), mask_93)  # WRF:3551
    mask_94 = jnp.logical_and(alive, jnp.logical_not(taken_93))
    mask_95 = jnp.logical_and(mask_94, jnp.logical_and(v['tmp1'] > _r(0.0), v['tmp1'] > v['qdep_satadj']))
    taken_95 = mask_95
    _put(v, 'ratio', None, _max(_r(0.0), v['qdep_satadj']) / v['tmp1'], mask_95)  # WRF:3554
    _put(v, 'ratio', None, _min(_r(1.0), v['ratio']), mask_95)  # WRF:3555
    _put(v, 'qidep', None, v['qidep'] * _expand(v['ratio'], 1), mask_95)  # WRF:3556
    _put(v, 'qinuc', None, v['qinuc'] * _expand(v['ratio'], 1), mask_95)  # WRF:3557
    _put(v, 'ninuc', None, v['ninuc'] * _expand(v['ratio'], 1), mask_95)  # WRF:3558
    for iice in range(1, (ncat) + (1 if (1) > 0 else -1), 1):
        _put(v, 'dum', None, _max(_get(v['qisub'], (iice - 1,)), _r(1e-20)), mask_94)  # WRF:3561
        _put(v, 'qisub', ((iice) - 1,), _get(v['qisub'], (iice - 1,)) * _min(_r(1.0), _max(_r(0.0), -v['qdep_satadj']) / _max(_sum(v['qisub']), _r(1e-20))), mask_94)  # WRF:3562
        _put(v, 'nisub', ((iice) - 1,), _get(v['nisub'], (iice - 1,)) * _min(_r(1.0), _get(v['qisub'], (iice - 1,)) / v['dum']), mask_94)  # WRF:3563
    if observer is not None: observer('r1', v)
    _put(v, 'sinks', None, (v['qcaut'] + v['qcacc'] + _sum(v['qccol']) + v['qcevp'] + _sum(v['qchetc']) + _sum(v['qcheti']) + _sum(v['qcshd'])) * v['dt'], alive)  # WRF:3571
    _put(v, 'sources', None, v['qc'] + (v['qccon'] + v['qcnuc']) * v['dt'], alive)  # WRF:3572
    mask_96 = jnp.logical_and(alive, jnp.logical_and(v['sinks'] > v['sources'], v['sinks'] >= _r(1e-20)))
    taken_96 = mask_96
    _put(v, 'ratio', None, v['sources'] / v['sinks'], mask_96)  # WRF:3574
    _put(v, 'qcaut', None, v['qcaut'] * v['ratio'], mask_96)  # WRF:3575
    _put(v, 'qcacc', None, v['qcacc'] * v['ratio'], mask_96)  # WRF:3576
    _put(v, 'qcevp', None, v['qcevp'] * v['ratio'], mask_96)  # WRF:3577
    _put(v, 'qccol', None, v['qccol'] * _expand(v['ratio'], 1), mask_96)  # WRF:3578
    _put(v, 'qcheti', None, v['qcheti'] * _expand(v['ratio'], 1), mask_96)  # WRF:3579
    _put(v, 'qcshd', None, v['qcshd'] * _expand(v['ratio'], 1), mask_96)  # WRF:3580
    _put(v, 'ncautc', None, v['ncautc'] * v['ratio'], mask_96)  # WRF:3584
    _put(v, 'ncacc', None, v['ncacc'] * v['ratio'], mask_96)  # WRF:3585
    _put(v, 'nccol', None, v['nccol'] * _expand(v['ratio'], 1), mask_96)  # WRF:3586
    _put(v, 'ncheti', None, v['ncheti'] * _expand(v['ratio'], 1), mask_96)  # WRF:3587
    _put(v, 'sinks', None, (v['qrevp'] + _sum(v['qrcol']) + _sum(v['qrhetc']) + _sum(v['qrheti']) + _sum(v['qrmul'])) * v['dt'], alive)  # WRF:3593
    _put(v, 'sources', None, v['qr'] + (v['qrcon'] + v['qcaut'] + v['qcacc'] + _sum(v['qimlt']) + _sum(v['qcshd'])) * v['dt'], alive)  # WRF:3594
    mask_97 = jnp.logical_and(alive, jnp.logical_and(v['sinks'] > v['sources'], v['sinks'] >= _r(1e-20)))
    taken_97 = mask_97
    _put(v, 'ratio', None, v['sources'] / v['sinks'], mask_97)  # WRF:3596
    _put(v, 'qrevp', None, v['qrevp'] * v['ratio'], mask_97)  # WRF:3597
    _put(v, 'qrcol', None, v['qrcol'] * _expand(v['ratio'], 1), mask_97)  # WRF:3598
    _put(v, 'qrheti', None, v['qrheti'] * _expand(v['ratio'], 1), mask_97)  # WRF:3599
    _put(v, 'qrmul', None, v['qrmul'] * _expand(v['ratio'], 1), mask_97)  # WRF:3600
    _put(v, 'nrevp', None, v['nrevp'] * v['ratio'], mask_97)  # WRF:3601
    _put(v, 'nrcol', None, v['nrcol'] * _expand(v['ratio'], 1), mask_97)  # WRF:3602
    _put(v, 'nrheti', None, v['nrheti'] * _expand(v['ratio'], 1), mask_97)  # WRF:3603
    for iice in range(1, (ncat) + (1 if (1) > 0 else -1), 1):
        _put(v, 'sinks', None, (_get(v['qisub'], (iice - 1,)) + _get(v['qimlt'], (iice - 1,))) * v['dt'], alive)  # WRF:3610
        _put(v, 'sources', None, _get(v['qitot'], (iice - 1,)) + (_get(v['qidep'], (iice - 1,)) + _get(v['qinuc'], (iice - 1,)) + _get(v['qrcol'], (iice - 1,)) + _get(v['qccol'], (iice - 1,)) + _get(v['qrhetc'], (iice - 1,)) + _get(v['qrheti'], (iice - 1,)) + _get(v['qchetc'], (iice - 1,)) + _get(v['qcheti'], (iice - 1,)) + _get(v['qrmul'], (iice - 1,))) * v['dt'], alive)  # WRF:3611
        for catcoll in range(1, (ncat) + (1 if (1) > 0 else -1), 1):
            _put(v, 'sources', None, v['sources'] + _get(v['qicol'], (catcoll - 1, iice - 1)) * v['dt'], alive)  # WRF:3615
            _put(v, 'sinks', None, v['sinks'] + _get(v['qicol'], (iice - 1, catcoll - 1)) * v['dt'], alive)  # WRF:3617
        mask_98 = jnp.logical_and(alive, jnp.logical_and(v['sinks'] > v['sources'], v['sinks'] >= _r(1e-20)))
        taken_98 = mask_98
        _put(v, 'ratio', None, v['sources'] / v['sinks'], mask_98)  # WRF:3620
        _put(v, 'qisub', ((iice) - 1,), _get(v['qisub'], (iice - 1,)) * v['ratio'], mask_98)  # WRF:3621
        _put(v, 'qimlt', ((iice) - 1,), _get(v['qimlt'], (iice - 1,)) * v['ratio'], mask_98)  # WRF:3622
        _put(v, 'nisub', ((iice) - 1,), _get(v['nisub'], (iice - 1,)) * v['ratio'], mask_98)  # WRF:3623
        _put(v, 'nimlt', ((iice) - 1,), _get(v['nimlt'], (iice - 1,)) * v['ratio'], mask_98)  # WRF:3624
        for catcoll in range(1, (ncat) + (1 if (1) > 0 else -1), 1):
            _put(v, 'qicol', ((iice) - 1, (catcoll) - 1,), _get(v['qicol'], (iice - 1, catcoll - 1)) * v['ratio'], mask_98)  # WRF:3626
            _put(v, 'nicol', ((iice) - 1, (catcoll) - 1,), _get(v['nicol'], (iice - 1, catcoll - 1)) * v['ratio'], mask_98)  # WRF:3627
    _put(v, 'sinks', None, (v['qccon'] + v['qrcon'] + v['qcnuc'] + _sum(v['qidep']) + _sum(v['qinuc'])) * v['dt'], alive)  # WRF:3633
    _put(v, 'sources', None, v['qv'] + (v['qcevp'] + v['qrevp'] + _sum(v['qisub'])) * v['dt'], alive)  # WRF:3634
    mask_99 = jnp.logical_and(alive, jnp.logical_and(v['sinks'] > v['sources'], v['sinks'] >= _r(1e-20)))
    taken_99 = mask_99
    _put(v, 'ratio', None, v['sources'] / v['sinks'], mask_99)  # WRF:3636
    _put(v, 'qccon', None, v['qccon'] * v['ratio'], mask_99)  # WRF:3637
    _put(v, 'qrcon', None, v['qrcon'] * v['ratio'], mask_99)  # WRF:3638
    _put(v, 'qcnuc', None, v['qcnuc'] * v['ratio'], mask_99)  # WRF:3639
    _put(v, 'qidep', None, v['qidep'] * _expand(v['ratio'], 1), mask_99)  # WRF:3640
    _put(v, 'qinuc', None, v['qinuc'] * _expand(v['ratio'], 1), mask_99)  # WRF:3641
    _put(v, 'ninuc', None, v['ninuc'] * _expand(v['ratio'], 1), mask_99)  # WRF:3642
    _put(v, 'ncnuc', None, v['ncnuc'] * v['ratio'], mask_99)  # WRF:3643
    if log_3momentice:
        for iice in range(1, (ncat) + (1 if (1) > 0 else -1), 1):
            _put(v, 'dumm3', ((iice) - 1,), _get(v['qitot'], (iice - 1,)) + (_get(v['qidep'], (iice - 1,)) + _get(v['qrcol'], (iice - 1,)) + _get(v['qccol'], (iice - 1,)) + _get(v['qrmul'], (iice - 1,)) - _get(v['qisub'], (iice - 1,)) - _get(v['qimlt'], (iice - 1,))) * v['dt'], alive)  # WRF:3662
            _put(v, 'dumm0', ((iice) - 1,), _get(v['nitot'], (iice - 1,)) + (-_get(v['nimlt'], (iice - 1,)) - _get(v['nisub'], (iice - 1,)) + _get(v['nimul'], (iice - 1,)) - _get(v['nislf'], (iice - 1,))) * v['dt'], alive)  # WRF:3665
        for iice in range(1, (ncat) + (1 if (1) > 0 else -1), 1):
            for catcoll in range(1, (ncat) + (1 if (1) > 0 else -1), 1):
                _put(v, 'dumm3', ((catcoll) - 1,), _get(v['dumm3'], (catcoll - 1,)) - _get(v['qicol'], (catcoll - 1, iice - 1)) * v['dt'], alive)  # WRF:3673
                _put(v, 'dumm3', ((iice) - 1,), _get(v['dumm3'], (iice - 1,)) + _get(v['qicol'], (catcoll - 1, iice - 1)) * v['dt'], alive)  # WRF:3674
                _put(v, 'dumm0', ((catcoll) - 1,), _get(v['dumm0'], (catcoll - 1,)) - _get(v['nicol'], (catcoll - 1, iice - 1)) * v['dt'], alive)  # WRF:3675
            mask_100 = jnp.logical_and(alive, _get(v['dumm3'], (iice - 1,)) >= v['qsmall'])
            taken_100 = mask_100
            mask_101 = jnp.logical_and(mask_100, _get(v['qitot'], (iice - 1,)) >= v['qsmall'])
            taken_101 = mask_101
            _put(v, 'dumm3', ((iice) - 1,), _r(6.0) / (v['f1pr16'] * v['pi']) * _get(v['dumm3'], (iice - 1,)), mask_101)  # WRF:3683
            mask_102 = jnp.logical_and(mask_100, jnp.logical_not(taken_101))
            _put(v, 'dumm3', ((iice) - 1,), _r(6.0) / (_r(900.0) * v['pi']) * _get(v['dumm3'], (iice - 1,)), mask_102)  # WRF:3686
            mask_103 = jnp.logical_and(mask_100, _get(v['qitot'], (iice - 1,)) >= v['qsmall'])
            taken_103 = mask_103
            _put(v, 'dum1', None, _r(6.0) / (v['f1pr16'] * v['pi']) * _get(v['qitot'], (iice - 1,)), mask_103)  # WRF:3692
            _put(v, 'mu_i', None, compute_mu_3moment(_get(v['nitot'], (iice - 1,)), v['dum1'], _get(v['zitot'], (iice - 1,)), v['mu_i_max']), mask_103)  # WRF:3693
            mask_104 = jnp.logical_and(mask_100, jnp.logical_not(taken_103))
            _put(v, 'mu_i', None, v['mu_i_initial'], mask_104)  # WRF:3696
            _put(v, 'zitot', ((iice) - 1,), g_of_mu(v['mu_i']) * _get(v['dumm3'], (iice - 1,)) ** 2 / _max(_get(v['dumm0'], (iice - 1,)), v['nsmall']), mask_100)  # WRF:3700
            mask_105 = jnp.logical_and(alive, jnp.logical_not(taken_100))
            _put(v, 'zitot', ((iice) - 1,), _r(0.0), mask_105)  # WRF:3703
            _put(v, 'tmp2', None, _get(v['nrhetc'], (iice - 1,)) + _get(v['nrheti'], (iice - 1,)), alive)  # WRF:3712
            mask_106 = jnp.logical_and(alive, v['tmp2'] >= v['qsmall'])
            taken_106 = mask_106
            _put(v, 'tmp1', None, (_get(v['qrhetc'], (iice - 1,)) + _get(v['qrheti'], (iice - 1,))) * _r(6.0) / (_r(900.0) * v['pi']), mask_106)  # WRF:3714
            _put(v, 'mu_i_new', None, v['mu_r'], mask_106)  # WRF:3715
            _put(v, 'zitot', ((iice) - 1,), _get(v['zitot'], (iice - 1,)) + g_of_mu(v['mu_i_new']) * v['tmp1'] ** 2 / v['tmp2'] * v['dt'], mask_106)  # WRF:3716
            _put(v, 'tmp2', None, _get(v['nchetc'], (iice - 1,)) + _get(v['ncheti'], (iice - 1,)), alive)  # WRF:3720
            mask_107 = jnp.logical_and(alive, v['tmp2'] >= v['qsmall'])
            taken_107 = mask_107
            _put(v, 'tmp1', None, (_get(v['qchetc'], (iice - 1,)) + _get(v['qcheti'], (iice - 1,))) * _r(6.0) / (_r(900.0) * v['pi']), mask_107)  # WRF:3722
            _put(v, 'mu_i_new', None, v['mu_c'], mask_107)  # WRF:3723
            _put(v, 'zitot', ((iice) - 1,), _get(v['zitot'], (iice - 1,)) + g_of_mu(v['mu_i_new']) * v['tmp1'] ** 2 / v['tmp2'] * v['dt'], mask_107)  # WRF:3724
            _put(v, 'tmp2', None, _get(v['ninuc'], (iice - 1,)), alive)  # WRF:3728
            mask_108 = jnp.logical_and(alive, v['tmp2'] >= v['qsmall'])
            taken_108 = mask_108
            _put(v, 'tmp1', None, _get(v['qinuc'], (iice - 1,)) * _r(6.0) / (_r(900.0) * v['pi']), mask_108)  # WRF:3730
            _put(v, 'mu_i_new', None, v['mu_i_initial'], mask_108)  # WRF:3731
            _put(v, 'zitot', ((iice) - 1,), _get(v['zitot'], (iice - 1,)) + g_of_mu(v['mu_i_new']) * v['tmp1'] ** 2 / v['tmp2'] * v['dt'], mask_108)  # WRF:3732
    for iice in range(1, (ncat) + (1 if (1) > 0 else -1), 1):
        mask_109 = jnp.logical_and(alive, _get(v['qitot'], (iice - 1,)) >= v['qsmall'])
        taken_109 = mask_109
        _put(v, 'tmp1', None, _r(1.0) / _get(v['qitot'], (iice - 1,)), mask_109)  # WRF:3760
        _put(v, 'rimevolume', ((iice) - 1,), _get(v['birim'], (iice - 1,)) * v['tmp1'], mask_109)  # WRF:3761
        _put(v, 'rimefraction', ((iice) - 1,), _get(v['qirim'], (iice - 1,)) * v['tmp1'], mask_109)  # WRF:3762
    if observer is not None: observer('r2', v)
    for iice in range(1, (ncat) + (1 if (1) > 0 else -1), 1):
        _put(v, 'qc', None, v['qc'] + (-_get(v['qchetc'], (iice - 1,)) - _get(v['qcheti'], (iice - 1,)) - _get(v['qccol'], (iice - 1,)) - _get(v['qcshd'], (iice - 1,))) * v['dt'], alive)  # WRF:3769
        if log_predictnc:
            _put(v, 'nc', None, _real_binary('add', v['nc'], _real_binary('mul', _real_binary('sub', _real_binary('sub', -_get(v['nccol'], (iice - 1,)), _get(v['nchetc'], (iice - 1,))), _get(v['ncheti'], (iice - 1,))), v['dt'])), alive)  # WRF:3771
        _put(v, 'qr', None, _real_binary('add', v['qr'], _real_binary('mul', _real_binary('sub', _real_binary('add', _real_binary('sub', _real_binary('sub', _real_binary('add', -_get(v['qrcol'], (iice - 1,)), _get(v['qimlt'], (iice - 1,))), _get(v['qrhetc'], (iice - 1,))), _get(v['qrheti'], (iice - 1,))), _get(v['qcshd'], (iice - 1,))), _get(v['qrmul'], (iice - 1,))), v['dt'])), alive)  # WRF:3774
        _put(v, 'nr', None, _real_binary('add', v['nr'], _real_binary('mul', _real_binary('add', _real_binary('add', _real_binary('add', _real_binary('sub', _real_binary('sub', -_get(v['nrcol'], (iice - 1,)), _get(v['nrhetc'], (iice - 1,))), _get(v['nrheti'], (iice - 1,))), _real_binary('mul', v['nmltratio'], _get(v['nimlt'], (iice - 1,)))), _get(v['nrshdr'], (iice - 1,))), _get(v['ncshdc'], (iice - 1,))), v['dt'])), alive)  # WRF:3778
        _put(v, 'birim', ((iice) - 1,), _real_binary('sub', _get(v['birim'], (iice - 1,)), _real_binary('mul', _real_binary('mul', _real_binary('add', _get(v['qisub'], (iice - 1,)), _get(v['qimlt'], (iice - 1,))), v['dt']), _get(v['rimevolume'], (iice - 1,)))), alive)  # WRF:3783
        _put(v, 'qirim', ((iice) - 1,), _real_binary('sub', _get(v['qirim'], (iice - 1,)), _real_binary('mul', _real_binary('mul', _real_binary('add', _get(v['qisub'], (iice - 1,)), _get(v['qimlt'], (iice - 1,))), v['dt']), _get(v['rimefraction'], (iice - 1,)))), alive)  # WRF:3785
        _put(v, 'qitot', ((iice) - 1,), _real_binary('sub', _get(v['qitot'], (iice - 1,)), _real_binary('mul', _real_binary('add', _get(v['qisub'], (iice - 1,)), _get(v['qimlt'], (iice - 1,))), v['dt'])), alive)  # WRF:3787
        _put(v, 'dum', None, _real_binary('mul', _real_binary('add', _real_binary('add', _real_binary('add', _real_binary('add', _real_binary('add', _real_binary('add', _get(v['qrcol'], (iice - 1,)), _get(v['qccol'], (iice - 1,))), _get(v['qrhetc'], (iice - 1,))), _get(v['qrheti'], (iice - 1,))), _get(v['qchetc'], (iice - 1,))), _get(v['qcheti'], (iice - 1,))), _get(v['qrmul'], (iice - 1,))), v['dt']), alive)  # WRF:3790
        _put(v, 'qitot', ((iice) - 1,), _real_binary('add', _real_binary('add', _get(v['qitot'], (iice - 1,)), _real_binary('mul', _real_binary('add', _get(v['qidep'], (iice - 1,)), _get(v['qinuc'], (iice - 1,))), v['dt'])), v['dum']), alive)  # WRF:3792
        _put(v, 'qirim', ((iice) - 1,), _real_binary('add', _get(v['qirim'], (iice - 1,)), v['dum']), alive)  # WRF:3793
        _put(v, 'birim', ((iice) - 1,), _real_binary('add', _get(v['birim'], (iice - 1,)), _real_binary('mul', _real_binary('add', _real_binary('add', _real_binary('mul', _get(v['qrcol'], (iice - 1,)), v['inv_rho_rimemax']), _real_binary('div', _get(v['qccol'], (iice - 1,)), _get(v['rhorime_c'], (iice - 1,)))), _real_binary('mul', _real_binary('add', _real_binary('add', _real_binary('add', _real_binary('add', _get(v['qrhetc'], (iice - 1,)), _get(v['qrheti'], (iice - 1,))), _get(v['qchetc'], (iice - 1,))), _get(v['qcheti'], (iice - 1,))), _get(v['qrmul'], (iice - 1,))), v['inv_rho_rimemax'])), v['dt'])), alive)  # WRF:3794
        _put(v, 'nitot', ((iice) - 1,), _real_binary('add', _get(v['nitot'], (iice - 1,)), _real_binary('mul', _real_binary('add', _real_binary('add', _real_binary('add', _real_binary('add', _real_binary('add', _real_binary('sub', _real_binary('sub', _real_binary('sub', _get(v['ninuc'], (iice - 1,)), _get(v['nimlt'], (iice - 1,))), _get(v['nisub'], (iice - 1,))), _get(v['nislf'], (iice - 1,))), _get(v['nrhetc'], (iice - 1,))), _get(v['nrheti'], (iice - 1,))), _get(v['nchetc'], (iice - 1,))), _get(v['ncheti'], (iice - 1,))), _get(v['nimul'], (iice - 1,))), v['dt'])), alive)  # WRF:3797
        if ncat > 1:
            for catcoll in range(1, (ncat) + (1 if (1) > 0 else -1), 1):
                if iice != catcoll:
                    mask_110 = jnp.logical_and(alive, _get(v['qitot'], (catcoll - 1,)) >= v['qsmall'])
                    taken_110 = mask_110
                    _put(v, 'qirim', ((iice) - 1,), _real_binary('add', _get(v['qirim'], (iice - 1,)), _real_binary('mul', _real_binary('mul', _get(v['qicol'], (catcoll - 1, iice - 1)), v['dt']), _get(v['rimefraction'], (catcoll - 1,)))), mask_110)  # WRF:3811
                    _put(v, 'birim', ((iice) - 1,), _real_binary('add', _get(v['birim'], (iice - 1,)), _real_binary('mul', _real_binary('mul', _get(v['qicol'], (catcoll - 1, iice - 1)), v['dt']), _get(v['rimevolume'], (catcoll - 1,)))), mask_110)  # WRF:3813
                    _put(v, 'qirim', ((catcoll) - 1,), _real_binary('sub', _get(v['qirim'], (catcoll - 1,)), _real_binary('mul', _real_binary('mul', _get(v['qicol'], (catcoll - 1, iice - 1)), v['dt']), _get(v['rimefraction'], (catcoll - 1,)))), mask_110)  # WRF:3816
                    _put(v, 'birim', ((catcoll) - 1,), _real_binary('sub', _get(v['birim'], (catcoll - 1,)), _real_binary('mul', _real_binary('mul', _get(v['qicol'], (catcoll - 1, iice - 1)), v['dt']), _get(v['rimevolume'], (catcoll - 1,)))), mask_110)  # WRF:3818
                    _put(v, 'qitot', ((catcoll) - 1,), _real_binary('sub', _get(v['qitot'], (catcoll - 1,)), _real_binary('mul', _get(v['qicol'], (catcoll - 1, iice - 1)), v['dt'])), alive)  # WRF:3821
                    _put(v, 'nitot', ((catcoll) - 1,), _real_binary('sub', _get(v['nitot'], (catcoll - 1,)), _real_binary('mul', _get(v['nicol'], (catcoll - 1, iice - 1)), v['dt'])), alive)  # WRF:3822
                    _put(v, 'qitot', ((iice) - 1,), _real_binary('add', _get(v['qitot'], (iice - 1,)), _real_binary('mul', _get(v['qicol'], (catcoll - 1, iice - 1)), v['dt'])), alive)  # WRF:3823
        mask_111 = jnp.logical_and(alive, _get(v['qirim'], (iice - 1,)) < _r(0.0))
        taken_111 = mask_111
        _put(v, 'qirim', ((iice) - 1,), _r(0.0), mask_111)  # WRF:3830
        _put(v, 'birim', ((iice) - 1,), _r(0.0), mask_111)  # WRF:3831
        mask_112 = jnp.logical_and(alive, _get(v['log_wetgrowth'], (iice - 1,)))
        taken_112 = mask_112
        _put(v, 'qirim', ((iice) - 1,), _get(v['qitot'], (iice - 1,)), mask_112)  # WRF:3836
        _put(v, 'birim', ((iice) - 1,), _real_binary('mul', _get(v['qirim'], (iice - 1,)), v['inv_rho_rimemax']), mask_112)  # WRF:3837
        mask_113 = jnp.logical_and(alive, jnp.logical_and(jnp.logical_and(_get(v['qitot'], (iice - 1,)) >= v['qsmall'], _get(v['birim'], (iice - 1,)) >= v['bsmall']), _get(v['qimlt'], (iice - 1,)) > _r(0.0)))
        taken_113 = mask_113
        _put(v, 'tmp1', None, _real_binary('div', _get(v['qirim'], (iice - 1,)), _get(v['birim'], (iice - 1,))), mask_113)  # WRF:3842
        _put(v, 'tmp2', None, _real_binary('add', _get(v['qitot'], (iice - 1,)), _real_binary('mul', _get(v['qimlt'], (iice - 1,)), v['dt'])), mask_113)  # WRF:3843
        _put(v, 'birim', ((iice) - 1,), _real_binary('div', _get(v['qirim'], (iice - 1,)), _real_binary('add', v['tmp1'], _real_binary('div', _real_binary('mul', _real_binary('mul', _real_binary('sub', _r(917.0), v['tmp1']), _get(v['qimlt'], (iice - 1,))), v['dt']), v['tmp2']))), mask_113)  # WRF:3844
        _put(v, 'qv', None, _real_binary('add', v['qv'], _real_binary('mul', _real_binary('sub', _real_binary('add', -_get(v['qidep'], (iice - 1,)), _get(v['qisub'], (iice - 1,))), _get(v['qinuc'], (iice - 1,))), v['dt'])), alive)  # WRF:3854
        _put(v, 'th', None, _real_binary('add', v['th'], _real_binary('mul', _real_binary('mul', v['invexn'], _real_binary('add', _real_binary('mul', _real_binary('mul', _real_binary('add', _real_binary('sub', _get(v['qidep'], (iice - 1,)), _get(v['qisub'], (iice - 1,))), _get(v['qinuc'], (iice - 1,))), v['xxls']), v['inv_cp']), _real_binary('mul', _real_binary('mul', _real_binary('sub', _real_binary('add', _real_binary('add', _real_binary('add', _real_binary('add', _real_binary('add', _real_binary('add', _get(v['qrcol'], (iice - 1,)), _get(v['qccol'], (iice - 1,))), _get(v['qchetc'], (iice - 1,))), _get(v['qcheti'], (iice - 1,))), _get(v['qrhetc'], (iice - 1,))), _get(v['qrheti'], (iice - 1,))), _get(v['qrmul'], (iice - 1,))), _get(v['qimlt'], (iice - 1,))), v['xlf']), v['inv_cp']))), v['dt'])), alive)  # WRF:3859
    _put(v, 'qc', None, _real_binary('add', v['qc'], _real_binary('mul', _real_binary('sub', _real_binary('add', _real_binary('add', _real_binary('sub', -v['qcacc'], v['qcaut']), v['qcnuc']), v['qccon']), v['qcevp']), v['dt'])), alive)  # WRF:3869
    _put(v, 'qr', None, _real_binary('add', v['qr'], _real_binary('mul', _real_binary('sub', _real_binary('add', _real_binary('add', v['qcacc'], v['qcaut']), v['qrcon']), v['qrevp']), v['dt'])), alive)  # WRF:3870
    if log_predictnc:
        _put(v, 'nc', None, _real_binary('add', v['nc'], _real_binary('mul', _real_binary('add', _real_binary('add', _real_binary('sub', -v['ncacc'], v['ncautc']), v['ncslf']), v['ncnuc']), v['dt'])), alive)  # WRF:3873
    else:
        _put(v, 'nc', None, _real_binary('mul', v['nccnst'], v['inv_rho']), alive)  # WRF:3875
    if iparam == 1 or iparam == 2:
        _put(v, 'nr', None, _real_binary('add', v['nr'], _real_binary('mul', _real_binary('sub', _real_binary('sub', _real_binary('mul', _r(0.5), v['ncautc']), v['nrslf']), v['nrevp']), v['dt'])), alive)  # WRF:3878
    else:
        _put(v, 'nr', None, _real_binary('add', v['nr'], _real_binary('mul', _real_binary('sub', _real_binary('sub', v['ncautr'], v['nrslf']), v['nrevp']), v['dt'])), alive)  # WRF:3880
    _put(v, 'qv', None, _real_binary('add', v['qv'], _real_binary('mul', _real_binary('add', _real_binary('add', _real_binary('sub', _real_binary('sub', -v['qcnuc'], v['qccon']), v['qrcon']), v['qcevp']), v['qrevp']), v['dt'])), alive)  # WRF:3883
    _put(v, 'th', None, _real_binary('add', v['th'], _real_binary('mul', _real_binary('mul', v['invexn'], _real_binary('mul', _real_binary('mul', _real_binary('sub', _real_binary('sub', _real_binary('add', _real_binary('add', v['qcnuc'], v['qccon']), v['qrcon']), v['qcevp']), v['qrevp']), v['xxlv']), v['inv_cp'])), v['dt'])), alive)  # WRF:3884
    mask_114 = jnp.logical_and(alive, v['qc'] < v['qsmall'])
    taken_114 = mask_114
    _put(v, 'qv', None, v['qv'] + v['qc'], mask_114)  # WRF:3890
    _put(v, 'th', None, v['th'] - v['invexn'] * v['qc'] * v['xxlv'] * v['inv_cp'], mask_114)  # WRF:3891
    _put(v, 'qc', None, _r(0.0), mask_114)  # WRF:3892
    _put(v, 'nc', None, _r(0.0), mask_114)  # WRF:3893
    mask_115 = jnp.logical_and(alive, jnp.logical_not(taken_114))
    _put(v, 'log_hydrometeorspresent', None, True, mask_115)  # WRF:3895
    mask_116 = jnp.logical_and(alive, v['qr'] < v['qsmall'])
    taken_116 = mask_116
    _put(v, 'qv', None, v['qv'] + v['qr'], mask_116)  # WRF:3899
    _put(v, 'th', None, v['th'] - v['invexn'] * v['qr'] * v['xxlv'] * v['inv_cp'], mask_116)  # WRF:3900
    _put(v, 'qr', None, _r(0.0), mask_116)  # WRF:3901
    _put(v, 'nr', None, _r(0.0), mask_116)  # WRF:3902
    mask_117 = jnp.logical_and(alive, jnp.logical_not(taken_116))
    _put(v, 'log_hydrometeorspresent', None, True, mask_117)  # WRF:3904
    for iice in range(1, (ncat) + (1 if (1) > 0 else -1), 1):
        mask_118 = jnp.logical_and(alive, _get(v['qitot'], (iice - 1,)) < v['qsmall'])
        taken_118 = mask_118
        _put(v, 'qv', None, v['qv'] + _get(v['qitot'], (iice - 1,)), mask_118)  # WRF:3909
        _put(v, 'th', None, v['th'] - v['invexn'] * _get(v['qitot'], (iice - 1,)) * v['xxls'] * v['inv_cp'], mask_118)  # WRF:3910
        _put(v, 'qitot', ((iice) - 1,), _r(0.0), mask_118)  # WRF:3911
        _put(v, 'nitot', ((iice) - 1,), _r(0.0), mask_118)  # WRF:3912
        _put(v, 'qirim', ((iice) - 1,), _r(0.0), mask_118)  # WRF:3913
        _put(v, 'birim', ((iice) - 1,), _r(0.0), mask_118)  # WRF:3914
        mask_119 = jnp.logical_and(alive, jnp.logical_not(taken_118))
        _put(v, 'log_hydrometeorspresent', None, True, mask_119)  # WRF:3916
    _put(v, 'qv', None, _max(_r(0.0), v['qv']), alive)  # WRF:3920
    call_out = _call_impose_max_total_ni(v['nitot'], v['max_total_ni'], v['inv_rho'], tables=tables, three=log_3momentice)
    _put(v, 'nitot', None, call_out[0], alive)  # WRF:3921
    return v
