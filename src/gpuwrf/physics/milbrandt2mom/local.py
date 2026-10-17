"""Generated literal Milbrandt-Yau local processes; regenerate with proofs/s2_milbrandt/generate_point.py.
Pristine source SHA256: 76d11c00976f3c7dac30ff6ebb3efac1698bf43c13131b3aae43e464f22d23e5
"""
import jax.numpy as jnp
from .math import (_dm_x, _ilamda_x, _des_of_ds, _n_cooper, _nos_thompson,
                   _nccnfnc, _sxfnc, _qsat, _dim)
from .rounding import _op, _pow, _exp, _log, _log10, real, _not

def local_processes(fields, pres, wz, dt, c, point_mask):
    t, q, qc, qr, qi, qn, qg, qh, nc, nr, ny, nn, ng, nh = fields
    _r = lambda x: real(x, t.dtype)
    _d = lambda x: jnp.asarray(x, jnp.float64)
    ccntype = 2
    icephase_on = warmphase_on = snow_on = autoconv_on = True
    dt = _r(dt)
    idt = _op('div',_r(1.),dt)
    de = _op('div',pres,_op('mul',_r(c['rgasd']),t))
    ide = _op('div',_r(1.),de)
    gamfact = jnp.sqrt(_op('div',_r(c['deo']),de))
    qsw = _qsat(t, pres, 0)
    qsi = _qsat(t, pres, 1)
    nc, nr, ny, nn, ng, nh = (_op('mul',x,de) for x in (nc, nr, ny, nn, ng, nh))
    aa11 = c['aa11']
    aa22 = c['aa22']
    abigg = c['abigg']
    afg = c['afg']
    afh = c['afh']
    afi = c['afi']
    afr = c['afr']
    afs = c['afs']
    alpha_c = c['alpha_c']
    alpha_g = c['alpha_g']
    alpha_h = c['alpha_h']
    alpha_i = c['alpha_i']
    alpha_r = c['alpha_r']
    alpha_s = c['alpha_s']
    avx = c['avx']
    bbigg = c['bbigg']
    bfg = c['bfg']
    bfh = c['bfh']
    bfi = c['bfi']
    bfr = c['bfr']
    bfs = c['bfs']
    bvx = c['bvx']
    capfact_i = c['capfact_i']
    capfact_s = c['capfact_s']
    cappa = c['cappa']
    cexr1 = c['cexr1']
    cexr2 = c['cexr2']
    cexr3 = c['cexr3']
    cexr4 = c['cexr4']
    cexr5 = c['cexr5']
    cexr6 = c['cexr6']
    cexr9 = c['cexr9']
    cexs1 = c['cexs1']
    cexs2 = c['cexs2']
    chlc = c['chlc']
    chlf = c['chlf']
    chls = c['chls']
    ck5 = c['ck5']
    ck6 = c['ck6']
    ckqg1 = c['ckqg1']
    ckqg2 = c['ckqg2']
    ckqg4 = c['ckqg4']
    ckqh1 = c['ckqh1']
    ckqh2 = c['ckqh2']
    ckqh4 = c['ckqh4']
    ckqi1 = c['ckqi1']
    ckqi2 = c['ckqi2']
    ckqi4 = c['ckqi4']
    ckqr1 = c['ckqr1']
    ckqr2 = c['ckqr2']
    ckqr3 = c['ckqr3']
    ckqs1 = c['ckqs1']
    ckqs2 = c['ckqs2']
    cmg = c['cmg']
    cmh = c['cmh']
    cmi = c['cmi']
    cmr = c['cmr']
    cms = c['cms']
    cms_d3 = c['cms_d3']
    cnsgthres = c['cnsgthres']
    consol = c['consol']
    cpd = c['cpd']
    cpi = c['cpi']
    cpv = c['cpv']
    cpw = c['cpw']
    debug_abort = c['debug_abort']
    debug_on = c['debug_on']
    deg = c['deg']
    deh = c['deh']
    dei = c['dei']
    delta = c['delta']
    deo = c['deo']
    desfix = c['desfix']
    desmax = c['desmax']
    dew = c['dew']
    dgmax = c['dgmax']
    dh_large = c['dh_large']
    dh_min = c['dh_min']
    dhh = c['dhh']
    dhmax = c['dhmax']
    dimax = c['dimax']
    dmg = c['dmg']
    dmh = c['dmh']
    dmi = c['dmi']
    dmr = c['dmr']
    dms = c['dms']
    dr_3cmpthrs = c['dr_3cmpthrs']
    dr_large = c['dr_large']
    drmax = c['drmax']
    drshed = c['drshed']
    ds_large = c['ds_large']
    dsmax = c['dsmax']
    dso = c['dso']
    eci = c['eci']
    eds = c['eds']
    eps = c['eps']
    eps1 = c['eps1']
    eps2 = c['eps2']
    epsn = c['epsn']
    epsq = c['epsq']
    epsq2 = c['epsq2']
    epsqg_sedi = c['epsqg_sedi']
    epsqh_sedi = c['epsqh_sedi']
    epsqi_sedi = c['epsqi_sedi']
    epsqr_sedi = c['epsqr_sedi']
    epsqs_sedi = c['epsqs_sedi']
    epsvis = c['epsvis']
    erh = c['erh']
    eri = c['eri']
    ers = c['ers']
    fdielec = c['fdielec']
    fds = c['fds']
    fv_dsmax = c['fv_dsmax']
    fv_dsmin = c['fv_dsmin']
    gc1 = c['gc1']
    gc11 = c['gc11']
    gc12 = c['gc12']
    gc13 = c['gc13']
    gc14 = c['gc14']
    gc15 = c['gc15']
    gc2 = c['gc2']
    gc3 = c['gc3']
    gc4 = c['gc4']
    gc5 = c['gc5']
    gc6 = c['gc6']
    gc7 = c['gc7']
    gc8 = c['gc8']
    gg09 = c['gg09']
    gg11 = c['gg11']
    gg12 = c['gg12']
    gg13 = c['gg13']
    gg31 = c['gg31']
    gg32 = c['gg32']
    gg33 = c['gg33']
    gg34 = c['gg34']
    gg35 = c['gg35']
    gg36 = c['gg36']
    gg40 = c['gg40']
    gg50 = c['gg50']
    gh09 = c['gh09']
    gh11 = c['gh11']
    gh12 = c['gh12']
    gh13 = c['gh13']
    gh31 = c['gh31']
    gh32 = c['gh32']
    gh33 = c['gh33']
    gh40 = c['gh40']
    gh50 = c['gh50']
    gi11 = c['gi11']
    gi20 = c['gi20']
    gi21 = c['gi21']
    gi22 = c['gi22']
    gi31 = c['gi31']
    gi32 = c['gi32']
    gi33 = c['gi33']
    gi34 = c['gi34']
    gi35 = c['gi35']
    gi36 = c['gi36']
    gi4 = c['gi4']
    gi40 = c['gi40']
    gi6 = c['gi6']
    gr17 = c['gr17']
    gr31 = c['gr31']
    gr32 = c['gr32']
    gr33 = c['gr33']
    gr34 = c['gr34']
    gr35 = c['gr35']
    gr36 = c['gr36']
    gr37 = c['gr37']
    gr50 = c['gr50']
    grav = c['grav']
    grpl_on = c['grpl_on']
    gs09 = c['gs09']
    gs11 = c['gs11']
    gs12 = c['gs12']
    gs13 = c['gs13']
    gs31 = c['gs31']
    gs32 = c['gs32']
    gs33 = c['gs33']
    gs34 = c['gs34']
    gs35 = c['gs35']
    gs36 = c['gs36']
    gs40 = c['gs40']
    gs40_d3 = c['gs40_d3']
    gs50 = c['gs50']
    hail_on = c['hail_on']
    icedep_on = c['icedep_on']
    icexc9 = c['icexc9']
    icexi9 = c['icexi9']
    icexr9 = c['icexr9']
    icexs2 = c['icexs2']
    ichlf = c['ichlf']
    icmg = c['icmg']
    icmh = c['icmh']
    icmi = c['icmi']
    icmr = c['icmr']
    icms = c['icms']
    ideg = c['ideg']
    ideh = c['ideh']
    idei = c['idei']
    idew = c['idew']
    idms = c['idms']
    igc1 = c['igc1']
    igc5 = c['igc5']
    igg31 = c['igg31']
    igg34 = c['igg34']
    igg99 = c['igg99']
    igh31 = c['igh31']
    igh34 = c['igh34']
    igh99 = c['igh99']
    igi31 = c['igi31']
    igr31 = c['igr31']
    igr34 = c['igr34']
    igs20 = c['igs20']
    igs20_d3 = c['igs20_d3']
    igs31 = c['igs31']
    igs34 = c['igs34']
    igs40 = c['igs40']
    ilammin1 = c['ilammin1']
    ilammin2 = c['ilammin2']
    imgo = c['imgo']
    imso = c['imso']
    imuc = c['imuc']
    kapa = c['kapa']
    karman = c['karman']
    kboltz = c['kboltz']
    kk1 = c['kk1']
    kk2 = c['kk2']
    knams = c['knams']
    lama0 = c['lama0']
    lamdas_min = c['lamdas_min']
    lcp = c['lcp']
    lfp = c['lfp']
    lsp = c['lsp']
    maxvis = c['maxvis']
    mgo = c['mgo']
    minzet = c['minzet']
    mio = c['mio']
    mso = c['mso']
    muc = c['muc']
    n_c_sm = c['n_c_sm']
    ngh_crit = c['ngh_crit']
    ni_max = c['ni_max']
    no_g_sm = c['no_g_sm']
    no_h_sm = c['no_h_sm']
    no_r_sm = c['no_r_sm']
    no_s_max = c['no_s_max']
    nti0 = c['nti0']
    omega = c['omega']
    outfreq = c['outfreq']
    p0 = c['p0']
    pi = c['pi']
    pi2 = c['pi2']
    piov4 = c['piov4']
    piov6 = c['piov6']
    primicenucl = c['primicenucl']
    ra = c['ra']
    rainaccr_on = c['rainaccr_on']
    rauw = c['rauw']
    rayt = c['rayt']
    rfact_fvfm = c['rfact_fvfm']
    rgasd = c['rgasd']
    rgasv = c['rgasv']
    rh = c['rh']
    ric = c['ric']
    satw_peak = c['satw_peak']
    sigcthrs = c['sigcthrs']
    sixth = c['sixth']
    snowspherical = c['snowspherical']
    stefan = c['stefan']
    stlo = c['stlo']
    t0 = c['t0']
    tc_fzrh = c['tc_fzrh']
    tcdk = c['tcdk']
    tgl = c['tgl']
    thrd = c['thrd']
    trpl = c['trpl']
    vgmax = c['vgmax']
    vhmax = c['vhmax']
    vimax = c['vimax']
    vrmax = c['vrmax']
    vsmax = c['vsmax']
    w_cngh = c['w_cngh']
    xdisp = c['xdisp']
    zfact = c['zfact']
    zmax_sedi = c['zmax_sedi']
    abw = _r(0.)
    activepoint = jnp.zeros_like(t, dtype=jnp.bool_)
    ccaccr = _r(0.)
    ccautr = _r(0.)
    ccscoc = _r(0.)
    cdiff = _r(0.)
    crscor = _r(0.)
    d_sll = _r(0.)
    dc = _r(0.)
    delqvs = _r(0.)
    des = _r(0.)
    dey = _r(0.)
    dg = _r(0.)
    dgrg = _r(0.)
    dgrh = _r(0.)
    dh = _r(0.)
    di = _r(0.)
    dirg = _r(0.)
    dirh = _r(0.)
    dr = _r(0.)
    draut = _r(0.)
    drinit = _r(0.)
    ds = _r(0.)
    dsrg = _r(0.)
    dsrh = _r(0.)
    dsrs = _r(0.)
    ec = _r(0.)
    ecg = _r(0.)
    ech = _r(0.)
    ecs = _r(0.)
    eig = _r(0.)
    eih = _r(0.)
    eii = _r(0.)
    eis = _r(0.)
    erg = _r(0.)
    esh = _r(0.)
    ess = _r(0.)
    f1 = _r(0.)
    f2 = _r(0.)
    ff = _r(0.)
    frac = _r(0.)
    ft = _r(0.)
    gam = _r(0.)
    gg = _r(0.)
    iabi = _r(0.)
    ides = _r(0.)
    ieih = _r(0.)
    iesh = _r(0.)
    ilamc = _r(0.)
    ilamc2 = _r(0.)
    ilamc3 = _r(0.)
    ilamc4 = _r(0.)
    ilamc5 = _r(0.)
    ilamc6 = _r(0.)
    ilamg = _r(0.)
    ilamg2 = _r(0.)
    ilamgb0 = _r(0.)
    ilamgb1 = _r(0.)
    ilamgb2 = _r(0.)
    ilamh = _r(0.)
    ilamh2 = _r(0.)
    ilamhb0 = _r(0.)
    ilamhb1 = _r(0.)
    ilamhb2 = _r(0.)
    ilami = _r(0.)
    ilami2 = _r(0.)
    ilami3 = _r(0.)
    ilami4 = _r(0.)
    ilami5 = _r(0.)
    ilamib0 = _r(0.)
    ilamib1 = _r(0.)
    ilamib2 = _r(0.)
    ilamr = _r(0.)
    ilamr2 = _r(0.)
    ilamr3 = _r(0.)
    ilamr4 = _r(0.)
    ilamr5 = _r(0.)
    ilamr6 = _r(0.)
    ilams = _r(0.)
    ilams2 = _r(0.)
    ilams_d3 = _r(0.)
    ilamsb0 = _r(0.)
    ilamsb1 = _r(0.)
    ilamsb2 = _r(0.)
    imukin = _r(0.)
    inc = _r(0.)
    ing = _r(0.)
    inh = _r(0.)
    inn = _r(0.)
    inr = _r(0.)
    iny = _r(0.)
    iqc = _r(0.)
    iqg = _r(0.)
    iqh = _r(0.)
    iqi = _r(0.)
    iqn = _r(0.)
    iqr = _r(0.)
    jj = _r(0.)
    ka = _r(0.)
    kdiff = _r(0.)
    kn = _r(0.)
    kstoke = _r(0.)
    l = _r(0.)
    lamr = _r(0.)
    log1 = jnp.zeros_like(t, dtype=jnp.bool_)
    log2 = jnp.zeros_like(t, dtype=jnp.bool_)
    log3 = jnp.zeros_like(t, dtype=jnp.bool_)
    log4 = jnp.zeros_like(t, dtype=jnp.bool_)
    mudyn = _r(0.)
    mukin = _r(0.)
    na = _r(0.)
    nclcg = _r(0.)
    nclch = _r(0.)
    nclcs = _r(0.)
    nclgr = _r(0.)
    nclgrg = _r(0.)
    nclgrh = _r(0.)
    nclig = _r(0.)
    nclih = _r(0.)
    nclir = _r(0.)
    nclirg = _r(0.)
    nclirh = _r(0.)
    nclis = _r(0.)
    nclrg = _r(0.)
    nclrh = _r(0.)
    nclri = _r(0.)
    nclrs = _r(0.)
    nclsh = _r(0.)
    nclsr = _r(0.)
    nclsrg = _r(0.)
    nclsrh = _r(0.)
    nclsrs = _r(0.)
    nclss = _r(0.)
    ncnsg = _r(0.)
    nfzci = _r(0.)
    ng_tail = _r(0.)
    ngcngh = _r(0.)
    nhcngh = _r(0.)
    nhfzrh = _r(0.)
    nicnis = _r(0.)
    nimgi = _r(0.)
    nimsi = _r(0.)
    nmlgr = _r(0.)
    nmlhr = _r(0.)
    nmlir = _r(0.)
    nmlsr = _r(0.)
    nnumax = _r(0.)
    nnuvi = _r(0.)
    no_g = _r(0.)
    no_h = _r(0.)
    no_i = _r(0.)
    no_r = _r(0.)
    no_s = _r(0.)
    nr_in = _r(0.)
    nrfzrh = _r(0.)
    nscnis = _r(0.)
    nshhr = _r(0.)
    nucont = _r(0.)
    nuconta = _r(0.)
    nucontb = _r(0.)
    nucontc = _r(0.)
    nudepsor = _r(0.)
    nvdvg = _r(0.)
    nvdvh = _r(0.)
    nvdvi = _r(0.)
    nvdvs = _r(0.)
    psia = _r(0.)
    qclcg = _r(0.)
    qclch = _r(0.)
    qclcs = _r(0.)
    qclgr = _r(0.)
    qclig = _r(0.)
    qclih = _r(0.)
    qclir = _r(0.)
    qclis = _r(0.)
    qclrg = _r(0.)
    qclrh = _r(0.)
    qclri = _r(0.)
    qclrs = _r(0.)
    qclsh = _r(0.)
    qclsr = _r(0.)
    qcngh = _r(0.)
    qcnis = _r(0.)
    qcnis1 = _r(0.)
    qcnis2 = _r(0.)
    qcnsg = _r(0.)
    qfzci = _r(0.)
    qfzrh = _r(0.)
    qhwet = _r(0.)
    qimgi = _r(0.)
    qimsi = _r(0.)
    qmlgr = _r(0.)
    qmlhr = _r(0.)
    qmlir = _r(0.)
    qmlsr = _r(0.)
    qnuvi = _r(0.)
    qrevp = _r(0.)
    qshhr = _r(0.)
    qvdvg = _r(0.)
    qvdvh = _r(0.)
    qvdvi = _r(0.)
    qvdvs = _r(0.)
    qvs0 = _r(0.)
    rainpresent = jnp.zeros_like(t, dtype=jnp.bool_)
    ratio = _r(0.)
    rcaccr = _r(0.)
    rcautr = _r(0.)
    rz = _r(0.)
    scthrd = _r(0.)
    si = _r(0.)
    sigc = _r(0.)
    simax = _r(0.)
    sink = _r(0.)
    sour = _r(0.)
    source = _r(0.)
    ssat = _r(0.)
    swmax = _r(0.)
    tau = _r(0.)
    tc = _r(0.)
    tcc = _r(0.)
    tmp1 = _r(0.)
    tmp10 = _r(0.)
    tmp2 = _r(0.)
    tmp3 = _r(0.)
    tmp4 = _r(0.)
    tmp5 = _r(0.)
    tmp6 = _r(0.)
    tmp8 = _r(0.)
    tmp9 = _r(0.)
    vdmax = _r(0.)
    ventg = _r(0.)
    venth = _r(0.)
    venti = _r(0.)
    ventr = _r(0.)
    vents = _r(0.)
    vg0 = _r(0.)
    vh0 = _r(0.)
    vi0 = _r(0.)
    vr0 = _r(0.)
    vs0 = _r(0.)
    x = _r(0.)
    # WRF:1448
    tmp1 = _r(_op('div', qsw, jnp.maximum(q, _r(1e-20))))
    # WRF:1449
    tmp2 = _r(_op('div', qsi, jnp.maximum(q, _r(1e-20))))
    _m1 = (True) & ((qc > epsq) & (nc < epsn))
    # WRF:1460
    nc = jnp.where(_m1, _r(n_c_sm), nc)
    _m2 = (True) & _not(_m1) & ((qc <= epsq) | (qc < epsq2) & (tmp1 < _r(0.9)))
    # WRF:1462
    tmp3 = jnp.where(_m2, _r(jnp.maximum(_r(0.0), qc)), tmp3)
    # WRF:1463
    q = jnp.where(_m2, _r(_op('add', q, tmp3)), q)
    # WRF:1464
    t = jnp.where(_m2, _r(_op('sub', t, _op('mul', lcp, tmp3))), t)
    # WRF:1465
    qc = jnp.where(_m2, _r(_r(0.0)), qc)
    # WRF:1466
    nc = jnp.where(_m2, _r(_r(0.0)), nc)
    _m3 = (True) & ((qr > epsq) & (nr < epsn))
    # WRF:1471
    nr = jnp.where(_m3, _r(_op('mul', _pow(_op('mul', no_r_sm, gr31), _op('div', _r(3.0), _op('add', _r(4.0), alpha_r))), _pow(_op('mul', _op('mul', _op('mul', _op('mul', gr31, igr34), de), qr), icmr), _op('div', _op('add', _r(1.0), alpha_r), _op('add', _r(4.0), alpha_r))))), nr)
    _m4 = (True) & _not(_m3) & ((qr <= epsq) | (qr < epsq2) & (tmp1 < _r(0.9)))
    # WRF:1474
    tmp3 = jnp.where(_m4, _r(jnp.maximum(_r(0.0), qr)), tmp3)
    # WRF:1475
    q = jnp.where(_m4, _r(_op('add', q, tmp3)), q)
    # WRF:1476
    t = jnp.where(_m4, _r(_op('sub', t, _op('mul', lcp, tmp3))), t)
    # WRF:1477
    qr = jnp.where(_m4, _r(_r(0.0)), qr)
    # WRF:1478
    nr = jnp.where(_m4, _r(_r(0.0)), nr)
    _m5 = (True) & ((qi > epsq) & (ny < epsn))
    # WRF:1484
    ny = jnp.where(_m5, _r(jnp.maximum(_op('mul', _r(2.0), epsn), _n_cooper(trpl, t))), ny)
    _m6 = (True) & _not(_m5) & ((qi <= epsq) | (qi < epsq2) & (tmp2 < _r(0.8)))
    # WRF:1486
    tmp3 = jnp.where(_m6, _r(jnp.maximum(_r(0.0), qi)), tmp3)
    # WRF:1487
    q = jnp.where(_m6, _r(_op('add', q, tmp3)), q)
    # WRF:1488
    t = jnp.where(_m6, _r(_op('sub', t, _op('mul', lsp, tmp3))), t)
    # WRF:1489
    qi = jnp.where(_m6, _r(_r(0.0)), qi)
    # WRF:1490
    ny = jnp.where(_m6, _r(_r(0.0)), ny)
    _m7 = (True) & ((qn > epsq) & (nn < epsn))
    # WRF:1495
    no_s = jnp.where(_m7, _r(_nos_thompson(trpl, t)), no_s)
    # WRF:1496
    nn = jnp.where(_m7, _r(_op('mul', _pow(_op('mul', no_s, gs31), _op('mul', dms, icexs2)), _pow(_op('mul', _op('mul', _op('mul', _op('mul', gs31, igs40), icms), de), qn), _op('mul', _op('add', _r(1.0), alpha_s), icexs2)))), nn)
    _m8 = (True) & _not(_m7) & ((qn <= epsq) | (qn < epsq2) & (tmp2 < _r(0.8)))
    # WRF:1499
    tmp3 = jnp.where(_m8, _r(jnp.maximum(_r(0.0), qn)), tmp3)
    # WRF:1500
    q = jnp.where(_m8, _r(_op('add', q, tmp3)), q)
    # WRF:1501
    t = jnp.where(_m8, _r(_op('sub', t, _op('mul', lsp, tmp3))), t)
    # WRF:1502
    qn = jnp.where(_m8, _r(_r(0.0)), qn)
    # WRF:1503
    nn = jnp.where(_m8, _r(_r(0.0)), nn)
    _m9 = (True) & ((qg > epsq) & (ng < epsn))
    # WRF:1508
    ng = jnp.where(_m9, _r(_op('mul', _pow(_op('mul', no_g_sm, gg31), _op('div', _r(3.0), _op('add', _r(4.0), alpha_g))), _pow(_op('mul', _op('mul', _op('mul', _op('mul', gg31, igg34), de), qg), icmg), _op('div', _op('add', _r(1.0), alpha_g), _op('add', _r(4.0), alpha_g))))), ng)
    _m10 = (True) & _not(_m9) & ((qg <= epsq) | (qg < epsq2) & (tmp2 < _r(0.8)))
    # WRF:1511
    tmp3 = jnp.where(_m10, _r(jnp.maximum(_r(0.0), qg)), tmp3)
    # WRF:1512
    q = jnp.where(_m10, _r(_op('add', q, tmp3)), q)
    # WRF:1513
    t = jnp.where(_m10, _r(_op('sub', t, _op('mul', lsp, tmp3))), t)
    # WRF:1514
    qg = jnp.where(_m10, _r(_r(0.0)), qg)
    # WRF:1515
    ng = jnp.where(_m10, _r(_r(0.0)), ng)
    _m11 = (True) & ((qh > epsq) & (nh < epsn))
    # WRF:1520
    nh = jnp.where(_m11, _r(_op('mul', _pow(_op('mul', no_h_sm, gh31), _op('div', _r(3.0), _op('add', _r(4.0), alpha_h))), _pow(_op('mul', _op('mul', _op('mul', _op('mul', gh31, igh34), de), qh), icmh), _op('div', _op('add', _r(1.0), alpha_h), _op('add', _r(4.0), alpha_h))))), nh)
    _m12 = (True) & _not(_m11) & ((qh <= epsq) | (qh < epsq2) & (tmp2 < _r(0.8)))
    # WRF:1523
    tmp3 = jnp.where(_m12, _r(jnp.maximum(_r(0.0), qh)), tmp3)
    # WRF:1524
    q = jnp.where(_m12, _r(_op('add', q, tmp3)), q)
    # WRF:1525
    t = jnp.where(_m12, _r(_op('sub', t, _op('mul', lsp, tmp3))), t)
    # WRF:1526
    qh = jnp.where(_m12, _r(_r(0.0)), qh)
    # WRF:1527
    nh = jnp.where(_m12, _r(_r(0.0)), nh)
    qc_in, qr_in, nc_in, nr_in = qc, qr, nc, nr
    # WRF:1591
    activepoint = jnp.where(point_mask, jnp.asarray(False), activepoint)
    # WRF:1594
    log1 = jnp.where(point_mask, jnp.asarray(_op('add', _op('add', _op('add', qi, qg), qn), qh) < epsq), log1)
    # WRF:1595
    log2 = jnp.where(point_mask, jnp.asarray(_op('add', qc, qr) < epsq), log2)
    # WRF:1596
    log3 = jnp.where(point_mask, jnp.asarray((t > trpl) & log1), log3)
    # WRF:1597
    log4 = jnp.where(point_mask, jnp.asarray(log1 & log2 & (q < qsi)), log4)
    _m1 = (point_mask) & (_not(log3 | log4) & icephase_on)
    # WRF:1599
    activepoint = jnp.where(_m1, jnp.asarray(True), activepoint)
    _m2 = (point_mask) & (activepoint)
    # WRF:1614
    tc = jnp.where(_m2, _r(_op('sub', t, trpl)), tc)
    _m3 = (_m2) & ((tc < -_r(120.0)) | (tc > _r(50.0)))
    # WRF:1619
    cdiff = jnp.where(_m2, _r(_op('div', _op('mul', _op('add', _r(2.2157e-05), _op('mul', _r(1.55e-07), tc)), _r(100000.0)), pres)), cdiff)
    # WRF:1620
    mudyn = jnp.where(_m2, _r(_op('mul', _op('mul', _r(1.72e-05), _op('div', _r(393.0), _op('add', t, _r(120.0)))), _pow(_op('div', t, trpl), _r(1.5)))), mudyn)
    # WRF:1621
    mukin = jnp.where(_m2, _r(_op('mul', mudyn, ide)), mukin)
    # WRF:1622
    imukin = jnp.where(_m2, _r(_op('div', _r(1.0), mukin)), imukin)
    # WRF:1623
    scthrd = jnp.where(_m2, _r(_pow(_op('div', mukin, cdiff), thrd)), scthrd)
    # WRF:1624
    ka = jnp.where(_m2, _r(_op('add', _r(0.023971), _op('mul', _r(7.8e-05), tc))), ka)
    # WRF:1625
    kdiff = jnp.where(_m2, _r(_op('sub', _op('add', _op('mul', _op('mul', _r(9.1018e-11), t), t), _op('mul', _r(8.8197e-08), t)), _r(1.0654e-05))), kdiff)
    # WRF:1626
    gam = jnp.where(_m2, _r(gamfact), gam)
    # WRF:1629
    eis = jnp.where(_m2, _r(jnp.minimum(_op('mul', _r(0.05), _exp(_op('mul', _r(0.1), tc))), _r(1.0))), eis)
    # WRF:1630
    eig = jnp.where(_m2, _r(jnp.minimum(_op('mul', _r(0.01), _exp(_op('mul', _r(0.1), tc))), _r(1.0))), eig)
    # WRF:1631
    eii = jnp.where(_m2, _r(_op('mul', _r(0.1), eis)), eii)
    # WRF:1632
    ess = jnp.where(_m2, _r(eis), ess)
    # WRF:1632
    eih = jnp.where(_m2, _r(eig), eih)
    # WRF:1632
    esh = jnp.where(_m2, _r(eig), esh)
    # WRF:1633
    ieih = jnp.where(_m2, _r(_op('div', _r(1.0), eih)), ieih)
    # WRF:1634
    iesh = jnp.where(_m2, _r(_op('div', _r(1.0), esh)), iesh)
    # WRF:1641
    qvs0 = jnp.where(_m2, _r(_qsat(trpl, pres, 0)), qvs0)
    # WRF:1642
    delqvs = jnp.where(_m2, _r(_op('sub', qvs0, q)), delqvs)
    _m4 = (_m2) & (qc > epsq)
    # WRF:1646
    iqc = jnp.where(_m4, _r(_op('div', _r(1.0), qc)), iqc)
    # WRF:1647
    inc = jnp.where(_m4, _r(_op('div', _r(1.0), nc)), inc)
    # WRF:1648
    dc = jnp.where(_m4, _r(_dm_x(de, qc, inc, icmr, thrd)), dc)
    # WRF:1650
    ilamc = jnp.where(_m4, _r(_ilamda_x(de, qc, inc, icexc9, thrd)), ilamc)
    # WRF:1651
    ilamc2 = jnp.where(_m4, _r(_op('mul', ilamc, ilamc)), ilamc2)
    # WRF:1652
    ilamc3 = jnp.where(_m4, _r(_op('mul', ilamc2, ilamc)), ilamc3)
    # WRF:1653
    ilamc4 = jnp.where(_m4, _r(_op('mul', ilamc2, ilamc2)), ilamc4)
    # WRF:1654
    ilamc5 = jnp.where(_m4, _r(_op('mul', ilamc3, ilamc2)), ilamc5)
    _m5 = (_m2) & _not(_m4)
    # WRF:1656
    dc = jnp.where(_m5, _r(_r(0.0)), dc)
    # WRF:1656
    ilamc3 = jnp.where(_m5, _r(_r(0.0)), ilamc3)
    # WRF:1657
    ilamc = jnp.where(_m5, _r(_r(0.0)), ilamc)
    # WRF:1657
    ilamc4 = jnp.where(_m5, _r(_r(0.0)), ilamc4)
    # WRF:1658
    ilamc2 = jnp.where(_m5, _r(_r(0.0)), ilamc2)
    # WRF:1658
    ilamc5 = jnp.where(_m5, _r(_r(0.0)), ilamc5)
    _m6 = (_m2) & (qr > epsq)
    # WRF:1663
    iqr = jnp.where(_m6, _r(_op('div', _r(1.0), qr)), iqr)
    # WRF:1664
    inr = jnp.where(_m6, _r(_op('div', _r(1.0), nr)), inr)
    # WRF:1665
    dr = jnp.where(_m6, _r(_dm_x(de, qr, inr, icmr, thrd)), dr)
    # WRF:1666
    ilamr = jnp.where(_m6, _r(jnp.maximum(ilammin1, _ilamda_x(de, qr, inr, icexr9, thrd))), ilamr)
    # WRF:1667
    tmp1 = jnp.where(_m6, _r(_op('div', _r(1.0), ilamr)), tmp1)
    # WRF:1668
    ilamr2 = jnp.where(_m6, _r(_pow(ilamr, 2)), ilamr2)
    # WRF:1669
    ilamr3 = jnp.where(_m6, _r(_pow(ilamr, 3)), ilamr3)
    # WRF:1670
    ilamr4 = jnp.where(_m6, _r(_pow(ilamr, 4)), ilamr4)
    # WRF:1671
    ilamr5 = jnp.where(_m6, _r(_pow(ilamr, 5)), ilamr5)
    _m7 = (_m6) & (dr > _r(4e-05))
    # WRF:1673
    vr0 = jnp.where(_m7, _r(_op('mul', _op('mul', gamfact, ckqr1), _pow(ilamr, bfr))), vr0)
    _m8 = (_m6) & _not(_m7)
    # WRF:1675
    vr0 = jnp.where(_m8, _r(_r(0.0)), vr0)
    _m9 = (_m2) & _not(_m6)
    # WRF:1678
    ilamr = jnp.where(_m9, _r(_r(0.0)), ilamr)
    # WRF:1678
    dr = jnp.where(_m9, _r(_r(0.0)), dr)
    # WRF:1678
    vr0 = jnp.where(_m9, _r(_r(0.0)), vr0)
    # WRF:1679
    ilamr2 = jnp.where(_m9, _r(_r(0.0)), ilamr2)
    # WRF:1679
    ilamr3 = jnp.where(_m9, _r(_r(0.0)), ilamr3)
    # WRF:1679
    ilamr4 = jnp.where(_m9, _r(_r(0.0)), ilamr4)
    # WRF:1679
    ilamr5 = jnp.where(_m9, _r(_r(0.0)), ilamr5)
    _m10 = (_m2) & (qi > epsq)
    # WRF:1684
    iqi = jnp.where(_m10, _r(_op('div', _r(1.0), qi)), iqi)
    # WRF:1685
    iny = jnp.where(_m10, _r(_op('div', _r(1.0), ny)), iny)
    # WRF:1686
    ilami = jnp.where(_m10, _r(jnp.maximum(ilammin2, _ilamda_x(de, qi, iny, icexi9, thrd))), ilami)
    # WRF:1687
    ilami2 = jnp.where(_m10, _r(_pow(ilami, 2)), ilami2)
    # WRF:1688
    ilami3 = jnp.where(_m10, _r(_pow(ilami, 3)), ilami3)
    # WRF:1689
    ilami4 = jnp.where(_m10, _r(_pow(ilami, 4)), ilami4)
    # WRF:1690
    ilami5 = jnp.where(_m10, _r(_pow(ilami, 5)), ilami5)
    # WRF:1691
    ilamib0 = jnp.where(_m10, _r(_pow(ilami, bfi)), ilamib0)
    # WRF:1692
    ilamib1 = jnp.where(_m10, _r(_pow(ilami, _op('add', bfi, _r(1.0)))), ilamib1)
    # WRF:1693
    ilamib2 = jnp.where(_m10, _r(_pow(ilami, _op('add', bfi, _r(2.0)))), ilamib2)
    # WRF:1694
    vi0 = jnp.where(_m10, _r(_op('mul', _op('mul', gamfact, ckqi1), ilamib0)), vi0)
    # WRF:1695
    di = jnp.where(_m10, _r(_dm_x(de, qi, iny, icmi, thrd)), di)
    _m11 = (_m2) & _not(_m10)
    # WRF:1697
    ilami = jnp.where(_m11, _r(_r(0.0)), ilami)
    # WRF:1697
    vi0 = jnp.where(_m11, _r(_r(0.0)), vi0)
    # WRF:1697
    di = jnp.where(_m11, _r(_r(0.0)), di)
    # WRF:1698
    ilami2 = jnp.where(_m11, _r(_r(0.0)), ilami2)
    # WRF:1698
    ilami3 = jnp.where(_m11, _r(_r(0.0)), ilami3)
    # WRF:1698
    ilami4 = jnp.where(_m11, _r(_r(0.0)), ilami4)
    # WRF:1698
    ilami5 = jnp.where(_m11, _r(_r(0.0)), ilami5)
    # WRF:1699
    ilamib0 = jnp.where(_m11, _r(_r(0.0)), ilamib0)
    # WRF:1699
    ilamib1 = jnp.where(_m11, _r(_r(0.0)), ilamib1)
    # WRF:1699
    ilamib2 = jnp.where(_m11, _r(_r(0.0)), ilamib2)
    _m12 = (_m2) & (qn > epsq)
    # WRF:1704
    iqn = jnp.where(_m12, _r(_op('div', _r(1.0), qn)), iqn)
    # WRF:1705
    inn = jnp.where(_m12, _r(_op('div', _r(1.0), nn)), inn)
    # WRF:1706
    ilams = jnp.where(_m12, _r(jnp.maximum(ilammin2, _ilamda_x(de, qn, inn, igs20, idms))), ilams)
    # WRF:1707
    ilams_d3 = jnp.where(_m12, _r(jnp.maximum(ilammin2, _ilamda_x(de, qn, inn, igs20_d3, thrd))), ilams_d3)
    # WRF:1708
    ilams2 = jnp.where(_m12, _r(_pow(ilams, 2)), ilams2)
    # WRF:1709
    ilamsb0 = jnp.where(_m12, _r(_pow(ilams, bfs)), ilamsb0)
    # WRF:1710
    ilamsb1 = jnp.where(_m12, _r(_pow(ilams, _op('add', bfs, _r(1.0)))), ilamsb1)
    # WRF:1711
    ilamsb2 = jnp.where(_m12, _r(_pow(ilams, _op('add', bfs, _r(2.0)))), ilamsb2)
    # WRF:1712
    vs0 = jnp.where(_m12, _r(_op('mul', _op('mul', gamfact, ckqs1), ilamsb0)), vs0)
    # WRF:1713
    ds = jnp.where(_m12, _r(jnp.minimum(dsmax, _dm_x(de, qn, inn, icms, idms))), ds)
    _m13 = (_m12) & (snowspherical)
    # WRF:1715
    des = jnp.where(_m13, _r(desfix), des)
    _m14 = (_m12) & _not(_m13)
    # WRF:1717
    des = jnp.where(_m14, _r(_des_of_ds(ds, desmax, eds, fds)), des)
    # WRF:1728
    no_s = jnp.where(_m12, _r(_op('div', _op('mul', nn, igs31), ilams_d3)), no_s)
    # WRF:1729
    vents = jnp.where(_m12, _r(_op('add', _op('mul', _op('mul', avx, gs32), _pow(ilams_d3, _r(2.0))), _op('mul', _op('mul', _op('mul', _op('mul', bvx, scthrd), jnp.sqrt(_op('mul', _op('mul', gamfact, afs), imukin))), gs09), _pow(ilams_d3, cexs1)))), vents)
    _m15 = (_m2) & _not(_m12)
    # WRF:1732
    ilams = jnp.where(_m15, _r(_r(0.0)), ilams)
    # WRF:1732
    vs0 = jnp.where(_m15, _r(_r(0.0)), vs0)
    # WRF:1732
    ds = jnp.where(_m15, _r(_r(0.0)), ds)
    # WRF:1732
    ilams2 = jnp.where(_m15, _r(_r(0.0)), ilams2)
    # WRF:1733
    ilamsb0 = jnp.where(_m15, _r(_r(0.0)), ilamsb0)
    # WRF:1733
    ilamsb1 = jnp.where(_m15, _r(_r(0.0)), ilamsb1)
    # WRF:1733
    ilamsb1 = jnp.where(_m15, _r(_r(0.0)), ilamsb1)
    # WRF:1734
    des = jnp.where(_m15, _r(desfix), des)
    # WRF:1736
    ides = jnp.where(_m2, _r(_op('div', _r(1.0), des)), ides)
    _m16 = (_m2) & (qg > epsq)
    # WRF:1741
    iqg = jnp.where(_m16, _r(_op('div', _r(1.0), qg)), iqg)
    # WRF:1742
    ing = jnp.where(_m16, _r(_op('div', _r(1.0), ng)), ing)
    # WRF:1743
    ilamg = jnp.where(_m16, _r(jnp.maximum(ilammin1, _ilamda_x(de, qg, ing, igg99, thrd))), ilamg)
    # WRF:1744
    ilamg2 = jnp.where(_m16, _r(_pow(ilamg, 2)), ilamg2)
    # WRF:1745
    ilamgb0 = jnp.where(_m16, _r(_pow(ilamg, bfg)), ilamgb0)
    # WRF:1746
    ilamgb1 = jnp.where(_m16, _r(_pow(ilamg, _op('add', bfg, _r(1.0)))), ilamgb1)
    # WRF:1747
    ilamgb2 = jnp.where(_m16, _r(_pow(ilamg, _op('add', bfg, _r(2.0)))), ilamgb2)
    # WRF:1749
    no_g = jnp.where(_m16, _r(_op('div', _op('mul', ng, igg31), ilamg)), no_g)
    # WRF:1750
    vg0 = jnp.where(_m16, _r(_op('mul', _op('mul', gamfact, ckqg1), ilamgb0)), vg0)
    # WRF:1751
    dg = jnp.where(_m16, _r(_dm_x(de, qg, ing, icmg, thrd)), dg)
    _m17 = (_m2) & _not(_m16)
    # WRF:1753
    ilamg = jnp.where(_m17, _r(_r(0.0)), ilamg)
    # WRF:1753
    vg0 = jnp.where(_m17, _r(_r(0.0)), vg0)
    # WRF:1753
    dg = jnp.where(_m17, _r(_r(0.0)), dg)
    # WRF:1753
    no_g = jnp.where(_m17, _r(_r(0.0)), no_g)
    # WRF:1754
    ilamg2 = jnp.where(_m17, _r(_r(0.0)), ilamg2)
    # WRF:1754
    ilamgb0 = jnp.where(_m17, _r(_r(0.0)), ilamgb0)
    # WRF:1754
    ilamgb1 = jnp.where(_m17, _r(_r(0.0)), ilamgb1)
    # WRF:1754
    ilamgb1 = jnp.where(_m17, _r(_r(0.0)), ilamgb1)
    _m18 = (_m2) & (qh > epsq)
    # WRF:1759
    iqh = jnp.where(_m18, _r(_op('div', _r(1.0), qh)), iqh)
    # WRF:1760
    inh = jnp.where(_m18, _r(_op('div', _r(1.0), nh)), inh)
    # WRF:1761
    ilamh = jnp.where(_m18, _r(jnp.maximum(ilammin1, _ilamda_x(de, qh, inh, igh99, thrd))), ilamh)
    # WRF:1762
    ilamh2 = jnp.where(_m18, _r(_pow(ilamh, 2)), ilamh2)
    # WRF:1763
    ilamhb0 = jnp.where(_m18, _r(_pow(ilamh, bfh)), ilamhb0)
    # WRF:1764
    ilamhb1 = jnp.where(_m18, _r(_pow(ilamh, _op('add', bfh, _r(1.0)))), ilamhb1)
    # WRF:1765
    ilamhb2 = jnp.where(_m18, _r(_pow(ilamh, _op('add', bfh, _r(2.0)))), ilamhb2)
    # WRF:1766
    no_h = jnp.where(_m18, _r(_op('div', _op('mul', nh, igh31), _pow(ilamh, _op('add', _r(1.0), alpha_h)))), no_h)
    # WRF:1767
    vh0 = jnp.where(_m18, _r(_op('mul', _op('mul', gamfact, ckqh1), ilamhb0)), vh0)
    # WRF:1768
    dh = jnp.where(_m18, _r(_dm_x(de, qh, inh, icmh, thrd)), dh)
    _m19 = (_m2) & _not(_m18)
    # WRF:1770
    ilamh = jnp.where(_m19, _r(_r(0.0)), ilamh)
    # WRF:1770
    vh0 = jnp.where(_m19, _r(_r(0.0)), vh0)
    # WRF:1770
    dh = jnp.where(_m19, _r(_r(0.0)), dh)
    # WRF:1770
    no_h = jnp.where(_m19, _r(_r(0.0)), no_h)
    # WRF:1771
    ilamhb0 = jnp.where(_m19, _r(_r(0.0)), ilamhb0)
    # WRF:1771
    ilamhb1 = jnp.where(_m19, _r(_r(0.0)), ilamhb1)
    # WRF:1771
    ilamhb1 = jnp.where(_m19, _r(_r(0.0)), ilamhb1)
    # WRF:1778
    qnuvi = jnp.where(_m2, _r(_r(0.0)), qnuvi)
    # WRF:1778
    qvdvi = jnp.where(_m2, _r(_r(0.0)), qvdvi)
    # WRF:1778
    qvdvs = jnp.where(_m2, _r(_r(0.0)), qvdvs)
    # WRF:1778
    qvdvg = jnp.where(_m2, _r(_r(0.0)), qvdvg)
    # WRF:1778
    qvdvh = jnp.where(_m2, _r(_r(0.0)), qvdvh)
    # WRF:1779
    qclcs = jnp.where(_m2, _r(_r(0.0)), qclcs)
    # WRF:1779
    qclcg = jnp.where(_m2, _r(_r(0.0)), qclcg)
    # WRF:1779
    qclch = jnp.where(_m2, _r(_r(0.0)), qclch)
    # WRF:1779
    qfzci = jnp.where(_m2, _r(_r(0.0)), qfzci)
    # WRF:1779
    qclri = jnp.where(_m2, _r(_r(0.0)), qclri)
    # WRF:1779
    qmlsr = jnp.where(_m2, _r(_r(0.0)), qmlsr)
    # WRF:1780
    qclrs = jnp.where(_m2, _r(_r(0.0)), qclrs)
    # WRF:1780
    qclrg = jnp.where(_m2, _r(_r(0.0)), qclrg)
    # WRF:1780
    qmlgr = jnp.where(_m2, _r(_r(0.0)), qmlgr)
    # WRF:1780
    qclrh = jnp.where(_m2, _r(_r(0.0)), qclrh)
    # WRF:1780
    qmlhr = jnp.where(_m2, _r(_r(0.0)), qmlhr)
    # WRF:1780
    qfzrh = jnp.where(_m2, _r(_r(0.0)), qfzrh)
    # WRF:1781
    qmlir = jnp.where(_m2, _r(_r(0.0)), qmlir)
    # WRF:1781
    qclsr = jnp.where(_m2, _r(_r(0.0)), qclsr)
    # WRF:1781
    qclsh = jnp.where(_m2, _r(_r(0.0)), qclsh)
    # WRF:1781
    qclgr = jnp.where(_m2, _r(_r(0.0)), qclgr)
    # WRF:1781
    qcngh = jnp.where(_m2, _r(_r(0.0)), qcngh)
    # WRF:1782
    qcnis = jnp.where(_m2, _r(_r(0.0)), qcnis)
    # WRF:1782
    qclir = jnp.where(_m2, _r(_r(0.0)), qclir)
    # WRF:1782
    qclis = jnp.where(_m2, _r(_r(0.0)), qclis)
    # WRF:1782
    qclih = jnp.where(_m2, _r(_r(0.0)), qclih)
    # WRF:1783
    qimsi = jnp.where(_m2, _r(_r(0.0)), qimsi)
    # WRF:1783
    qimgi = jnp.where(_m2, _r(_r(0.0)), qimgi)
    # WRF:1783
    qcnsg = jnp.where(_m2, _r(_r(0.0)), qcnsg)
    # WRF:1783
    qhwet = jnp.where(_m2, _r(_r(0.0)), qhwet)
    # WRF:1785
    nclcs = jnp.where(_m2, _r(_r(0.0)), nclcs)
    # WRF:1785
    nclcg = jnp.where(_m2, _r(_r(0.0)), nclcg)
    # WRF:1785
    nclch = jnp.where(_m2, _r(_r(0.0)), nclch)
    # WRF:1785
    nfzci = jnp.where(_m2, _r(_r(0.0)), nfzci)
    # WRF:1785
    nmlhr = jnp.where(_m2, _r(_r(0.0)), nmlhr)
    # WRF:1785
    nhcngh = jnp.where(_m2, _r(_r(0.0)), nhcngh)
    # WRF:1786
    nclri = jnp.where(_m2, _r(_r(0.0)), nclri)
    # WRF:1786
    nclrs = jnp.where(_m2, _r(_r(0.0)), nclrs)
    # WRF:1786
    nclrg = jnp.where(_m2, _r(_r(0.0)), nclrg)
    # WRF:1786
    nclrh = jnp.where(_m2, _r(_r(0.0)), nclrh)
    # WRF:1786
    nmlsr = jnp.where(_m2, _r(_r(0.0)), nmlsr)
    # WRF:1786
    nmlgr = jnp.where(_m2, _r(_r(0.0)), nmlgr)
    # WRF:1787
    nmlir = jnp.where(_m2, _r(_r(0.0)), nmlir)
    # WRF:1787
    nshhr = jnp.where(_m2, _r(_r(0.0)), nshhr)
    # WRF:1787
    nnuvi = jnp.where(_m2, _r(_r(0.0)), nnuvi)
    # WRF:1787
    nvdvi = jnp.where(_m2, _r(_r(0.0)), nvdvi)
    # WRF:1787
    nvdvh = jnp.where(_m2, _r(_r(0.0)), nvdvh)
    # WRF:1787
    qclig = jnp.where(_m2, _r(_r(0.0)), qclig)
    # WRF:1788
    nclir = jnp.where(_m2, _r(_r(0.0)), nclir)
    # WRF:1788
    nclis = jnp.where(_m2, _r(_r(0.0)), nclis)
    # WRF:1788
    nclig = jnp.where(_m2, _r(_r(0.0)), nclig)
    # WRF:1788
    nclih = jnp.where(_m2, _r(_r(0.0)), nclih)
    # WRF:1788
    nimsi = jnp.where(_m2, _r(_r(0.0)), nimsi)
    # WRF:1788
    nimgi = jnp.where(_m2, _r(_r(0.0)), nimgi)
    # WRF:1789
    nicnis = jnp.where(_m2, _r(_r(0.0)), nicnis)
    # WRF:1789
    nscnis = jnp.where(_m2, _r(_r(0.0)), nscnis)
    # WRF:1789
    nvdvs = jnp.where(_m2, _r(_r(0.0)), nvdvs)
    # WRF:1789
    ncnsg = jnp.where(_m2, _r(_r(0.0)), ncnsg)
    # WRF:1789
    nclgr = jnp.where(_m2, _r(_r(0.0)), nclgr)
    # WRF:1789
    nclsrh = jnp.where(_m2, _r(_r(0.0)), nclsrh)
    # WRF:1790
    nclss = jnp.where(_m2, _r(_r(0.0)), nclss)
    # WRF:1790
    nclsr = jnp.where(_m2, _r(_r(0.0)), nclsr)
    # WRF:1790
    nclsh = jnp.where(_m2, _r(_r(0.0)), nclsh)
    # WRF:1790
    nclsrs = jnp.where(_m2, _r(_r(0.0)), nclsrs)
    # WRF:1790
    nclgrg = jnp.where(_m2, _r(_r(0.0)), nclgrg)
    # WRF:1790
    ngcngh = jnp.where(_m2, _r(_r(0.0)), ngcngh)
    # WRF:1791
    nvdvg = jnp.where(_m2, _r(_r(0.0)), nvdvg)
    # WRF:1791
    nclirg = jnp.where(_m2, _r(_r(0.0)), nclirg)
    # WRF:1791
    nclsrg = jnp.where(_m2, _r(_r(0.0)), nclsrg)
    # WRF:1791
    nclgrh = jnp.where(_m2, _r(_r(0.0)), nclgrh)
    # WRF:1791
    nrfzrh = jnp.where(_m2, _r(_r(0.0)), nrfzrh)
    # WRF:1791
    nhfzrh = jnp.where(_m2, _r(_r(0.0)), nhfzrh)
    # WRF:1792
    nclirh = jnp.where(_m2, _r(_r(0.0)), nclirh)
    # WRF:1794
    dirg = jnp.where(_m2, _r(_r(0.0)), dirg)
    # WRF:1794
    dirh = jnp.where(_m2, _r(_r(0.0)), dirh)
    # WRF:1794
    dsrs = jnp.where(_m2, _r(_r(0.0)), dsrs)
    # WRF:1794
    dsrg = jnp.where(_m2, _r(_r(0.0)), dsrg)
    # WRF:1794
    dsrh = jnp.where(_m2, _r(_r(0.0)), dsrh)
    # WRF:1794
    dgrg = jnp.where(_m2, _r(_r(0.0)), dgrg)
    # WRF:1794
    dgrh = jnp.where(_m2, _r(_r(0.0)), dgrh)
    # WRF:1798
    si = jnp.where(_m2, _r(_op('div', q, qsi)), si)
    # WRF:1799
    iabi = jnp.where(_m2, _r(_op('div', _r(1.0), _op('add', _op('div', _op('mul', chls, chls), _op('mul', _op('mul', ka, rgasv), _pow(t, 2))), _op('div', _r(1.0), _op('mul', _op('mul', de, qsi), cdiff))))), iabi)
    _m20 = (_m2) & (qn > epsq)
    _m21 = (_m20) & (qc > epsq)
    # WRF:1810
    ecs = jnp.where(_m21, _r(_op('mul', _op('mul', jnp.minimum(dc, _r(3e-05)), _r(33330.0)), jnp.sqrt(_op('mul', jnp.minimum(ds, _r(0.001)), _r(1000.0))))), ecs)
    # WRF:1811
    qclcs = jnp.where(_m21, _r(_op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', dt, gam), afs), cmr), ecs), piov4), ide), _op('mul', nc, nn)), igc5), igs31), _op('add', _op('add', _op('mul', _op('mul', _op('mul', gc13, gs13), ilamc3), ilamsb2), _op('mul', _op('mul', _op('mul', _op('mul', _r(2.0), gc14), gs12), ilamc4), ilamsb1)), _op('mul', _op('mul', _op('mul', gc15, gs11), ilamc5), ilamsb0)))), qclcs)
    # WRF:1815
    nclcs = jnp.where(_m21, _r(_op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', dt, gam), afs), piov4), ecs), _op('mul', nc, nn)), igc5), igs31), _op('add', _op('add', _op('mul', _op('mul', gc5, gs13), ilamsb2), _op('mul', _op('mul', _op('mul', _op('mul', _r(2.0), gc11), gs12), ilamc), ilamsb1)), _op('mul', _op('mul', _op('mul', gc12, gs11), ilamc2), ilamsb0)))), nclcs)
    _m22 = (_m21) & (_not(snowspherical))
    # WRF:1827
    tmp1 = jnp.where(_m22, _r(_r(0.6366)), tmp1)
    # WRF:1828
    qclcs = jnp.where(_m22, _r(_op('mul', tmp1, qclcs)), qclcs)
    # WRF:1829
    nclcs = jnp.where(_m22, _r(_op('mul', tmp1, nclcs)), nclcs)
    # WRF:1832
    qclcs = jnp.where(_m21, _r(jnp.minimum(qclcs, qc)), qclcs)
    # WRF:1833
    nclcs = jnp.where(_m21, _r(jnp.minimum(nclcs, nc)), nclcs)
    _m23 = (_m20) & _not(_m21)
    # WRF:1835
    qclcs = jnp.where(_m23, _r(_r(0.0)), qclcs)
    # WRF:1835
    nclcs = jnp.where(_m23, _r(_r(0.0)), nclcs)
    _m24 = (_m20) & (qi > epsq)
    # WRF:1840
    tmp1 = jnp.where(_m24, _r(_op('sub', vs0, vi0)), tmp1)
    # WRF:1841
    tmp3 = jnp.where(_m24, _r(jnp.sqrt(_op('add', _op('mul', tmp1, tmp1), _op('mul', _op('mul', _r(0.04), vs0), vi0)))), tmp3)
    # WRF:1843
    qclis = jnp.where(_m24, _r(_op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', dt, cmi), ide), pi), _r(6.0)), eis), _op('mul', ny, nn)), tmp3), igi31), igs31), _op('add', _op('add', _op('mul', _op('mul', _r(0.5), ilams2), ilami3), _op('mul', _op('mul', _r(2.0), ilams), ilami4)), _op('mul', _r(5.0), ilami5)))), qclis)
    # WRF:1846
    nclis = jnp.where(_m24, _r(_op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', dt, piov4), eis), _op('mul', ny, nn)), gi31), gs31), tmp3), _op('add', _op('add', _op('mul', _op('mul', gi33, gs31), ilami2), _op('mul', _op('mul', _op('mul', _op('mul', _r(2.0), gi32), gs32), ilami), ilams)), _op('mul', _op('mul', gi31, gs33), ilams2)))), nclis)
    # WRF:1849
    qclis = jnp.where(_m24, _r(jnp.minimum(qclis, qi)), qclis)
    # WRF:1850
    nclis = jnp.where(_m24, _r(jnp.minimum(_op('mul', qclis, _op('mul', ny, iqi)), nclis)), nclis)
    _m25 = (_m20) & _not(_m24)
    # WRF:1852
    qclis = jnp.where(_m25, _r(_r(0.0)), qclis)
    # WRF:1852
    nclis = jnp.where(_m25, _r(_r(0.0)), nclis)
    # WRF:1856
    nclss = jnp.where(_m20, _r(_op('mul', _op('mul', _op('mul', _op('mul', dt, _r(0.93952)), ess), _pow(_op('mul', de, qn), _op('mul', _op('add', _r(2.0), bfs), thrd))), _pow(nn, _op('mul', _op('sub', _r(4.0), bfs), thrd)))), nclss)
    # WRF:1862
    nclss = jnp.where(_m20, _r(jnp.minimum(nclss, _op('mul', _r(0.5), nn))), nclss)
    _m26 = (_m2) & _not(_m20)
    # WRF:1865
    qclcs = jnp.where(_m26, _r(_r(0.0)), qclcs)
    # WRF:1865
    nclcs = jnp.where(_m26, _r(_r(0.0)), nclcs)
    # WRF:1865
    qclis = jnp.where(_m26, _r(_r(0.0)), qclis)
    # WRF:1865
    nclis = jnp.where(_m26, _r(_r(0.0)), nclis)
    # WRF:1865
    nclss = jnp.where(_m26, _r(_r(0.0)), nclss)
    _m27 = (_m2) & (qg > epsq)
    _m28 = (_m27) & (qc > epsq)
    # WRF:1875
    kstoke = jnp.where(_m28, _r(_op('div', _op('mul', _op('mul', _op('mul', dew, vg0), dc), dc), _op('mul', _op('mul', _r(9.0), mudyn), dg))), kstoke)
    # WRF:1876
    kstoke = jnp.where(_m28, _r(jnp.maximum(_r(1.5), jnp.minimum(_r(10.0), kstoke))), kstoke)
    # WRF:1877
    ecg = jnp.where(_m28, _r(_op('mul', _r(0.55), _log10(_op('mul', _r(2.51), kstoke)))), ecg)
    # WRF:1879
    qclcg = jnp.where(_m28, _r(_op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', dt, gam), afg), cmr), ecg), piov4), ide), _op('mul', nc, ng)), igc5), igg31), _op('add', _op('add', _op('mul', _op('mul', _op('mul', gc13, gg13), ilamc3), ilamgb2), _op('mul', _op('mul', _op('mul', _op('mul', _r(2.0), gc14), gg12), ilamc4), ilamgb1)), _op('mul', _op('mul', _op('mul', gc15, gg11), ilamc5), ilamgb0)))), qclcg)
    # WRF:1883
    nclcg = jnp.where(_m28, _r(_op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', dt, gam), afg), piov4), ecg), _op('mul', nc, ng)), igc5), igg31), _op('add', _op('add', _op('mul', _op('mul', gc5, gg13), ilamgb2), _op('mul', _op('mul', _op('mul', _op('mul', _r(2.0), gc11), gg12), ilamc), ilamgb1)), _op('mul', _op('mul', _op('mul', gc12, gg11), ilamc2), ilamgb0)))), nclcg)
    # WRF:1886
    qclcg = jnp.where(_m28, _r(jnp.minimum(qclcg, qc)), qclcg)
    # WRF:1887
    nclcg = jnp.where(_m28, _r(jnp.minimum(nclcg, nc)), nclcg)
    _m29 = (_m27) & _not(_m28)
    # WRF:1889
    qclcg = jnp.where(_m29, _r(_r(0.0)), qclcg)
    # WRF:1889
    nclcg = jnp.where(_m29, _r(_r(0.0)), nclcg)
    _m30 = (_m27) & (qi > epsq)
    # WRF:1894
    tmp1 = jnp.where(_m30, _r(_op('sub', vg0, vi0)), tmp1)
    # WRF:1895
    tmp3 = jnp.where(_m30, _r(jnp.sqrt(_op('add', _op('mul', tmp1, tmp1), _op('mul', _op('mul', _r(0.04), vg0), vi0)))), tmp3)
    # WRF:1897
    qclig = jnp.where(_m30, _r(_op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', dt, cmi), ide), pi), _r(6.0)), eig), _op('mul', ny, ng)), tmp3), igi31), igg31), _op('add', _op('add', _op('mul', _op('mul', _r(0.5), ilamg2), ilami3), _op('mul', _op('mul', _r(2.0), ilamg), ilami4)), _op('mul', _r(5.0), ilami5)))), qclig)
    # WRF:1899
    nclig = jnp.where(_m30, _r(_op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', dt, piov4), eig), _op('mul', ny, ng)), gi31), gg31), tmp3), _op('add', _op('add', _op('mul', _op('mul', gi33, gg31), ilami2), _op('mul', _op('mul', _op('mul', _op('mul', _r(2.0), gi32), gg32), ilami), ilamg)), _op('mul', _op('mul', gi31, gg33), ilamg2)))), nclig)
    # WRF:1902
    qclig = jnp.where(_m30, _r(jnp.minimum(qclig, qi)), qclig)
    # WRF:1903
    nclig = jnp.where(_m30, _r(jnp.minimum(_op('mul', qclig, _op('mul', ny, iqi)), nclig)), nclig)
    _m31 = (_m27) & _not(_m30)
    # WRF:1905
    qclig = jnp.where(_m31, _r(_r(0.0)), qclig)
    # WRF:1905
    nclig = jnp.where(_m31, _r(_r(0.0)), nclig)
    # WRF:1909
    ventg = jnp.where(_m27, _r(_op('add', _op('mul', _op('mul', _op('mul', avx, gg32), ilamg), ilamg), _op('mul', _op('mul', _op('mul', _op('mul', bvx, scthrd), jnp.sqrt(_op('mul', _op('mul', gam, afg), imukin))), gg09), _pow(ilamg, _op('add', _op('add', _r(2.5), _op('mul', _r(0.5), bfg)), alpha_g))))), ventg)
    # WRF:1913
    qvdvg = jnp.where(_m27, _r(_op('mul', _op('mul', _op('mul', dt, ide), iabi), _op('mul', _op('mul', _op('mul', pi2, _op('sub', si, _r(1.0))), no_g), ventg))), qvdvg)
    # WRF:1915
    vdmax = jnp.where(_m27, _r(_op('div', _op('sub', q, qsi), _op('add', _r(1.0), _op('div', _op('mul', ck6, qsi), _pow(_op('sub', t, _r(7.66)), 2))))), vdmax)
    _m32 = (_m27) & (si >= _r(1.0))
    # WRF:1917
    qvdvg = jnp.where(_m32, _r(jnp.minimum(jnp.maximum(qvdvg, _r(0.0)), vdmax)), qvdvg)
    _m33 = (_m27) & _not(_m32)
    _m34 = (_m33) & (vdmax < _r(0.0))
    # WRF:1919
    qvdvg = jnp.where(_m34, _r(jnp.maximum(qvdvg, vdmax)), qvdvg)
    # WRF:1923
    nvdvg = jnp.where(_m27, _r(_r(0.0)), nvdvg)
    _m35 = (_m2) & _not(_m27)
    # WRF:1926
    qclcg = jnp.where(_m35, _r(_r(0.0)), qclcg)
    # WRF:1926
    qclrg = jnp.where(_m35, _r(_r(0.0)), qclrg)
    # WRF:1926
    qclig = jnp.where(_m35, _r(_r(0.0)), qclig)
    # WRF:1927
    nclcg = jnp.where(_m35, _r(_r(0.0)), nclcg)
    # WRF:1927
    nclrg = jnp.where(_m35, _r(_r(0.0)), nclrg)
    # WRF:1927
    nclig = jnp.where(_m35, _r(_r(0.0)), nclig)
    _m36 = (_m2) & (qh > epsq)
    _m37 = (_m36) & (qc > epsq)
    # WRF:1935
    ech = jnp.where(_m37, _r(_exp(_op('mul', _op('mul', -_r(8.68e-07), _pow(dc, -_r(1.6))), dh))), ech)
    # WRF:1937
    qclch = jnp.where(_m37, _r(_op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', dt, gam), afh), cmr), ech), piov4), ide), _op('mul', nc, nh)), igc5), igh31), _op('add', _op('add', _op('mul', _op('mul', _op('mul', gc13, gh13), ilamc3), ilamhb2), _op('mul', _op('mul', _op('mul', _op('mul', _r(2.0), gc14), gh12), ilamc4), ilamhb1)), _op('mul', _op('mul', _op('mul', gc15, gh11), ilamc5), ilamhb0)))), qclch)
    # WRF:1941
    nclch = jnp.where(_m37, _r(_op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', dt, gam), afh), piov4), ech), _op('mul', nc, nh)), igc5), igh31), _op('add', _op('add', _op('mul', _op('mul', gc5, gh13), ilamhb2), _op('mul', _op('mul', _op('mul', _op('mul', _r(2.0), gc11), gh12), ilamc), ilamhb1)), _op('mul', _op('mul', _op('mul', gc12, gh11), ilamc2), ilamhb0)))), nclch)
    # WRF:1944
    qclch = jnp.where(_m37, _r(jnp.minimum(qclch, qc)), qclch)
    # WRF:1945
    nclch = jnp.where(_m37, _r(jnp.minimum(nclch, nc)), nclch)
    _m38 = (_m36) & _not(_m37)
    # WRF:1947
    qclch = jnp.where(_m38, _r(_r(0.0)), qclch)
    # WRF:1947
    nclch = jnp.where(_m38, _r(_r(0.0)), nclch)
    _m39 = (_m36) & (qr > epsq)
    # WRF:1952
    tmp1 = jnp.where(_m39, _r(_op('sub', vh0, vr0)), tmp1)
    # WRF:1953
    tmp3 = jnp.where(_m39, _r(jnp.sqrt(_op('add', _op('mul', tmp1, tmp1), _op('mul', _op('mul', _r(0.04), vh0), vr0)))), tmp3)
    # WRF:1954
    qclrh = jnp.where(_m39, _r(_op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', dt, cmr), erh), piov4), ide), _op('mul', nh, nr)), igr31), igh31), tmp3), _op('add', _op('add', _op('mul', _op('mul', gr36, gh31), ilamr5), _op('mul', _op('mul', _op('mul', _op('mul', _r(2.0), gr35), gh32), ilamr4), ilamh)), _op('mul', _op('mul', _op('mul', gr34, gh33), ilamr3), ilamh2)))), qclrh)
    # WRF:1957
    nclrh = jnp.where(_m39, _r(_op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', dt, piov4), erh), _op('mul', nh, nr)), igr31), igh31), tmp3), _op('add', _op('add', _op('mul', _op('mul', gr33, gh31), ilamr2), _op('mul', _op('mul', _op('mul', _op('mul', _r(2.0), gr32), gh32), ilamr), ilamh)), _op('mul', _op('mul', gr31, gh33), ilamh2)))), nclrh)
    # WRF:1960
    qclrh = jnp.where(_m39, _r(jnp.minimum(qclrh, qr)), qclrh)
    # WRF:1961
    nclrh = jnp.where(_m39, _r(jnp.minimum(nclrh, _op('mul', qclrh, _op('mul', nr, iqr)))), nclrh)
    _m40 = (_m36) & _not(_m39)
    # WRF:1963
    qclrh = jnp.where(_m40, _r(_r(0.0)), qclrh)
    # WRF:1963
    nclrh = jnp.where(_m40, _r(_r(0.0)), nclrh)
    _m41 = (_m36) & (qi > epsq)
    # WRF:1968
    tmp1 = jnp.where(_m41, _r(_op('sub', vh0, vi0)), tmp1)
    # WRF:1969
    tmp3 = jnp.where(_m41, _r(jnp.sqrt(_op('add', _op('mul', tmp1, tmp1), _op('mul', _op('mul', _r(0.04), vh0), vi0)))), tmp3)
    # WRF:1971
    qclih = jnp.where(_m41, _r(_op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', dt, cmi), ide), pi), _r(6.0)), eih), _op('mul', ny, nh)), tmp3), igi31), igh31), _op('add', _op('add', _op('mul', _op('mul', _r(0.5), ilamh2), ilami3), _op('mul', _op('mul', _r(2.0), ilamh), ilami4)), _op('mul', _r(5.0), ilami5)))), qclih)
    # WRF:1974
    nclih = jnp.where(_m41, _r(_op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', dt, piov4), eih), _op('mul', ny, nh)), gi31), gh31), tmp3), _op('add', _op('add', _op('mul', _op('mul', gi33, gh31), ilami2), _op('mul', _op('mul', _op('mul', _op('mul', _r(2.0), gi32), gh32), ilami), ilamh)), _op('mul', _op('mul', gi31, gh33), ilamh2)))), nclih)
    # WRF:1977
    qclih = jnp.where(_m41, _r(jnp.minimum(qclih, qi)), qclih)
    # WRF:1978
    nclih = jnp.where(_m41, _r(jnp.minimum(_op('mul', qclih, _op('mul', ny, iqi)), nclih)), nclih)
    _m42 = (_m36) & _not(_m41)
    # WRF:1980
    qclih = jnp.where(_m42, _r(_r(0.0)), qclih)
    # WRF:1980
    nclih = jnp.where(_m42, _r(_r(0.0)), nclih)
    _m43 = (_m36) & (qn > epsq)
    # WRF:1985
    tmp1 = jnp.where(_m43, _r(_op('sub', vh0, vs0)), tmp1)
    # WRF:1986
    tmp3 = jnp.where(_m43, _r(jnp.sqrt(_op('add', _op('mul', tmp1, tmp1), _op('mul', _op('mul', _r(0.04), vh0), vs0)))), tmp3)
    # WRF:1987
    tmp4 = jnp.where(_m43, _r(_op('mul', ilams2, ilams2)), tmp4)
    _m44 = (_m43) & (snowspherical)
    # WRF:1991
    qclsh = jnp.where(_m44, _r(_op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', dt, cms), ide), pi), _r(6.0)), esh), _op('mul', nn, nh)), tmp3), igs31), igh31), _op('add', _op('add', _op('mul', _op('mul', _op('mul', _r(0.5), ilamh2), ilams2), ilams), _op('mul', _op('mul', _r(2.0), ilamh), tmp4)), _op('mul', _op('mul', _r(5.0), tmp4), ilams)))), qclsh)
    _m45 = (_m43) & _not(_m44)
    # WRF:1995
    qclsh = jnp.where(_m45, _r(_op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', dt, cms), ide), pi), _r(0.25)), esh), tmp3), nn), nh), igs31), igh31), _op('add', _op('add', _op('mul', _op('mul', _op('mul', gh33, gs33), _pow(ilamh, _r(2.0))), _pow(ilams, _r(2.0))), _op('mul', _op('mul', _op('mul', _op('mul', _r(2.0), gh32), gs34), ilamh), _pow(ilams, _r(3.0)))), _op('mul', _op('mul', gh31, gs35), _pow(ilams, _r(4.0)))))), qclsh)
    # WRF:2000
    nclsh = jnp.where(_m43, _r(_op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', dt, piov4), esh), _op('mul', nn, nh)), gs31), gh31), tmp3), _op('add', _op('add', _op('mul', _op('mul', gs33, gh31), ilams2), _op('mul', _op('mul', _op('mul', _op('mul', _r(2.0), gs32), gh32), ilams), ilamh)), _op('mul', _op('mul', gs31, gh33), ilamh2)))), nclsh)
    # WRF:2003
    qclsh = jnp.where(_m43, _r(jnp.minimum(qclsh, qn)), qclsh)
    # WRF:2004
    nclsh = jnp.where(_m43, _r(jnp.minimum(jnp.minimum(_op('mul', _op('mul', nn, iqn), qclsh), nclsh), nn)), nclsh)
    _m46 = (_m36) & _not(_m43)
    # WRF:2006
    qclsh = jnp.where(_m46, _r(_r(0.0)), qclsh)
    # WRF:2006
    nclsh = jnp.where(_m46, _r(_r(0.0)), nclsh)
    # WRF:2010
    venth = jnp.where(_m36, _r(_op('add', _op('mul', _op('mul', avx, gh32), _pow(ilamh, _op('add', _r(2.0), alpha_h))), _op('mul', _op('mul', _op('mul', _op('mul', bvx, scthrd), jnp.sqrt(_op('mul', _op('mul', gam, afh), imukin))), gh09), _pow(ilamh, _op('add', _op('add', _r(2.5), _op('mul', _r(0.5), bfh)), alpha_h))))), venth)
    # WRF:2012
    qhwet = jnp.where(_m36, _r(jnp.maximum(_r(0.0), _op('add', _op('mul', _op('div', _op('mul', _op('mul', _op('mul', _op('mul', dt, pi2), _op('sub', _op('mul', _op('mul', _op('mul', de, chlc), cdiff), delqvs), _op('mul', ka, tc))), no_h), ide), _op('add', chlf, _op('mul', cpw, tc))), venth), _op('mul', _op('add', _op('mul', qclih, ieih), _op('mul', qclsh, iesh)), _op('sub', _r(1.0), _op('div', _op('mul', cpi, tc), _op('add', chlf, _op('mul', cpw, tc)))))))), qhwet)
    # WRF:2018
    qvdvh = jnp.where(_m36, _r(_op('mul', _op('mul', _op('mul', dt, ide), iabi), _op('mul', _op('mul', _op('mul', pi2, _op('sub', si, _r(1.0))), no_h), venth))), qvdvh)
    # WRF:2020
    vdmax = jnp.where(_m36, _r(_op('div', _op('sub', q, qsi), _op('add', _r(1.0), _op('div', _op('mul', ck6, qsi), _pow(_op('sub', t, _r(7.66)), 2))))), vdmax)
    _m47 = (_m36) & (si >= _r(1.0))
    # WRF:2022
    qvdvh = jnp.where(_m47, _r(jnp.minimum(jnp.maximum(qvdvh, _r(0.0)), vdmax)), qvdvh)
    _m48 = (_m36) & _not(_m47)
    _m49 = (_m48) & (vdmax < _r(0.0))
    # WRF:2024
    qvdvh = jnp.where(_m49, _r(jnp.maximum(qvdvh, vdmax)), qvdvh)
    # WRF:2027
    nvdvh = jnp.where(_m36, _r(_r(0.0)), nvdvh)
    _m50 = (_m2) & _not(_m36)
    # WRF:2030
    qclch = jnp.where(_m50, _r(_r(0.0)), qclch)
    # WRF:2030
    qclrh = jnp.where(_m50, _r(_r(0.0)), qclrh)
    # WRF:2030
    qclih = jnp.where(_m50, _r(_r(0.0)), qclih)
    # WRF:2030
    qclsh = jnp.where(_m50, _r(_r(0.0)), qclsh)
    # WRF:2030
    qhwet = jnp.where(_m50, _r(_r(0.0)), qhwet)
    # WRF:2031
    nclch = jnp.where(_m50, _r(_r(0.0)), nclch)
    # WRF:2031
    nclrh = jnp.where(_m50, _r(_r(0.0)), nclrh)
    # WRF:2031
    nclsh = jnp.where(_m50, _r(_r(0.0)), nclsh)
    # WRF:2031
    nclih = jnp.where(_m50, _r(_r(0.0)), nclih)
    _m51 = (_m2) & ((t > trpl) & warmphase_on)
    # WRF:2041
    qmlir = jnp.where(_m51, _r(qi), qmlir)
    # WRF:2042
    qi = jnp.where(_m51, _r(_r(0.0)), qi)
    # WRF:2043
    nmlir = jnp.where(_m51, _r(ny), nmlir)
    _m52 = (_m51) & (qn > epsq)
    # WRF:2047
    qmlsr = jnp.where(_m52, _r(_op('mul', dt, _op('add', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', pi2, ide), ichlf), no_s), vents), _op('sub', _op('mul', ka, tc), _op('mul', _op('mul', chlc, cdiff), delqvs))), _op('mul', _op('mul', _op('mul', _op('mul', cpw, ichlf), tc), _op('add', qclcs, qclrs)), idt)))), qmlsr)
    # WRF:2049
    qmlsr = jnp.where(_m52, _r(jnp.minimum(jnp.maximum(qmlsr, _r(0.0)), qn)), qmlsr)
    # WRF:2050
    nmlsr = jnp.where(_m52, _r(_op('mul', _op('mul', nn, iqn), qmlsr)), nmlsr)
    _m53 = (_m51) & _not(_m52)
    # WRF:2052
    qmlsr = jnp.where(_m53, _r(_r(0.0)), qmlsr)
    # WRF:2052
    nmlsr = jnp.where(_m53, _r(_r(0.0)), nmlsr)
    _m54 = (_m51) & (qg > epsq)
    # WRF:2057
    qmlgr = jnp.where(_m54, _r(_op('mul', dt, _op('add', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', pi2, ide), ichlf), no_g), ventg), _op('sub', _op('mul', ka, tc), _op('mul', _op('mul', chlc, cdiff), delqvs))), _op('mul', _op('mul', _op('mul', _op('mul', cpw, ichlf), tc), _op('add', qclcg, qclrg)), idt)))), qmlgr)
    # WRF:2059
    qmlgr = jnp.where(_m54, _r(jnp.minimum(jnp.maximum(qmlgr, _r(0.0)), qg)), qmlgr)
    # WRF:2060
    nmlgr = jnp.where(_m54, _r(_op('mul', _op('mul', ng, iqg), qmlgr)), nmlgr)
    _m55 = (_m51) & _not(_m54)
    # WRF:2062
    qmlgr = jnp.where(_m55, _r(_r(0.0)), qmlgr)
    # WRF:2062
    nmlgr = jnp.where(_m55, _r(_r(0.0)), nmlgr)
    _m56 = (_m51) & ((qh > epsq) & (tc > _r(5.0)))
    # WRF:2067
    venth = jnp.where(_m56, _r(_op('add', _op('mul', _op('mul', avx, gh32), _pow(ilamh, _op('add', _r(2.0), alpha_h))), _op('mul', _op('mul', _op('mul', _op('mul', bvx, scthrd), jnp.sqrt(_op('mul', _op('mul', gam, afh), imukin))), gh09), _pow(ilamh, _op('add', _op('add', _r(2.5), _op('mul', _r(0.5), bfh)), alpha_h))))), venth)
    # WRF:2069
    qmlhr = jnp.where(_m56, _r(_op('mul', dt, _op('add', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', pi2, ide), ichlf), no_h), venth), _op('sub', _op('mul', ka, tc), _op('mul', _op('mul', chlc, cdiff), delqvs))), _op('mul', _op('mul', _op('mul', _op('div', cpw, chlf), tc), _op('add', qclch, qclrh)), idt)))), qmlhr)
    # WRF:2071
    qmlhr = jnp.where(_m56, _r(jnp.minimum(jnp.maximum(qmlhr, _r(0.0)), qh)), qmlhr)
    # WRF:2072
    nmlhr = jnp.where(_m56, _r(_op('mul', _op('mul', nh, iqh), qmlhr)), nmlhr)
    _m57 = (_m56) & (qclrh > _r(0.0))
    # WRF:2073
    nmlhr = jnp.where(_m57, _r(_op('mul', nmlhr, _r(0.1))), nmlhr)
    _m58 = (_m51) & _not(_m56)
    # WRF:2075
    qmlhr = jnp.where(_m58, _r(_r(0.0)), qmlhr)
    # WRF:2075
    nmlhr = jnp.where(_m58, _r(_r(0.0)), nmlhr)
    # WRF:2079
    qnuvi = jnp.where(_m51, _r(_r(0.0)), qnuvi)
    # WRF:2079
    qfzci = jnp.where(_m51, _r(_r(0.0)), qfzci)
    # WRF:2079
    qvdvi = jnp.where(_m51, _r(_r(0.0)), qvdvi)
    # WRF:2079
    qvdvs = jnp.where(_m51, _r(_r(0.0)), qvdvs)
    # WRF:2080
    qclis = jnp.where(_m51, _r(_r(0.0)), qclis)
    # WRF:2080
    qcnis1 = jnp.where(_m51, _r(_r(0.0)), qcnis1)
    # WRF:2080
    qcnis2 = jnp.where(_m51, _r(_r(0.0)), qcnis2)
    # WRF:2080
    qclri = jnp.where(_m51, _r(_r(0.0)), qclri)
    # WRF:2081
    qcngh = jnp.where(_m51, _r(_r(0.0)), qcngh)
    # WRF:2081
    qimsi = jnp.where(_m51, _r(_r(0.0)), qimsi)
    # WRF:2081
    qimgi = jnp.where(_m51, _r(_r(0.0)), qimgi)
    # WRF:2081
    qclir = jnp.where(_m51, _r(_r(0.0)), qclir)
    # WRF:2082
    qclrs = jnp.where(_m51, _r(_r(0.0)), qclrs)
    # WRF:2082
    qclgr = jnp.where(_m51, _r(_r(0.0)), qclgr)
    # WRF:2082
    qclrg = jnp.where(_m51, _r(_r(0.0)), qclrg)
    # WRF:2082
    qcnis = jnp.where(_m51, _r(_r(0.0)), qcnis)
    # WRF:2083
    qcnsg = jnp.where(_m51, _r(_r(0.0)), qcnsg)
    # WRF:2083
    qclsr = jnp.where(_m51, _r(_r(0.0)), qclsr)
    # WRF:2085
    nnuvi = jnp.where(_m51, _r(_r(0.0)), nnuvi)
    # WRF:2085
    nfzci = jnp.where(_m51, _r(_r(0.0)), nfzci)
    # WRF:2085
    nclgr = jnp.where(_m51, _r(_r(0.0)), nclgr)
    # WRF:2085
    nclrg = jnp.where(_m51, _r(_r(0.0)), nclrg)
    # WRF:2085
    ngcngh = jnp.where(_m51, _r(_r(0.0)), ngcngh)
    # WRF:2086
    nclis = jnp.where(_m51, _r(_r(0.0)), nclis)
    # WRF:2086
    nvdvi = jnp.where(_m51, _r(_r(0.0)), nvdvi)
    # WRF:2086
    nvdvs = jnp.where(_m51, _r(_r(0.0)), nvdvs)
    # WRF:2086
    nclri = jnp.where(_m51, _r(_r(0.0)), nclri)
    # WRF:2086
    nclsr = jnp.where(_m51, _r(_r(0.0)), nclsr)
    # WRF:2087
    ncnsg = jnp.where(_m51, _r(_r(0.0)), ncnsg)
    # WRF:2087
    nhcngh = jnp.where(_m51, _r(_r(0.0)), nhcngh)
    # WRF:2087
    nicnis = jnp.where(_m51, _r(_r(0.0)), nicnis)
    # WRF:2087
    nscnis = jnp.where(_m51, _r(_r(0.0)), nscnis)
    # WRF:2088
    nimsi = jnp.where(_m51, _r(_r(0.0)), nimsi)
    # WRF:2088
    nimgi = jnp.where(_m51, _r(_r(0.0)), nimgi)
    # WRF:2088
    nclir = jnp.where(_m51, _r(_r(0.0)), nclir)
    # WRF:2088
    nclrs = jnp.where(_m51, _r(_r(0.0)), nclrs)
    _m59 = (_m2) & _not(_m51)
    # WRF:2095
    qmlir = jnp.where(_m59, _r(_r(0.0)), qmlir)
    # WRF:2095
    qmlsr = jnp.where(_m59, _r(_r(0.0)), qmlsr)
    # WRF:2095
    qmlgr = jnp.where(_m59, _r(_r(0.0)), qmlgr)
    # WRF:2095
    qmlhr = jnp.where(_m59, _r(_r(0.0)), qmlhr)
    # WRF:2096
    nmlir = jnp.where(_m59, _r(_r(0.0)), nmlir)
    # WRF:2096
    nmlsr = jnp.where(_m59, _r(_r(0.0)), nmlsr)
    # WRF:2096
    nmlgr = jnp.where(_m59, _r(_r(0.0)), nmlgr)
    # WRF:2096
    nmlhr = jnp.where(_m59, _r(_r(0.0)), nmlhr)
    _m60 = (_m59) & ((tc < tc_fzrh) & (qr > epsq) & hail_on)
    # WRF:2104
    nrfzrh = jnp.where(_m60, _r(_op('mul', _op('mul', _op('mul', _op('mul', _op('mul', -dt, bbigg), _op('sub', _exp(_op('mul', abigg, tc)), _r(1.0))), de), qr), idew)), nrfzrh)
    # WRF:2105
    rz = jnp.where(_m60, _r(_r(1.0)), rz)
    # WRF:2112
    nhfzrh = jnp.where(_m60, _r(_op('mul', rz, nrfzrh)), nhfzrh)
    # WRF:2113
    qfzrh = jnp.where(_m60, _r(_op('mul', nrfzrh, _op('mul', qr, inr))), qfzrh)
    _m61 = (_m59) & _not(_m60)
    # WRF:2115
    qfzrh = jnp.where(_m61, _r(_r(0.0)), qfzrh)
    # WRF:2115
    nrfzrh = jnp.where(_m61, _r(_r(0.0)), nrfzrh)
    # WRF:2115
    nhfzrh = jnp.where(_m61, _r(_r(0.0)), nhfzrh)
    _m62 = (_m59) & (qc > epsq)
    # WRF:2123
    tmp2 = jnp.where(_m62, _r(_op('mul', tc, tc)), tmp2)
    # WRF:2123
    tmp3 = jnp.where(_m62, _r(_op('mul', tmp2, tc)), tmp3)
    # WRF:2123
    tmp4 = jnp.where(_m62, _r(_op('mul', tmp2, tmp2)), tmp4)
    # WRF:2124
    jj = jnp.where(_m62, _r(_pow(_r(10.0), jnp.maximum(-_r(20.0), _op('sub', _op('sub', _op('sub', _op('sub', -_r(606.3952), _op('mul', _r(52.6611), tc)), _op('mul', _r(1.7439), tmp2)), _op('mul', _r(0.0265), tmp3)), _op('mul', _r(0.0001536), tmp4))))), jj)
    # WRF:2126
    tmp1 = jnp.where(_m62, _r(_op('mul', _r(1000000.0), _op('mul', _op('mul', de, _op('mul', qc, inc)), icmr))), tmp1)
    # WRF:2127
    frac = jnp.where(_m62, _r(_op('sub', _r(1.0), _exp(_op('mul', _op('mul', _op('mul', -jj, piov6), tmp1), dt)))), frac)
    _m63 = (_m62) & (tc > -_r(30.0))
    # WRF:2128
    frac = jnp.where(_m63, _r(_r(0.0)), frac)
    _m64 = (_m62) & (tc < -_r(50.0))
    # WRF:2129
    frac = jnp.where(_m64, _r(_r(1.0)), frac)
    # WRF:2130
    qfzci = jnp.where(_m62, _r(_op('mul', frac, qc)), qfzci)
    # WRF:2131
    nfzci = jnp.where(_m62, _r(_op('mul', frac, nc)), nfzci)
    _m65 = (_m59) & _not(_m62)
    # WRF:2133
    qfzci = jnp.where(_m65, _r(_r(0.0)), qfzci)
    # WRF:2133
    nfzci = jnp.where(_m65, _r(_r(0.0)), nfzci)
    # WRF:2137
    nnuvi = jnp.where(_m59, _r(_r(0.0)), nnuvi)
    # WRF:2137
    qnuvi = jnp.where(_m59, _r(_r(0.0)), qnuvi)
    _m66 = (_m59) & (primicenucl == 1)
    # WRF:2140
    nudepsor = jnp.where(_m66, _r(_r(0.0)), nudepsor)
    # WRF:2140
    nucont = jnp.where(_m66, _r(_r(0.0)), nucont)
    _m67 = (_m66) & (qsi > _r(1e-20))
    # WRF:2142
    simax = jnp.where(_m67, _r(jnp.minimum(si, _op('div', _op('mul', satw_peak, qsw), qsi))), simax)
    _m68 = (_m66) & _not(_m67)
    # WRF:2144
    simax = jnp.where(_m68, _r(_r(0.0)), simax)
    # WRF:2146
    tmp1 = jnp.where(_m66, _r(_op('sub', t, _r(7.66))), tmp1)
    # WRF:2147
    nnumax = jnp.where(_m66, _r(jnp.maximum(_r(0.0), _op('div', _op('mul', _op('div', de, mio), _op('sub', q, qsi)), _op('add', _r(1.0), _op('mul', ck6, _op('div', qsi, _op('mul', tmp1, tmp1))))))), nnumax)
    _m69 = (_m66) & ((tc < -_r(5.0)) & (si > _r(1.0)))
    # WRF:2151
    nudepsor = jnp.where(_m69, _r(jnp.maximum(_r(0.0), _op('sub', _op('mul', _r(1000.0), _exp(_op('sub', _op('mul', _r(12.96), _op('sub', simax, _r(1.0))), _r(0.639)))), ny))), nudepsor)
    _m70 = (_m66) & ((qc > epsq) & (tc < -_r(2.0)) & (wz > _r(0.001)))
    # WRF:2155
    gg = jnp.where(_m70, _r(_op('div', _op('mul', _r(1.0), idew), _op('add', _op('div', _op('div', _op('mul', rgasv, t), _op('div', _op('mul', qsw, pres), eps1)), cdiff), _op('mul', _op('div', _op('div', chlc, ka), t), _op('sub', _op('div', _op('div', chlc, rgasv), t), _r(1.0)))))), gg)
    # WRF:2157
    swmax = jnp.where(_m70, _r(_sxfnc(wz, tc, pres, qsw, qsi, ccntype, 1)), swmax)
    _m71 = (_m70) & (qsw > _r(1e-20))
    # WRF:2159
    ssat = jnp.where(_m71, _r(_op('sub', jnp.minimum(_op('div', q, qsw), swmax), _r(1.0))), ssat)
    _m72 = (_m70) & _not(_m71)
    # WRF:2161
    ssat = jnp.where(_m72, _r(_r(0.0)), ssat)
    # WRF:2163
    tcc = jnp.where(_m70, _r(_op('add', tc, _op('div', _op('mul', _op('mul', gg, ssat), chlc), kdiff))), tcc)
    # WRF:2164
    na = jnp.where(_m70, _r(_exp(_op('sub', _r(4.11), _op('mul', _r(0.262), tcc)))), na)
    # WRF:2165
    kn = jnp.where(_m70, _r(_op('div', _op('mul', _op('mul', lama0, t), p0), _op('mul', _op('mul', t0, pres), ra))), kn)
    # WRF:2166
    psia = jnp.where(_m70, _r(_op('mul', _op('div', _op('mul', -kboltz, tcc), _op('mul', _op('mul', _op('mul', _r(6.0), pi), ra), mudyn)), _op('add', _r(1.0), kn))), psia)
    # WRF:2167
    ft = jnp.where(_m70, _r(_op('div', _op('div', _op('mul', _op('mul', _r(0.4), _op('add', _op('add', _r(1.0), _op('mul', _r(1.45), kn)), _op('mul', _op('mul', _r(0.4), kn), _exp(_op('div', -_r(1.0), kn))))), _op('add', ka, _op('mul', _op('mul', _r(2.5), kn), kapa))), _op('add', _r(1.0), _op('mul', _r(3.0), kn))), _op('add', _op('add', _op('mul', _r(2.0), ka), _op('mul', _op('mul', _r(5.0), kapa), kn)), kapa))), ft)
    # WRF:2169
    dc = jnp.where(_m70, _r(_pow(_op('mul', _op('mul', de, _op('mul', qc, inc)), icmr), thrd)), dc)
    # WRF:2170
    f1 = jnp.where(_m70, _r(_op('mul', _op('mul', _op('mul', pi2, dc), na), nc)), f1)
    # WRF:2171
    f2 = jnp.where(_m70, _r(_op('mul', _op('div', ka, pres), _op('sub', tc, tcc))), f2)
    # WRF:2172
    nuconta = jnp.where(_m70, _r(_op('mul', _op('div', _op('mul', _op('mul', _op('mul', -f1, f2), rgasv), t), chlc), ide)), nuconta)
    # WRF:2173
    nucontb = jnp.where(_m70, _r(_op('mul', _op('mul', _op('mul', f1, f2), ft), ide)), nucontb)
    # WRF:2174
    nucontc = jnp.where(_m70, _r(_op('mul', f1, psia)), nucontc)
    # WRF:2175
    nucont = jnp.where(_m70, _r(jnp.maximum(_r(0.0), _op('mul', _op('add', _op('add', nuconta, nucontb), nucontc), dt))), nucont)
    _m73 = (_m66) & (icephase_on)
    # WRF:2179
    nnuvi = jnp.where(_m73, _r(jnp.minimum(nnumax, _op('add', nudepsor, nucont))), nnuvi)
    # WRF:2180
    qnuvi = jnp.where(_m73, _r(_op('mul', _op('mul', mio, ide), nnuvi)), qnuvi)
    # WRF:2181
    qnuvi = jnp.where(_m73, _r(jnp.minimum(qnuvi, q)), qnuvi)
    _m74 = (_m59) & _not(_m66) & (primicenucl == 2)
    _m75 = (_m74) & ((tc < -_r(5.0)) & (si > _r(1.08)))
    # WRF:2186
    nnuvi = jnp.where(_m75, _r(jnp.maximum(_op('sub', _n_cooper(trpl, t), ny), _r(0.0))), nnuvi)
    # WRF:2187
    qnuvi = jnp.where(_m75, _r(jnp.minimum(_op('mul', _op('mul', mio, ide), nnuvi), q)), qnuvi)
    _m76 = (_m59) & (qi > epsq)
    # WRF:2203
    no_i = jnp.where(_m76, _r(_op('div', _op('mul', ny, igi31), ilami)), no_i)
    # WRF:2204
    venti = jnp.where(_m76, _r(_op('add', _op('mul', _op('mul', _op('mul', avx, gi32), ilami), ilami), _op('mul', _op('mul', _op('mul', _op('mul', bvx, scthrd), jnp.sqrt(_op('mul', _op('mul', gam, afi), imukin))), gi6), _pow(ilami, _op('add', _op('add', _r(2.5), _op('mul', _r(0.5), bfi)), alpha_i))))), venti)
    # WRF:2208
    qvdvi = jnp.where(_m76, _r(_op('mul', _op('mul', _op('mul', _op('mul', dt, ide), capfact_i), iabi), _op('mul', _op('mul', _op('mul', pi2, _op('sub', si, _r(1.0))), no_i), venti))), qvdvi)
    # WRF:2211
    vdmax = jnp.where(_m76, _r(_op('div', _op('sub', q, qsi), _op('add', _r(1.0), _op('div', _op('mul', ck6, qsi), _pow(_op('sub', t, _r(7.66)), 2))))), vdmax)
    _m77 = (_m76) & (si >= _r(1.0))
    # WRF:2213
    qvdvi = jnp.where(_m77, _r(jnp.minimum(jnp.maximum(qvdvi, _r(0.0)), vdmax)), qvdvi)
    _m78 = (_m76) & _not(_m77)
    _m79 = (_m78) & (vdmax < _r(0.0))
    # WRF:2215
    qvdvi = jnp.where(_m79, _r(jnp.maximum(qvdvi, vdmax)), qvdvi)
    _m80 = (_m76) & (_not(icedep_on))
    # WRF:2218
    qvdvi = jnp.where(_m80, _r(_r(0.0)), qvdvi)
    # WRF:2219
    nvdvi = jnp.where(_m76, _r(jnp.minimum(_r(0.0), _op('mul', _op('mul', ny, iqi), qvdvi))), nvdvi)
    _m81 = (_m76) & ((_op('add', qi, qvdvi) > epsq) & (_op('add', ny, nvdvi) > epsn))
    # WRF:2223
    tmp5 = jnp.where(_m81, _r(ilami), tmp5)
    # WRF:2224
    tmp6 = jnp.where(_m81, _r(no_i), tmp6)
    # WRF:2226
    tmp1 = jnp.where(_m81, _r(_op('add', qi, qvdvi)), tmp1)
    # WRF:2227
    tmp2 = jnp.where(_m81, _r(_op('add', ny, nvdvi)), tmp2)
    # WRF:2228
    tmp3 = jnp.where(_m81, _r(_op('div', _r(1.0), tmp2)), tmp3)
    # WRF:2229
    ilami = jnp.where(_m81, _r(jnp.maximum(ilammin2, _ilamda_x(de, tmp1, tmp3, icexi9, thrd))), ilami)
    # WRF:2230
    no_i = jnp.where(_m81, _r(_op('div', _op('mul', tmp2, igi31), ilami)), no_i)
    # WRF:2233
    tmp4 = jnp.where(_m81, _r(_exp(_op('div', -dso, ilami))), tmp4)
    # WRF:2234
    nicnis = jnp.where(_m81, _r(_op('mul', _op('mul', no_i, ilami), tmp4)), nicnis)
    # WRF:2235
    nscnis = jnp.where(_m81, _r(nicnis), nscnis)
    # WRF:2236
    qcnis = jnp.where(_m81, _r(_op('mul', _op('mul', _op('mul', cmi, no_i), tmp4), _op('add', _op('add', _op('add', _op('mul', _pow(dso, 3), ilami), _op('mul', _op('mul', _r(3.0), _pow(dso, 2)), _pow(ilami, 2))), _op('mul', _op('mul', _r(6.0), dso), _pow(ilami, 3))), _op('mul', _r(6.0), _pow(ilami, 4))))), qcnis)
    # WRF:2238
    ilami = jnp.where(_m81, _r(tmp5), ilami)
    # WRF:2239
    no_i = jnp.where(_m81, _r(tmp6), no_i)
    _m82 = (_m76) & (_not(snow_on))
    # WRF:2243
    qcnis = jnp.where(_m82, _r(_r(0.0)), qcnis)
    # WRF:2244
    nicnis = jnp.where(_m82, _r(_r(0.0)), nicnis)
    # WRF:2245
    nscnis = jnp.where(_m82, _r(_r(0.0)), nscnis)
    _m83 = (_m76) & ((qr > epsq) & (qi > epsq))
    # WRF:2250
    tmp1 = jnp.where(_m83, _r(_op('sub', vr0, vi0)), tmp1)
    # WRF:2251
    tmp3 = jnp.where(_m83, _r(jnp.sqrt(_op('add', _op('mul', tmp1, tmp1), _op('mul', _op('mul', _r(0.04), vr0), vi0)))), tmp3)
    # WRF:2253
    qclir = jnp.where(_m83, _r(_op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', dt, cmi), eri), piov4), ide), _op('mul', nr, ny)), igi31), igr31), tmp3), _op('add', _op('add', _op('mul', _op('mul', gi36, gr31), ilami5), _op('mul', _op('mul', _op('mul', _op('mul', _r(2.0), gi35), gr32), ilami4), ilamr)), _op('mul', _op('mul', _op('mul', gi34, gr33), ilami3), ilamr2)))), qclir)
    # WRF:2257
    nclri = jnp.where(_m83, _r(_op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', dt, piov4), eri), _op('mul', nr, ny)), igi31), igr31), tmp3), _op('add', _op('add', _op('mul', _op('mul', gi33, gr31), ilami2), _op('mul', _op('mul', _op('mul', _op('mul', _r(2.0), gi32), gr32), ilami), ilamr)), _op('mul', _op('mul', gi31, gr33), ilamr2)))), nclri)
    # WRF:2260
    qclri = jnp.where(_m83, _r(_op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', dt, cmr), eri), piov4), ide), _op('mul', ny, nr)), igr31), igi31), tmp3), _op('add', _op('add', _op('mul', _op('mul', gr36, gi31), ilamr5), _op('mul', _op('mul', _op('mul', _op('mul', _r(2.0), gr35), gi32), ilamr4), ilami)), _op('mul', _op('mul', _op('mul', gr34, gi33), ilamr3), ilami2)))), qclri)
    # WRF:2265
    nclir = jnp.where(_m83, _r(jnp.minimum(_op('mul', qclir, _op('mul', ny, iqi)), nclri)), nclir)
    # WRF:2266
    qclri = jnp.where(_m83, _r(jnp.minimum(qclri, qr)), qclri)
    # WRF:2266
    qclir = jnp.where(_m83, _r(jnp.minimum(qclir, qi)), qclir)
    # WRF:2267
    nclri = jnp.where(_m83, _r(jnp.minimum(nclri, nr)), nclri)
    # WRF:2267
    nclir = jnp.where(_m83, _r(jnp.minimum(nclir, ny)), nclir)
    # WRF:2270
    tmp1 = jnp.where(_m83, _r(jnp.maximum(di, dr)), tmp1)
    # WRF:2271
    dey = jnp.where(_m83, _r(_op('div', _op('add', _op('mul', _op('mul', _op('mul', dei, di), di), di), _op('mul', _op('mul', _op('mul', dew, dr), dr), dr)), _op('mul', _op('mul', tmp1, tmp1), tmp1))), dey)
    _m84 = (_m83) & ((dey > _op('mul', _r(0.5), _op('add', deg, deh))) & (dr > dr_3cmpthrs) & hail_on)
    # WRF:2273
    dirg = jnp.where(_m84, _r(_r(0.0)), dirg)
    # WRF:2273
    dirh = jnp.where(_m84, _r(_r(1.0)), dirh)
    _m85 = (_m83) & _not(_m84)
    # WRF:2275
    dirg = jnp.where(_m85, _r(_r(1.0)), dirg)
    # WRF:2275
    dirh = jnp.where(_m85, _r(_r(0.0)), dirh)
    _m86 = (_m83) & (_not(grpl_on))
    # WRF:2277
    dirg = jnp.where(_m86, _r(_r(0.0)), dirg)
    _m87 = (_m76) & _not(_m83)
    # WRF:2280
    qclir = jnp.where(_m87, _r(_r(0.0)), qclir)
    # WRF:2280
    nclir = jnp.where(_m87, _r(_r(0.0)), nclir)
    # WRF:2280
    qclri = jnp.where(_m87, _r(_r(0.0)), qclri)
    # WRF:2281
    nclri = jnp.where(_m87, _r(_r(0.0)), nclri)
    # WRF:2281
    dirh = jnp.where(_m87, _r(_r(0.0)), dirh)
    # WRF:2281
    dirg = jnp.where(_m87, _r(_r(0.0)), dirg)
    # WRF:2285
    ff = jnp.where(_m76, _r(_r(0.0)), ff)
    _m88 = (_m76) & ((tc >= -_r(8.0)) & (tc <= -_r(5.0)))
    # WRF:2286
    ff = jnp.where(_m88, _r(_op('mul', _op('mul', _r(350000000.0), _op('add', tc, _r(8.0))), thrd)), ff)
    _m89 = (_m76) & ((tc > -_r(5.0)) & (tc < -_r(3.0)))
    # WRF:2287
    ff = jnp.where(_m89, _r(_op('mul', _op('mul', _r(350000000.0), _op('sub', -_r(3.0), tc)), _r(0.5))), ff)
    # WRF:2288
    nimsi = jnp.where(_m76, _r(_op('mul', _op('mul', de, ff), qclcs)), nimsi)
    # WRF:2289
    nimgi = jnp.where(_m76, _r(_op('mul', _op('mul', de, ff), qclcg)), nimgi)
    # WRF:2290
    qimsi = jnp.where(_m76, _r(_op('mul', _op('mul', mio, ide), nimsi)), qimsi)
    # WRF:2291
    qimgi = jnp.where(_m76, _r(_op('mul', _op('mul', mio, ide), nimgi)), qimgi)
    _m90 = (_m59) & _not(_m76)
    # WRF:2295
    qvdvi = jnp.where(_m90, _r(_r(0.0)), qvdvi)
    # WRF:2295
    qcnis = jnp.where(_m90, _r(_r(0.0)), qcnis)
    # WRF:2296
    qimsi = jnp.where(_m90, _r(_r(0.0)), qimsi)
    # WRF:2296
    qimgi = jnp.where(_m90, _r(_r(0.0)), qimgi)
    # WRF:2296
    qclri = jnp.where(_m90, _r(_r(0.0)), qclri)
    # WRF:2296
    qclir = jnp.where(_m90, _r(_r(0.0)), qclir)
    # WRF:2297
    nvdvi = jnp.where(_m90, _r(_r(0.0)), nvdvi)
    # WRF:2297
    nclir = jnp.where(_m90, _r(_r(0.0)), nclir)
    # WRF:2297
    nimsi = jnp.where(_m90, _r(_r(0.0)), nimsi)
    # WRF:2298
    nicnis = jnp.where(_m90, _r(_r(0.0)), nicnis)
    # WRF:2298
    nscnis = jnp.where(_m90, _r(_r(0.0)), nscnis)
    # WRF:2298
    nimgi = jnp.where(_m90, _r(_r(0.0)), nimgi)
    # WRF:2298
    nclri = jnp.where(_m90, _r(_r(0.0)), nclri)
    _m91 = (_m59) & (qn > epsq)
    # WRF:2311
    qvdvs = jnp.where(_m91, _r(_op('mul', _op('mul', _op('mul', _op('mul', dt, ide), capfact_s), iabi), _op('sub', _op('mul', _op('mul', _op('mul', pi2, _op('sub', si, _r(1.0))), no_s), vents), _op('mul', _op('mul', _op('div', _op('mul', chls, chlf), _op('mul', _op('mul', ka, rgasv), _pow(t, 2))), qclcs), idt)))), qvdvs)
    # WRF:2315
    vdmax = jnp.where(_m91, _r(_op('div', _op('sub', q, qsi), _op('add', _r(1.0), _op('div', _op('mul', ck6, qsi), _pow(_op('sub', t, _r(7.66)), 2))))), vdmax)
    _m92 = (_m91) & (si >= _r(1.0))
    # WRF:2317
    qvdvs = jnp.where(_m92, _r(jnp.minimum(jnp.maximum(qvdvs, _r(0.0)), vdmax)), qvdvs)
    _m93 = (_m91) & _not(_m92)
    _m94 = (_m93) & (vdmax < _r(0.0))
    # WRF:2319
    qvdvs = jnp.where(_m94, _r(jnp.maximum(qvdvs, vdmax)), qvdvs)
    # WRF:2322
    nvdvs = jnp.where(_m91, _r(-jnp.minimum(_r(0.0), _op('mul', _op('mul', nn, iqn), qvdvs))), nvdvs)
    _m95 = (_m91) & ((qclcs > _r(0.0)) & (qclcs > _op('mul', cnsgthres, qvdvs)) & grpl_on)
    # WRF:2326
    tmp1 = jnp.where(_m95, _r(_r(100.0)), tmp1)
    # WRF:2329
    qcnsg = jnp.where(_m95, _r(jnp.minimum(_op('add', qn, qclcs), _op('mul', qclcs, _op('div', _op('mul', tmp1, qclcs), qn)))), qcnsg)
    # WRF:2334
    ncnsg = jnp.where(_m95, _r(_op('div', _op('mul', de, qcnsg), _op('add', qn, qclcs))), ncnsg)
    _m96 = (_m91) & _not(_m95)
    # WRF:2336
    qcnsg = jnp.where(_m96, _r(_r(0.0)), qcnsg)
    # WRF:2337
    ncnsg = jnp.where(_m96, _r(_r(0.0)), ncnsg)
    _m97 = (_m91) & ((qr > epsq) & (qn > epsq) & (tc < -_r(5.0)))
    # WRF:2342
    tmp1 = jnp.where(_m97, _r(_op('sub', vs0, vr0)), tmp1)
    # WRF:2343
    tmp2 = jnp.where(_m97, _r(jnp.sqrt(_op('add', _op('mul', tmp1, tmp1), _op('mul', _op('mul', _r(0.04), vs0), vr0)))), tmp2)
    # WRF:2344
    tmp6 = jnp.where(_m97, _r(_op('mul', _op('mul', ilams2, ilams2), ilams)), tmp6)
    # WRF:2346
    qclrs = jnp.where(_m97, _r(_op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', dt, cmr), ers), piov4), ide), nn), nr), igr31), igs31), tmp2), _op('add', _op('add', _op('mul', _op('mul', gr36, gs31), ilamr5), _op('mul', _op('mul', _op('mul', _op('mul', _r(2.0), gr35), gs32), ilamr4), ilams)), _op('mul', _op('mul', _op('mul', gr34, gs33), ilamr3), ilams2)))), qclrs)
    # WRF:2350
    nclrs = jnp.where(_m97, _r(_op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', dt, _r(0.25)), pi), ers), _op('mul', nn, nr)), igr31), igs31), tmp2), _op('add', _op('add', _op('mul', _op('mul', gr33, gs31), ilamr2), _op('mul', _op('mul', _op('mul', _op('mul', _r(2.0), gr32), gs32), ilamr), ilams)), _op('mul', _op('mul', gr31, gs33), ilams2)))), nclrs)
    _m98 = (_m97) & (snowspherical)
    # WRF:2355
    qclsr = jnp.where(_m98, _r(_op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', dt, cms), ers), piov4), ide), _op('mul', nr, nn)), igs31), igr31), tmp2), _op('add', _op('add', _op('mul', _op('mul', gs36, gr31), tmp6), _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _r(2.0), gs35), gr32), ilams2), ilams2), ilamr)), _op('mul', _op('mul', _op('mul', _op('mul', gs34, gr33), ilams2), ilams), ilamr2)))), qclsr)
    _m99 = (_m97) & _not(_m98)
    # WRF:2360
    qclsr = jnp.where(_m99, _r(_op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', dt, cms), ide), pi), _r(0.25)), ers), tmp2), nn), nr), igs31), igr31), _op('add', _op('add', _op('mul', _op('mul', _op('mul', gr33, gs33), _pow(ilamr, _r(2.0))), _pow(ilams, _r(2.0))), _op('mul', _op('mul', _op('mul', _op('mul', _r(2.0), gr32), gs34), ilamr), _pow(ilams, _r(3.0)))), _op('mul', _op('mul', gr31, gs35), _pow(ilams, _r(4.0)))))), qclsr)
    # WRF:2366
    nclsr = jnp.where(_m97, _r(jnp.minimum(_op('mul', qclsr, _op('mul', nn, iqn)), nclrs)), nclsr)
    # WRF:2367
    qclrs = jnp.where(_m97, _r(jnp.minimum(qclrs, qr)), qclrs)
    # WRF:2367
    qclsr = jnp.where(_m97, _r(jnp.minimum(qclsr, qn)), qclsr)
    # WRF:2368
    nclrs = jnp.where(_m97, _r(jnp.minimum(nclrs, nr)), nclrs)
    # WRF:2368
    nclsr = jnp.where(_m97, _r(jnp.minimum(nclsr, nn)), nclsr)
    # WRF:2371
    dsrs = jnp.where(_m97, _r(_r(0.0)), dsrs)
    # WRF:2371
    dsrg = jnp.where(_m97, _r(_r(0.0)), dsrg)
    # WRF:2371
    dsrh = jnp.where(_m97, _r(_r(0.0)), dsrh)
    # WRF:2372
    tmp1 = jnp.where(_m97, _r(jnp.maximum(ds, dr)), tmp1)
    # WRF:2373
    tmp2 = jnp.where(_m97, _r(_op('mul', _op('mul', tmp1, tmp1), tmp1)), tmp2)
    # WRF:2374
    dey = jnp.where(_m97, _r(_op('div', _op('add', _op('mul', _op('mul', _op('mul', des, ds), ds), ds), _op('mul', _op('mul', _op('mul', dew, dr), dr), dr)), tmp2)), dey)
    _m100 = (_m97) & (dey <= _op('mul', _r(0.5), _op('add', des, deg)))
    # WRF:2375
    dsrs = jnp.where(_m100, _r(_r(1.0)), dsrs)
    _m101 = (_m97) & ((dey > _op('mul', _r(0.5), _op('add', des, deg))) & (dey < _op('mul', _r(0.5), _op('add', deg, deh))))
    # WRF:2376
    dsrg = jnp.where(_m101, _r(_r(1.0)), dsrg)
    _m102 = (_m97) & (dey >= _op('mul', _r(0.5), _op('add', deg, deh)))
    # WRF:2378
    dsrh = jnp.where(_m102, _r(_r(1.0)), dsrh)
    _m103 = (_m102) & (_not(hail_on) | (dr < dr_3cmpthrs))
    # WRF:2380
    dsrg = jnp.where(_m103, _r(_r(1.0)), dsrg)
    # WRF:2380
    dsrh = jnp.where(_m103, _r(_r(0.0)), dsrh)
    _m104 = (_m97) & (_not(grpl_on))
    # WRF:2383
    dsrg = jnp.where(_m104, _r(_r(0.0)), dsrg)
    _m105 = (_m91) & _not(_m97)
    # WRF:2386
    qclrs = jnp.where(_m105, _r(_r(0.0)), qclrs)
    # WRF:2386
    qclsr = jnp.where(_m105, _r(_r(0.0)), qclsr)
    # WRF:2386
    nclrs = jnp.where(_m105, _r(_r(0.0)), nclrs)
    # WRF:2386
    nclsr = jnp.where(_m105, _r(_r(0.0)), nclsr)
    _m106 = (_m59) & _not(_m91)
    # WRF:2391
    qvdvs = jnp.where(_m106, _r(_r(0.0)), qvdvs)
    # WRF:2391
    qclcs = jnp.where(_m106, _r(_r(0.0)), qclcs)
    # WRF:2391
    qcnsg = jnp.where(_m106, _r(_r(0.0)), qcnsg)
    # WRF:2391
    qclsr = jnp.where(_m106, _r(_r(0.0)), qclsr)
    # WRF:2391
    qclrs = jnp.where(_m106, _r(_r(0.0)), qclrs)
    # WRF:2392
    nvdvs = jnp.where(_m106, _r(_r(0.0)), nvdvs)
    # WRF:2392
    nclcs = jnp.where(_m106, _r(_r(0.0)), nclcs)
    # WRF:2392
    nclsr = jnp.where(_m106, _r(_r(0.0)), nclsr)
    # WRF:2392
    nclrs = jnp.where(_m106, _r(_r(0.0)), nclrs)
    # WRF:2392
    ncnsg = jnp.where(_m106, _r(_r(0.0)), ncnsg)
    _m107 = (_m59) & ((qg > epsq) & (ng > epsn))
    _m108 = (_m107) & ((_op('add', qclcg, qclrg) > _r(0.0)) & hail_on)
    # WRF:2405
    tmp1 = jnp.where(_m108, _r(_op('add', _op('mul', _op('mul', _r(11000.0), de), _op('add', qc, qr)), _r(1.0))), tmp1)
    # WRF:2406
    tmp1 = jnp.where(_m108, _r(jnp.maximum(_r(1.0), tmp1)), tmp1)
    # WRF:2407
    d_sll = jnp.where(_m108, _r(_op('mul', _op('mul', _r(2.0), _r(0.01)), _op('sub', _exp(jnp.minimum(_r(20.0), _op('div', -tc, tmp1))), _r(1.0)))), d_sll)
    # WRF:2408
    d_sll = jnp.where(_m108, _r(jnp.minimum(_r(1.0), jnp.maximum(_r(0.0001), d_sll))), d_sll)
    # WRF:2409
    tmp1 = jnp.where(_m108, _r(ilamg), tmp1)
    # WRF:2410
    tmp2 = jnp.where(_m108, _r(no_g), tmp2)
    # WRF:2412
    tmp3 = jnp.where(_m108, _r(_op('add', _op('add', qg, qclcg), qclrg)), tmp3)
    # WRF:2413
    ilamg = jnp.where(_m108, _r(_exp(_op('mul', thrd, _log(_op('div', _op('mul', de, tmp3), _op('mul', _op('mul', ng, 6), cmg)))))), ilamg)
    # WRF:2414
    no_g = jnp.where(_m108, _r(_op('div', ng, ilamg)), no_g)
    # WRF:2415
    tmp4 = jnp.where(_m108, _r(_exp(_op('div', -d_sll, ilamg))), tmp4)
    # WRF:2416
    ng_tail = jnp.where(_m108, _r(_op('mul', _op('mul', no_g, ilamg), tmp4)), ng_tail)
    _m109 = (_m108) & (ng_tail > ngh_crit)
    # WRF:2418
    ngcngh = jnp.where(_m109, _r(jnp.minimum(ng, ng_tail)), ngcngh)
    # WRF:2419
    qcngh = jnp.where(_m109, _r(jnp.minimum(qg, _op('mul', _op('mul', _op('mul', cmg, no_g), tmp4), _op('add', _op('add', _op('add', _op('mul', _pow(d_sll, 3), ilamg), _op('mul', _op('mul', _r(3.0), _pow(d_sll, 2)), _pow(ilamg, 2))), _op('mul', _op('mul', _r(6.0), d_sll), _pow(ilamg, 3))), _op('mul', _r(6.0), _pow(ilamg, 4)))))), qcngh)
    # WRF:2421
    rz = jnp.where(_m109, _r(_r(1.0)), rz)
    # WRF:2425
    nhcngh = jnp.where(_m109, _r(_op('mul', rz, ngcngh)), nhcngh)
    _m110 = (_m108) & _not(_m109)
    # WRF:2427
    qcngh = jnp.where(_m110, _r(_r(0.0)), qcngh)
    # WRF:2428
    ngcngh = jnp.where(_m110, _r(_r(0.0)), ngcngh)
    # WRF:2429
    nhcngh = jnp.where(_m110, _r(_r(0.0)), nhcngh)
    # WRF:2431
    ilamg = jnp.where(_m108, _r(tmp1), ilamg)
    # WRF:2432
    no_g = jnp.where(_m108, _r(tmp2), no_g)
    _m111 = (_m107) & ((qr > epsq) & (tc < -_r(5.0)))
    # WRF:2438
    tmp1 = jnp.where(_m111, _r(_op('sub', vg0, vr0)), tmp1)
    # WRF:2439
    tmp2 = jnp.where(_m111, _r(jnp.sqrt(_op('add', _op('mul', tmp1, tmp1), _op('mul', _op('mul', _r(0.04), vg0), vr0)))), tmp2)
    # WRF:2440
    tmp8 = jnp.where(_m111, _r(_op('mul', ilamg2, ilamg)), tmp8)
    # WRF:2441
    tmp9 = jnp.where(_m111, _r(_op('mul', tmp8, ilamg)), tmp9)
    # WRF:2442
    tmp10 = jnp.where(_m111, _r(_op('mul', tmp9, ilamg)), tmp10)
    # WRF:2445
    kstoke = jnp.where(_m111, _r(_op('div', _op('mul', _op('mul', _op('mul', dew, jnp.abs(_op('sub', vg0, vr0))), dr), dr), _op('mul', _op('mul', _r(9.0), mudyn), dg))), kstoke)
    # WRF:2446
    kstoke = jnp.where(_m111, _r(jnp.maximum(_r(1.5), jnp.minimum(_r(10.0), kstoke))), kstoke)
    # WRF:2447
    erg = jnp.where(_m111, _r(_op('mul', _r(0.55), _log10(_op('mul', _r(2.51), kstoke)))), erg)
    # WRF:2449
    qclrg = jnp.where(_m111, _r(_op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', dt, cmr), erg), piov4), ide), _op('mul', ng, nr)), igr31), igg31), tmp2), _op('add', _op('add', _op('mul', _op('mul', gr36, gg31), ilamr5), _op('mul', _op('mul', _op('mul', _op('mul', _r(2.0), gr35), gg32), ilamr4), ilamg)), _op('mul', _op('mul', _op('mul', gr34, gg33), ilamr3), ilamg2)))), qclrg)
    # WRF:2453
    nclrg = jnp.where(_m111, _r(_op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', dt, piov4), erg), _op('mul', ng, nr)), igr31), igg31), tmp2), _op('add', _op('add', _op('mul', _op('mul', gr33, gg31), ilamr2), _op('mul', _op('mul', _op('mul', _op('mul', _r(2.0), gr32), gg32), ilamr), ilamg)), _op('mul', _op('mul', gr31, gg33), ilamg2)))), nclrg)
    # WRF:2456
    qclgr = jnp.where(_m111, _r(_op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', _op('mul', dt, cmg), erg), piov4), ide), _op('mul', nr, ng)), igg31), igr31), tmp2), _op('add', _op('add', _op('mul', _op('mul', gg36, gr31), tmp10), _op('mul', _op('mul', _op('mul', _op('mul', _r(2.0), gg35), gr32), tmp9), ilamr)), _op('mul', _op('mul', _op('mul', gg34, gr33), tmp8), ilamr2)))), qclgr)
    # WRF:2460
    nclgr = jnp.where(_m111, _r(jnp.minimum(nclrg, _op('mul', qclgr, _op('mul', ng, iqg)))), nclgr)
    # WRF:2461
    qclrg = jnp.where(_m111, _r(jnp.minimum(qclrg, qr)), qclrg)
    # WRF:2461
    qclgr = jnp.where(_m111, _r(jnp.minimum(qclgr, qg)), qclgr)
    # WRF:2462
    nclrg = jnp.where(_m111, _r(jnp.minimum(nclrg, nr)), nclrg)
    # WRF:2462
    nclgr = jnp.where(_m111, _r(jnp.minimum(nclgr, ng)), nclgr)
    # WRF:2465
    tmp1 = jnp.where(_m111, _r(jnp.maximum(dg, dr)), tmp1)
    # WRF:2466
    tmp2 = jnp.where(_m111, _r(_op('mul', _op('mul', tmp1, tmp1), tmp1)), tmp2)
    # WRF:2467
    dey = jnp.where(_m111, _r(_op('div', _op('add', _op('mul', _op('mul', _op('mul', deg, dg), dg), dg), _op('mul', _op('mul', _op('mul', dew, dr), dr), dr)), tmp2)), dey)
    _m112 = (_m111) & ((dey > _op('mul', _r(0.5), _op('add', deg, deh))) & (dr > dr_3cmpthrs) & hail_on)
    # WRF:2469
    dgrg = jnp.where(_m112, _r(_r(0.0)), dgrg)
    # WRF:2469
    dgrh = jnp.where(_m112, _r(_r(1.0)), dgrh)
    _m113 = (_m111) & _not(_m112)
    # WRF:2471
    dgrg = jnp.where(_m113, _r(_r(1.0)), dgrg)
    # WRF:2471
    dgrh = jnp.where(_m113, _r(_r(0.0)), dgrh)
    _m114 = (_m107) & _not(_m111)
    # WRF:2474
    qclgr = jnp.where(_m114, _r(_r(0.0)), qclgr)
    # WRF:2474
    qclrg = jnp.where(_m114, _r(_r(0.0)), qclrg)
    # WRF:2474
    nclgr = jnp.where(_m114, _r(_r(0.0)), nclgr)
    # WRF:2474
    nclrg = jnp.where(_m114, _r(_r(0.0)), nclrg)
    _m115 = (_m59) & _not(_m107)
    # WRF:2479
    qcngh = jnp.where(_m115, _r(_r(0.0)), qcngh)
    # WRF:2479
    qclgr = jnp.where(_m115, _r(_r(0.0)), qclgr)
    # WRF:2479
    qclrg = jnp.where(_m115, _r(_r(0.0)), qclrg)
    # WRF:2479
    ngcngh = jnp.where(_m115, _r(_r(0.0)), ngcngh)
    # WRF:2480
    nhcngh = jnp.where(_m115, _r(_r(0.0)), nhcngh)
    # WRF:2480
    nclgr = jnp.where(_m115, _r(_r(0.0)), nclgr)
    # WRF:2480
    nclrg = jnp.where(_m115, _r(_r(0.0)), nclrg)
    _m116 = (_m59) & (qh > epsq)
    _m117 = (_m116) & ((qhwet < _op('add', _op('add', _op('add', qclch, qclrh), qclih), qclsh)) & (tc > -_r(40.0)))
    # WRF:2490
    qclih = jnp.where(_m117, _r(jnp.minimum(_op('mul', qclih, ieih), qi)), qclih)
    # WRF:2491
    nclih = jnp.where(_m117, _r(jnp.minimum(_op('mul', nclih, ieih), ny)), nclih)
    # WRF:2492
    qclsh = jnp.where(_m117, _r(jnp.minimum(_op('mul', qclsh, iesh), qn)), qclsh)
    # WRF:2493
    nclsh = jnp.where(_m117, _r(jnp.minimum(_op('mul', nclsh, iesh), nn)), nclsh)
    # WRF:2494
    tmp3 = jnp.where(_m117, _r(qclrh), tmp3)
    # WRF:2495
    qclrh = jnp.where(_m117, _r(_op('sub', qhwet, _op('add', _op('add', qclch, qclih), qclsh))), qclrh)
    # WRF:2496
    qshhr = jnp.where(_m117, _r(_op('sub', tmp3, qclrh)), qshhr)
    # WRF:2497
    nshhr = jnp.where(_m117, _r(_op('div', _op('mul', de, qshhr), _op('mul', _op('mul', _op('mul', cmr, drshed), drshed), drshed))), nshhr)
    _m118 = (_m116) & _not(_m117)
    # WRF:2499
    nshhr = jnp.where(_m118, _r(_r(0.0)), nshhr)
    _m119 = (_m59) & _not(_m116)
    # WRF:2502
    nshhr = jnp.where(_m119, _r(_r(0.0)), nshhr)
    # WRF:2514
    tmp1 = jnp.where(_m2, _r(_op('add', _op('sub', _op('sub', _op('sub', _op('sub', _op('sub', _op('sub', _op('add', nn, nscnis), nvdvs), ncnsg), nmlsr), nclss), nclsr), nclsh), nclsrs)), tmp1)
    _m120 = (_m2) & (tmp1 < epsn)
    # WRF:2516
    qclcs = jnp.where(_m120, _r(_r(0.0)), qclcs)
    # WRF:2517
    nclcs = jnp.where(_m120, _r(_r(0.0)), nclcs)
    # WRF:2518
    qclrs = jnp.where(_m120, _r(_r(0.0)), qclrs)
    # WRF:2519
    nclrs = jnp.where(_m120, _r(_r(0.0)), nclrs)
    _m121 = (_m120) & (dsrs == _r(1.0))
    # WRF:2521
    qclrs = jnp.where(_m121, _r(_r(0.0)), qclrs)
    # WRF:2522
    qclsr = jnp.where(_m121, _r(_r(0.0)), qclsr)
    # WRF:2526
    tmp2 = jnp.where(_m2, _r(_op('sub', _op('add', _op('add', _op('add', _op('sub', _op('sub', _op('sub', _op('add', ng, ncnsg), nclgr), nvdvg), nmlgr), nclirg), nclsrg), nclgrg), ngcngh)), tmp2)
    _m122 = (_m2) & (tmp2 < epsn)
    # WRF:2528
    qclcg = jnp.where(_m122, _r(_r(0.0)), qclcg)
    # WRF:2529
    nclcg = jnp.where(_m122, _r(_r(0.0)), nclcg)
    # WRF:2530
    qclrg = jnp.where(_m122, _r(_r(0.0)), qclrg)
    # WRF:2531
    nclrg = jnp.where(_m122, _r(_r(0.0)), nclrg)
    _m123 = (_m122) & (dirg == _r(1.0))
    # WRF:2533
    qclri = jnp.where(_m123, _r(_r(0.0)), qclri)
    # WRF:2534
    qclir = jnp.where(_m123, _r(_r(0.0)), qclir)
    _m124 = (_m122) & (dsrg == _r(1.0))
    # WRF:2537
    qclrs = jnp.where(_m124, _r(_r(0.0)), qclrs)
    # WRF:2538
    qclsr = jnp.where(_m124, _r(_r(0.0)), qclsr)
    # WRF:2542
    tmp3 = jnp.where(_m2, _r(_op('add', _op('add', _op('add', _op('sub', _op('sub', _op('add', _op('add', nh, nhfzrh), nhcngh), nmlhr), nvdvh), nclirh), nclsrh), nclgrh)), tmp3)
    _m125 = (_m2) & (tmp3 < epsn)
    # WRF:2544
    qclch = jnp.where(_m125, _r(_r(0.0)), qclch)
    # WRF:2545
    nclch = jnp.where(_m125, _r(_r(0.0)), nclch)
    # WRF:2546
    qclrh = jnp.where(_m125, _r(_r(0.0)), qclrh)
    # WRF:2547
    nclrh = jnp.where(_m125, _r(_r(0.0)), nclrh)
    _m126 = (_m125) & (dirh == _r(1.0))
    # WRF:2549
    qclri = jnp.where(_m126, _r(_r(0.0)), qclri)
    # WRF:2550
    qclir = jnp.where(_m126, _r(_r(0.0)), qclir)
    _m127 = (_m125) & (dsrh == _r(1.0))
    # WRF:2553
    qclrs = jnp.where(_m127, _r(_r(0.0)), qclrs)
    # WRF:2554
    qclsr = jnp.where(_m127, _r(_r(0.0)), qclsr)
    for _depletion_pass in range(2):
        # WRF:2565
        source = jnp.where(_m2, _r(_op('add', _op('add', _op('add', _op('add', q, _dim(-qvdvi, _r(0.0))), _dim(-qvdvs, _r(0.0))), _dim(-qvdvg, _r(0.0))), _dim(-qvdvh, _r(0.0)))), source)
        # WRF:2566
        sink = jnp.where(_m2, _r(_op('add', _op('add', qnuvi, _dim(qvdvi, _r(0.0))), _dim(qvdvs, _r(0.0)))), sink)
        # WRF:2567
        sour = jnp.where(_m2, _r(jnp.maximum(source, _r(0.0))), sour)
        _m128 = (_m2) & (sink > sour)
        # WRF:2569
        ratio = jnp.where(_m128, _r(_op('div', sour, sink)), ratio)
        # WRF:2570
        qnuvi = jnp.where(_m128, _r(_op('mul', ratio, qnuvi)), qnuvi)
        # WRF:2570
        nnuvi = jnp.where(_m128, _r(_op('mul', ratio, nnuvi)), nnuvi)
        _m129 = (_m128) & (qvdvi > _r(0.0))
        # WRF:2572
        qvdvi = jnp.where(_m129, _r(_op('mul', ratio, qvdvi)), qvdvi)
        # WRF:2572
        nvdvi = jnp.where(_m129, _r(_op('mul', ratio, nvdvi)), nvdvi)
        _m130 = (_m128) & (qvdvs > _r(0.0))
        # WRF:2575
        qvdvs = jnp.where(_m130, _r(_op('mul', ratio, qvdvs)), qvdvs)
        # WRF:2575
        nvdvs = jnp.where(_m130, _r(_op('mul', ratio, nvdvs)), nvdvs)
        # WRF:2577
        qvdvg = jnp.where(_m128, _r(_op('mul', ratio, qvdvg)), qvdvg)
        # WRF:2577
        nvdvg = jnp.where(_m128, _r(_op('mul', ratio, nvdvg)), nvdvg)
        # WRF:2578
        qvdvh = jnp.where(_m128, _r(_op('mul', ratio, qvdvh)), qvdvh)
        # WRF:2578
        nvdvh = jnp.where(_m128, _r(_op('mul', ratio, nvdvh)), nvdvh)
        # WRF:2582
        source = jnp.where(_m2, _r(qc), source)
        # WRF:2583
        sink = jnp.where(_m2, _r(_op('add', _op('add', _op('add', qclcs, qclcg), qclch), qfzci)), sink)
        # WRF:2584
        sour = jnp.where(_m2, _r(jnp.maximum(source, _r(0.0))), sour)
        _m131 = (_m2) & (sink > sour)
        # WRF:2586
        ratio = jnp.where(_m131, _r(_op('div', sour, sink)), ratio)
        # WRF:2587
        qfzci = jnp.where(_m131, _r(_op('mul', ratio, qfzci)), qfzci)
        # WRF:2587
        nfzci = jnp.where(_m131, _r(_op('mul', ratio, nfzci)), nfzci)
        # WRF:2588
        qclcs = jnp.where(_m131, _r(_op('mul', ratio, qclcs)), qclcs)
        # WRF:2588
        nclcs = jnp.where(_m131, _r(_op('mul', ratio, nclcs)), nclcs)
        # WRF:2589
        qclcg = jnp.where(_m131, _r(_op('mul', ratio, qclcg)), qclcg)
        # WRF:2589
        nclcg = jnp.where(_m131, _r(_op('mul', ratio, nclcg)), nclcg)
        # WRF:2590
        qclch = jnp.where(_m131, _r(_op('mul', ratio, qclch)), qclch)
        # WRF:2590
        nclch = jnp.where(_m131, _r(_op('mul', ratio, nclch)), nclch)
        # WRF:2594
        source = jnp.where(_m2, _r(_op('add', _op('add', _op('add', _op('add', qr, qmlsr), qmlgr), qmlhr), qmlir)), source)
        # WRF:2595
        sink = jnp.where(_m2, _r(_op('add', _op('add', _op('add', _op('add', qclri, qclrs), qclrg), qclrh), qfzrh)), sink)
        # WRF:2596
        sour = jnp.where(_m2, _r(jnp.maximum(source, _r(0.0))), sour)
        _m132 = (_m2) & (sink > sour)
        # WRF:2598
        ratio = jnp.where(_m132, _r(_op('div', sour, sink)), ratio)
        # WRF:2599
        qclrg = jnp.where(_m132, _r(_op('mul', ratio, qclrg)), qclrg)
        # WRF:2599
        qclri = jnp.where(_m132, _r(_op('mul', ratio, qclri)), qclri)
        # WRF:2599
        nclri = jnp.where(_m132, _r(_op('mul', ratio, nclri)), nclri)
        # WRF:2600
        qclrs = jnp.where(_m132, _r(_op('mul', ratio, qclrs)), qclrs)
        # WRF:2600
        nclrs = jnp.where(_m132, _r(_op('mul', ratio, nclrs)), nclrs)
        # WRF:2601
        nclrg = jnp.where(_m132, _r(_op('mul', ratio, nclrg)), nclrg)
        # WRF:2601
        qclrh = jnp.where(_m132, _r(_op('mul', ratio, qclrh)), qclrh)
        # WRF:2601
        nclrh = jnp.where(_m132, _r(_op('mul', ratio, nclrh)), nclrh)
        # WRF:2602
        qfzrh = jnp.where(_m132, _r(_op('mul', ratio, qfzrh)), qfzrh)
        # WRF:2602
        nrfzrh = jnp.where(_m132, _r(_op('mul', ratio, nrfzrh)), nrfzrh)
        # WRF:2602
        nhfzrh = jnp.where(_m132, _r(_op('mul', ratio, nhfzrh)), nhfzrh)
        _m133 = (_m132) & (ratio == _r(0.0))
        # WRF:2604
        dirg = jnp.where(_m133, _r(_r(0.0)), dirg)
        # WRF:2604
        dirh = jnp.where(_m133, _r(_r(0.0)), dirh)
        # WRF:2604
        dgrg = jnp.where(_m133, _r(_r(0.0)), dgrg)
        # WRF:2604
        dgrh = jnp.where(_m133, _r(_r(0.0)), dgrh)
        # WRF:2605
        dsrs = jnp.where(_m133, _r(_r(0.0)), dsrs)
        # WRF:2605
        dsrg = jnp.where(_m133, _r(_r(0.0)), dsrg)
        # WRF:2605
        dsrh = jnp.where(_m133, _r(_r(0.0)), dsrh)
        # WRF:2610
        source = jnp.where(_m2, _r(_op('add', _op('add', _op('add', qi, qnuvi), _dim(qvdvi, _r(0.0))), qfzci)), source)
        # WRF:2611
        sink = jnp.where(_m2, _r(_op('add', _op('add', _op('add', _op('add', _op('add', _op('add', qcnis, qclir), _dim(-qvdvi, _r(0.0))), qclis), qclig), qclih), qmlir)), sink)
        # WRF:2612
        sour = jnp.where(_m2, _r(jnp.maximum(source, _r(0.0))), sour)
        _m134 = (_m2) & (sink > sour)
        # WRF:2614
        ratio = jnp.where(_m134, _r(_op('div', sour, sink)), ratio)
        # WRF:2615
        qmlir = jnp.where(_m134, _r(_op('mul', ratio, qmlir)), qmlir)
        # WRF:2615
        nmlir = jnp.where(_m134, _r(_op('mul', ratio, nmlir)), nmlir)
        _m135 = (_m134) & (qvdvi < _r(0.0))
        # WRF:2617
        qvdvi = jnp.where(_m135, _r(_op('mul', ratio, qvdvi)), qvdvi)
        # WRF:2617
        nvdvi = jnp.where(_m135, _r(_op('mul', ratio, nvdvi)), nvdvi)
        # WRF:2619
        qcnis = jnp.where(_m134, _r(_op('mul', ratio, qcnis)), qcnis)
        # WRF:2619
        nicnis = jnp.where(_m134, _r(_op('mul', ratio, nicnis)), nicnis)
        # WRF:2619
        nscnis = jnp.where(_m134, _r(_op('mul', ratio, nscnis)), nscnis)
        # WRF:2620
        qclir = jnp.where(_m134, _r(_op('mul', ratio, qclir)), qclir)
        # WRF:2620
        nclir = jnp.where(_m134, _r(_op('mul', ratio, nclir)), nclir)
        # WRF:2620
        qclig = jnp.where(_m134, _r(_op('mul', ratio, qclig)), qclig)
        # WRF:2621
        qclis = jnp.where(_m134, _r(_op('mul', ratio, qclis)), qclis)
        # WRF:2621
        nclis = jnp.where(_m134, _r(_op('mul', ratio, nclis)), nclis)
        # WRF:2622
        qclih = jnp.where(_m134, _r(_op('mul', ratio, qclih)), qclih)
        # WRF:2622
        nclih = jnp.where(_m134, _r(_op('mul', ratio, nclih)), nclih)
        _m136 = (_m134) & (ratio == _r(0.0))
        # WRF:2624
        dirg = jnp.where(_m136, _r(_r(0.0)), dirg)
        # WRF:2624
        dirh = jnp.where(_m136, _r(_r(0.0)), dirh)
        # WRF:2629
        source = jnp.where(_m2, _r(_op('add', _op('add', _op('add', _op('add', _op('add', qn, qcnis), _dim(qvdvs, _r(0.0))), qclis), _op('mul', dsrs, _op('add', qclrs, qclsr))), qclcs)), source)
        # WRF:2630
        sink = jnp.where(_m2, _r(_op('add', _op('add', _op('add', _op('add', _dim(-qvdvs, _r(0.0)), qcnsg), qmlsr), qclsr), qclsh)), sink)
        # WRF:2631
        sour = jnp.where(_m2, _r(jnp.maximum(source, _r(0.0))), sour)
        _m137 = (_m2) & (sink > sour)
        # WRF:2633
        ratio = jnp.where(_m137, _r(_op('div', sour, sink)), ratio)
        _m138 = (_m137) & (qvdvs <= _r(0.0))
        # WRF:2635
        qvdvs = jnp.where(_m138, _r(_op('mul', ratio, qvdvs)), qvdvs)
        # WRF:2635
        nvdvs = jnp.where(_m138, _r(_op('mul', ratio, nvdvs)), nvdvs)
        # WRF:2637
        qcnsg = jnp.where(_m137, _r(_op('mul', ratio, qcnsg)), qcnsg)
        # WRF:2637
        ncnsg = jnp.where(_m137, _r(_op('mul', ratio, ncnsg)), ncnsg)
        # WRF:2637
        qmlsr = jnp.where(_m137, _r(_op('mul', ratio, qmlsr)), qmlsr)
        # WRF:2638
        nmlsr = jnp.where(_m137, _r(_op('mul', ratio, nmlsr)), nmlsr)
        # WRF:2638
        qclsr = jnp.where(_m137, _r(_op('mul', ratio, qclsr)), qclsr)
        # WRF:2638
        nclsr = jnp.where(_m137, _r(_op('mul', ratio, nclsr)), nclsr)
        # WRF:2639
        qclsh = jnp.where(_m137, _r(_op('mul', ratio, qclsh)), qclsh)
        # WRF:2639
        nclsh = jnp.where(_m137, _r(_op('mul', ratio, nclsh)), nclsh)
        _m139 = (_m137) & (ratio == _r(0.0))
        # WRF:2641
        dsrs = jnp.where(_m139, _r(_r(0.0)), dsrs)
        # WRF:2641
        dsrg = jnp.where(_m139, _r(_r(0.0)), dsrg)
        # WRF:2641
        dsrh = jnp.where(_m139, _r(_r(0.0)), dsrh)
        # WRF:2646
        source = jnp.where(_m2, _r(_op('add', _op('add', _op('add', _op('add', _op('add', _op('add', _op('add', qg, qcnsg), _dim(qvdvg, _r(0.0))), _op('mul', dirg, _op('add', qclri, qclir))), _op('mul', dgrg, _op('add', qclrg, qclgr))), qclcg), _op('mul', dsrg, _op('add', qclrs, qclsr))), qclig)), source)
        # WRF:2648
        sink = jnp.where(_m2, _r(_op('add', _op('add', _op('add', _dim(-qvdvg, _r(0.0)), qmlgr), qcngh), qclgr)), sink)
        # WRF:2649
        sour = jnp.where(_m2, _r(jnp.maximum(source, _r(0.0))), sour)
        _m140 = (_m2) & (sink > sour)
        # WRF:2651
        ratio = jnp.where(_m140, _r(_op('div', sour, sink)), ratio)
        # WRF:2652
        qvdvg = jnp.where(_m140, _r(_op('mul', ratio, qvdvg)), qvdvg)
        # WRF:2652
        nvdvg = jnp.where(_m140, _r(_op('mul', ratio, nvdvg)), nvdvg)
        # WRF:2652
        qmlgr = jnp.where(_m140, _r(_op('mul', ratio, qmlgr)), qmlgr)
        # WRF:2653
        nmlgr = jnp.where(_m140, _r(_op('mul', ratio, nmlgr)), nmlgr)
        # WRF:2653
        qcngh = jnp.where(_m140, _r(_op('mul', ratio, qcngh)), qcngh)
        # WRF:2653
        ngcngh = jnp.where(_m140, _r(_op('mul', ratio, ngcngh)), ngcngh)
        # WRF:2654
        qclgr = jnp.where(_m140, _r(_op('mul', ratio, qclgr)), qclgr)
        # WRF:2654
        nclgr = jnp.where(_m140, _r(_op('mul', ratio, nclgr)), nclgr)
        # WRF:2654
        nhcngh = jnp.where(_m140, _r(_op('mul', ratio, nhcngh)), nhcngh)
        _m141 = (_m140) & (ratio == _r(0.0))
        # WRF:2656
        dgrg = jnp.where(_m141, _r(_r(0.0)), dgrg)
        # WRF:2656
        dgrh = jnp.where(_m141, _r(_r(0.0)), dgrh)
        # WRF:2661
        source = jnp.where(_m2, _r(_op('add', _op('add', _op('add', _op('add', _op('add', _op('add', _op('add', _op('add', _op('add', _op('add', qh, _dim(qvdvh, _r(0.0))), qclch), qclrh), _op('mul', dirh, _op('add', qclri, qclir))), qclih), qclsh), _op('mul', dsrh, _op('add', qclrs, qclsr))), qcngh), _op('mul', dgrh, _op('add', qclrg, qclgr))), qfzrh)), source)
        # WRF:2663
        sink = jnp.where(_m2, _r(_op('add', _dim(-qvdvh, _r(0.0)), qmlhr)), sink)
        # WRF:2664
        sour = jnp.where(_m2, _r(jnp.maximum(source, _r(0.0))), sour)
        _m142 = (_m2) & (sink > sour)
        # WRF:2666
        ratio = jnp.where(_m142, _r(_op('div', sour, sink)), ratio)
        # WRF:2667
        qvdvh = jnp.where(_m142, _r(_op('mul', ratio, qvdvh)), qvdvh)
        # WRF:2667
        nvdvh = jnp.where(_m142, _r(_op('mul', ratio, nvdvh)), nvdvh)
        # WRF:2668
        qmlhr = jnp.where(_m142, _r(_op('mul', ratio, qmlhr)), qmlhr)
        # WRF:2668
        nmlhr = jnp.where(_m142, _r(_op('mul', ratio, nmlhr)), nmlhr)
    # WRF:2675
    nclirg = jnp.where(_m2, _r(_r(0.0)), nclirg)
    # WRF:2675
    nclirh = jnp.where(_m2, _r(_r(0.0)), nclirh)
    # WRF:2675
    nclsrs = jnp.where(_m2, _r(_r(0.0)), nclsrs)
    # WRF:2675
    nclsrg = jnp.where(_m2, _r(_r(0.0)), nclsrg)
    # WRF:2676
    nclsrh = jnp.where(_m2, _r(_r(0.0)), nclsrh)
    # WRF:2676
    nclgrg = jnp.where(_m2, _r(_r(0.0)), nclgrg)
    # WRF:2676
    nclgrh = jnp.where(_m2, _r(_r(0.0)), nclgrh)
    _m143 = (_m2) & (_op('add', qclir, qclri) > _r(0.0))
    # WRF:2679
    tmp1 = jnp.where(_m143, _r(jnp.maximum(dr, di)), tmp1)
    # WRF:2680
    tmp2 = jnp.where(_m143, _r(_op('mul', _op('mul', _op('mul', tmp1, tmp1), tmp1), piov6)), tmp2)
    # WRF:2681
    nclirg = jnp.where(_m143, _r(_op('div', _op('mul', _op('mul', dirg, de), _op('add', qclir, qclri)), _op('mul', deg, tmp2))), nclirg)
    # WRF:2682
    nclirh = jnp.where(_m143, _r(_op('div', _op('mul', _op('mul', dirh, de), _op('add', qclir, qclri)), _op('mul', deh, tmp2))), nclirh)
    _m144 = (_m2) & (_op('add', qclsr, qclrs) > _r(0.0))
    # WRF:2686
    tmp1 = jnp.where(_m144, _r(jnp.maximum(dr, ds)), tmp1)
    # WRF:2687
    tmp2 = jnp.where(_m144, _r(_op('mul', _op('mul', _op('mul', tmp1, tmp1), tmp1), piov6)), tmp2)
    # WRF:2688
    nclsrs = jnp.where(_m144, _r(_op('div', _op('mul', _op('mul', dsrs, de), _op('add', qclsr, qclrs)), _op('mul', des, tmp2))), nclsrs)
    # WRF:2689
    nclsrg = jnp.where(_m144, _r(_op('div', _op('mul', _op('mul', dsrg, de), _op('add', qclsr, qclrs)), _op('mul', deg, tmp2))), nclsrg)
    # WRF:2690
    nclsrh = jnp.where(_m144, _r(_op('div', _op('mul', _op('mul', dsrh, de), _op('add', qclsr, qclrs)), _op('mul', deh, tmp2))), nclsrh)
    _m145 = (_m2) & (_op('add', qclgr, qclrg) > _r(0.0))
    # WRF:2694
    tmp1 = jnp.where(_m145, _r(jnp.maximum(dr, dg)), tmp1)
    # WRF:2695
    tmp2 = jnp.where(_m145, _r(_op('mul', _op('mul', _op('mul', tmp1, tmp1), tmp1), piov6)), tmp2)
    # WRF:2696
    nclgrg = jnp.where(_m145, _r(_op('div', _op('mul', _op('mul', dgrg, de), _op('add', qclgr, qclrg)), _op('mul', deg, tmp2))), nclgrg)
    # WRF:2697
    nclgrh = jnp.where(_m145, _r(_op('div', _op('mul', _op('mul', dgrh, de), _op('add', qclgr, qclrg)), _op('mul', deh, tmp2))), nclgrh)
    # WRF:2708
    q = jnp.where(_m2, _r(_op('sub', _op('sub', _op('sub', _op('sub', _op('sub', q, qnuvi), qvdvi), qvdvs), qvdvg), qvdvh)), q)
    # WRF:2709
    qc = jnp.where(_m2, _r(_op('sub', _op('sub', _op('sub', _op('sub', qc, qclcs), qclcg), qclch), qfzci)), qc)
    # WRF:2710
    qr = jnp.where(_m2, _r(_op('add', _op('sub', _op('add', _op('sub', _op('add', _op('sub', _op('sub', _op('add', _op('sub', qr, qclri), qmlsr), qclrs), qclrg), qmlgr), qclrh), qmlhr), qfzrh), qmlir)), qr)
    # WRF:2711
    qi = jnp.where(_m2, _r(_op('add', _op('add', _op('sub', _op('sub', _op('sub', _op('sub', _op('sub', _op('sub', _op('add', _op('add', _op('add', qi, qnuvi), qvdvi), qfzci), qcnis), qclir), qclis), qclig), qmlir), qclih), qimsi), qimgi)), qi)
    # WRF:2713
    qg = jnp.where(_m2, _r(_op('add', _op('add', _op('add', _op('add', _op('sub', _op('sub', _op('sub', _op('sub', _op('add', _op('add', _op('add', qg, qcnsg), qvdvg), qclcg), qclgr), qmlgr), qcngh), qimgi), qclig), _op('mul', dirg, _op('add', qclri, qclir))), _op('mul', dgrg, _op('add', qclrg, qclgr))), _op('mul', dsrg, _op('add', qclrs, qclsr)))), qg)
    # WRF:2715
    qn = jnp.where(_m2, _r(_op('add', _op('sub', _op('add', _op('sub', _op('sub', _op('sub', _op('sub', _op('add', _op('add', _op('add', qn, qcnis), qvdvs), qclcs), qcnsg), qmlsr), qimsi), qclsr), qclis), qclsh), _op('mul', dsrs, _op('add', qclrs, qclsr)))), qn)
    # WRF:2717
    qh = jnp.where(_m2, _r(_op('add', _op('add', _op('add', _op('add', _op('add', _op('add', _op('add', _op('add', _op('add', _op('sub', _op('add', qh, _op('mul', dirh, _op('add', qclri, qclir))), qmlhr), qvdvh), qclch), _op('mul', dsrh, _op('add', qclrs, qclsr))), qclih), qclsh), qfzrh), qclrh), qcngh), _op('mul', dgrh, _op('add', qclrg, qclgr)))), qh)
    # WRF:2721
    nc = jnp.where(_m2, _r(_op('sub', _op('sub', _op('sub', _op('sub', nc, nclcs), nclcg), nclch), nfzci)), nc)
    # WRF:2722
    nr = jnp.where(_m2, _r(_op('add', _op('add', _op('sub', _op('add', _op('add', _op('add', _op('sub', _op('sub', _op('sub', _op('sub', nr, nclri), nclrs), nclrg), nclrh), nmlsr), nmlgr), nmlhr), nrfzrh), nmlir), nshhr)), nr)
    # WRF:2724
    ny = jnp.where(_m2, _r(_op('sub', _op('add', _op('add', _op('sub', _op('sub', _op('sub', _op('sub', _op('sub', _op('add', _op('add', _op('add', ny, nnuvi), nvdvi), nfzci), nclir), nclis), nclig), nclih), nmlir), nimsi), nimgi), nicnis)), ny)
    # WRF:2726
    nn = jnp.where(_m2, _r(_op('add', _op('sub', _op('sub', _op('sub', _op('sub', _op('sub', _op('sub', _op('add', nn, nscnis), nvdvs), ncnsg), nmlsr), nclss), nclsr), nclsh), nclsrs)), nn)
    # WRF:2727
    ng = jnp.where(_m2, _r(_op('sub', _op('add', _op('add', _op('add', _op('sub', _op('sub', _op('sub', _op('add', ng, ncnsg), nclgr), nvdvg), nmlgr), nclirg), nclsrg), nclgrg), ngcngh)), ng)
    # WRF:2728
    nh = jnp.where(_m2, _r(_op('add', _op('add', _op('add', _op('sub', _op('sub', _op('add', _op('add', nh, nhfzrh), nhcngh), nmlhr), nvdvh), nclirh), nclsrh), nclgrh)), nh)
    # WRF:2730
    t = jnp.where(_m2, _r(_op('add', _op('add', t, _op('mul', lfp, _op('add', _op('add', _op('add', _op('sub', _op('sub', _op('sub', _op('add', _op('add', _op('sub', _op('add', _op('add', _op('add', qclri, qclcs), qclrs), qfzci), qmlsr), qclcg), qclrg), qmlir), qmlgr), qmlhr), qclch), qclrh), qfzrh))), _op('mul', lsp, _op('add', _op('add', _op('add', _op('add', qnuvi, qvdvi), qvdvs), qvdvg), qvdvh)))), t)
    # WRF:2734
    tmp1 = jnp.where(_m2, _r(_op('div', pres, _op('mul', rgasd, t))), tmp1)
    _m146 = (_m2) & ((qc > epsq) & (nc < epsn))
    # WRF:2740
    nc = jnp.where(_m146, _r(n_c_sm), nc)
    _m147 = (_m2) & _not(_m146) & (qc <= epsq)
    # WRF:2742
    q = jnp.where(_m147, _r(_op('add', q, qc)), q)
    # WRF:2743
    t = jnp.where(_m147, _r(_op('sub', t, _op('mul', lcp, qc))), t)
    # WRF:2744
    qc = jnp.where(_m147, _r(_r(0.0)), qc)
    # WRF:2745
    nc = jnp.where(_m147, _r(_r(0.0)), nc)
    _m148 = (_m2) & ((qr > epsq) & (nr < epsn))
    # WRF:2750
    nr = jnp.where(_m148, _r(_op('mul', _pow(_op('mul', no_r_sm, gr31), _op('div', _r(3.0), _op('add', _r(4.0), alpha_r))), _pow(_op('mul', _op('mul', _op('mul', _op('mul', gr31, igr34), tmp1), qr), icmr), _op('div', _op('add', _r(1.0), alpha_r), _op('add', _r(4.0), alpha_r))))), nr)
    _m149 = (_m2) & _not(_m148) & (qr <= epsq)
    # WRF:2753
    q = jnp.where(_m149, _r(_op('add', q, qr)), q)
    # WRF:2754
    t = jnp.where(_m149, _r(_op('sub', t, _op('mul', lcp, qr))), t)
    # WRF:2755
    qr = jnp.where(_m149, _r(_r(0.0)), qr)
    # WRF:2756
    nr = jnp.where(_m149, _r(_r(0.0)), nr)
    _m150 = (_m2) & ((qi > epsq) & (ny < epsn))
    # WRF:2762
    ny = jnp.where(_m150, _r(jnp.maximum(_op('mul', _r(2.0), epsn), _n_cooper(trpl, t))), ny)
    _m151 = (_m2) & _not(_m150) & (qi <= epsq)
    # WRF:2764
    q = jnp.where(_m151, _r(_op('add', q, qi)), q)
    # WRF:2765
    t = jnp.where(_m151, _r(_op('sub', t, _op('mul', lsp, qi))), t)
    # WRF:2766
    qi = jnp.where(_m151, _r(_r(0.0)), qi)
    # WRF:2767
    ny = jnp.where(_m151, _r(_r(0.0)), ny)
    _m152 = (_m2) & ((qn > epsq) & (nn < epsn))
    # WRF:2772
    no_s = jnp.where(_m152, _r(_nos_thompson(trpl, t)), no_s)
    # WRF:2773
    nn = jnp.where(_m152, _r(_op('mul', _pow(_op('mul', no_s, gs31), _op('mul', dms, icexs2)), _pow(_op('mul', _op('mul', _op('mul', _op('mul', gs31, igs40), icms), tmp1), qn), _op('mul', _op('add', _r(1.0), alpha_s), icexs2)))), nn)
    _m153 = (_m2) & _not(_m152) & (qn <= epsq)
    # WRF:2776
    q = jnp.where(_m153, _r(_op('add', q, qn)), q)
    # WRF:2777
    t = jnp.where(_m153, _r(_op('sub', t, _op('mul', lsp, qn))), t)
    # WRF:2778
    qn = jnp.where(_m153, _r(_r(0.0)), qn)
    # WRF:2779
    nn = jnp.where(_m153, _r(_r(0.0)), nn)
    _m154 = (_m2) & ((qg > epsq) & (ng < epsn))
    # WRF:2784
    ng = jnp.where(_m154, _r(_op('mul', _pow(_op('mul', no_g_sm, gg31), _op('div', _r(3.0), _op('add', _r(4.0), alpha_g))), _pow(_op('mul', _op('mul', _op('mul', _op('mul', gg31, igg34), tmp1), qg), icmg), _op('div', _op('add', _r(1.0), alpha_g), _op('add', _r(4.0), alpha_g))))), ng)
    _m155 = (_m2) & _not(_m154) & (qg <= epsq)
    # WRF:2787
    q = jnp.where(_m155, _r(_op('add', q, qg)), q)
    # WRF:2788
    t = jnp.where(_m155, _r(_op('sub', t, _op('mul', lsp, qg))), t)
    # WRF:2789
    qg = jnp.where(_m155, _r(_r(0.0)), qg)
    # WRF:2790
    ng = jnp.where(_m155, _r(_r(0.0)), ng)
    _m156 = (_m2) & ((qh > epsq) & (nh < epsn))
    # WRF:2795
    nh = jnp.where(_m156, _r(_op('mul', _pow(_op('mul', no_h_sm, gh31), _op('div', _r(3.0), _op('add', _r(4.0), alpha_h))), _pow(_op('mul', _op('mul', _op('mul', _op('mul', gh31, igh34), tmp1), qh), icmh), _op('div', _op('add', _r(1.0), alpha_h), _op('add', _r(4.0), alpha_h))))), nh)
    _m157 = (_m2) & _not(_m156) & (qh <= epsq)
    # WRF:2798
    q = jnp.where(_m157, _r(_op('add', q, qh)), q)
    # WRF:2799
    t = jnp.where(_m157, _r(_op('sub', t, _op('mul', lsp, qh))), t)
    # WRF:2800
    qh = jnp.where(_m157, _r(_r(0.0)), qh)
    # WRF:2801
    nh = jnp.where(_m157, _r(_r(0.0)), nh)
    _m158 = (_m2) & ((qh > epsq) & (nh > epsn))
    # WRF:2805
    tmp1 = jnp.where(_m158, _r(_op('div', _r(1.0), nh)), tmp1)
    # WRF:2806
    dh = jnp.where(_m158, _r(_dm_x(de, qh, tmp1, icmh, thrd)), dh)
    _m159 = (_m158) & (dh < dh_min)
    # WRF:2808
    qg = jnp.where(_m159, _r(_op('add', qg, qh)), qg)
    # WRF:2809
    ng = jnp.where(_m159, _r(_op('add', ng, nh)), ng)
    # WRF:2810
    qh = jnp.where(_m159, _r(_r(0.0)), qh)
    # WRF:2811
    nh = jnp.where(_m159, _r(_r(0.0)), nh)
    # WRF:2815
    q = jnp.where(_m2, _r(jnp.maximum(q, _r(0.0))), q)
    # WRF:2816
    ny = jnp.where(_m2, _r(jnp.minimum(ny, ni_max)), ny)
    _m160 = (_m2) & ((t < _r(173.0)) | (t > _r(323.0)))
    _m1 = (point_mask) & (warmphase_on)
    # WRF:2851
    rcautr = jnp.where(_m1, _r(_r(0.0)), rcautr)
    # WRF:2851
    ccaccr = jnp.where(_m1, _r(_r(0.0)), ccaccr)
    # WRF:2851
    dc = jnp.where(_m1, _r(_r(0.0)), dc)
    # WRF:2851
    ilamc = jnp.where(_m1, _r(_r(0.0)), ilamc)
    # WRF:2851
    l = jnp.where(_m1, _r(_r(0.0)), l)
    # WRF:2852
    rcaccr = jnp.where(_m1, _r(_r(0.0)), rcaccr)
    # WRF:2852
    ccscoc = jnp.where(_m1, _r(_r(0.0)), ccscoc)
    # WRF:2852
    dr = jnp.where(_m1, _r(_r(0.0)), dr)
    # WRF:2852
    ilamr = jnp.where(_m1, _r(_r(0.0)), ilamr)
    # WRF:2852
    tau = jnp.where(_m1, _r(_r(0.0)), tau)
    # WRF:2853
    ccautr = jnp.where(_m1, _r(_r(0.0)), ccautr)
    # WRF:2853
    crscor = jnp.where(_m1, _r(_r(0.0)), crscor)
    # WRF:2853
    sigc = jnp.where(_m1, _r(_r(0.0)), sigc)
    # WRF:2853
    drinit = jnp.where(_m1, _r(_r(0.0)), drinit)
    # WRF:2854
    ilamc3 = jnp.where(_m1, _r(_r(0.0)), ilamc3)
    # WRF:2854
    ilamc6 = jnp.where(_m1, _r(_r(0.0)), ilamc6)
    # WRF:2854
    ilamr3 = jnp.where(_m1, _r(_r(0.0)), ilamr3)
    # WRF:2854
    ilamr6 = jnp.where(_m1, _r(_r(0.0)), ilamr6)
    # WRF:2856
    rainpresent = jnp.where(_m1, jnp.asarray((qr_in > epsq) & (nr_in > epsn)), rainpresent)
    _m2 = (_m1) & ((qc_in > epsq) & (nc_in > epsn))
    # WRF:2859
    inc = jnp.where(_m2, _r(_op('div', _r(1.0), nc_in)), inc)
    # WRF:2860
    ilamc = jnp.where(_m2, _r(_ilamda_x(de, qc_in, inc, icexc9, thrd)), ilamc)
    # WRF:2861
    ilamc3 = jnp.where(_m2, _r(_op('mul', _op('mul', ilamc, ilamc), ilamc)), ilamc3)
    # WRF:2862
    ilamc6 = jnp.where(_m2, _r(_op('mul', ilamc3, ilamc3)), ilamc6)
    # WRF:2863
    dc = jnp.where(_m2, _r(_op('mul', ilamc, _pow(_op('mul', gc2, igc1), thrd))), dc)
    # WRF:2864
    sigc = jnp.where(_m2, _r(_op('mul', ilamc, _pow(_op('sub', _op('mul', gc3, igc1), _op('mul', _op('mul', gc2, igc1), _op('mul', gc2, igc1))), sixth))), sigc)
    # WRF:2865
    l = jnp.where(_m2, _r(_op('mul', _op('mul', _op('mul', _r(0.027), de), qc_in), _op('sub', _op('mul', _op('mul', _op('mul', _op('mul', _r(6.25e+18), sigc), sigc), sigc), dc), _r(0.4)))), l)
    _m3 = (_m2) & (sigc > sigcthrs)
    # WRF:2866
    tau = jnp.where(_m3, _r(_op('div', _r(3.7), _op('mul', _op('mul', de, qc_in), _op('sub', _op('mul', _r(500000.0), sigc), _r(7.5))))), tau)
    _m4 = (_m1) & (rainpresent)
    # WRF:2870
    inr = jnp.where(_m4, _r(_op('div', _r(1.0), nr_in)), inr)
    # WRF:2871
    dr = jnp.where(_m4, _r(_dm_x(de, qr_in, inr, icmr, thrd)), dr)
    _m5 = (_m4) & (dr > _r(0.003))
    # WRF:2875
    tmp1 = jnp.where(_m5, _r(_op('sub', dr, _r(0.003))), tmp1)
    # WRF:2876
    tmp2 = jnp.where(_m5, _r(_op('div', dr, drmax)), tmp2)
    # WRF:2877
    tmp3 = jnp.where(_m5, _r(_op('mul', _op('mul', tmp2, tmp2), tmp2)), tmp3)
    # WRF:2878
    nr_in = jnp.where(_m5, _r(_op('mul', nr_in, jnp.maximum(_op('add', _r(1.0), _op('mul', _op('mul', _r(20000.0), tmp1), tmp1)), tmp3))), nr_in)
    # WRF:2879
    inr = jnp.where(_m5, _r(_op('div', _r(1.0), nr_in)), inr)
    # WRF:2880
    dr = jnp.where(_m5, _r(_dm_x(de, qr_in, inr, icmr, thrd)), dr)
    # WRF:2882
    ilamr = jnp.where(_m4, _r(_ilamda_x(de, qr_in, inr, icexr9, thrd)), ilamr)
    # WRF:2883
    ilamr3 = jnp.where(_m4, _r(_op('mul', _op('mul', ilamr, ilamr), ilamr)), ilamr3)
    # WRF:2884
    ilamr6 = jnp.where(_m4, _r(_op('mul', ilamr3, ilamr3)), ilamr6)
    _m6 = (_m1) & ((qc_in > epsq) & (sigc > sigcthrs) & autoconv_on)
    # WRF:2889
    rcautr = jnp.where(_m6, _r(jnp.minimum(jnp.maximum(_op('div', l, tau), _r(0.0)), _op('mul', qc, idt))), rcautr)
    # WRF:2890
    drinit = jnp.where(_m6, _r(jnp.maximum(_r(8.3e-05), _op('div', _r(0.00126), _op('sub', _op('mul', _r(500000.0), sigc), _r(3.5))))), drinit)
    # WRF:2891
    draut = jnp.where(_m6, _r(jnp.maximum(drinit, dr)), draut)
    # WRF:2892
    ccautr = jnp.where(_m6, _r(_op('div', _op('mul', rcautr, de), _op('mul', _op('mul', _op('mul', cmr, draut), draut), draut))), ccautr)
    # WRF:2907
    ccscoc = jnp.where(_m6, _r(jnp.minimum(_op('mul', _op('mul', _op('mul', _op('mul', _op('mul', kk2, nc_in), nc_in), gc3), igc1), ilamc6), _op('mul', nc_in, idt))), ccscoc)
    _m7 = (_m1) & (((qr_in > _op('mul', _op('mul', _r(1.2), jnp.maximum(l, _r(0.0))), ide)) | (dr > jnp.maximum(_r(5e-06), drinit))) & rainaccr_on & rainpresent)
    _m8 = (_m7) & ((qc_in > epsq) & (l > _r(0.0)))
    _m9 = (_m8) & (dr >= _r(0.0001))
    # WRF:2917
    ccaccr = jnp.where(_m9, _r(_op('mul', _op('mul', kk1, _op('mul', nc_in, nr_in)), _op('add', _op('mul', _op('mul', gc2, igc1), ilamc3), _op('mul', _op('mul', gr34, igr31), ilamr3)))), ccaccr)
    # WRF:2918
    rcaccr = jnp.where(_m9, _r(_op('mul', _op('mul', _op('mul', _op('mul', _op('mul', cmr, ide), kk1), _op('mul', nc_in, nr_in)), ilamc3), _op('add', _op('mul', _op('mul', gc3, igc1), ilamc3), _op('mul', _op('mul', _op('mul', _op('mul', gc2, igc1), gr34), igr31), ilamr3)))), rcaccr)
    _m10 = (_m8) & _not(_m9)
    # WRF:2921
    ccaccr = jnp.where(_m10, _r(_op('mul', _op('mul', kk2, _op('mul', nc_in, nr_in)), _op('add', _op('mul', _op('mul', gc3, igc1), ilamc6), _op('mul', _op('mul', gr37, igr31), ilamr6)))), ccaccr)
    # WRF:2926
    tmp1 = jnp.where(_m10, _r(_op('mul', cmr, ide)), tmp1)
    # WRF:2927
    tmp2 = jnp.where(_m10, _r(_op('mul', _op('mul', kk2, _op('mul', nc_in, nr_in)), ilamc3)), tmp2)
    # WRF:2928
    rcaccr = jnp.where(_m10, _r(_op('mul', tmp1, tmp2)), rcaccr)
    # WRF:2929
    tmp1 = jnp.where(_m10, _r(_op('mul', gc4, igr31)), tmp1)
    # WRF:2930
    tmp1 = jnp.where(_m10, _r(_op('mul', tmp1, ilamc6)), tmp1)
    # WRF:2931
    tmp2 = jnp.where(_m10, _r(_op('mul', gc2, igc1)), tmp2)
    # WRF:2932
    tmp2 = jnp.where(_m10, _r(_op('mul', _op('mul', tmp2, gr37), igr31)), tmp2)
    # WRF:2933
    tmp2 = jnp.where(_m10, _r(_op('mul', tmp2, ilamr6)), tmp2)
    # WRF:2934
    rcaccr = jnp.where(_m10, _r(_op('mul', rcaccr, _op('add', tmp1, tmp2))), rcaccr)
    # WRF:2937
    ccaccr = jnp.where(_m8, _r(jnp.minimum(ccaccr, _op('mul', nc, idt))), ccaccr)
    # WRF:2938
    rcaccr = jnp.where(_m8, _r(jnp.minimum(rcaccr, _op('mul', qc, idt))), rcaccr)
    # WRF:2942
    tmp1 = jnp.where(_m7, _r(_op('mul', nr_in, nr_in)), tmp1)
    _m11 = (_m7) & (dr >= _r(0.0001))
    # WRF:2944
    crscor = jnp.where(_m11, _r(_op('mul', _op('mul', _op('mul', _op('mul', kk1, tmp1), gr34), igr31), ilamr3)), crscor)
    _m12 = (_m7) & _not(_m11)
    # WRF:2946
    crscor = jnp.where(_m12, _r(_op('mul', _op('mul', _op('mul', _op('mul', kk2, tmp1), gr37), igr31), ilamr6)), crscor)
    # WRF:2949
    ec = jnp.where(_m7, _r(_r(1.0)), ec)
    _m13 = (_m7) & (ilamr > _r(0.0003))
    # WRF:2951
    ec = jnp.where(_m13, _r(_op('sub', _r(2.0), _exp(_op('mul', _r(2300.0), _op('sub', ilamr, _r(0.0003)))))), ec)
    # WRF:2953
    crscor = jnp.where(_m7, _r(jnp.minimum(_op('mul', ec, crscor), _op('mul', _op('mul', _r(0.5), nr), idt))), crscor)
    # WRF:2958
    source = jnp.where(_m1, _r(qc), source)
    # WRF:2959
    sink = jnp.where(_m1, _r(_op('mul', _op('add', rcautr, rcaccr), dt)), sink)
    _m14 = (_m1) & (sink > source)
    # WRF:2961
    ratio = jnp.where(_m14, _r(_op('div', source, sink)), ratio)
    # WRF:2962
    rcautr = jnp.where(_m14, _r(_op('mul', ratio, rcautr)), rcautr)
    # WRF:2963
    rcaccr = jnp.where(_m14, _r(_op('mul', ratio, rcaccr)), rcaccr)
    # WRF:2964
    ccaccr = jnp.where(_m14, _r(_op('mul', ratio, ccaccr)), ccaccr)
    # WRF:2968
    qc = jnp.where(_m1, _r(jnp.maximum(_r(0.0), _op('add', qc, _op('mul', _op('sub', -rcautr, rcaccr), dt)))), qc)
    # WRF:2969
    qr = jnp.where(_m1, _r(jnp.maximum(_r(0.0), _op('add', qr, _op('mul', _op('add', rcautr, rcaccr), dt)))), qr)
    # WRF:2970
    nc = jnp.where(_m1, _r(jnp.maximum(_r(0.0), _op('add', nc, _op('mul', _op('sub', -ccaccr, ccscoc), dt)))), nc)
    # WRF:2971
    nr = jnp.where(_m1, _r(jnp.maximum(_r(0.0), _op('add', nr, _op('mul', _op('sub', ccautr, crscor), dt)))), nr)
    _m15 = (_m1) & ((qr > epsq) & (nr > epsn))
    # WRF:2974
    inr = jnp.where(_m15, _r(_op('div', _r(1.0), nr)), inr)
    # WRF:2975
    dr = jnp.where(_m15, _r(_dm_x(de, qr, inr, icmr, thrd)), dr)
    _m16 = (_m15) & (dr > _r(0.003))
    # WRF:2977
    tmp1 = jnp.where(_m16, _r(_op('sub', dr, _r(0.003))), tmp1)
    # WRF:2977
    tmp2 = jnp.where(_m16, _r(_op('mul', tmp1, tmp1)), tmp2)
    # WRF:2978
    tmp3 = jnp.where(_m16, _r(_op('div', dr, drmax)), tmp3)
    # WRF:2978
    tmp4 = jnp.where(_m16, _r(_op('mul', _op('mul', tmp3, tmp3), tmp3)), tmp4)
    # WRF:2979
    nr = jnp.where(_m16, _r(_op('mul', nr, jnp.maximum(_op('add', _r(1.0), _op('mul', _r(20000.0), tmp2)), tmp4))), nr)
    _m17 = (_m15) & _not(_m16) & (dr < dhh)
    # WRF:2982
    qc = jnp.where(_m17, _r(_op('add', qc, qr)), qc)
    # WRF:2983
    nc = jnp.where(_m17, _r(_op('add', nc, nr)), nc)
    # WRF:2984
    qr = jnp.where(_m17, _r(_r(0.0)), qr)
    # WRF:2984
    nr = jnp.where(_m17, _r(_r(0.0)), nr)
    _m18 = (_m1) & _not(_m15)
    # WRF:2987
    qr = jnp.where(_m18, _r(_r(0.0)), qr)
    # WRF:2987
    nr = jnp.where(_m18, _r(_r(0.0)), nr)
    # WRF:2992
    qsw = jnp.where(_m1, _r(_qsat(t, pres, 0)), qsw)
    _m19 = (_m1) & (qsw > _r(1e-20))
    # WRF:2994
    ssat = jnp.where(_m19, _r(_op('sub', _op('div', q, qsw), _r(1.0))), ssat)
    _m20 = (_m1) & _not(_m19)
    # WRF:2996
    ssat = jnp.where(_m20, _r(_r(0.0)), ssat)
    # WRF:2998
    x = jnp.where(_m1, _r(_op('sub', q, qsw)), x)
    # WRF:3001
    x = jnp.where(_m1, _r(_op('div', x, _op('add', _r(1.0), _op('div', _op('mul', _pow(_op('sub', _r(3148400.0), _op('mul', _r(2370.0), t)), 2), qsw), _op('mul', _op('mul', _op('mul', _r(1005.0), _op('add', _r(1.0), _op('mul', _r(0.887), q))), _r(461.5)), _pow(t, 2)))))), x)
    # WRF:3003
    x = jnp.where(_m1, _r(jnp.maximum(x, -qc)), x)
    # WRF:3004
    qc = jnp.where(_m1, _r(_op('add', qc, x)), qc)
    # WRF:3005
    q = jnp.where(_m1, _r(_op('sub', q, x)), q)
    # WRF:3006
    t = jnp.where(_m1, _r(_op('add', t, _op('mul', lcp, x))), t)
    _m21 = (_m1) & (x > _r(0.0))
    _m22 = (_m21) & (wz > _r(0.001))
    # WRF:3014
    nc = jnp.where(_m22, _r(jnp.maximum(nc, _nccnfnc(wz, t, pres, ccntype))), nc)
    _m23 = (_m21) & _not(_m22)
    # WRF:3017
    nc = jnp.where(_m23, _r(jnp.maximum(nc, n_c_sm)), nc)
    _m24 = (_m1) & _not(_m21)
    _m25 = (_m24) & (qc > epsq)
    # WRF:3022
    nc = jnp.where(_m25, _r(jnp.maximum(_r(0.0), _op('add', nc, _op('div', _op('mul', x, nc), jnp.maximum(qc, epsq))))), nc)
    _m26 = (_m24) & _not(_m25)
    # WRF:3024
    nc = jnp.where(_m26, _r(_r(0.0)), nc)
    _m27 = (_m1) & ((qc > epsq) & (nc < epsn))
    # WRF:3027
    nc = jnp.where(_m27, _r(n_c_sm), nc)
    # WRF:3031
    qsw = jnp.where(_m1, _r(_qsat(t, pres, 0)), qsw)
    _m28 = (_m1) & ((q < qsw) & (qr > epsq) & (nr > epsn))
    # WRF:3033
    ssat = jnp.where(_m28, _r(_op('sub', _op('div', q, qsw), _r(1.0))), ssat)
    # WRF:3034
    tc = jnp.where(_m28, _r(_op('sub', t, trpl)), tc)
    # WRF:3035
    cdiff = jnp.where(_m28, _r(_op('div', _op('mul', jnp.maximum(_r(1.62e-05), _op('add', _r(2.2157e-05), _op('mul', _r(1.55e-07), tc))), _r(100000.0)), pres)), cdiff)
    # WRF:3036
    mudyn = jnp.where(_m28, _r(jnp.maximum(_r(1.51e-05), _op('add', _r(1.7153e-05), _op('mul', _r(5e-08), tc)))), mudyn)
    # WRF:3037
    ka = jnp.where(_m28, _r(jnp.maximum(_r(0.0207), _op('add', _r(0.023971), _op('mul', _r(7.8e-05), tc)))), ka)
    # WRF:3038
    mukin = jnp.where(_m28, _r(_op('mul', mudyn, ide)), mukin)
    # WRF:3039
    imukin = jnp.where(_m28, _r(_op('div', _r(1.0), mukin)), imukin)
    # WRF:3040
    scthrd = jnp.where(_m28, _r(_pow(_op('div', mukin, cdiff), thrd)), scthrd)
    # WRF:3041
    x = jnp.where(_m28, _r(_op('sub', qsw, q)), x)
    # WRF:3044
    x = jnp.where(_m28, _r(_op('div', x, _op('add', _r(1.0), _op('div', _op('mul', _pow(_op('sub', _r(3148400.0), _op('mul', _r(2370.0), t)), 2), qsw), _op('mul', _op('mul', _op('mul', _r(1005.0), _op('add', _r(1.0), _op('mul', _r(0.887), q))), _r(461.5)), _pow(t, 2)))))), x)
    # WRF:3046
    de = jnp.where(_m28, _r(_op('div', pres, _op('mul', rgasd, t))), de)
    # WRF:3047
    ide = jnp.where(_m28, _r(_op('div', _r(1.0), de)), ide)
    # WRF:3048
    gam = jnp.where(_m28, _r(jnp.sqrt(_op('mul', deo, ide))), gam)
    # WRF:3049
    tmp1 = jnp.where(_m28, _r(_op('div', _r(1.0), nr)), tmp1)
    # WRF:3050
    ilamr = jnp.where(_m28, _r(_ilamda_x(de, qr, tmp1, icexr9, thrd)), ilamr)
    # WRF:3051
    lamr = jnp.where(_m28, _r(_op('div', _r(1.0), ilamr)), lamr)
    # WRF:3054
    no_r = jnp.where(_m28, _r(_op('mul', _op('mul', _d(nr), _pow(_d(lamr), _d(_op('add', _r(1.0), alpha_r)))), igr31)), no_r)
    # WRF:3056
    ventr = jnp.where(_m28, _r(_op('add', _op('mul', _op('mul', avx, gr32), _pow(ilamr, cexr5)), _op('mul', _op('mul', _op('mul', _op('mul', bvx, scthrd), jnp.sqrt(_op('mul', _op('mul', gam, afr), imukin))), gr17), _pow(ilamr, cexr6)))), ventr)
    # WRF:3057
    abw = jnp.where(_m28, _r(_op('add', _op('div', _pow(chlc, 2), _op('mul', _op('mul', ka, rgasv), _pow(t, 2))), _op('div', _r(1.0), _op('mul', _op('mul', de, qsw), cdiff)))), abw)
    # WRF:3058
    qrevp = jnp.where(_m28, _r(jnp.minimum(qr, _op('mul', -dt, _op('div', _op('mul', _op('mul', _op('mul', _op('mul', ide, pi2), ssat), no_r), ventr), abw)))), qrevp)
    # WRF:3059
    tmp1 = jnp.where(_m28, _r(qr), tmp1)
    # WRF:3060
    t = jnp.where(_m28, _r(_op('sub', t, _op('mul', lcp, qrevp))), t)
    # WRF:3061
    q = jnp.where(_m28, _r(_op('add', q, qrevp)), q)
    # WRF:3062
    qr = jnp.where(_m28, _r(_op('sub', qr, qrevp)), qr)
    # WRF:3063
    nr = jnp.where(_m28, _r(jnp.maximum(_r(0.0), _op('sub', nr, _op('div', _op('mul', qrevp, nr), tmp1)))), nr)
    _m29 = (_m28) & ((qr < epsq) | (nr < epsn))
    # WRF:3066
    q = jnp.where(_m29, _r(_op('add', q, qr)), q)
    # WRF:3067
    t = jnp.where(_m29, _r(_op('sub', t, _op('mul', qr, lcp))), t)
    # WRF:3068
    qr = jnp.where(_m29, _r(_r(0.0)), qr)
    # WRF:3069
    nr = jnp.where(_m29, _r(_r(0.0)), nr)
    # WRF:3074
    tc = jnp.where(_m1, _r(_op('sub', t, trpl)), tc)
    _m30 = (_m1) & ((qc > epsq) & (tc < -_r(30.0)) & icephase_on)
    _m31 = (_m30) & (tc < -_r(35.0))
    # WRF:3087
    frac = jnp.where(_m31, _r(_r(1.0)), frac)
    _m32 = (_m30) & _not(_m31)
    # WRF:3089
    frac = jnp.where(_m32, _r(_r(0.0)), frac)
    # WRF:3092
    qfzci = jnp.where(_m30, _r(_op('mul', frac, qc)), qfzci)
    # WRF:3093
    nfzci = jnp.where(_m30, _r(_op('mul', frac, nc)), nfzci)
    # WRF:3094
    qc = jnp.where(_m30, _r(_op('sub', qc, qfzci)), qc)
    # WRF:3095
    nc = jnp.where(_m30, _r(_op('sub', nc, nfzci)), nc)
    # WRF:3096
    qi = jnp.where(_m30, _r(_op('add', qi, qfzci)), qi)
    # WRF:3097
    ny = jnp.where(_m30, _r(_op('add', ny, nfzci)), ny)
    # WRF:3098
    t = jnp.where(_m30, _r(_op('add', t, _op('mul', lfp, qfzci))), t)
    _m33 = (_m30) & ((qc > epsq) & (nc < epsn))
    # WRF:3101
    nc = jnp.where(_m33, _r(n_c_sm), nc)
    _m34 = (_m30) & _not(_m33) & (qc <= epsq)
    # WRF:3103
    q = jnp.where(_m34, _r(_op('add', q, qc)), q)
    # WRF:3104
    t = jnp.where(_m34, _r(_op('sub', t, _op('mul', lcp, qc))), t)
    # WRF:3105
    qc = jnp.where(_m34, _r(_r(0.0)), qc)
    # WRF:3106
    nc = jnp.where(_m34, _r(_r(0.0)), nc)
    _m35 = (_m30) & ((qi > epsq) & (ny < epsn))
    # WRF:3111
    ny = jnp.where(_m35, _r(jnp.maximum(_op('mul', _r(2.0), epsn), _n_cooper(trpl, t))), ny)
    _m36 = (_m30) & _not(_m35) & (qi <= epsq)
    # WRF:3113
    q = jnp.where(_m36, _r(_op('add', q, qi)), q)
    # WRF:3114
    t = jnp.where(_m36, _r(_op('sub', t, _op('mul', lsp, qi))), t)
    # WRF:3115
    qi = jnp.where(_m36, _r(_r(0.0)), qi)
    # WRF:3116
    ny = jnp.where(_m36, _r(_r(0.0)), ny)
    return (t, q, qc, qr, qi, qn, qg, qh, nc, nr, ny, nn, ng, nh), de, ide
