# o2-fdda — grid analysis nudging (grid_fdda=1) pristine-WRF oracle

* `oracle/fdda_oracle_driver.F90` — calls the UNMODIFIED pristine `phys/module_fdda_psufddagd.F` (FDDAGD) inside verbatim
  transcriptions of the WRF caller chain: auxinput10 read → `fddagd_driver` control (itimestep==1 OLD overwrite,
  fgdtzero, STEPFG reset+recompute) → `calculate_phy_tend` coupling (calc_mu_uv interior muu/muv, mut=mub+mu_2) →
  `phy_fg_ten` ownership (add_a2a / add_c2c_u / add_c2c_v, specified) → `phy_prep_part2` decoupling (calc_mu_uv_1 muts).
* `oracle/build.sh <workdir>` — lane-dir copy of the pristine source (E104), WRF configure.wrf flags (`-O2 -ftree-vectorize
  -funroll-loops`, no `-march` → no FMA), REAL (r4) and `-fdefault-real-8` (r8) builds. Compiler: conda gfortran 14.3 (E129).
* `oracle/make_savepoints.py <build> <out>` — real-structured crops of ORIGINAL CPU-WRF history frames (Swiss RD11
  winter 10×8, PROD D5 summer day/night 12×10 with 55 % land, 44 levels); analyses = CPU-WRF frames of other hours,
  model states = frames of in-between hours. Regimes: plain (specified), if_no_pbl_nudging uv/t/q + if_zfac + STEPFG=10
  hold chain + the end_h_sfc/|dtramp_min| past-end quirk, dtramp>0 ramp (coef forced 0, tfac→0, early return),
  dtramp<0 ramp + fgdtzero + STEPFG=5. `savepoints/MANIFEST.json` = sha256 + E154 census (kpbl spread, land, nonzero).
* Parity (tests/v040/fdda/test_o2_fdda_oracle_parity.py): port == oracle bitwise where no product can be FMA-contracted
  (coef=0 recompute calls); elsewhere |port−WRF| ≤ 4·eps·(|WRF| + M·g·A) — XLA (CPU and GPU) contracts
  `old*(1-coef)+new*coef` and `c1h*mu+c2h` into FMAs, gfortran without -march does not (measured: the CPU port equals an
  FMA emulation bit for bit). Clock divisions (xtime, FLOOR(curr_mins2/interval), coef, tfac) are REAL with a fenced
  divisor (XLA turns x/const into x*(1/const), E130); only WRF's DOUBLE curr_mins2 = REAL(curr_secs2_r8/60d0) is f64
  (rv-fdda F2; exact f64 census pinned in tests/v040/fdda/test_o2_fdda_wiring.py).

## Coupled CPU checks on the real Swiss case (RD11, 42×42×44, dt 18 s)

* `make_wrffdda.py` writes a real-structured `wrffdda_d01` from the ORIGINAL CPU-WRF hourly frames (record n:
  OLD = frame(t_n), NEW = frame(t_n+1h)); no real.exe `grid_fdda=1` output exists for our cases.
  It supplies original WRF grid metadata, a versioned provenance TITLE and `FieldType=104` for every REAL field.
  Earlier files lacked FieldType: original WRF returned before reading values and nudged toward zero analyses despite
  reporting SUCCESS. Such runs cannot serve as coupled fidelity references. See `proofs/t_fdda` for a paired actual
  WRF restart-operand audit and a metadata-only repair retaining all 13 numerical payloads.
* PSU runs retain `HFX_FDDA` from the driver call before coupling, with WRF's literal
  `p/(287*th_phy)*1004*RTHNDGDTEN*dz8w`; it uses dry potential temperature, not actual temperature.
  Both pipelines publish that held 3-D field; restart requires it, and the daily pipeline retains the global clock
  and analysis cursor across hourly/auxiliary outputs. Spectral nudging retains its existing 15-leaf carry schema.
* `probe.py` (from proofs/o1_nlbind) loads each arm through the REAL CLI loader `nested_pipeline._load_domains`
  and records the location-stripped jaxpr sha of the production step (`_advance_chunk_fori`, 1 step); with
  `O1_EXEC_STEPS=N` it executes N steps (legacy CPU path, `GPUWRF_FAST_DEFAULTS=0`); `O2_MUTANT=no_qv` drops
  the FDDA qv source (coupled-level deletion check).
* `smoke_check.py` compares the FDDA arm with the control: in WRF's first interval OLD = the step-1 model state,
  so the target is wrfinput·(1−coef) + NEW·coef; FDDA−control must correlate with (target − state) per lane.
