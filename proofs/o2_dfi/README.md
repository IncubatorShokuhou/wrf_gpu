# o2-dfi — WRF TDFI (`dfi_opt=3`) evidence

All runs on CPU (no GPU in wave O2). Case: Swiss 42×42 d01 (RD11 CPU inputs, dt 18 s,
release physics). Oracle: CPU-WRF 4.7.1 dmpar (the RD11 executable, 1 rank) with
`tests/fixtures/o2_dfi_swiss_tdfi.namelist` (= oracle A: Dolph `dfi_nfilter=7`, cutoff 1200 s,
backward 00:00 → 23:42, forward 23:51 → 00:09, `dfi_write_filtered_input`).

| asset | what |
|---|---|
| `build_dfcoef_oracle.py`, `dfcoef_driver.F90`, `dfcoef_oracle.json` | verbatim `share/dfi.F` `dfcoef`…`cnr`, 64 window/filter cases; `gpuwrf.dfi.filter` reproduces all 64 bitwise |
| `build_start_em_fixture.py` | crop of WRF's `wrfinput_initialized_d01` → `tests/fixtures/o2_dfi_start_em_crop.npz` (start_em P/AL/P_HYD, 0 ulp) |
| `run_tdfi_e2e.py` | TDFI pre-pass (`gpuwrf.dfi.driver.run_tdfi`) on the Swiss case; `DIAG_INTERPRET=1` runs the release-default Pallas kernels in interpret mode on CPU |
| `compare_filtered_state.py` | port filtered `wrfinput` vs WRF `wrfinput_initialized_d01`: RMSE, WRF increment, residual fraction |
| `run_forecast.py`, `compare_forecast.py` | forecast from a filtered input dir; frame RMSE vs WRF wrfout |

Short-window oracle (W2): the same namelist with `dfi_bckstop` = start − 36 s,
`dfi_fwdstop` = start + 18 s, `run_seconds = 18` (2 backward + 2 forward steps). Stage-start
errors that a 60 + 60-step Dolph window averages out show up there (the forward-stage W
re-set was found that way).

## Results (release defaults, tree 9aadad3a9; residual = RMSE(port − WRF) / RMSE(WRF − input))

| field | oracle A (60 + 60 steps) | W2 (2 + 2 steps) |
|---|---|---|
| U / V / W | .006 / .005 / .009 | .015 / .013 / .004 |
| PH / MU / P / AL | .008 / .015 / .051 / .005 | .002 / .008 / .031 / .014 |
| THM / QVAPOR / QCLOUD | .006 / .003 / .009 | .006 / .005 / – |
| TSLB / SMOIS / LAI / TSK | .000 / .000 / .032 / .074 | .000 / .000 / .032 / .063 |
| T2 / PSFC / U10 / V10 / Q2 | .048 / .025 / .008 / .013 / .19 | .014 / .004 / .005 / .002 / .17 |

Forecast (1 h, release path) RMSE vs the CPU-WRF DFI forecast, port from its own filtered
input / port from the unfiltered input: t0 T2 .093 / 1.93 K, PSFC .74 / 29.5 Pa; +1 h T2 .121 /
.237 K, U10 .016 / .123 m/s, V10 .025 / .108, PSFC .28 / 1.48 Pa, THM .0047 / .042 K, MU .29 /
1.44 Pa (every 10-min frame 2–9× closer).

Known residuals: wrfinput `T` (WRF writes the forward-stage `th_phy_m_t0`, the port the dry
theta of the filtered THM; .67), SNOW (.67 of a 8.5e-4 kg/m² increment); legacy carry
(`GPUWRF_FAST_DEFAULTS=0`) has no in-step history diagnostics, so T2/Q2/U10/V10/PSFC keep the
input values there. Backward-stage parity per step: notes `.agent/kernel/lanes/o2-dfi.notes.md`.
