# Radiation cloud fraction: `icloud` (v0.4.1, experimental)

`icloud` (`&physics`, one value for all domains) selects how WRF's radiation driver builds
the cloud fraction `CLDFRA` and the cloud water/ice that RRTMG sees. The port follows WRF
4.7.1 `phys/module_radiation_driver.F` exactly.

| `icloud` | WRF routine | What radiation sees | Port status |
| --- | --- | --- | --- |
| 1 (default) | `cal_cldfra1` (Xu–Randall), then the `icloud_bl=1` MYNN merge | After the first step `CLDFRA = CLDFRA_BL` (MYNN sub-grid cloud PDF); MYNN `QC_BL/QI_BL` added where resolved cloud is absent | Default suite |
| 2 | `cal_cldfra2` | Binary: 1 where `qc+qi > 1e-6 kg/kg`, else 0 | CPU-oracle-qualified; one 24 h WN3 3-nest GPU forecast within D6 of original CPU-WRF |
| 3 | `cal_cldfra3` (G. Thompson) | Scale-aware RH cloud fraction (land/ocean RH thresholds from `dx`, `dy`), plus "fake" radiation-only cloud water/ice in partly cloudy layers, capped at 1 kg/m² per column | CPU-oracle-qualified; one 24 h WN3 3-nest GPU forecast within D6 of original CPU-WRF |
| 0 | — | No clouds in RRTMG | Not bound (refused) |

## What changes with `icloud = 3`

* `icloud=3` **replaces the whole `icloud=1` block**, including the `icloud_bl` MYNN merge
  (WRF 4.7.1 `module_radiation_driver.F:1404-1431` sits inside the `ICLOUD == 1` branch).
  MYNN keeps computing its own sub-grid cloud (`CLDFRA_BL`, `QC_BL`) for its mixing; radiation
  just does not use it. This is WRF's behaviour, not a port choice.
* The extra cloud water/ice exists only for the radiation call (WRF restores `qc`/`qi`
  afterwards). Snow is read, never changed. Water vapour is not changed
  (`modify_qvapor=.false.` in the radiation caller).
* History `CLDFRA` is the radiation input fraction, as in WRF.
* Caller operands as in WRF: `T = t_phy`, `P = p_hyd`, `dz8w` from `(PH+PHB)/9.81`, `XLAND`,
  `gridkm = sqrt((dx/1000)^2 + (dy/1000)^2)`.

## Restrictions (fail closed)

* RRTMG (`ra_lw_physics = ra_sw_physics = 4`) only.
* `icloud=3` needs QC and QI: `mp_physics = 0, 1, 3` are refused (WRF refuses Kessler/WSM3 in
  `check_a_mundo`, and stops without QC/QI). Ferrier (5/85/95), CAM-MG (11) and P3 (50–55)
  have scheme-specific radiation cloud views and are unqualified with `icloud = 2/3`.
* `cu_physics = 7/10` (cloud mediation unqualified). `cu_physics = 11` forces `icloud = 1`
  exactly as WRF does.

## Evidence

* `proofs/w41_icloud/oracle_gate.py`: 640 real columns (Swiss winter stratus/ice/snow and WN3
  0227 Canary trade-inversion Sc/Cu, plus 12 labelled adversarial edits) against the verbatim
  WRF 4.7.1 routines. Literal REAL4 port **byte-identical** (cldfra, qc, qi); compiled port
  inside WRF's own FMA-vs-no-FMA spread plus the one-ulp input envelope; every branch of
  `cal_cldfra3`/`find_cloudLayers`/`adjust_*` active (the IWP > 1 kg/m² rescale is reachable
  only with a non-physical dz stretch, kept as a labelled stress row).
* Disclosed: XLA's `log10` in the small-condensate branch (`1e-10 < qc+qi < 1e-6`) differs from
  glibc's correctly rounded `log10f` by up to 15 ulp of the cloud fraction (≤ 1.2e-7).
  WRF's own cold-temperature saturation polynomial (Thompson `RSIF`) has an FMA spread of up to
  3.8e-4 in cloud fraction, so this is far below WRF's own build-to-build variation.
* Coupled forecast: WN3 0227 Canary 3-nest (9/3/1 km), 24 h, `icloud = 3`, release defaults, on an
  RTX 5090 against an original CPU-WRF 4.7.1 run of the same namelist: frozen D6 manifest PASS on all
  three domains (25 frames, no failures, all fields finite). Worst-hour RMSE on the 1 km domain: T2
  0.09 K, PSFC 2.5 Pa, 10 m wind 0.33 m/s, RAINNC 0.05 mm. `CLDFRA` has no D6 bound; at +24 h the
  cloudy-cell share is 4.04/4.05 % (d01), 3.58/3.60 % (d02), 8.51/8.52 % (d03) CPU/GPU, pattern
  correlation 0.97-0.99 (`proofs/w41_icloud/cldfra_report.py`). E41 cubin gate PASS (max stack 72 B).
* Coupled forecast, `icloud = 2`: the same case and recipe with only `icloud` changed, against an
  original CPU-WRF 4.7.1 `icloud = 2` run: frozen D6 manifest PASS on all three domains (25 frames, no
  failures, all fields finite, all 375 CPU history fields present). Worst-hour RMSE on the 1 km domain:
  T2 0.08 K, PSFC 3.7 Pa, 10 m wind 0.29 m/s, RAINNC 0.03 mm (9/3 km domains: T2 0.010/0.015 K).
  `CLDFRA` is binary in both runs (no value other than 0 or 1 in any frame); at +24 h the cloudy-cell
  share is 0.68/0.67 % (d01), 1.11/1.11 % (d02), 4.10/4.05 % (d03) CPU/GPU, pattern correlation
  0.93-0.95. E41 cubin gate PASS (max stack 32 B). Measured on the v0.4.1 tree including the SWDDIR
  fix; an earlier run on the tree before that fix also passed. `proofs/w41_icloud2/`.
* Cost: one `cal_cldfra3` call is 0.8-1.1 ms per domain on the GPU (≤ 114 MB scratch), once per
  radiation call (radt = 30 min → about 6 ms per forecast hour for three domains). `cal_cldfra2` is a
  single threshold: about 0.03 ms per call and domain, no scratch memory. No new state is carried, so
  restart files are unchanged.
* Tests: `tests/v041/test_icloud.py`.

## Support-matrix row (v0.4.1 draft)

| Option | Status / scope | Receipt |
| --- | --- | --- |
| `icloud=3`, `icloud=2` | Experimental. CPU-oracle-qualified (literal BYTE vs pristine WRF 4.7.1 on 640 real columns); RRTMG 4/4 only; `icloud=3` and `icloud=2`: each a WN3 0227 3-nest 24 h GPU forecast with D6 PASS 3/3 vs an original CPU-WRF run of the same namelist; E41 PASS. | [Oracle gate](../proofs/w41_icloud/), [icloud=2 forecast](../proofs/w41_icloud2/), [doc](radiation-cloud-fraction.md) |
