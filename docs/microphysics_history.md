# Microphysics history coverage

The full WRF history stream retains its default Thompson inventory. Selected
microphysics schemes extend that inventory with fields taken directly from the
existing State or explicitly supplied held diagnostics. A training/subset stream
can request the same fields. No microphysics is rerun at an output alarm.

| mp_physics | Additional fields supplied from existing State |
|---|---|
| 40 | QNCLOUD (Nc), QNSNOW (Ns), QNGRAUPEL (Ng) |
| 24 | QHAIL (qh) |
| 28 | QNCLOUD (Nc), QNWFA (nwfa), QNIFA (nifa) |
| 16 | QNCLOUD (Nc), QNCCN (Nn) |
| 26 | QNCLOUD (Nc), QNCCN (Nn), QHAIL (qh) |
| 18 | QNDROP (Nc), QNCCN (Nn), QNSNOW (Ns), QNGRAUPEL (Ng), QNHAIL (Nh), QHAIL (qh), QVGRAUPEL (qvolg), QVHAIL (qvolh) |
| 22 | QNDROP (Nc), QNCCN (Nn), QNSNOW (Ns), QNGRAUPEL (Ng), QVGRAUPEL (qvolg) |
| 19 | QHAIL (qh), QNCCN (Nn), QVGRAUPEL (qvolg); no predicted hydrometeor numbers or hail volume |
| 21 | No extra scalars; six bulk moist species only |

NSSL names its droplet scalar **QNDROP**, whereas WDM/Morrison/Thompson aerosol
name it QNCLOUD. Values are transcribed without density conversion: these State
leaves are the corresponding WRF scalar values returned by their adapters.
History dimensions, units, descriptions, MemoryOrder and stagger match the
original CPU-WRF qualification files. WRF's older build uses spaces for the
Registry `#` characters in number-concentration units; this metadata is retained.

Legacy19/21 suppress number fields belonging to other schemes in full,
operational and subset streams. Legacy21 also suppresses all hail/volume/CCN
extras. See the [original one-moment inventory](../proofs/s2_mpmore/nssl1mom/HISTORY.md).

The following CPU history slots are **not provided by the operational port**.
They are omitted, including in the full stream, unless the caller supplies an
explicit same-name State or held diagnostic value. They are not replaced with
zeros merely because a particular CPU reference has inactive values.

| mp_physics | Fields not provided | Reason |
|---|---|---|
| 40 | CCN1_GS–CCN7_GS, EFCG, EFIG, EFSG, WACT | The reachable operational adapter selects aercu_opt=0. The aerosol branch that produces these diagnostics is inactive and no held outputs exist. The aercu_opt=2 column API has these diagnostics, but that configuration requires the separate MSKF/CESM integration. |
| 28 | QNWFA2D, QNIFA2D | The adapter computes emission amounts internally, but does not retain the WRF initialization-time emission fields. Recomputing from the evolving geopotential at output would not reproduce those held fields. |

`QNBCA`/`TAOD5502D` (mp=28/38) and `MSKF_REFL_10CM` (mp=40) are written as WRF's
own +0 under the admitted branches (`wif_input_opt/=2`, `aer_opt/=3`,
`do_radar_ref=0`; `gpuwrf.io.producer_history`, proofs/t_classd). Morrison mp=10/40
with `do_radar_ref=1` hold `REFL_10CM` from WRF `refl10cm_hm` (full column at history
alarms, `MAX(-35,dBZ)`); mp=40 writes `MSKF_REFL_10CM` equal to it (no MSKF cumulus
hydrometeors under `aercu_opt=0`).

NSSL `mp_physics=17/18/19/21/22` hold `REFL_10CM` with WRF's NSSL `makediag`
lifetime (full depth at history alarms, bottom level on the first step or with
`nwp_diagnostics=1`; WRF `check_a_mundo` forces `compute_radar_ref=1` for every
NSSL selector). The one-moment selectors 19/21 use `radardd02`'s `ipconc=0`
intercept branches (fp32 byte-exact against the pristine S5->S6 column fixtures,
`proofs/t_classd/`). `REFD_MAX` is WRF's untouched allocation zero when
`nwp_diagnostics=0` (`gpuwrf.io.producer_history`).

These omissions remain visible to the full CPU/GPU inventory gate. This writer
change does not grant complete coupled-forecast qualification or close a D6 rain
failure. HFX_FDDA belongs to the grid-FDDA producer and its separate qualification.
mp16's CPU reference does not allocate QHAIL; mp26 does.

Tests use inventories/metadata from each of the six original CPU-WRF runs and
small real final-frame samples, committed under `proofs/t_mpout/`. They exercise
full, operational and subset output, reject NaN/Inf, and verify absent sources
are not fabricated. The samples qualify output transcription, not model physics.

Performance and memory: no model graph, prognostic carry, restart schema, kernel
launch or timestep allocation changes. Output preparation reuses existing host
arrays from the one batched device-get boundary. New finite checks scan only the
selected extra fields on the host. Additional history volume is four bytes per
grid point per emitted REAL field (plus NetCDF metadata); no discarded optional
species are copied merely to support a selected subset. GPU output timing remains
unmeasured; this is a coverage fix, not a speed claim.

On the original CPU-WRF 44×42×42 output arrays, five measured interleaved pairs
(after one warm-up pair) gave median host preparation 7.337→7.857 ms for mp40 and
7.194→7.632 ms for mp18. Preparation plus NetCDF/fsync was 197.079→195.681 ms
and 218.180→227.692 ms, respectively. These are host output measurements under
shared load, not forecast timings. The additional 3/8 REAL volumes total
0.931/2.484 MB per frame and share their existing State arrays; all 375 common
serialized variables remain byte-exact. The two changed IO modules are outside
the actual 267-file traced-source closure, so this change does not alter the
model's source fingerprint. The official native-facade Swiss jaxpr remains
`2bf12cb8`. Evidence and the explicit residual inventory are in
`proofs/t_mpout/QUALIFICATION.json`.
