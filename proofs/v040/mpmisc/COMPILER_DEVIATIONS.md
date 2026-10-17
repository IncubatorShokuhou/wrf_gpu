# Ferrier CPU compiler qualification

Candidate qualification is experimental. GPU execution, native E41, forecast
D6 and station validation remain pending. Literal shared-equation REAL32
execution with scalar libm expf/logf/powf is BYTE against all public outputs of
36 independent original-WRF columns, including GID 1/2, 1/18/54/60 s steps,
2/3/4 successive own steps, active rime-factor carry, strong cloud
autoconversion, heavy rain and rime factor 20 (MP30).

The ordinary CPU compiler contracts FPVS's `50*T-8999` table index into FMA.
Original-WRF native-FMA and independent +/-1 input-ULP ensembles were measured;
all nine recorded forcing vectors of the FMA arm were byte-equal to the generic
arm. On the actual Pallas CPU interpreter (MP22), a few results extend beyond
the union of those independently measured intervals:

| field | maximum excess beyond interval | units |
|---|---:|---|
| QVAPOR | 2.561137080192566e-9 | kg/kg |
| QCLOUD | 1.862645149230957e-9 | kg/kg |
| QICE / CWM | 3.725290298461914e-9 | kg/kg |
| QRAIN | 9.313225746154785e-10 | kg/kg |
| QRIMEF | 7.450580596923828e-9 | kg/kg |
| F_ICE_PHY | 5.960464477539063e-8 | dimensionless |
| F_RIMEF_PHY | 9.5367431640625e-6 | dimensionless |

These are disclosed compiled deviations under the manager's standing compiler
policy, not passes of the original strict 128-eps gate or of the raw ULP
interval. No field is clipped or repaired to obtain agreement. D6 decides
forecast qualification. MP22 is the predecessor fixed-iteration artifact.
MP30 checked the early-exit candidate's actual Pallas CPU executable against
the sealed original-WRF intervals and the exact predecessor excess ceilings:
all six original groups pass (32.73 s including compilation and test overhead).
MP35 separately checked all 12 new stress columns against ten-input-ULP
and original native-FMA intervals. All outputs are finite; stress_mp15 is
entirely inside the independent intervals, while the other two groups have
the small excesses included above (7 and 4 cells). Native CPU compilation
plus 3/2/4 calls took 12.29/9.03/18.37 s. Original physics source and forcing
bytes were verified; no GPU speed claim follows.

Receipts: `<USER_HOME>/wrf_gpu2_lanes/s2-mpmisc/MP22/bounds.json`,
`MP25/byte.json`, `MP26/gate.json`, `mp30_literal.log` and
`mp30_compiled.log`, and `MP35/bounds.json`. Tracked
`ferrier_compiler_bounds.npz/json` and `ferrier_stress_compiler_bounds.npz/json` contain only
the original-WRF interval arrays and provenance, with the predecessor excesses
explicitly separated. Tests also reject a 1% latent-heating coefficient mutant.
The legacy v018 Ferrier oracle used
GSMDT=60 minutes with DT=60 seconds; it is retained as historical evidence and
superseded by the MPDT=0 caller references in tests/v040/mpmisc.
