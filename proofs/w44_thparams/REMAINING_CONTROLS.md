# Remaining ALISIOS Thompson controls — source audit

Nt_c has CPU and actual GPU component evidence; its product-root gate is open.
The other controls below have separate CPU evidence or a named implementation
blocker. They are not covered by the four-parameter GPU result.
All source locations refer to pristine `module_mp_thompson.F`.

| Control | Original definition/use | Dependency and next step |
|---|---|---|
| Ice coefficient `av_i` | Private REAL PARAMETER :161; active sedimentation uses :3681/:3686 | Pure scalar, no initialization/table dependency. Separate CPU qualification is in `AV_I.md` |
| Autoconversion threshold | Literal `rc(k) > 0.01e-3`, :2180 | CPU-qualified; cloud-water **density**, kg m⁻³, not mixing ratio. No tables. Values below R1 are refused. See `AUTOCONVERSION.md` |
| Autoconversion rate | Literal `zeta = 0.027*rc(k)*zeta1`, :2186 | CPU-qualified actual coefficient, not a relative factor; zero is allowed, joint WRF limiters retained. See `AUTOCONVERSION.md` |
| Rain coefficient `av_r` | Private REAL PARAMETER :143; init :786–788/:801, sedimentation :3618/:3626 | CPU-qualified with all four dependent initializers; see `AV_R.md`. Table mentions at :4187/:4403 are **commented out**; active table rain velocities use the fixed Gunn–Kinzer polynomial |
| Snow coefficient `av_s` | Private REAL PARAMETER :146; init :794/:797/:805/:809, rates :2766, sedimentation :3721 | Also changes qr_acr_qs integration bounds/relative velocities (:4407/:4410), Efsw (:4933), and aerosol collection (:4979). Changing only its sedimentation use does not reproduce a changed av_s PARAMETER |

The exact snow blocker is table dependence: the rain/snow collection integral
uses nonlinear velocity crossings and av_s-dependent integration limits. A
linear scaling of existing table entries is not WRF-faithful. A changed-WRF
reference must regenerate the affected tables, since their filenames carry
no coefficient identity and a normal init can read stale default tables.

A sedimentation-only snow multiplier is a different, explicit model extension.
It would require an agreed definition and a correspondingly modified Fortran
reference; it must not be presented as runtime `av_s`. No such substitution is
implemented here. No extra warm-cache or performance claim is made.
