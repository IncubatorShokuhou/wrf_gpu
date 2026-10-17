# CAM-MG GPU refusal fallback

Manager-selected fallback only: GPU `mp_physics=11` refuses before JAX imports
in the CLI and before backend/device initialization in API calls (imports alone
are permitted by the manager's review scope). CPU remains admitted. This patch
does not fix the native GPU precipitation bug.

`QUALIFICATION.json` binds the measured source, all 353 completed shared test
nodes, the 12 genuine deletion-mutant failures with one healthy control, both
full frozen default jaxpr hashes, and 18 raw receipts by SHA-256.
The default physics/dynamics source is unchanged. No GPU fidelity/speed/VRAM
claim is made. Incomplete OOM/time-cap attempts and the initial mutation setup
refusal are retained and excluded from passing counts.

Tests use CPU cores 12/13, nice 19, JAX cache disabled. Heavy shared consumers
use the official RAM wrapper and per-test cache cleanup; manager-authorized
12 GiB closes the intrinsic mp38/NTU requirements. Completed nodes are reused
across resumed runs. All tests and assertions are unchanged.

The support-matrix row proposal and integration conditions are in the lane
handoff. Current known issue: `KNOWN_ISSUES.md#cam-mg-gpu-precipitation`.

## Entry-order repair after the critic's FIX verdict

`FIX_QUALIFICATION.json` supersedes the original readiness decision under the
manager's explicit scope: strict pre-JAX CLI refusal; API imports allowed, but
admission precedes any backend query, DFI, loading, prefetch, restore or compile.
Public/private nested and daily loaders check all case/domain selections;
batch environment/configuration inputs are admitted before the first model load.
CLI case options use their last occurrence, including equals forms; abbreviated
`--namelist`/`--input-dir` spellings fail closed before JAX on GPU. An explicit
safe namelist cannot hide mp11 in the actual case or ensemble directories.
CPU admission is a host no-op and all model-step arithmetic remains unchanged.
