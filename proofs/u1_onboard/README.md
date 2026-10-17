# Onboarding CPU qualification

The source-bound receipt in `receipt.json` records the lane's CPU gates.

- Actual GFS → original WPS/WRF 4.7.1 `real.exe` → public CLI dry-run passed for Tenerife, Alps and Sydney (40×40×44, 9 km, 3 h).
- The local Docker image passed the same chain, with zero JAX backend initializations. Its 19 constrained JAX/CUDA/Triton packages match the captured release stack.
- Final rebased source passed 53 onboarding/CLI/verification-consumer checks; the eight shared consumer files passed 249 checks. The forced missing-field-PASS mutant failed a genuine assertion.
- All 258 protected model blobs and the 229-file traced source/cache identity are unchanged. No numerical model validation was repeated for the final metadata-only fixes.
- Actual IFS soil GRIB uses an encoding incompatible with the bundled WPS Vtable. IFS availability is complete; AIFS lacks soil layers 3/4 and snow water equivalent. Both automatic preparation paths are explicitly refused.

GPU execution, fresh-region forecast fidelity and consumer-card size limits remain unmeasured. The memory profiles are labelled inferred. The future GPU smoke plan is recorded in the artifact directory; it requires a manager window and official locking.

Reproduce the CPU workflow with `docs/onboarding.md`. Runtime code and container assets are tracked; the final tests ran from a clean Git archive. Large disposable contexts, snapshots and test caches were removed after use. Small source receipts and prepared cases remain for the critic and future GPU smoke. No image was pushed.
