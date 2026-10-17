# MERGE — compiler-guard follow-up

MERGE `33d0bf38dc55ed4a18e4651eaa0a897352b4164d`. This closes the sole FIX on `60675e705` when the follow-up is applied. Scope is only the compiler availability delta; the previous final-batch verdict and evidence remain immutable.

[M CPU/source] 10/10 probes PASS, including three real original-WRF harness builds with FC unset and no PATH gfortran, configured through WRF_PRISTINE_FC, WRF_ENV and GPUWRF_WRF_ENV. Three supported-setting probes also prove present invalid objects fail rather than skip. Actual missing objects/table/compiler retain their explicit skips. Restoring only the old guard yields 3/3 genuine AssertionErrors with zero setup errors or false numerical failure labels.

The eight original assertion ASTs, all skip calls/reasons, and the entire build-and-assertion tail are exact. Production source, scripts, data and fixtures have zero delta; the unchanged real builder blob is740426aad3e46370d846781e7c3b85a5ed6febaf. Lookup follows the builder: executable pristine override only on the pristine-object fallback, empty-FC environment-directory fallback (including alias), then PATH fallback for empty/non-executable FC. Relative -x checks use the build working directory.

Immutable snapshot33d0bf38d, official6GiB/no-swap RAM slot, JAX CPU, timeout600, QUIET guard, private cache/basetemp, cores8,9,24,25 nice19. Peak measured reviewer RSS183396KiB; all finished scratch cleaned. No production performance/memory/compile change by source identity; no GPU or forecast claims. Full CPU suite/release remains incomplete.

Receipts: R08/R09 JSON and lossless gzip XML/logs; compiler build logs and standalone regression/mutant probes included. Raw and compressed SHA bindings are in RAW_RECEIPTS.json; all archived files are sealed by SHA256.json.
