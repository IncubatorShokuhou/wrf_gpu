# MERGE — lane/t-heavytests at 75ba9e1b9

The sole FIX finding is resolved. Heavy tests now receive an explicit skip marker with the resource requirement and `--run-heavy`/24G instruction. The default `-q -rs` run visibly reports that reason. Subprocess controls assert it for default and `-m heavy` selection; opt-in still executes the genuine assertion sentinel.

Independent final changed-path check: 16 PASS, 1 expected heavy SKIP, 0 errors, RSS 181,672 KiB, zero swap, official6G slot, CPU2,3 nice19, JAX CPU, cache disabled. The existing unregistered `slow` warning is unchanged.

SRC/scripts/data/pyproject and all four numerical harness files are BYTE-identical to reviewed2d3a593a5. SRC is also identical to current main. Carry forward the independent 22 positive checks and two genuine mutant assertion kills, SMS2.965GiB/KF5.065GiB under6G, unchanged17-output pristine oracle/tolerances, and source-bound seven-sibling24G evidence. No redundant numerical/GPU execution.

Performance/memory conclusions remain unchanged: CPU test compilation trades work for lower SMS peak; KF avoids unused child initialization. This correction adds only collection-time skip markers and small subprocess assertions; production launches/traffic/allocations/compile behavior are unchanged.

Final receipts/source-byte admission: `FINAL_REVIEW.json` and `fix.{log,xml,time}`. Initial finding/evidence remain in `INITIAL_VERDICT.md` and `REVIEW.json`. No outstanding required fixes.
