# Upstream WRF 4.7.1 observations — YSU (lane t-ysuqc)

Policy (wave S1): pristine bodies are never edited; the oracle is built with compiler-defined
init (`-finit-real=zero`) and a sensitivity build (`-finit-real=inf`).

1. **RQIBLTEN unassigned when FLAG_QI=.false.** `phys/physics_mmm/bl_ysu.F90:1205-1220` writes
   `qitnp` (INTENT(OUT)) only `if(f_qi)`; `module_bl_ysu.F:467` copies the undefined tile array into
   RQIBLTEN. `zero` vs `inf` builds differ only there (2,464 cells, all FLAG_QI false Kessler rows).
   Impact: none — WRF's tendency update consumes RQIBLTEN only for schemes with P_QI. The port
   returns no `qi` rate when QI3D is absent. Same applies to RQCBLTEN for FLAG_QC false.
2. **Possible out-of-bounds reads (not reached by any oracle column).** The cloud-top branch reads
   `thlix/thx/qvx/qcxl(i,kpbl(i)+1)` (`:851`), out of bounds if kpbl = kte; the top-down loop can
   leave pblflg true with kpbl = 1 if the only unstable theta-li level is kte-1, after which
   `za(i,kpbl-1)` (`:765`) reads `za(i,0)`. The port clips these indices. Neither occurs in the 218 cases.
3. **get_pblh exit safeguard** (`:1661`, `:1685`) overwrites a PBL height found at k = kte-2 with
   zw(2); real columns never get there, the port reproduces it literally (unit test against a
   literal do-while transcription).
