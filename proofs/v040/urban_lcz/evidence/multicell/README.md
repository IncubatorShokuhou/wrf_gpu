# Pristine WRF multi-cell diagnostic (RV03)

This is a retained diagnostic, not a whole-tile parity PASS. The original critic
files, source manifest, drivers, verbatim caller, anisotropic and isotropic
savepoints and result are preserved byte-for-byte and bound by SEAL.json.
The original raw isotropic dump was deleted after scoring; its historical hash
remains in the unchanged critic receipt. No original reference was modified.

Twenty-five regimes ×30 steps use two identical cells in one pristine
noahmp_urban tile. The second cell receives the first cell's complete array
inputs/state before the call. All five scored first-cell fields are BYTE with
the frozen one-cell fixture. Anisotropic LF=(.1,.3,.5,.7) produces material
second-cell differences, including HFX 6.273636 W/m2, U10 0.992088 m/s and
TSK 0.301880 K. Isotropic LF=(.1,.1,.1,.1) produces exactly zero differences in
all five fields. Scored values are finite. Bounds are unchanged and the five
anisotropic differences exceed the existing one-cell REAL4 bounds.

Reproduction sources are multicell.py (expands the existing build_oracle.py
driver to two cells) and score_multicell.py (scores and builds the isotropic
control). They retain their historical absolute critic paths as part of the
seal. To rerun, use a separate lane artifact directory, replace the historical
snapshot path with the source tree and the critic output root with that artifact
directory in private copies of these scripts. Run them sequentially with
JAX_PLATFORMS=cpu, the allowed CPU affinity/nice19, QUIET guard and timeout;
remove the disposable Fortran build tree after scoring. The dependency is
proofs/v040/urban_lcz/build_oracle.py and its retained pristine module/table
sources, flags and original one-cell fixtures. driver.f90 and
driver_isotropic.f90 are also retained for direct compilation with manifest
flags and the unmodified pristine modules. Do not run the historical scripts
against another lane's directories or reset outputs before each cell.
