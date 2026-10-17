# Legacy22 history inventory

The original WRF Registry allocates six moist species, six hydrometeor/CCN
number fields and graupel volume for legacy22. The added history fields are
`QNDROP` (State.Nc), `QNCCN` (Nn), `QNSNOW` (Ns), `QNGRAUPEL` (Ng), and
`QVGRAUPEL` (qvolg). Existing `QNICE` and `QNRAIN` retain Ni and Nr. Full,
operational and requested-subset files use the same names and original WRF
metadata. NSSL droplets use `QNDROP`; no `QNCLOUD` alias is emitted for22.

Hail mass, hail number and hail volume are absent. `REFL_10CM` and `REFD_MAX`
have no held operational source; these fields are omitted unless the caller
supplies actual diagnostics. The writer never invents zero diagnostics.
Nonfinite held values are refused before writing.

The history test uses original CPU-WRF18 samples of the shared Registry fields
to check transcription and metadata. Those samples establish no22 physical
trajectory claim. The independent22 physics oracles remain separate.
