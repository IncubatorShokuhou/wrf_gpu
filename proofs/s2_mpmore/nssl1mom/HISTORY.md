# Legacy19/21 history inventory

Original WRF legacy19 is one-moment microphysics with hail mass and predicted
graupel density. Its extra history fields are `QHAIL` (State.qh), `QNCCN`
(Nn) and `QVGRAUPEL` (qvolg). CCN is transported as a passive scalar and held
unchanged by the microphysics after the original first-step zeroing. No
hydrometeor numbers or hail-volume scalar are predicted. Legacy21 has six
bulk moist species and none of these extra scalars.

Legacy19 uses the original default CCN1. WRF does not override an explicit
CCN0 for19; that distinct Registry configuration is refused by this port.

Full, operational and requested-subset files suppress QNDROP/QNCLOUD, QNICE,
QNRAIN, QNSNOW, QNGRAUPEL, QNHAIL and QVHAIL for both variants. Legacy21 also
suppresses QHAIL, QNCCN and QVGRAUPEL. Uniform State slots used by other schemes
must not create fabricated variables for these Registry configurations.
Actual represented values reject NaN/Inf before writing.

REFL_10CM and REFD_MAX lack a held operational source and are omitted unless
the caller supplies actual diagnostics. Original NSSL one-moment initialization
does not enable effective cloud/ice/snow radius handoff to RRTMG.

Tests exercise emitted full/operational/subset files using original CPU-WRF18
samples of shared Registry fields for byte transcription and metadata only.
Those samples do not establish a19/21 physical trajectory claim. Independent
19/21 REAL4/REAL8 column savepoints qualify the numerical microphysics.
