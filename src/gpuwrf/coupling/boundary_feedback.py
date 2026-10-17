"""Child->parent two-way nest feedback operators.

This module ports the v0.5.0 WRF ``copy_fcn`` feedback operator onto the current
v0.10.0 state contract.  It is intentionally separate from the default
one-way live nesting path: callers must opt in with an explicit feedback gate.

The implemented geometric core is the WRF odd-ratio area average:

* mass fields: each parent cell receives the mean of the ``ratio**2`` child mass
  cells that tile it;
* U faces: each parent U face receives the mean of the coincident child U faces
  along ``j``;
* V faces: each parent V face receives the mean of the coincident child V faces
  along ``i``.

The child specified boundary ring is excluded from the feedback overlap.  The
per-call device op is a static ``jnp.take`` plus mean plus scatter; the gather
indices are precomputed once from the static domain geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.state import State


@dataclass(frozen=True)
class FeedbackWeights:
    """Static WRF ``copy_fcn`` gather plan for one horizontal staggering."""

    parent_lin: jax.Array
    child_lin: jax.Array
    pwe: int
    psn: int
    cwe: int
    csn: int
    stencil: int
    ratio: int
    stagger: str

    def tree_flatten(self):
        return (self.parent_lin, self.child_lin), (
            self.pwe,
            self.psn,
            self.cwe,
            self.csn,
            self.stencil,
            self.ratio,
            self.stagger,
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        parent_lin, child_lin = children
        pwe, psn, cwe, csn, stencil, ratio, stagger = aux
        return cls(parent_lin, child_lin, pwe, psn, cwe, csn, stencil, ratio, stagger)


jax.tree_util.register_pytree_node_class(FeedbackWeights)


@dataclass(frozen=True)
class FeedbackSmoother:
    """Static WRF ``sm121`` 1-2-1 feedback-zone smoother plan for one staggering.

    The smoother runs on the PARENT (coarse) grid AFTER the child overlap has been
    fed back, over exactly the WRF ``sm121`` interior of the nest-overlap region.
    The interior bounds are 0-based half-open ``[lo, hi)`` slices into the parent
    field; ``stagger`` controls the WRF ``istag``/``jstag`` end trim.
    """

    # 0-based inclusive..exclusive interior bounds on the parent grid (the cells
    # that WRF's sm121 actually overwrites).
    i_lo: int
    i_hi: int
    j_lo: int
    j_hi: int
    stagger: str

    def tree_flatten(self):
        return (), (self.i_lo, self.i_hi, self.j_lo, self.j_hi, self.stagger)

    @classmethod
    def tree_unflatten(cls, aux, children):
        del children
        i_lo, i_hi, j_lo, j_hi, stagger = aux
        return cls(i_lo, i_hi, j_lo, j_hi, stagger)


jax.tree_util.register_pytree_node_class(FeedbackSmoother)


@dataclass(frozen=True)
class StateFeedbackWeights:
    """Feedback plans for mass, U-face, and V-face leaves of one nest edge.

    Each ``*_smooth`` member is the matching WRF ``sm121`` feedback-zone smoother
    for that staggering; ``smooth_option`` selects the WRF ``smooth_option``
    (0 = none, 1 = 1-2-1 ``sm121``, 2 = ``smdsm`` smooth/desmooth). These default to ``None``
    only for legacy/hand-built weights; :func:`build_state_feedback_weights`
    always populates them.
    """

    mass: FeedbackWeights
    u: FeedbackWeights
    v: FeedbackWeights
    mass_smooth: FeedbackSmoother | None = None
    u_smooth: FeedbackSmoother | None = None
    v_smooth: FeedbackSmoother | None = None
    smooth_option: int = 1
    point: FeedbackWeights | None = None

    def tree_flatten(self):
        children = (self.mass, self.u, self.v, self.point)
        aux = (self.mass_smooth, self.u_smooth, self.v_smooth, int(self.smooth_option))
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        mass, u, v, point = children
        mass_smooth, u_smooth, v_smooth, smooth_option = aux
        return cls(
            mass=mass,
            u=u,
            v=v,
            mass_smooth=mass_smooth,
            u_smooth=u_smooth,
            v_smooth=v_smooth,
            smooth_option=smooth_option,
            point=point,
        )


jax.tree_util.register_pytree_node_class(StateFeedbackWeights)


@dataclass(frozen=True)
class ConservationResult:
    """Integral conservation diagnostic for one feedback field."""

    leaf: str
    stagger: str
    n_cells: int
    child_overlap_integral: float
    parent_overlap_integral: float
    abs_residual: float
    rel_residual: float
    conserved: bool


def _parent_child_extents(
    *,
    parent_we: int,
    parent_sn: int,
    child_we: int,
    child_sn: int,
    stagger: str,
) -> tuple[int, int, int, int]:
    pwe = int(parent_we) + (1 if stagger == "U" else 0)
    psn = int(parent_sn) + (1 if stagger == "V" else 0)
    cwe = int(child_we) + (1 if stagger == "U" else 0)
    csn = int(child_sn) + (1 if stagger == "V" else 0)
    return pwe, psn, cwe, csn


def build_feedback_weights(
    *,
    parent_grid_ratio: int,
    i_parent_start: int,
    j_parent_start: int,
    parent_we: int,
    parent_sn: int,
    child_we: int,
    child_sn: int,
    stagger: str = "",
    spec_zone: int = 1,
) -> FeedbackWeights:
    """Precompute WRF ``copy_fcn`` child->parent feedback weights.

    ``parent_we``/``parent_sn`` and ``child_we``/``child_sn`` are mass-grid
    horizontal extents.  ``stagger`` is ``""`` for mass/w fields, ``"U"`` for
    U faces, and ``"V"`` for V faces.
    """

    if stagger not in ("", "U", "V"):
        raise ValueError(f"unknown feedback stagger {stagger!r}")
    ratio = int(parent_grid_ratio)
    if ratio <= 1:
        raise ValueError(f"feedback requires parent_grid_ratio > 1, got {ratio}")
    if ratio % 2 == 0:
        raise NotImplementedError("feedback currently supports the WRF odd-ratio copy_fcn branch")

    ipos = int(i_parent_start)
    jpos = int(j_parent_start)
    sz = int(spec_zone)
    pwe, psn, cwe, csn = _parent_child_extents(
        parent_we=parent_we,
        parent_sn=parent_sn,
        child_we=child_we,
        child_sn=child_sn,
        stagger=stagger,
    )

    istag = 0 if stagger == "U" else 1
    jstag = 0 if stagger == "V" else 1
    n_parent_i = int(child_we) // ratio
    n_parent_j = int(child_sn) // ratio

    cj_lo = jpos + sz
    cj_hi = jpos + n_parent_j - jstag - sz
    ci_lo = ipos + sz
    ci_hi = ipos + n_parent_i - istag - sz
    stencil = ratio * ratio if stagger == "" else ratio

    parent_lin: list[int] = []
    child_lin: list[list[int]] = []
    for cj in range(cj_lo, cj_hi + 1):
        for ci in range(ci_lo, ci_hi + 1):
            if stagger == "":
                ni = (ci - ipos) * ratio + ratio // 2 + 1
                nj = (cj - jpos) * ratio + ratio // 2 + 1
                donors = []
                for jj in range(ratio):
                    jp = jj - ratio // 2
                    for ii in range(ratio):
                        ip = ii - ratio // 2
                        donors.append((nj + jp - 1) * cwe + (ni + ip - 1))
            elif stagger == "U":
                ni = (ci - ipos) * ratio + 1
                nj = (cj - jpos) * ratio + ratio // 2 + 1
                donors = []
                for jj in range(ratio):
                    jp = jj - ratio // 2
                    donors.append((nj + jp - 1) * cwe + (ni - 1))
            else:
                ni = (ci - ipos) * ratio + ratio // 2 + 1
                nj = (cj - jpos) * ratio + 1
                donors = []
                for ii in range(ratio):
                    ip = ii - ratio // 2
                    donors.append((nj - 1) * cwe + (ni + ip - 1))

            for donor in donors:
                drow, dcol = divmod(donor, cwe)
                if not (0 <= drow < csn and 0 <= dcol < cwe):
                    raise IndexError(
                        f"feedback donor out of bounds: child=({drow},{dcol}) extent=({csn},{cwe})"
                    )
            parent_lin.append((cj - 1) * pwe + (ci - 1))
            child_lin.append(donors)

    child_arr = np.asarray(child_lin, dtype=np.int32).reshape((-1, stencil))
    return FeedbackWeights(
        parent_lin=jnp.asarray(np.asarray(parent_lin, dtype=np.int32)),
        child_lin=jnp.asarray(child_arr),
        pwe=pwe,
        psn=psn,
        cwe=cwe,
        csn=csn,
        stencil=stencil,
        ratio=ratio,
        stagger=stagger,
    )


def build_feedback_smoother(
    *,
    parent_grid_ratio: int,
    i_parent_start: int,
    j_parent_start: int,
    child_we: int,
    child_sn: int,
    stagger: str = "",
) -> FeedbackSmoother:
    """Precompute the WRF ``sm121`` feedback-zone smoother interior for one stagger.

    WRF (``share/interp_fcn.F::sm121``, serial single-tile) smooths the PARENT
    grid over the nest-overlap interior::

        i in [ipos+2 .. ipos+(child_we)/nri-2-istag]  (1-based, inclusive)
        j in [jpos+2 .. jpos+(child_sn)/nrj-2-jstag]   (1-based, inclusive)

    with ``istag = 0`` for an x-staggered (U) field, ``1`` otherwise; ``jstag = 0``
    for a y-staggered (V) field, ``1`` otherwise.  We return those bounds converted
    to 0-based half-open Python slices into the parent field.
    """

    if stagger not in ("", "U", "V"):
        raise ValueError(f"unknown feedback stagger {stagger!r}")
    ratio = int(parent_grid_ratio)
    ipos = int(i_parent_start)
    jpos = int(j_parent_start)
    n_parent_i = int(child_we) // ratio
    n_parent_j = int(child_sn) // ratio
    istag = 0 if stagger == "U" else 1
    jstag = 0 if stagger == "V" else 1

    # WRF 1-based inclusive bounds.
    i_lo_1 = ipos + 2
    i_hi_1 = ipos + n_parent_i - 2 - istag
    j_lo_1 = jpos + 2
    j_hi_1 = jpos + n_parent_j - 2 - jstag

    # 0-based half-open slices ([lo, hi) over the cells WRF overwrites).
    i_lo = i_lo_1 - 1
    i_hi = i_hi_1  # inclusive 1-based -> exclusive 0-based == (i_hi_1-1)+1
    j_lo = j_lo_1 - 1
    j_hi = j_hi_1
    return FeedbackSmoother(
        i_lo=int(max(i_lo, 0)),
        i_hi=int(max(i_hi, i_lo)),
        j_lo=int(max(j_lo, 0)),
        j_hi=int(max(j_hi, j_lo)),
        stagger=str(stagger),
    )


def point_feedback_weights(mass: FeedbackWeights) -> FeedbackWeights:
    """WRF copy_fcnm/copy_fcni odd-ratio point sample (no averaging).

    WRF chooses fine index `(coarse-ipos)*ratio + 2`, including ratio 5;
    it is the second donor in each direction, not a generic geometric center.
    Build once outside the timestep loop.
    """
    from dataclasses import replace
    # copy_fcnm: ni=(ci-ipos)*nri+istag+1, likewise j. Along
    # the face the area plan already has offset 1; across it choose donor 1.
    donor = 1 if mass.stagger else mass.ratio + 1
    return replace(mass,child_lin=mass.child_lin[:,donor:donor+1],stencil=1)


def build_state_feedback_weights(
    *,
    parent_grid_ratio: int,
    i_parent_start: int,
    j_parent_start: int,
    parent_grid: GridSpec,
    child_grid: GridSpec,
    spec_zone: int = 1,
    smooth_option: int = 1,
) -> StateFeedbackWeights:
    """Build mass/U/V feedback weights + ``sm121`` smoothers for a ``State`` edge."""

    if int(smooth_option) not in (0, 1, 2):
        raise ValueError(f"smooth_option must be 0, 1 or 2, got {smooth_option}")

    common = dict(
        parent_grid_ratio=int(parent_grid_ratio),
        i_parent_start=int(i_parent_start),
        j_parent_start=int(j_parent_start),
        parent_we=int(parent_grid.nx),
        parent_sn=int(parent_grid.ny),
        child_we=int(child_grid.nx),
        child_sn=int(child_grid.ny),
        spec_zone=int(spec_zone),
    )
    smooth_common = dict(
        parent_grid_ratio=int(parent_grid_ratio),
        i_parent_start=int(i_parent_start),
        j_parent_start=int(j_parent_start),
        child_we=int(child_grid.nx),
        child_sn=int(child_grid.ny),
    )
    mass = build_feedback_weights(stagger="", **common)
    return StateFeedbackWeights(
        mass=mass,
        u=build_feedback_weights(stagger="U", **common),
        v=build_feedback_weights(stagger="V", **common),
        mass_smooth=build_feedback_smoother(stagger="", **smooth_common),
        u_smooth=build_feedback_smoother(stagger="U", **smooth_common),
        v_smooth=build_feedback_smoother(stagger="V", **smooth_common),
        smooth_option=int(smooth_option),
        point=point_feedback_weights(mass),
    )


@jax.jit
def _feedback_values(child: jax.Array, weights: FeedbackWeights) -> jax.Array:
    child_idx = weights.child_lin.reshape(-1)
    if child.ndim == 3:
        z = int(child.shape[0])
        flat = child.reshape(z, weights.csn * weights.cwe)
        gathered = jnp.take(flat, child_idx, axis=1)
        gathered = gathered.reshape(z, weights.parent_lin.shape[0], weights.stencil)
    elif child.ndim == 2:
        flat = child.reshape(weights.csn * weights.cwe)
        gathered = jnp.take(flat, child_idx, axis=0)
        gathered = gathered.reshape(weights.parent_lin.shape[0], weights.stencil)
    else:
        raise ValueError(f"feedback expects a 2D or 3D child field, got {child.shape}")
    # copy_fcn sums WEIGHTED donors in i-fast, j-slow order. A reduce-mean
    # changes REAL rounding, particularly for nearly cancelling perturbations.
    # One leaf-local executable keeps the ordered arithmetic off eager dispatch.
    scale = jnp.asarray(1.0 / weights.stencil, dtype=child.dtype)
    values = jnp.zeros(gathered.shape[:-1], dtype=child.dtype)
    for donor in range(weights.stencil):
        term = jax.lax.optimization_barrier(scale * gathered[..., donor])
        values = values + term
    return values


def feedback_to_parent_grid(child: jax.Array, weights: FeedbackWeights) -> jax.Array:
    """Return a parent-shaped feedback product, zero outside the overlap."""

    values = _feedback_values(child, weights)
    if child.ndim == 3:
        z = int(child.shape[0])
        out = jnp.zeros((z, weights.psn * weights.pwe), dtype=child.dtype)
        out = out.at[:, weights.parent_lin].set(values)
        return out.reshape(z, weights.psn, weights.pwe)
    out = jnp.zeros((weights.psn * weights.pwe,), dtype=child.dtype)
    out = out.at[weights.parent_lin].set(values)
    return out.reshape(weights.psn, weights.pwe)


def feedback_mask(weights: FeedbackWeights) -> jax.Array:
    """Parent-grid mask with one where feedback writes."""

    mask = jnp.zeros((weights.psn * weights.pwe,), dtype=jnp.float64)
    mask = mask.at[weights.parent_lin].set(1.0)
    return mask.reshape(weights.psn, weights.pwe)


def apply_feedback(
    parent: jax.Array,
    child: jax.Array,
    weights: FeedbackWeights,
    *,
    feedback: bool = True,
) -> jax.Array:
    """Write child-overlap averages onto the parent field.

    With ``feedback=False`` this returns ``parent`` unchanged, which is the
    baseline one-way nesting gate.
    """

    if not bool(feedback):
        return parent
    values = _feedback_values(child, weights)
    if parent.ndim == 3:
        z = int(parent.shape[0])
        flat = parent.reshape(z, weights.psn * weights.pwe)
        flat = flat.at[:, weights.parent_lin].set(values.astype(parent.dtype))
        return flat.reshape(z, weights.psn, weights.pwe)
    if parent.ndim == 2:
        flat = parent.reshape(weights.psn * weights.pwe)
        flat = flat.at[weights.parent_lin].set(values.astype(parent.dtype))
        return flat.reshape(weights.psn, weights.pwe)
    raise ValueError(f"apply_feedback expects a 2D or 3D parent field, got {parent.shape}")


def sm121_smooth(field: jax.Array, smoother: FeedbackSmoother) -> jax.Array:
    """WRF ``sm121`` 1-2-1 feedback-zone smoother on a parent field.

    Faithful port of ``share/interp_fcn.F::sm121`` (serial single-tile,
    ``smooth_passes = 1``).  ``field`` is the PARENT-grid leaf, 2D ``(ny, nx)`` or
    3D ``(nz, ny, nx)``; axis ``-1`` is the WRF ``i`` (west-east) and axis ``-2``
    is the WRF ``j`` (south-north).  Only the nest-overlap interior is mutated; the
    intermediate ``cfldnew`` is the raw field with the j-pass applied over the
    interior (so the interior i-pass reads raw values just outside the j-region,
    exactly as WRF does with its 3-cell ``cfldnew`` halo).

    VRAM note (v0.13): an alternative interior-slab + ``jnp.concatenate``
    formulation was measured to be bit-identical but PEAK-WORSE under eager dispatch
    (the concat of the j-blend slab with the raw halo columns materialises more live
    buffers than the two ``.at[interior].set`` writes below, especially for the
    large d01->d02 overlap).  The full-field ``.at[].set`` form here is kept because
    it is the lower-peak eager form; see proofs/v013/twoway_vram.* for the A/B.
    """

    i_lo, i_hi = int(smoother.i_lo), int(smoother.i_hi)
    j_lo, j_hi = int(smoother.j_lo), int(smoother.j_hi)
    if i_hi <= i_lo or j_hi <= j_lo:
        return field  # degenerate overlap (no interior cells to smooth)

    arr = field
    ndim = arr.ndim
    if ndim == 2:
        arr = arr[jnp.newaxis, ...]
    elif ndim != 3:
        raise ValueError(f"sm121_smooth expects a 2D or 3D field, got {field.shape}")

    ny = arr.shape[-2]
    nx = arr.shape[-1]
    # WRF interior must have a one-cell stencil halo available on every side.
    if i_lo < 1 or i_hi > nx - 1 or j_lo < 1 or j_hi > ny - 1:
        # Region touches the array border with no halo: clamp the smoothing region
        # one cell inward so the 1-2-1 stencil never reads out of bounds.  This
        # never triggers for real Canary nests (the overlap sits inside the parent
        # with ample margin) but keeps the op total.
        i_lo = max(i_lo, 1)
        i_hi = min(i_hi, nx - 1)
        j_lo = max(j_lo, 1)
        j_hi = min(j_hi, ny - 1)
        if i_hi <= i_lo or j_hi <= j_lo:
            out = arr[0] if ndim == 2 else arr
            return out

    # j-pass: cfldnew = raw field, then overwrite the interior with the j 1-2-1.
    cfldnew = arr
    j_blend = 0.25 * (
        arr[:, j_lo + 1 : j_hi + 1, i_lo:i_hi]
        + 2.0 * arr[:, j_lo:j_hi, i_lo:i_hi]
        + arr[:, j_lo - 1 : j_hi - 1, i_lo:i_hi]
    )
    cfldnew = cfldnew.at[:, j_lo:j_hi, i_lo:i_hi].set(j_blend.astype(arr.dtype))

    # i-pass: overwrite the interior of the field with the i 1-2-1 of cfldnew.
    i_blend = 0.25 * (
        cfldnew[:, j_lo:j_hi, i_lo + 1 : i_hi + 1]
        + 2.0 * cfldnew[:, j_lo:j_hi, i_lo:i_hi]
        + cfldnew[:, j_lo:j_hi, i_lo - 1 : i_hi - 1]
    )
    out = arr.at[:, j_lo:j_hi, i_lo:i_hi].set(i_blend.astype(arr.dtype))

    if ndim == 2:
        return out[0]
    return out


def smdsm_smooth(field: jax.Array, smoother: FeedbackSmoother) -> jax.Array:
    """WRF option 2: j then i smoothing (0.50), then desmoothing (-0.52).

    Each directional pass reads the completed previous field. The untouched
    overlap halo is read again in every pass (share/interp_fcn.F::smdsm).
    """
    i0, i1, j0, j1 = smoother.i_lo, smoother.i_hi, smoother.j_lo, smoother.j_hi
    if i1 <= i0 or j1 <= j0:
        return field
    if field.ndim not in (2, 3):
        raise ValueError(f"smdsm_smooth expects a 2D or 3D field, got {field.shape}")
    if i0 < 1 or j0 < 1 or i1 >= field.shape[-1] or j1 >= field.shape[-2]:
        raise ValueError("feedback smoother requires a one-cell parent halo")
    arr = field
    for nu in (0.50, -0.52):
        center = arr[..., j0:j1, i0:i1]
        blend = center + nu * ((arr[..., j0+1:j1+1, i0:i1]
                               + arr[..., j0-1:j1-1, i0:i1]) * 0.5 - center)
        arr = arr.at[..., j0:j1, i0:i1].set(blend)
        center = arr[..., j0:j1, i0:i1]
        blend = center + nu * ((arr[..., j0:j1, i0+1:i1+1]
                               + arr[..., j0:j1, i0-1:i1-1]) * 0.5 - center)
        arr = arr.at[..., j0:j1, i0:i1].set(blend)
    return arr


@partial(jax.jit, static_argnames=("option",))
def _feedback_leaf(parent, child, weights, smoother, option):
    fed = apply_feedback(parent, child, weights)
    if smoother is not None and option:
        fed = (sm121_smooth if option == 1 else smdsm_smooth)(fed, smoother)
    return fed


def _base_pressure(state: State) -> jax.Array:
    return state.p_total - state.p_perturbation


def _base_geopotential(state: State) -> jax.Array:
    return state.ph_total - state.ph_perturbation


def _base_mu(state: State) -> jax.Array:
    return state.mu_total - state.mu_perturbation


def _apply_state_feedback_and_base(
    parent: State,
    child: State,
    weights: StateFeedbackWeights,
    *,
    feedback: bool = True,
    parent_base=None,
    child_base=None,
):
    """Apply optional two-way feedback to the current ``State`` contract.

    The default runtime passes ``feedback=False``. Registry-selected prognostic,
    moisture and number fields plus PB/PHB/MUB are fed back. Diagnostic pressure and
    QKE (distinct from WRF's advected qke_adv) stay held. Total aliases are rebuilt
    from the fed-back bases and perturbations.
    """

    if not bool(feedback):
        return parent, parent_base

    mass = weights.mass
    point = weights.point if weights.point is not None else point_feedback_weights(mass)
    option = int(getattr(weights, "smooth_option", 1))
    if option not in (0, 1, 2):
        raise ValueError(f"smooth_option must be 0, 1 or 2, got {option}")
    mass_sm = getattr(weights, "mass_smooth", None)
    u_sm = getattr(weights, "u_smooth", None)
    v_sm = getattr(weights, "v_smooth", None)
    base_mask = None
    if parent_base is not None:
        base_mask = jnp.zeros((mass.psn*mass.pwe,),dtype=bool).at[mass.parent_lin].set(True)
        base_mask = base_mask.reshape(mass.psn,mass.pwe)
        if option and mass_sm is not None:
            base_mask = base_mask.at[mass_sm.j_lo:mass_sm.j_hi,mass_sm.i_lo:mass_sm.i_hi].set(True)

    def _fb(parent_leaf, child_leaf, fb_weights, smoother):
        return _feedback_leaf(parent_leaf, child_leaf, fb_weights, smoother, option)

    u = _fb(parent.u, child.u, weights.u, u_sm)
    v = _fb(parent.v, child.v, weights.v, v_sm)
    w = _fb(parent.w, child.w, mass, mass_sm)
    # WRF feeds perturbation T, while State stores the full theta_m.
    theta_offset=jnp.asarray(300.,dtype=parent.theta.dtype)
    theta = _fb(parent.theta-theta_offset, child.theta-theta_offset, mass, mass_sm)+theta_offset
    qv = _fb(parent.qv, child.qv, mass, mass_sm)
    ph_pert = _fb(parent.ph_perturbation, child.ph_perturbation, mass, mass_sm)
    mu_pert = _fb(parent.mu_perturbation, child.mu_perturbation, mass, mass_sm)
    # Registry: p=irh (diagnostic, no feedback); pb/phb/mub=irhdus (feedback
    # AND smoothing). Prefer the resident authoritative base over total-prime.
    pphb = parent_base.phb if parent_base is not None else _base_geopotential(parent)
    cphb = child_base.phb if child_base is not None else _base_geopotential(child)
    pmub = parent_base.mub if parent_base is not None else _base_mu(parent)
    cmub = child_base.mub if child_base is not None else _base_mu(child)
    ppb = parent_base.pb if parent_base is not None else _base_pressure(parent)
    cpb = child_base.pb if child_base is not None else _base_pressure(child)
    # These resident bases are stored as DOUBLE by the port, but WRF declares
    # the feedback fields REAL. Round the operator to REAL, retain the storage
    # dtype and the exact parent values outside the feedback overlap.
    def _base_fb(pbase, cbase):
        if parent_base is None:
            return _fb(pbase, cbase, mass, mass_sm)
        result = _fb(pbase.astype(jnp.float32), cbase.astype(jnp.float32), mass, mass_sm)
        return jnp.where(base_mask, result.astype(pbase.dtype), pbase)
    phb = _base_fb(pphb, cphb)
    mub = _base_fb(pmub, cmub)
    pb = _base_fb(ppb, cpb)
    base = parent_base.replace(pb=pb, phb=phb, mub=mub) if parent_base is not None else None
    if base is not None and getattr(parent_base,"alb",None) is not None:
        if getattr(child_base,"alb",None) is None:
            raise ValueError("two-way feedback requires both resident WRF ALB fields")
        base=base.replace(alb=_base_fb(parent_base.alb,child_base.alb))

    # Rebuild each changed total once and share its transitional legacy alias.
    # Perturbation pressure is held; its total alias follows the updated PB.
    p_val = pb + parent.p_perturbation
    ph_val = phb + ph_pert
    mu_val = mub + mu_pert

    updates = {
        "u": u,
        "v": v,
        "w": w,
        "theta": theta,
        "qv": qv,
        "p_total": p_val,
        "p": p_val,
        "ph_perturbation": ph_pert,
        "ph_total": ph_val,
        "ph": ph_val,
        "mu_perturbation": mu_pert,
        "mu_total": mu_val,
        "mu": mu_val,
    }
    # v0.17 ADR-032 graupel/hail substrate (qh/Nh/qvolg/qvolh) and v0.16
    # aerosol-aware Thompson (nwfa/nifa) join the two-moment scalar feedback
    # loop. The hasattr guard makes each a no-op for any state that does not
    # carry the leaf, and for a run without them the leaves are zero
    # (_fb(0,0,...) == 0), so the nest feedback is byte-unchanged.
    for name in (
        "qc", "qr", "qi", "qs", "qg", "Ni", "Nr", "Ns", "Ng", "Nc", "Nn",
        "qh", "Nh", "qvolg", "qvolh", "nwfa", "nifa",
    ):
        if hasattr(parent, name) and hasattr(child, name):
            parent_value = getattr(parent, name)
            child_value = getattr(child, name)
            if parent_value is not None and child_value is not None:
                updates[name] = _fb(parent_value, child_value, mass, mass_sm)
    # Registry accumulated precipitation has `u` but NO `s` flag.
    for name in ("rain_acc", "rainc_acc", "snow_acc", "graupel_acc", "hail_acc"):
        pv, cv = getattr(parent, name, None), getattr(child, name, None)
        if pv is not None and cv is not None:
            updates[name] = _feedback_leaf(pv, cv, mass, None, 0)
    # Registry explicit u=(copy_fcnm), not the default area average.
    for name in ("t_skin", "soil_moisture", "xland", "lu_index"):
        pv, cv = getattr(parent, name, None), getattr(child, name, None)
        if pv is not None and cv is not None:
            updates[name] = _feedback_leaf(pv, cv, point, None, 0)
    if "xland" in updates and getattr(parent,"landmask",None) is not None:
        updates["landmask"]=(updates["xland"]<1.5).astype(parent.landmask.dtype)
    return parent.replace(_cast=False, **updates), base


def apply_state_feedback(parent: State, child: State, weights: StateFeedbackWeights,
                         *, feedback: bool = True) -> State:
    """Registry feedback on state fields; pressure is refreshed by the next step."""
    return _apply_state_feedback_and_base(parent, child, weights, feedback=feedback)[0]


def apply_carry_feedback(parent, child, weights: StateFeedbackWeights):
    """Feedback state, resident PB/PHB/MUB and unsmoothed WRF h_diabatic."""
    fed, base = _apply_state_feedback_and_base(
        parent.state, child.state, weights,
        parent_base=parent.base_state, child_base=child.base_state)
    updates = {"state": fed, "base_state": base}
    if parent.h_diabatic is not None and child.h_diabatic is not None:
        # Registry h_diabatic=rdu: no `s` flag, hence no smoother.
        updates["h_diabatic"] = _feedback_leaf(parent.h_diabatic, child.h_diabatic, weights.mass, None, 0)
    pl, cl = getattr(parent,"noahmp_land",None),getattr(child,"noahmp_land",None)
    if pl is not None and cl is not None:
        # registry.noahmp / EM_COMMON explicit copy_fcnm (ISNOW copy_fcni).
        # LAI, TAUSS, SMCWTD, QSFC, ZNT, ALBEDO/EMISS and runoff have NO u flag.
        point = weights.point if weights.point is not None else point_feedback_weights(weights.mass)
        fields=("tslb","smois","sh2o","isnow","tsno","snice","snliq","zsnso",
                "snowh","sneqv","sneqvo","albold","tv","tg","tah","eah",
                "canliq","canice","fwet","sai","cm","ch","t_skin")
        updates["noahmp_land"]=pl.replace(**{name:_feedback_leaf(getattr(pl,name),getattr(cl,name),point,None,0)
                                             for name in fields})
    from gpuwrf.coupling.feedback_aux import feedback_group
    ph,ch=getattr(parent,'history_diagnostics',None),getattr(child,'history_diagnostics',None)
    if ph is not None and ch is not None:
        updates['history_diagnostics']=ph._replace(**{name:_feedback_leaf(
            getattr(ph,name),getattr(ch,name),weights.mass,None,0)
            for name in ('t2','q2','u10','v10','psfc')})
    for name in ("land_history","energy_accumulators","feedback_history"):
        pv,cv=getattr(parent,name,None),getattr(child,name,None)
        if pv is not None and cv is not None:
            updates[name]=feedback_group(pv,cv,weights)
    return parent.replace(**updates)


def feedback_overlap_conservation(
    child: jax.Array,
    weights: FeedbackWeights,
    *,
    leaf: str,
    rel_tol: float = 1.0e-12,
) -> ConservationResult:
    """Check integral conservation over the feedback overlap."""

    values = _feedback_values(child, weights)
    child_idx = weights.child_lin.reshape(-1)
    if child.ndim == 3:
        z = int(child.shape[0])
        flat = child.reshape(z, weights.csn * weights.cwe)
        donors = jnp.take(flat, child_idx, axis=1).reshape(
            z, weights.parent_lin.shape[0], weights.stencil
        )
        child_sum = jnp.sum(donors)
        parent_sum = jnp.sum(values) * weights.stencil
    else:
        flat = child.reshape(weights.csn * weights.cwe)
        donors = jnp.take(flat, child_idx, axis=0).reshape(
            weights.parent_lin.shape[0], weights.stencil
        )
        child_sum = jnp.sum(donors)
        parent_sum = jnp.sum(values) * weights.stencil

    child_integral = float(child_sum)
    parent_integral = float(parent_sum)
    abs_residual = abs(parent_integral - child_integral)
    rel_residual = abs_residual / max(abs(child_integral), 1.0e-300)
    return ConservationResult(
        leaf=str(leaf),
        stagger=weights.stagger,
        n_cells=int(weights.parent_lin.shape[0]),
        child_overlap_integral=child_integral,
        parent_overlap_integral=parent_integral,
        abs_residual=abs_residual,
        rel_residual=rel_residual,
        conserved=bool(rel_residual <= float(rel_tol)),
    )


__all__ = [
    "ConservationResult",
    "FeedbackSmoother",
    "FeedbackWeights",
    "StateFeedbackWeights",
    "apply_feedback",
    "apply_state_feedback",
    "apply_carry_feedback",
    "build_feedback_smoother",
    "build_feedback_weights",
    "build_state_feedback_weights",
    "feedback_mask",
    "feedback_overlap_conservation",
    "feedback_to_parent_grid",
    "sm121_smooth",
    "smdsm_smooth",
    "point_feedback_weights",
]
