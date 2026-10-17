"""Registry registry.stoch p2c fields: parent-held full SINT, never child RNG."""
import jax

from gpuwrf.nesting.interp import interp_sint_full


def force_stochastic(child, parent, weights, ratio):
    patterns = list(child.patterns)
    for index, target in enumerate(patterns):
        source = parent.patterns[index]
        if target is None or source is None:
            continue
        source_fields = source.field if index == 0 else (source.field,)
        target_fields = target.field if index == 0 else (target.field,)
        fields = []
        for component,(coarse,fine) in enumerate(zip(source_fields,target_fields)):
            xstag,ystag = index == 0 and component == 0, index == 0 and component == 1
            plan = weights.u if xstag else weights.v if ystag else weights.mass
            interpolated = interp_sint_full(coarse,plan,parent_grid_ratio=ratio,xstag=xstag,ystag=ystag)
            if interpolated.shape[0] != fine.shape[0]:
                raise ValueError("stochastic parent/child vertical layouts differ")
            # interp_fcn_sint writes only nide-istag/njde-jstag. The scalar
            # high-side memory ghosts remain their child initialization values.
            fields.append(fine.at[:,:interpolated.shape[1],:interpolated.shape[2]].set(interpolated))
        patterns[index] = target._replace(field=tuple(fields) if index == 0 else fields[0])
    return child._replace(patterns=tuple(patterns))


_COMPILED = {}


def force_child_stochastic(child_carry,parent_carry,weights,ratio,batched=False):
    child,parent = child_carry.stochastic,parent_carry.stochastic
    if child is None or parent is None:
        return child_carry
    key = (id(weights),int(ratio),bool(batched))
    hit = _COMPILED.get(key)
    if hit is None or hit[0] is not weights:
        one = lambda c,p: force_stochastic(c,p,weights,int(ratio))
        hit = weights,jax.jit(jax.vmap(one) if batched else one)
        _COMPILED[key] = hit
    return child_carry.replace(stochastic=hit[1](child,parent))
