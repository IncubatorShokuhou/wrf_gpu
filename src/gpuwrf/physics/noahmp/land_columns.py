"""Layer-profile ABI for optional Noah-MP routines in the existing column kernel.

The retained column helper accepts only per-column fields. Split profile planes
before that boundary, rebuild them inside the native routine, and return planes
through its existing dtype packs. This changes layout, not arithmetic.
"""
from itertools import product
import jax
import jax.numpy as jnp
from gpuwrf.kernels.phys_noahmp_columns import column_call,in_column_kernel


def layered_column_call(fn,args,*,grid_shape,name):
    leaves,tree=jax.tree.flatten(args)
    inputs=[];layouts=[]
    for leaf in leaves:
        if hasattr(leaf,'shape') and len(leaf.shape)>2:
            leading=tuple(leaf.shape[:-2]);count=1
            for n in leading:count*=n
            layouts.append((leading,count))
            inputs.extend(leaf[index] for index in product(*(range(n) for n in leading)))
        else:
            layouts.append(((),1));inputs.append(leaf)
    abstract=jax.eval_shape(fn,*args)
    out_leaves,out_tree=jax.tree.flatten(abstract)
    out_layouts=[tuple(leaf.shape[:-2]) for leaf in out_leaves]
    def native(*columns):
        rebuilt=[];offset=0
        for leading,count in layouts:
            if leading:
                rebuilt.append(stack_layers(columns[offset:offset+count]).reshape((*leading,-1)))
            else:rebuilt.append(columns[offset])
            offset+=count
        result=fn(*jax.tree.unflatten(tree,rebuilt))
        flat=[]
        for leaf,leading in zip(jax.tree.leaves(result),out_layouts,strict=True):
            if leading:
                planes=jnp.split(leaf.reshape((-1,*leaf.shape[len(leading):])),leaf.size//leaf.shape[-1],axis=0)
                flat.extend(plane.reshape(leaf.shape[len(leading):]) for plane in planes)
            else:flat.append(leaf)
        return tuple(flat)
    values=column_call(native,tuple(inputs),grid_shape=grid_shape,name=name)
    restored=[];offset=0
    for leading in out_layouts:
        count=1
        for n in leading:count*=n
        restored.append(jnp.stack(values[offset:offset+count],axis=0).reshape((*leading,*grid_shape)) if leading else values[offset])
        offset+=count
    return jax.tree.unflatten(out_tree,restored)


def pick_layer(array,index):
    """Same per-column gather; native tensors use constant layer slices/selects."""
    array=jnp.asarray(array)
    if not in_column_kernel():return jnp.take_along_axis(array,index[None,...],axis=0)[0]
    value=layer_row(array,-1)
    for k in range(array.shape[0]-1):value=jnp.where(index==k,layer_row(array,k),value)
    return value


def stack_layers(values):
    """Original stack outside native code; singleton-last-axis Triton joins.

    Pallas supports concatenate only for two [..., 1] operands. Each join
    appends a binary layer digit; transpose it before the existing digits so
    the final reshape retains the original layer order (no arithmetic).
    """
    if not in_column_kernel():return jnp.stack(values,axis=0)
    values=[jnp.asarray(v)[...,None] for v in values]
    if len(values)&(len(values)-1):raise ValueError('Native land profile length must be a power of two')
    while len(values)>1:
        joined=[]
        for k in range(0,len(values),2):
            pair=jnp.concatenate((values[k][...,None],values[k+1][...,None]),axis=-1)
            pair=jnp.swapaxes(pair,-1,-2)
            joined.append(pair.reshape((*pair.shape[:-2],-1)))
        values=joined
    return jnp.moveaxis(values[0],-1,0)


def layer_row(array,index):
    """Static layer read: Pallas implements split, but not tensor slice."""
    if not in_column_kernel() or isinstance(array,(list,tuple)):return array[index]
    return jnp.split(array,array.shape[0],axis=0)[index].reshape(array.shape[1:])


def set_layer(array,index,value):
    from gpuwrf.kernels.phys_noahmp_layers import set_layer as retained
    if not in_column_kernel():return retained(array,index,value)
    ids=jnp.arange(array.shape[0],dtype=jnp.int32).reshape((-1,)+(1,)*(array.ndim-1))
    return jnp.where(ids==index%array.shape[0],jnp.asarray(value,array.dtype)[None,...],array)


def add_layer(array,index,value):
    from gpuwrf.kernels.phys_noahmp_layers import add_layer as retained
    if not in_column_kernel():return retained(array,index,value)
    return set_layer(array,index,layer_row(array,index)+value)


def iterate_six(body,carry):
    """Keep WRF's soil substeps in one native loop, avoiding six code copies."""
    if in_column_kernel():return jax.lax.fori_loop(0,6,body,carry)
    for k in range(6):carry=body(k,carry)
    return carry
