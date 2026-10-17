"""Static level slots containing independent horizontal column vectors.

No dense vertical tensor is assembled inside Triton. This is a trace-time
Python container; its elements are registers, never host field values.
"""


class ColumnLevels(list):
    @property
    def shape(self):return (len(self),*self[0].shape)
    @property
    def ndim(self):return 1+self[0].ndim
    @property
    def dtype(self):return self[0].dtype
