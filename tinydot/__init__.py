import ctypes as _ct

from .lib import LIB as _LIB
from .tensor import Tensor
from .matrix import Matrix
from .vector import Vector

def _get_cls(object=None, shape=None):
  if isinstance(object, Tensor) or len(shape) > 2:
    return Tensor
  elif isinstance(object, Matrix) or len(shape) == 2:
    return Matrix
  elif isinstance(object, Vector) or len(shape) == 1:
    return Vector
  else:
    raise TypeError(f"Can't get class of type {type(object)}")

def sqrt(tensor):
  pointer = _LIB().sqrt(tensor.pointer)
  try:
    return _get_cls(object=tensor)(pointer=pointer)
  except TypeError:
    return f"Can't take element-wise square root of type {type(tensor)}"

# def uniform(low=0.0, high=1.0, shape=None):
#   pointer = None
#   if shape:
#     rank = len(shape)
#     c_data = rank * _ct.c_uint
#     pointer = _LIB().uniform(rank, (c_data)(*shape), low, high)
#   else:
#     pointer = _LIB().uniform(0, 0, low, high)
  
#   return _get_cls(shape=shape)(pointer=pointer)
  
def random(*args):
  rank = 1 if isinstance(args, int) else len(args)
  pointer = _LIB().random(rank, (rank * _ct.c_uint)(*args))
  return _get_cls(shape=args)(pointer=pointer)
 
# def prod(tensor):
#   pointer = _LIB().prod(tensor.pointer)
#   return _get_cls(object=tensor)(pointer=pointer)