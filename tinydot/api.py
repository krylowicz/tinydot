import ctypes as _ct

from .lib import LIB as _LIB
from .tensor import Tensor
from .matrix import Matrix
from .vector import Vector

# This file contains tinydot API for initialization and generic math methods for tinydot's data structures.

def _get_cls(object=None, shape=None):
  if isinstance(object, Tensor) or len(shape) > 2:
    return Tensor
  elif isinstance(object, Matrix) or len(shape) == 2:
    return Matrix
  elif isinstance(object, Vector) or len(shape) == 1:
    return Vector
  # TODO - add scalar type
  else:
    raise TypeError(f"Can't get class of type {type(object)}")

def sqrt(tensor):
  pointer = _LIB().sqrt(tensor.pointer)
  try:
    return _get_cls(object=tensor)(pointer=pointer)
  except TypeError:
    return f"Can't take element-wise square root of type {type(tensor)}"

def uniform(low=0.0, high=1.0, shape=None):
  pointer = None
  if shape:
    rank = len(shape)
    c_data = rank * _ct.c_uint
    pointer = _LIB().uniform(rank, (c_data)(*shape), low, high)
  else:
    # TODO - change to scalar (now it's a vector)
    pointer = _LIB().uniform(1, (1 * _ct.c_uint)(1), low, high)
  
  return _get_cls(shape=shape)(pointer=pointer)
  
def random(*shape):
  rank = 1 if isinstance(shape, int) else len(shape)
  pointer = _LIB().random(rank, (rank * _ct.c_uint)(*shape))
  return _get_cls(shape=shape)(pointer=pointer)
 
def prod(tensor):
  return _LIB().prod(tensor.pointer)

# TODO - fix tests
def zeros(*shape):
  rank = 1 if isinstance(shape, int) else len(shape)
  c_data = rank * _ct.c_uint
  pointer = _LIB().zeros(rank, (c_data)(*shape))
  return _get_cls(shape=shape)(pointer=pointer)

def ones(*shape):
  rank = 1 if isinstance(shape, int) else len(shape)
  c_data = rank * _ct.c_uint
  pointer = _LIB().ones(rank, (c_data)(*shape))
  return _get_cls(shape=shape)(pointer=pointer)

# TODO - zeros, ones, dot, add, mul, copy

# def _is_on_axis(flat_tensor, i, axis):
#   if axis < 0:
#     axis += flat_tensor.rank
#   elif axis >= flat_tensor.rank:
#     raise IndexError(f"Axis {axis} is out of bounds")
#   return i % flat_tensor.shape[axis] != 0
