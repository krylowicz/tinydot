import ctypes as _ct

from .lib import LIB as _LIB
from .tensor import Tensor
from .matrix import Matrix
from .vector import Vector

# TODO - TESTS FOR API

# This file contains API for initialization and generic math methods for tinydot's data structures.

def _get_cls(object=None, shape=None):
  if isinstance(object, Tensor) or len(shape) > 2:
    return Tensor
  elif isinstance(object, Matrix) or len(shape) == 2:
    return Matrix
  elif isinstance(object, Vector) or len(shape) == 1:
    return Vector
  else:
    raise TypeError(f"Can't get class of type {type(object)}")

def copy(tensor):
  pointer = _LIB().copy(tensor.pointer)
  return _get_cls(object=tensor)(pointer=pointer)

# TODO - make it work for all cases
def add(t1, t2):
  if Tensor.match_shapes(t1, t2):
    pointer = _LIB().add(t1.pointer, t2.pointer)
    return _get_cls(object=t1)(pointer=pointer)
  else:
    raise ValueError("Tensors must have matching shapes")

def sum(tensor, axis=None):
  if axis is None:
    return _LIB().sum(tensor.pointer, None)
  else:
    return _LIB().sum(tensor.pointer, axis)

def exp(tensor):
  pointer = _LIB().exp(tensor.pointer)
  return _get_cls(object=tensor)(pointer=pointer)

def sqrt(tensor):
  pointer = _LIB().sqrt(tensor.pointer)
  try:
    return _get_cls(object=tensor)(pointer=pointer)
  except TypeError:
    return f"Can't take element-wise square root of type {type(tensor)}"

def maximum(tensor, max):
  pointer = None
  if isinstance(max, (int, float)):
    pointer = _LIB().maximum_scalar(tensor.pointer, max)
  elif Tensor.match_shapes(tensor, max):
    pointer = _LIB().maximum(tensor.pointer, max)
  return _get_cls(object=tensor)(pointer=pointer)

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

def arange(*, start=0, stop, step=1, shape):
  # check if interval matches shape
  rank = 1 if isinstance(shape, int) else len(shape)
  c_data = rank * _ct.c_uint
  pointer = _LIB().arange(start, stop, rank, (c_data)(*shape), step)
  return _get_cls(shape=shape)(pointer=pointer)
