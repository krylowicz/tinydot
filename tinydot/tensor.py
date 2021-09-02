from ctypes import *
from tinydot.lib import LIB
from tinydot.utils import flatten

class TensorData(Structure):
  _fields_ = [
    ('rank', c_uint),
    ('length', c_uint),
    ('shape', POINTER(c_uint)),
    ('data', POINTER(c_double))
  ]

  def __str__(self):
    return f"{[self.data[i] for i in range(self.length)]}"

class Tensor:
  def __init__(self, shape=None, pointer=None):
    if pointer:
      self.from_pointer(pointer)
    elif shape:
      if all(isinstance(v, (int, float)) for v in shape):
        self.from_shape(shape)
      else:
        raise TypeError(f"Can't create tensor from shape of type {type(shape[0])}")

  def __repr__(self):
    return self.pointer

  def __str__(self):
    return f"<Tensor with shape {self.shape}>"

  def __del__(self):
    try:
      LIB().destory(self.pointer) 
    except AttributeError:
      return

  def from_pointer(self, pointer):
    self.pointer = pointer
    tensor = self.get()
    self.rank = tensor.rank
    self.shape = tensor.shape

  def from_shape(self, shape):
    self.shape = shape
    self.rank = len(shape)
    return LIB().init(self.rank, (c_uint * self.rank)(*shape))

  # TODO - add set method

  def get(pointer):
    return TensorData.from_address(pointer)
