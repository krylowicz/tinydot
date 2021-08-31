from ctypes import *
from tinydot.lib import LIB

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
      # TODO - destroy pointer form c
      pass
    except AttributeError:
      return

  def from_pointer(self, pointer):
    self.pointer = pointer
    # TODO - add get method
    tensor = self.get()
    self.rank = tensor.rank
    self.shape = tensor.shape


  def from_shape(self, shape):
    # TODO - add flatten data
    data = flatten(data)
    self.rank = len(shape)
    # TODO - add tensor init from c library

