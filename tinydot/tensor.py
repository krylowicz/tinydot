from ctypes import *
from tinydot.lib import LIB
from tinydot.utils import flatten, get_index, reshape

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
  
  @property
  def length(self):
    return TensorData.from_address(self.pointer).length
    
  @property
  def data(self):
    return reshape([self.get().data[i] for i in range(self.length)], self.shape)

  def from_pointer(self, pointer):
    self.pointer = pointer
    tensor = self.get()
    self.rank = tensor.rank
    self.shape = tensor.shape

  def from_shape(self, shape):
    self.shape = shape
    self.rank = len(shape)
    self.pointer = LIB().init(self.rank, (c_int * self.rank)(*shape))

  def set(self, data):
    data = flatten(data)
    c_data = c_double * len(data)
    LIB().set(c_void_p(self.pointer), (c_data)(*data))

  def get(self, coord=None):
    if coord:
      index = get_index(coord, self.shape)
      return TensorData.from_address(self.pointer).data[index]
    return TensorData.from_address(self.pointer)

  @staticmethod
  def match_shapes(t1, t2):
    if t1.rank != t2.rank:
      return False
    return all(t1.shape[i] == t2.shape[i] for i in range(t1.rank))

