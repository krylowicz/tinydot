from ctypes import *
from tinydot.lib import LIB
from tinydot._utils import _flatten, _get_index, _reshape

class TensorData(Structure):
  _fields_ = [
    ('rank', c_uint),
    ('length', c_uint),
    ('shape', POINTER(c_uint)),
    ('data', POINTER(c_double))
  ]

class Tensor:
  def __init__(self, shape=None, pointer=None):
    if pointer:
      self.pointer = None
      self.rank = None
      self.shape = None
      self.from_pointer(pointer)
    elif shape:
      if all(isinstance(v, (int, float)) for v in shape):
        self.from_shape(shape)
      else:
        raise TypeError(f"Can't create tensor from shape of type {type(shape[0])}")

  def __repr__(self):
    return f"{self.pointer}"

  def __str__(self):
    return f"<Tensor with shape {self.shape}>"

  def __add__(self, other):
    return self.add(self, other)

  def __mul__(self, other):
    return self.mul(self, other)

  def __del__(self):
    try:
      LIB().destory(self.pointer) 
    except AttributeError:
      return
  
  @property
  def length(self):
    return self.get().length
   
  # TODO - print tensor with correct shape
  @property
  def data(self):
    return _reshape([self.get().data[i] for i in range(self.length)], self.shape)

  def from_pointer(self, pointer):
    self.pointer = pointer
    tensor = self.get()
    self.rank = tensor.rank
    self.shape = [tensor.shape[i] for i in range(self.rank)]

  def from_shape(self, shape):
    self.shape = shape
    self.rank = len(shape)
    self.pointer = LIB().init(self.rank, (c_int * self.rank)(*shape))

  def set(self, data):
    data = _flatten(data)
    c_data = c_double * len(data)
    LIB().set(c_void_p(self.pointer), (c_data)(*data))

  def get(self, coord=None):
    if coord:
      index = _get_index(coord, self.shape)
      return TensorData.from_address(self.pointer).data[index]
    return TensorData.from_address(self.pointer)

  def reshape(self, *shape):
    if -1 in shape and shape.count(-1) > 1:
      raise ValueError("Can only specify one unknown dimension")
    self.shape = shape

  def copy(self):
    pointer = LIB().copy(self.pointer)
    return Tensor(pointer=pointer)

  @classmethod
  def add(cls, t1, t2):
    if Tensor.match_shapes(t1, t2):
      pointer = LIB().add(t1.pointer, t2.pointer)
      return cls(pointer=pointer)
    else:
      raise ValueError("Tensors must have matching shapes")

  @classmethod
  def mul(cls, t, scalar):
    if isinstance(scalar, (int, float)):
      pointer = LIB().mul(t.pointer, scalar)
      return cls(pointer=pointer)
    else:
      raise TypeError(f"Can't multiply tensor with type {type(scalar)}")

  # TODO - Tensor dot product
  # @classmethod
  # def dot(cls, t1, t2):
  #   if Tensor.match_shapes(t1, t2):
  #     pointer = LIB().dot(t1.pointer, t2.pointer)
  #     return cls(pointer=pointer)
  #   else:
  #     raise ValueError("Tensors must have matching shapes")

  # def _is_on_axis(flat_tensor, i, axis):
  #   if axis < 0:
  #     axis += flat_tensor.rank
  #   elif axis >= flat_tensor.rank:
  #     raise IndexError(f"Axis {axis} is out of bounds")
  #   return i % flat_tensor.shape[axis] != 0

  @classmethod
  def ones(cls, *shape):
    rank = 1 if isinstance(shape, int) else len(shape)
    c_data = c_uint * rank
    pointer = LIB().ones(rank, (c_data)(*shape))
    return cls(pointer=pointer)

  @classmethod
  def zeros(cls, *shape):
    rank = 1 if isinstance(shape, int) else len(shape)
    c_data = c_uint * rank
    pointer = LIB().zeros(rank, (c_data)(*shape))
    return cls(pointer=pointer)

  # TODO - add random with seed
  @classmethod
  def random(cls, *shape):
    rank = 1 if isinstance(shape, int) else len(shape)
    c_data = c_int * rank
    pointer = LIB().random(rank, (c_data)(*shape))
    return cls(pointer=pointer)

  @classmethod
  def uniform(cls, low=0.0, high=1.0, shape=None):
    if shape:
      rank = len(shape)
      c_data = rank * c_uint
      pointer = LIB().uniform(rank, (c_data)(*shape), low, high)
    else:
      # TODO - change to scalar (now it's a vector)
      pointer = LIB().uniform(1, (1 * c_uint)(1), low, high)
    return cls(pointer=pointer)
  
  @classmethod
  def prod(cls, t):
    return LIB().prod(t.pointer)

  @classmethod
  def sqrt(cls, t):
    return LIB().sqrt(t.pointer)

  @staticmethod
  def match_shapes(t1, t2):
    if t1.rank != t2.rank:
      return False
    return all(t1.shape[i] == t2.shape[i] for i in range(t1.rank))

