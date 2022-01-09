from ctypes import *
from tinydot.lib import LIB
from tinydot.tensor import Tensor

class Matrix(Tensor):
  def __init__(self, data=None, pointer=None):
    if pointer:
      super().__init__(pointer=pointer)
      self.rows = len(self.data)
      self.cols = len(self.data[0])
    else:
      self.rows = len(data)
      self.cols = len(data[0])
      super().__init__([self.rows, self.cols])
      self.set(data)

  def __str__(self):
    return f"<Matrix with shape {self.shape}>"

  @property
  def norm(self):
    return LIB().norm(self.pointer)

  @property
  def trace(self):
    return LIB().trace(self.pointer)

  @property
  def det(self):
    return LIB().det(self.pointer)

  @property
  def T(self):
    return self.transpose()

  def transpose(self):
    # TODO - return shape from C
    # shape comes from parent class
    self.shape = self.shape[::-1]
    return LIB().T(self.pointer)

  @classmethod
  def identity(cls, shape):
    rank = len(shape)
    c_data = c_int * rank
    pointer = LIB().identity(rank, (c_data)(*shape))
    return cls(pointer=pointer)

  @classmethod
  def matmul(cls, a, b):
    # TODO - add * or @ operator
    pointer = LIB().matmul(a.pointer, b.pointer)
    return cls(pointer=pointer)

  @classmethod
  def dot(cls, a, b):
    return cls.matmul(a, b)