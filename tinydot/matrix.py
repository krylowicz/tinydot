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

  def __matmul__(self, other):
    return self.matmul(self, other)

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
    new_shape = LIB().T(self.pointer)
    self.shape = [new_shape[i] for i in range(len(self.shape))]
    return self

  @classmethod
  def identity(cls, shape):
    rank = len(shape)
    c_data = c_int * rank
    pointer = LIB().identity(rank, (c_data)(*shape))
    return cls(pointer=pointer)

  @classmethod
  def matmul(cls, a, b):
    if a.shape[0] != b.shape[1]:
      raise ValueError("The number of columns in the first matrix must be equal to the number of rows in the second matrix")
    pointer = LIB().matmul(a.pointer, b.pointer)
    return cls(pointer=pointer)

  @classmethod
  def dot(cls, a, b):
    return cls.matmul(a, b)