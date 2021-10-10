from ctypes import *
from tinydot.lib import LIB
from tinydot.tensor import Tensor

class Matrix(Tensor):
  def __init__(self, data):
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
  def T(self):
    self.shape = self.shape[::-1]
    return LIB().T(self.pointer)

  @property
  def det(self):
    return LIB().det(self.pointer, self.copy().pointer)

