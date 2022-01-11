from ctypes import *
from tinydot.lib import LIB
from tinydot.tensor import Tensor
from tinydot.matrix import Matrix

class Vector(Matrix):
  def __init__(self, *data, pointer=None, **kwargs):
    if pointer:
      Tensor.__init__(self, pointer=pointer)
    else:
      Tensor.__init__(self, [len(data)])
      self.set(data)

  def __sub__(self, other):
    if isinstance(other, Vector):
      return self.sub(self, other)
    else:
      raise TypeError(f"Can't subtract Vector with type {type(other)}")

  def __mul__(self, other):
    if isinstance(other, Vector):
      return self.dot(self, other)
    elif isinstance(other, (float, int)):
      return self.mul(self, other)
    else:
      raise TypeError(f"Can't multiply Vector with type {type(other)}")

  # overrides matrix norm method
  @property
  def norm(self):
    return LIB().v_norm(self.pointer)

  def rotate(self, angle):
    LIB().v_rotate(self.pointer, angle)

  @classmethod
  def dot(cls, v1, v2):
    if Tensor.match_shapes(v1, v2):
      return LIB().v_dot(v1.pointer, v2.pointer)
    else:
      raise ValueError("Vectors must have matching lengths")

  @classmethod
  def sub(cls, v1, v2):
    if Tensor.match_shapes(v1, v2):
      pointer = LIB().v_sub(v1.pointer, v2.pointer)
      return cls(pointer=pointer)
    else:
      raise ValueError("Vectors must have matching lengths")
 
  @classmethod
  def angle(cls, v1, v2, degrees=False):
    if Tensor.match_shapes(v1, v2):
      if degrees:
        return LIB().v_angle(v1.pointer, v2.pointer, 1)
      return LIB().v_angle(v1.pointer, v2.pointer, 0)
    else:
      raise ValueError("Vectors must have matching lengths")