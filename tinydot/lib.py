from pathlib import Path
from ctypes import *

class Singleton(type):
  _instances = {}

  def __call__(cls, *args, **kwargs):
    if cls not in cls._instances:
      cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
    return cls._instances[cls]

class LIB(metaclass=Singleton):
  def __init__(self):
    self.lib = CDLL(Path().resolve() / 'build/c_lib.so')

    #tensor / generic methods
    self.init    = self.c_wrapper('tensor_init',        c_void_p, [c_uint, POINTER(c_int)])
    self.destroy = self.c_wrapper('tensor_destroy',     None,     [c_void_p])
    self.copy    = self.c_wrapper('tensor_copy',        c_void_p, [c_void_p])
    self.set     = self.c_wrapper('tensor_set',         None,     [c_void_p, POINTER(c_double)])
    self.add     = self.c_wrapper('tensor_add',         c_void_p, [c_void_p, c_void_p])
    self.mul     = self.c_wrapper('mul',                c_void_p, [c_void_p, c_double])
    self.zeros   = self.c_wrapper('zeros',              c_void_p, [c_uint, POINTER(c_int)])
    self.ones    = self.c_wrapper('ones',               c_void_p, [c_uint, POINTER(c_int)])
    self.rand    = self.c_wrapper('tensor_rand',        c_void_p, [c_uint, POINTER(c_int), c_uint])

    #matrix methods
    self.norm    = self.c_wrapper('matrix_norm',        c_double,  [c_void_p])
    self.trace   = self.c_wrapper('matrix_trace',       c_double,  [c_void_p])
    self.T       = self.c_wrapper('matrix_transpose',   None,      [c_void_p])
    self.det     = self.c_wrapper('matrix_determinant', c_double,  [c_void_p])
    self.identity= self.c_wrapper('matrix_identity',    c_void_p,  [c_uint, POINTER(c_int)])
    self.matmul  = self.c_wrapper('matmul',             c_void_p,  [c_void_p, c_void_p])

  def c_wrapper(self, funcname, restype, argtypes):
    func = self.lib.__getattr__(funcname)
    func.restype = restype
    func.argtypes = argtypes
    return func

