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
    # TODO - find better way to do this
    self.lib = CDLL(Path().resolve() / '../build/c_lib.so')

    #api methods
    self.sqrt           = self.c_wrapper('api_sqrt',           c_void_p, [c_void_p])
    self.random         = self.c_wrapper('api_random',         c_void_p, [c_uint, POINTER(c_uint)])
    self.prod           = self.c_wrapper('api_prod',           c_float,  [c_void_p])
    self.uniform        = self.c_wrapper('api_uniform',        c_void_p, [c_uint, POINTER(c_uint), c_double, c_double])
    # api_zeros and api_ones names give an error while linking?
    self.zeros          = self.c_wrapper('zeros',              c_void_p, [c_uint, POINTER(c_uint)])
    self.ones           = self.c_wrapper('ones',               c_void_p, [c_uint, POINTER(c_uint)])
    self.maximum        = self.c_wrapper('maximum',            c_void_p, [c_void_p, c_void_p])
    self.maximum_scalar = self.c_wrapper('maximum_scalar',     c_void_p, [c_void_p, c_int])
    self.exp            = self.c_wrapper('exponent',           c_void_p, [c_void_p])

    #tensor / generic methods
    self.init           = self.c_wrapper('tensor_init',        c_void_p, [c_uint, POINTER(c_int)])
    self.destroy        = self.c_wrapper('tensor_destroy',     None,     [c_void_p])
    self.copy           = self.c_wrapper('tensor_copy',        c_void_p, [c_void_p])
    self.set            = self.c_wrapper('tensor_set',         None,     [c_void_p, POINTER(c_double)])
    self.add            = self.c_wrapper('tensor_add',         c_void_p, [c_void_p, c_void_p])
    self.mul            = self.c_wrapper('mul',                c_void_p, [c_void_p, c_double])

    #matrix methods
    self.norm           = self.c_wrapper('matrix_norm',        c_double,  [c_void_p])
    self.trace          = self.c_wrapper('matrix_trace',       c_double,  [c_void_p])
    self.T              = self.c_wrapper('matrix_transpose',   POINTER(c_uint),      [c_void_p])
    self.det            = self.c_wrapper('matrix_determinant', c_double,  [c_void_p])
    self.identity       = self.c_wrapper('matrix_identity',    c_void_p,  [c_uint, POINTER(c_int)])
    self.matmul         = self.c_wrapper('matmul',             c_void_p,  [c_void_p, c_void_p])

    #vector methods
    self.v_dot          = self.c_wrapper('vector_dot',         c_double,  [c_void_p, c_void_p])
    self.v_sub          = self.c_wrapper('vector_sub',         c_void_p,  [c_void_p, c_void_p])
    self.v_norm         = self.c_wrapper('vector_norm',        c_double,  [c_void_p])
    self.v_angle        = self.c_wrapper('vector_angle',       c_double,  [c_void_p, c_void_p, c_uint])
    self.v_rotate       = self.c_wrapper('vector_rotate',      c_void_p,  [c_void_p, c_double])

  def c_wrapper(self, funcname, restype, argtypes):
    func = self.lib.__getattr__(funcname)
    func.restype = restype
    func.argtypes = argtypes
    return func

