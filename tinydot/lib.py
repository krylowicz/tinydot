from ctypes import *

class Singleton(type):
  _instances = {}

  def __call__(cls, *args, **kwargs):
    if cls not in cls._instances:
      cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
    return cls._instances[cls]

class LIB(metaclass=Singleton):
  def __init__(self):
    self.lib = CDLL('../build/c_lib.so')
    
    #tensor / generic methods
    self.init = self.c_wrapper('tensor_init', c_void_p, [c_uint, POINTER(c_double)])
    self.destroy = self.c_wrapper('tensor_destroy', None, [c_void_p])
    self.set = self.c_wrapper('tensor_set', None, [c_void_p, POINTER(c_double)])

  def c_wrapper(self, funcname, restype, argtypes):
    func = self.lib.__getattr__(funcname)
    func.restype = restype
    func.argtypes = argtypes
    return func

