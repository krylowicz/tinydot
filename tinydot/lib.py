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
  

  def c_wrapper(self, funcname, restype, argtypes):
    func = self.lib.__getattr__(funcname)
    func.restype = restype
    func.argtypes = argtypes
    return func

