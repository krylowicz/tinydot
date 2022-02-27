import os
from setuptools import setup, Extension
from distutils.command.build_ext import build_ext as build_ext_setuptools

C_SOURCES = []

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

with os.scandir('tinydot/lib') as it:
  for entry in it:
    if entry.is_file() and not entry.name.startswith('.') and not entry.name.endswith('.h'):
      C_SOURCES.append(entry.path)

class CTypesExtension(Extension): pass
class build_ext(build_ext_setuptools):
  def build_extension(self, ext):
    self._ctypes = isinstance(ext, CTypesExtension)
    return super().build_extension(ext)

  def get_export_symbols(self, ext):
    if self._ctypes:
      return ext.export_symbols
    return super().get_export_symbols(ext)

  def get_ext_filename(self, ext_name):
    if self._ctypes:
      print('get_ext_filename', ext_name)
      return f'{ext_name}.so'
    return super().get_ext_filename(ext_name)

setup(
  name='tinydot',
  version='0.1.0',
  description='Tiny linear algebra library',
  author='Kacper Krylowicz',
  license='MIT',
  long_description=long_description,
  long_description_content_type='text/markdown',
  packages=['tinydot'],
  install_requires=[],
  python_requires='>=3.8',
  extras_require={
    'testing': [
      'pytest'
    ],
  },
  include_package_data=True,
  ext_modules=[CTypesExtension('tinydot_lib', C_SOURCES)],
  cmdclass={'build_ext': build_ext},
)
