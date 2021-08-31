import os
from setuptools import setup

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

setup(name='tinydot',
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
      include_package_data=True)

