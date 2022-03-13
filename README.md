<p align='center'>
    <img src='https://user-images.githubusercontent.com/22550143/158031381-0561f9fb-7db6-481f-aaab-ca8bd30697b6.svg' width="300px" height="300px">
</p>

<hr />

tinydot is a Python wrapper for my tiny linear algebra library writen in C. The wrapper interacts with C code via `ctypes`.

## How?

Everything in tinydot is based on nd Tensor structure, i.e. a matrix is a 2d tensor with shape [rows, cols].

tinydot also support basic operations on Tensors.

## Installation
If you would like to play around with this framework, you can install it using pip directly from GitHub by running 
```
pip install git+https://github.com/krylowicz/tinydot.git`.
```

Here's an example:
```python
import tinydot as td
 
v1 = td.Vector(1, 3)
v2 = td.Vector(2, 4)

dot_product = td.Vector.dot(v1, v2)
v1_norm = v1.norm
v2.rotate(47) #inplace counterclockwise vector rotation by theta degrees (use -theta for clockwise)

matrix = td.Matrix([[1, 2], [3, 4], [5, 6]])
matrix.T #transposes the matrix inplace
trace = matrix.trace
matrix_norm = matrix.norm
```

## Testing
Tinydot is written in test driven development fashion using `pytest`. To run the tests execute the following command:

```shell
$ python -m pytest
```

You should run the command above in the root directory
