<p align='center'>
    <img src='https://user-images.githubusercontent.com/22550143/114083373-2c029e80-98af-11eb-855f-ae82c81a5c2e.png'>
</p>

<hr />

Tinydot is a Python wrapper for my tiny linear algebra library writen in C. The wrapper interacts with C code via `ctypes`.

## How?

Everything in tinydot is based on nd Tensor structure, i. e. a matrix is just a 2d tensor with shape [rows, cols].

Maybe I will add some tensor algebra in the future.

## Installation
I am planning to upload tinydot to pip but meanwhile if you would like to play around with this framework, here's an example:

```python
from tinydot.vector import Vector
from tinydot.matrix import Matrix
 
v1 = Vector(1, 3)
v2 = Vector(2, 4)

dot_product = Vector.dot(v1, v2)
v1_norm = v1.norm
v2.rotate(47) #inplace counterclockwise vector rotation by theta degrees (use -theta for clockwise)

matrix = Matrix([[1, 2], [3, 4], [5, 6]])
matrix.T #transposes the matrix inplace
trace = matrix.trace
matrix_norm = matrix.norm
```

You have to compile C code to .so file. You can do it by running `make` command in the root directory.

## Testing
Tinydot is written in test driven development fashion using `pytest`. To run the tests execute the following command:

```shell
$ python -m pytest
```

You should run the command above in the test directory
