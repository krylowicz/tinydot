import unittest
from tinydot.matrix import Matrix

class TestMatrix(unittest.TestCase):
  def test_norm(self):
    matrix = Matrix([
      [5, -4, 2], 
      [-1, 2, 3],
      [-2, 1, 0]
    ])
    self.assertEqual(matrix.norm, 8.0)

  def test_transpose(self):
    matrix = Matrix([
      [1, 2],
      [3, 4],
      [5, 6]
    ])
    matrix.T
    self.assertEqual(matrix.data, [
      [1.0, 3.0, 5.0],
      [2.0, 4.0, 6.0]
    ])

  def test_trace(self):
    matrix = Matrix([
      [1, 0, 3], 
      [11, 5, 2], 
      [6, 12, -5]
    ])
    self.assertEqual(matrix.trace, 1)

  def test_det(self):
    m1 = Matrix([
      [1, 2],
      [3, 4]
    ])
    self.assertEqual(m1.det, -2)

    m2 = Matrix([
      [6, 1, 1],
      [4, -2, 5],
      [2, 8, 7]
    ])
    self.assertEqual(m2.det, -306)
  
  def test_identity(self):
    matrix = Matrix.identity([4, 4])
    self.assertEqual(matrix.data, [
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0]
    ])
  
  # TODO: matrix inverse
  # def test_inv(self):
  #   matrix = Matrix([
  #     [5., 3., 1.],
  #     [3., 9., 4.],
  #     [1., 3., 5.]
  #   ])
    
  #   inv = matrix.inv()

  #   self.assertEqual(inv.data, [
  #     [ 0.25,  -0.091,  0.023],
  #     [-0.083,  0.182, -0.129],
  #     [ 0.,    -0.091,  0.273]
  #   ])

  def test_matmul(self):
    A = Matrix([
      [1, 2, 3],
      [3, 4, 2],
      [3, 2, 1]
    ])

    B = Matrix([
      [1, 1, 1],
      [3, 4, 2],
      [3, 2, 1]
    ])

    C = Matrix.matmul(A, B)
    self.assertEqual(C.data, [
      [16.0, 15.0, 8.0],
      [21.0, 23.0, 13.0],
      [12.0, 13.0, 8.0]
    ])
  

  def test_dot(self):
    A = Matrix([
      [1, 2, 3],
      [3, 4, 2],
      [3, 2, 1]
    ])

    B = Matrix([
      [1, 1, 1],
      [3, 4, 2],
      [3, 2, 1]
    ])
    
    C = Matrix.dot(A, B)
    self.assertEqual(C.data, [
      [16.0, 15.0, 8.0],
      [21.0, 23.0, 13.0],
      [12.0, 13.0, 8.0]
    ])
