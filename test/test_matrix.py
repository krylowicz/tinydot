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

