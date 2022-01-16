import unittest
import tinydot as td
from tinydot.tensor import Tensor

class TestTensor(unittest.TestCase):
  def test_wrong_shape_type(self):
    with self.assertRaises(TypeError):
      Tensor(['string'])

  def test_from_shape(self):
    tensor = Tensor([3, 2, 5])
    self.assertEqual(tensor.shape, [3, 2, 5])

  def test_data(self):
    tensor = Tensor([2, 2])
    tensor.set([[1, 2], [3, 4]])
    self.assertEqual(tensor.data, [[1.0, 2.0], [3.0, 4.0]])

  def test_reshape(self):
    tensor = Tensor([3, 2])
    tensor.set([
      [1, 2],
      [3, 4],
      [5, 6]
    ])
    tensor.reshape(2, 3)
    self.assertEqual(tensor.data, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    tensor.reshape(-1, 6)
    self.assertEqual(tensor.data, [[1., 2., 3., 4., 5., 6.]])

  def test_copy(self):
    t1 = td.ones(3, 5, 2)
    t2 = t1.copy()
    self.assertEqual(t1.data, t2.data)

  def test_get_value_at_index(self):
    t1 = Tensor([2, 3])
    t1.set([[1, 2, 3], [4, 5, 6]])
    self.assertEqual(t1.get([1, 1]), 5.0)
    
    t2 = Tensor([3, 2, 1])
    t2.set([
      [[1], [2]],
      [[3], [4]],
      [[5], [6]]
    ])
    self.assertEqual(t2.get([2, 0, 0]), 5.0)

  def test_zeros(self):
    tensor = td.zeros(3, 2, 1)  
    self.assertEqual(tensor.data, [
      [[0], [0]],
      [[0], [0]],
      [[0], [0]]
    ])

  def test_ones(self):
    tensor = td.ones(3, 2)
    self.assertEqual(tensor.data, [
      [1, 1],
      [1, 1],
      [1, 1]
    ])

  # def test_uniform(self):
  #   tensor = Tensor.uniform(64, 32, shape=(2, 2))
  #   self.assertEqual(tensor.data, [[0.0, 0.0], [0.0, 0.0]])

  def test_prod(self):
    tensor = Tensor([2, 2])
    tensor.set([
      [4, 3],
      [2, 1]
    ])

    prod = td.prod(tensor)
    self.assertEqual(prod, 24.0)

  def test_add(self):
    t1 = Tensor([2, 2])
    t1.set([
      [1, 2],
      [3, 4]
    ])
    
    t2 = Tensor([2, 2])
    t2.set([
      [4, 3],
      [2, 1]
    ])
    
    tensor = Tensor.add(t1, t2)
    self.assertEqual(tensor.data, [[5.0, 5.0], [5.0, 5.0]])

  def test_mul(self):
    t1 = Tensor([2, 2])
    t1.set([
      [4, 3],
      [2, 1]
    ])

    tensor = Tensor.mul(t1, 2.0)
    self.assertEqual(tensor.data, [[8.0, 6.0], [4.0, 2.0]])

  def test_maximum(self):
    t1 = Tensor([2, 2])
    t1.set([
      [4, -3],
      [-2, 1]
    ])

    tensor = td.maximum(t1, 0)
    self.assertEqual(tensor.data, [[4.0, 0.0], [0.0, 1.0]])