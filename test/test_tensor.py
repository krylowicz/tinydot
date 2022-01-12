import unittest
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
    t1 = Tensor.ones([3, 5, 2])
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
    tensor = Tensor.zeros([3, 2, 1])  
    self.assertEqual(tensor.data, [
      [[0], [0]],
      [[0], [0]],
      [[0], [0]]
    ])

  def test_ones(self):
    tensor = Tensor.ones([3, 2])
    self.assertEqual(tensor.data, [
      [1, 1],
      [1, 1],
      [1, 1]
    ])

  def test_rand_seed(self):
    tensor = Tensor.rand([3, 2], seed=42)
    self.assertEqual(tensor.data, [
      [0.00032870750889587566, 0.5245871020129822],
      [0.7354235321913956, 0.26330554078487006],
      [0.37622397131110724, 0.19628582577979464]
    ])

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