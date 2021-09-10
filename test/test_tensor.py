import unittest
from tinydot.tensor import Tensor

class TestTensor(unittest.TestCase):
  def test_wrong_shape_type(self):
    with self.assertRaises(TypeError):
      tensor = Tensor(['string'])

  def test_from_shape(self):
    tensor = Tensor([3, 2, 5])
    self.assertEqual(tensor.shape, [3, 2, 5])

  def test_print_data(self):
    tensor = Tensor([2, 2])
    tensor.set([[1, 2], [3, 4]])
    # TODO - need a better way
    self.assertEqual(tensor.get().__str__(), '[1.0, 2.0, 3.0, 4.0]')

  def test_get_value_at_index(self):
    t1 = Tensor([2, 3])
    t1.set([[1, 2, 3], [4, 5, 6]])
    self.assertEqual(tensor.get([1, 1]), 5.0)
    
    t2 = Tensor([3, 2, 1])
    t2.set([
      [[1.0], [2.0], [3,0]],
      [[4.0], [5.0], [6.0]]
    ])
    self.assertEqual(tensor.get([1, 1, 0]), 5.0)

