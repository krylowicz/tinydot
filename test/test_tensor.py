import unittest
from tinydot.tensor import Tensor

class TestTensor(unittest.TestCase):
  def test_wrong_shape_type(self):
    with self.assertRaises(TypeError):
      tensor = Tensor(['string'])

  def test_from_shape(self):
    tensor = Tensor([3, 5, 2])
    self.assertEqual(tensor.shape, [3, 5, 2])

  def test_print_data(self):
    tensor = Tensor([2, 2])
    tensor.set([[1, 2], [3, 4]])
    self.assertEqual(tensor.get().__str__(), '[1.0, 2.0, 3.0, 4.0]')

