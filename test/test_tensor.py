import unittest
from tinydot.tensor import Tensor

class TestTensor(unittest.TestCase):
  def test_wrong_shape_type(self):
    with self.assertRaises(TypeError):
      tensor = Tensor(['string'])

  def test_from_shape(self):
    tensor = Tensor([2])
    tensor.set([1, 2])
    self.assertEqual(tensor.shape, [2])
  
