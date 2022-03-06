import unittest
import tinydot as td
from tinydot.api import _get_cls
from tinydot._utils import _flatten

class TestApi(unittest.TestCase):
  def test_arange(self):
    x = td.arange(stop=24, shape=(2, 3, 4))
    
    self.assertEqual(x.shape, [2, 3, 4])
    self.assertEqual(x.data, [
      [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]],
      [[12.0, 13.0, 14.0, 15.0], [16.0, 17.0, 18.0, 19.0], [20.0, 21.0, 22.0, 23.0]]
    ])

    with self.assertRaises(ValueError):
      td.arange(stop=7, shape=(2, 3))

  def test_copy(self):
    x = td.arange(stop=24, shape=(2, 3, 4))
    y = td.copy(x)

    self.assertNotEqual(x.pointer, y.pointer)

  def test_get_cls(self):
    x = td.arange(stop=6, shape=(2, 3))
    self.assertEqual(_get_cls(x), td.Matrix)

  def test_sum(self):
    x = td.arange(stop=24, shape=(2, 3, 4))
    
    a = td.sum(x, axis=0)
    self.assertEqual(a.data, [
      [12.0, 14.0, 16.0, 18.0],
      [20.0, 22.0, 24.0, 26.0],
      [28.0, 30.0, 32.0, 34.0],
    ])

    b = td.sum(x, axis=1)
    self.assertEqual(b.data, [
      [12.0, 15.0, 18.0, 21.0],
      [48.0, 51.0, 54.0, 57.0]
    ])

    c = td.sum(x, axis=2)
    self.assertEqual(c.data, [
      [6.0, 22.0, 38.0],
      [54.0, 70.0, 86.0]
    ])

  def test_ones(self):
    x = td.ones(2, 3)

    self.assertEqual(x.data, [
      [1.0, 1.0, 1.0],
      [1.0, 1.0, 1.0]
    ])
    self.assertEqual(type(x), td.Matrix)

  # TODO: how to test uniform?

  def test_maximum(self):
    x = td.arange(stop=24, shape=(2, 3, 4))
    # TODO: arange returns wrong class type
    y = td.maximum(x, x.T)

    self.assertEqual(y.data, [
      [12.0, 14.0, 16.0, 18.0],
      [20.0, 22.0, 24.0, 26.0],
      [28.0, 30.0, 32.0, 34.0]
    ])

    # write a test for maximum scalar

  # TODO: add minimum function & test

  def test_sqrt(self):
    x = td.arange(stop=24, shape=(2, 3, 4))
    
    correct_y = [
      [0.0, 1.0, 1.41421356],
      [1.73205081, 2.0, 2.23606798]
    ]

    sqrt_y = td.sqrt(x).data

    for sqrt_y, correct_y in zip(_flatten(sqrt_y), _flatten(correct_y)):
      self.assertAlmostEqual(sqrt_y, correct_y, places=2)

  def test_exp(self):
    x = td.arange(stop=6, shape=(2, 3))
    
    correct_y = [
      [1.0, 2.71828183, 7.3890561], 
      [20.08553692, 54.59815003, 148.4131591]
    ]

    exp_y = td.exp(x).data

    for exp_y, correct_y in zip(_flatten(exp_y), _flatten(correct_y)):
      self.assertAlmostEqual(exp_y, correct_y, places=2)