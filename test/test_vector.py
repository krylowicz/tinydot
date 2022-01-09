import unittest
from tinydot.vector import Vector

class TestVector(unittest.TestCase):
  def test_length(self):
    v = Vector(1, -4, 3)
    self.assertEqual(v.length, 3)

  def test_add_wrong_length(self):
    with self.assertRaises(Exception):
      v1 = Vector(1, 6)
      v2 = Vector(-1, 5, 6)
      v1 + v2

  def test_add(self):
    v1 = Vector(1, 6)
    v2 = Vector(-2, 5)
    res = v1 + v2
    self.assertEqual(res.data, [-1.0, 11.0])

  def test_sub(self):
    v1 = Vector(1, -5.2, 8)
    v2 = Vector(6, 2, 8.5)
    res = v1 - v2
    self.assertEqual(res.data, [-5, -7.2, -0.5])

  def test_scalar_mul(self):
    v1 = Vector(2.3, -8.19)
    res = v1 * -4
    self.assertEqual(res.data, [-9.2, 32.76])

  def test_dot(self):
    v1 = Vector(1, 3)
    v2 = Vector(2, 4)
    res1 = v1 * v2
    self.assertEqual(res1, 14)

    v3 = Vector(5.931, -4.234)
    v4 = Vector(9.072, 4.432)
    res2 = v3 * v4
    self.assertEqual(res2, 35.040943999999996)

  def test_norm(self):
    vector = Vector(2, -3, 6)
    self.assertEqual(vector.norm, 7)

  def test_zeros(self):
    vector = Vector.zeros(4)
    self.assertEqual(vector.data, [0.0, 0.0, 0.0, 0.0])

  def test_ones(self):
    vector = Vector.ones(3)
    self.assertEqual(vector.data, [1.0, 1.0, 1.0])

  def test_angle(self):
    v1 = Vector(4, 3)
    v2 = Vector(2, 5)
    self.assertEqual(Vector.angle(v1, v2), 0.5467888408892473)

    v3 = Vector(4, 2)
    v4 = Vector(-5, 1)
    self.assertEqual(Vector.angle(v3, v4, degrees=True), 142.12501634890182)

  def test_rotate(self):
    v1 = Vector(3, 4)
    v1.rotate(30)
    self.assertEqual(v1.data, (4.598076211353316, 1.964101615137755))

    v2 = Vector(3.7, -8.259)
    v2.rotate(-47)
    self.assertEqual(v2.data, (8.56364415390397347, -2.926615759765244))