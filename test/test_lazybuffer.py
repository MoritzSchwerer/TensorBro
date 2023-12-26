import unittest

from tensorbro.lazy import LazyBuffer
from tensorbro.ops import BinaryOps

class TestLazyBuffer(unittest.TestCase):
    def test_lazy_buffer_full(self):
        LazyBuffer.full(10, (10, 10), device='CPU')
        self.assertTrue(True)

    def test_lazy_buffer_rand(self):
        LazyBuffer.rand((10, 10), device="CPU", seed=99)
        self.assertTrue(True)

    def test_elem_op_same_shape_un_realized(self):
        l1 = LazyBuffer.full(10, (10, 10), device='CPU')
        l2 = LazyBuffer.full(10, (10, 10), device='CPU')
        mul = l1.elementwise(BinaryOps.MUL, l2)
        self.assertEqual(mul.op.op, BinaryOps.MUL)
        self.assertEqual(mul.shape, (10, 10))
        self.assertEqual(mul.device, 'CPU')
        self.assertFalse(mul.is_realized)


if __name__ == "__main__":
    unittest.main()
