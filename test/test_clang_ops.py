import unittest
import numpy as np

from tensorbro import LazyBuffer
from tensorbro.ops import BinaryOps, UnaryOps, MovementOps
from tensorbro.linearizer import linearize

class TestLazyOpsMovement(unittest.TestCase):
    def setUp(self):
        self.l1 = LazyBuffer.full(10, (10, 10), device="CLANG")

    def test_reshape(self):
        res = self.l1.reshape(20, 5)
        linearize(res.schedule())()
        np_res = np.ones((20, 5)) * 10
        clang_res = np.frombuffer(res.base, np.float32).reshape(*res.shape)
        np.testing.assert_allclose(np_res, clang_res)

    def test_expand(self):
        lb = LazyBuffer.full(10, (100, 1), device="CLANG")
        res = lb.movement(MovementOps.EXPAND, (100, 20))
        linearize(res.schedule())()
        self.assertEqual(res.shape, (100, 20))
        self.assertEqual(res.st.stride, (1, 20))
        

class TestLazyOpsUnary(unittest.TestCase):
    def setUp(self):
        self.l1 = LazyBuffer.full(10, (10, 10), device="CLANG")

    def test_unary_neg(self):
        res = self.l1.elementwise(UnaryOps.NEG)
        linearize(res.schedule())()
        np_res = -np.ones((10, 10)) * 10
        clang_res = np.frombuffer(res.base, np.float32).reshape(*res.shape)
        np.testing.assert_allclose(np_res, clang_res)

    def test_unary_neg_strided(self):
        l1 = LazyBuffer.rand((10, 1, 1), device="CLANG")
        res = l1.expand(10, 5, 8).elementwise(UnaryOps.NEG)
        linearize(res.schedule())()
        np_res = -np.frombuffer(l1.base, np.float32).reshape(10, 1, 1)
        clang_res = np.frombuffer(res.base, np.float32).reshape(10, 1, 1)
        np.testing.assert_allclose(np_res, clang_res)

    def test_unary_sin(self):
        res = self.l1.elementwise(UnaryOps.SIN)
        linearize(res.schedule())()
        np_res = np.sin(np.ones((10, 10)) * 10)
        clang_res = np.frombuffer(res.base, np.float32).reshape(*res.shape)
        np.testing.assert_allclose(np_res, clang_res)

    def test_unary_sin_strided(self):
        l1 = LazyBuffer.rand((10, 1, 1), device="CLANG")
        res = l1.expand(10, 5, 8).elementwise(UnaryOps.SIN)
        linearize(res.schedule())()
        np_res = np.sin(np.frombuffer(l1.base, np.float32).reshape(10, 1, 1))
        clang_res = np.frombuffer(res.base, np.float32).reshape(10, 1, 1)
        np.testing.assert_allclose(np_res, clang_res)

    def test_unary_sqrt(self):
        res = self.l1.elementwise(UnaryOps.SQRT)
        linearize(res.schedule())()
        np_res = np.sqrt(np.ones((10, 10)) * 10)
        clang_res = np.frombuffer(res.base, np.float32).reshape(*res.shape)
        np.testing.assert_allclose(np_res, clang_res)

    def test_unary_sqrt_strided(self):
        l1 = LazyBuffer.rand((10, 1, 1), device="CLANG")
        res = l1.expand(10, 5, 8).elementwise(UnaryOps.SQRT)
        linearize(res.schedule())()
        np_res = np.sqrt(np.frombuffer(l1.base, np.float32).reshape(10, 1, 1))
        clang_res = np.frombuffer(res.base, np.float32).reshape(10, 1, 1)
        np.testing.assert_allclose(np_res, clang_res)

    def test_unary_exp2(self):
        res = self.l1.elementwise(UnaryOps.EXP2)
        linearize(res.schedule())()
        np_res = np.exp2(np.ones((10, 10)) * 10)
        clang_res = np.frombuffer(res.base, np.float32).reshape(*res.shape)
        np.testing.assert_allclose(np_res, clang_res)

    def test_unary_exp2_strided(self):
        l1 = LazyBuffer.rand((10, 1, 1), device="CLANG")
        res = l1.expand(10, 5, 8).elementwise(UnaryOps.EXP2)
        linearize(res.schedule())()
        np_res = np.exp2(np.frombuffer(l1.base, np.float32).reshape(10, 1, 1))
        clang_res = np.frombuffer(res.base, np.float32).reshape(10, 1, 1)
        np.testing.assert_allclose(np_res, clang_res)

    def test_unary_log2(self):
        res = self.l1.elementwise(UnaryOps.LOG2)
        linearize(res.schedule())()
        np_res = np.log2(np.ones((10, 10)) * 10)
        clang_res = np.frombuffer(res.base, np.float32).reshape(*res.shape)
        np.testing.assert_allclose(np_res, clang_res)

    def test_unary_log2_strided(self):
        l1 = LazyBuffer.rand((10, 1, 1), device="CLANG")
        res = l1.expand(10, 5, 8).elementwise(UnaryOps.LOG2)
        linearize(res.schedule())()
        np_res = np.log2(np.frombuffer(l1.base, np.float32).reshape(10, 1, 1))
        clang_res = np.frombuffer(res.base, np.float32).reshape(10, 1, 1)
        np.testing.assert_allclose(np_res, clang_res)


class TestLazyOpsBinary(unittest.TestCase):
    def setUp(self):
        self.l1 = LazyBuffer.full(10, (10, 10), device="CLANG")
        self.l2 = LazyBuffer.full(10, (10, 10), device="CLANG")


    def test_elemwise_mul(self):
        res = self.l1.elementwise(BinaryOps.MUL, self.l2)
        linearize(res.schedule())()
        np_res = (np.ones((10, 10)) * 10) * (np.ones((10, 10)) * 10)
        clang_res = np.frombuffer(res.base, np.float32).reshape(*res.shape)
        np.testing.assert_allclose(np_res, clang_res)

    def test_elemwise_mul_strided(self):
        l1 = LazyBuffer.rand((10, 10, 5), device="CLANG")
        l2 = LazyBuffer.rand((1, 10, 1), device="CLANG")
        l2.movement(MovementOps.EXPAND, (10, 10, 5))

        res = l1.elementwise(BinaryOps.MUL, l2)
        linearize(res.schedule())()

        np1 = np.frombuffer(l1.base, np.float32).reshape(*l1.shape)
        np2 = np.frombuffer(l2.base, np.float32).reshape(1, 10, 1)
        np3 = np.tile(np2, (10, 1, 5))

        clang_res = np.frombuffer(res.base, np.float32).reshape(*res.shape)
        np.testing.assert_allclose(np1 * np3, clang_res)

    def test_elemwise_add(self):
        res = self.l1.elementwise(BinaryOps.ADD, self.l2)
        linearize(res.schedule())()
        np_res = (np.ones((10, 10)) * 10) + (np.ones((10, 10)) * 10)
        clang_res = np.frombuffer(res.base, np.float32).reshape(*res.shape)
        np.testing.assert_allclose(np_res, clang_res)

    def test_elemwise_add_strided(self):
        l1 = LazyBuffer.rand((10, 10, 5), device="CLANG")
        l2 = LazyBuffer.rand((1, 10, 1), device="CLANG")
        l2.movement(MovementOps.EXPAND, (10, 10, 5))

        res = l1.elementwise(BinaryOps.ADD, l2)
        linearize(res.schedule())()

        np1 = np.frombuffer(l1.base, np.float32).reshape(*l1.shape)
        np2 = np.frombuffer(l2.base, np.float32).reshape(1, 10, 1)
        np3 = np.tile(np2, (10, 1, 5))

        clang_res = np.frombuffer(res.base, np.float32).reshape(*res.shape)
        np.testing.assert_allclose(np1 + np3, clang_res)

    def test_elemwise_sub(self):
        res = self.l1.elementwise(BinaryOps.SUB, self.l2)
        linearize(res.schedule())()
        np_res = (np.ones((10, 10)) * 10) - (np.ones((10, 10)) * 10)
        clang_res = np.frombuffer(res.base, np.float32).reshape(*res.shape)
        np.testing.assert_allclose(np_res, clang_res)

    def test_elemwise_sub_strided(self):
        l1 = LazyBuffer.rand((10, 10, 5), device="CLANG")
        l2 = LazyBuffer.rand((1, 10, 1), device="CLANG")
        l2.movement(MovementOps.EXPAND, (10, 10, 5))

        res = l1.elementwise(BinaryOps.SUB, l2)
        linearize(res.schedule())()

        np1 = np.frombuffer(l1.base, np.float32).reshape(*l1.shape)
        np2 = np.frombuffer(l2.base, np.float32).reshape(1, 10, 1)
        np3 = np.tile(np2, (10, 1, 5))

        clang_res = np.frombuffer(res.base, np.float32).reshape(*res.shape)
        np.testing.assert_allclose(np1 - np3, clang_res)

    def test_elemwise_div(self):
        res = self.l1.elementwise(BinaryOps.DIV, self.l2)
        linearize(res.schedule())()
        np_res = (np.ones((10, 10)) * 10) / (np.ones((10, 10)) * 10)
        clang_res = np.frombuffer(res.base, np.float32).reshape(*res.shape)
        np.testing.assert_allclose(np_res, clang_res)

    def test_elemwise_div_strided(self):
        l1 = LazyBuffer.rand((10, 10, 5), device="CLANG")
        l2 = LazyBuffer.rand((1, 10, 1), device="CLANG")
        l2.movement(MovementOps.EXPAND, (10, 10, 5))

        res = l1.elementwise(BinaryOps.DIV, l2)
        linearize(res.schedule())()

        np1 = np.frombuffer(l1.base, np.float32).reshape(*l1.shape)
        np2 = np.frombuffer(l2.base, np.float32).reshape(1, 10, 1)
        np3 = np.tile(np2, (10, 1, 5))

        clang_res = np.frombuffer(res.base, np.float32).reshape(*res.shape)
        np.testing.assert_allclose(np1 / np3, clang_res)

    def test_elemwise_max(self):
        res = self.l1.elementwise(BinaryOps.MAX, self.l2)
        linearize(res.schedule())()
        np_res = np.maximum(np.ones((10, 10)) * 10, np.ones((10, 10)) * 10)
        clang_res = np.frombuffer(res.base, np.float32).reshape(*res.shape)
        np.testing.assert_allclose(np_res, clang_res)


    def test_elemwise_max_strided(self):
        l1 = LazyBuffer.rand((10, 10, 5), device="CLANG")
        l2 = LazyBuffer.rand((1, 10, 1), device="CLANG")
        l2.movement(MovementOps.EXPAND, (10, 10, 5))

        res = l1.elementwise(BinaryOps.MAX, l2)
        linearize(res.schedule())()

        np1 = np.frombuffer(l1.base, np.float32).reshape(*l1.shape)
        np2 = np.frombuffer(l2.base, np.float32).reshape(1, 10, 1)
        np3 = np.tile(np2, (10, 1, 5))

        clang_res = np.frombuffer(res.base, np.float32).reshape(*res.shape)
        np.testing.assert_allclose(np.maximum(np1, np3), clang_res)
