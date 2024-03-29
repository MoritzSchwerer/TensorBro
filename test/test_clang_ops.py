import unittest
import numpy as np

from tensorbro import LazyBuffer
from tensorbro.ops import BinaryOps, UnaryOps, MovementOps, ReduceOps
from tensorbro.linearizer import linearize

class TestMatmul(unittest.TestCase):
    def setUp(self):
        self.l1 = LazyBuffer.rand((10, 20), device="CLANG")
        self.l2 = LazyBuffer.rand((20, 10), device="CLANG")

    # def test_matmul_2d(self):
    #     res = self.l1.matmul(self.l2)
    #     linearize(res.schedule())()
    #
    #     np1 = np.frombuffer(self.l1.base, np.float32).reshape(10, 20)
    #     np2 = np.frombuffer(self.l2.base, np.float32).reshape(20, 10)
    #     np_res = np1 @ np2
    #     self.assertEqual(np_res.shape, res.shape)
    #
    #     clang_res = np.frombuffer(res.base, np.float32).reshape(res.shape)
    #     np.testing.assert_allclose(np_res, clang_res, rtol=1e-5)
    #
    # def test_matmul_3d(self):
    #     l1 = LazyBuffer.rand((5, 10, 20), device="CLANG")
    #     l2 = LazyBuffer.rand((20, 10), device="CLANG")
    #     res = l1.matmul(l2)
    #     linearize(res.schedule())()
    #
    #     np1 = np.frombuffer(l1.base, np.float32).reshape(5, 10, 20)
    #     np2 = np.frombuffer(l2.base, np.float32).reshape(20, 10)
    #     np_res = np1 @ np2
    #     self.assertEqual(np_res.shape, res.shape)
    #
    #     clang_res = np.frombuffer(res.base, np.float32).reshape(res.shape)
    #     np.testing.assert_allclose(np_res, clang_res, rtol=1e-5)

class TestLazyOpsReduce(unittest.TestCase):
    def setUp(self):
        self.l1 = LazyBuffer.rand((5, 10, 20), device="CLANG")

    def test_sum_1(self):
        res = self.l1.reduce(ReduceOps.SUM, 0)
        linearize(res.schedule())()

        np_res = np.frombuffer(self.l1.base, np.float32).reshape(5, 10, 20).sum(0)
        self.assertEqual(np_res.shape, res.shape)

        clang_res = np.frombuffer(res.base, np.float32).reshape(res.shape)
        np.testing.assert_allclose(np_res, clang_res)

    def test_sum_strided_1(self):
        l1 = LazyBuffer.rand((1, 10, 1), device="CLANG")
        res = l1.expand(20, 10, 5).reduce(ReduceOps.SUM, 0)
        linearize(res.schedule())()

        np_res = np.tile(np.frombuffer(l1.base, np.float32).reshape(1, 10, 1), (20, 1, 5)).sum(0)
        self.assertEqual(np_res.shape, res.shape)

        clang_res = np.frombuffer(res.base, np.float32).reshape(res.shape)
        np.testing.assert_allclose(np_res, clang_res)

    def test_sum_strided_2(self):
        l1 = LazyBuffer.rand((1, 10, 1), device="CLANG")
        res = l1.expand(20, 10, 5).reduce(ReduceOps.SUM, 1)
        linearize(res.schedule())()

        np_res = np.tile(np.frombuffer(l1.base, np.float32).reshape(1, 10, 1), (20, 1, 5)).sum(1)
        self.assertEqual(np_res.shape, res.shape)

        clang_res = np.frombuffer(res.base, np.float32).reshape(res.shape)
        np.testing.assert_allclose(np_res, clang_res)

    def test_sum_strided_3(self):
        l1 = LazyBuffer.rand((1, 10, 1), device="CLANG")
        res = l1.expand(20, 10, 5).reduce(ReduceOps.SUM, 2)
        linearize(res.schedule())()

        np_res = np.tile(np.frombuffer(l1.base, np.float32).reshape(1, 10, 1), (20, 1, 5)).sum(2)
        self.assertEqual(np_res.shape, res.shape)

        clang_res = np.frombuffer(res.base, np.float32).reshape(res.shape)
        np.testing.assert_allclose(np_res, clang_res)

    def test_max_1(self):
        res = self.l1.reduce(ReduceOps.MAX, 0)
        linearize(res.schedule())()

        np_res = np.frombuffer(self.l1.base, np.float32).reshape(5, 10, 20).max(0)
        self.assertEqual(np_res.shape, res.shape)

        clang_res = np.frombuffer(res.base, np.float32).reshape(res.shape)
        np.testing.assert_allclose(np_res, clang_res)

    def test_max_strided_1(self):
        l1 = LazyBuffer.rand((1, 10, 1), device="CLANG")
        res = l1.expand(20, 10, 5).reduce(ReduceOps.MAX, 0)
        linearize(res.schedule())()

        np_res = np.tile(np.frombuffer(l1.base, np.float32).reshape(1, 10, 1), (20, 1, 5)).max(0)
        self.assertEqual(np_res.shape, res.shape)

        clang_res = np.frombuffer(res.base, np.float32).reshape(res.shape)
        np.testing.assert_allclose(np_res, clang_res)

    def test_max_strided_2(self):
        l1 = LazyBuffer.rand((1, 10, 1), device="CLANG")
        res = l1.expand(20, 10, 5).reduce(ReduceOps.MAX, 1)
        linearize(res.schedule())()

        np_res = np.tile(np.frombuffer(l1.base, np.float32).reshape(1, 10, 1), (20, 1, 5)).max(1)
        self.assertEqual(np_res.shape, res.shape)

        clang_res = np.frombuffer(res.base, np.float32).reshape(res.shape)
        np.testing.assert_allclose(np_res, clang_res)

    def test_max_strided_3(self):
        l1 = LazyBuffer.rand((1, 10, 1), device="CLANG")
        res = l1.expand(20, 10, 5).reduce(ReduceOps.MAX, 2)
        linearize(res.schedule())()

        np_res = np.tile(np.frombuffer(l1.base, np.float32).reshape(1, 10, 1), (20, 1, 5)).max(2)
        self.assertEqual(np_res.shape, res.shape)

        clang_res = np.frombuffer(res.base, np.float32).reshape(res.shape)
        np.testing.assert_allclose(np_res, clang_res)



class TestLazyOpsMovement(unittest.TestCase):
    def setUp(self):
        self.l1 = LazyBuffer.full(10, (10, 10), device="CLANG")
        self.l2 = LazyBuffer.rand((2, 4, 3), device="CLANG")

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

    def test_permute_1(self):
        res = self.l2.permute(0, 2, 1)
        linearize(res.schedule())()

        np_res = np.frombuffer(self.l2.base, np.float32).reshape(2, 4, 3).transpose(0, 2, 1)
        clang_res = np.frombuffer(res.base, np.float32).reshape(*res.shape)
        self.assertEqual(np_res.shape, clang_res.shape)
        np.testing.assert_allclose(np_res, clang_res)

    def test_permute_2(self):
        res = self.l2.permute(2, 1, 0)
        linearize(res.schedule())()

        np_res = np.frombuffer(self.l2.base, np.float32).reshape(2, 4, 3).transpose(2, 1, 0)
        clang_res = np.frombuffer(res.base, np.float32).reshape(*res.shape)
        self.assertEqual(np_res.shape, clang_res.shape)
        np.testing.assert_allclose(np_res, clang_res)

    def test_permute_strided_1(self):
        res = self.l2.reshape(*self.l2.shape, 1).expand(2, 4, 3, 5).permute(0, 2, 1, 3)
        linearize(res.schedule())()

        np_res = np.tile(np.frombuffer(self.l2.base, np.float32).reshape(2, 4, 3, 1), (1, 1, 1, 5)).transpose(0, 2, 1, 3)
        clang_res = np.frombuffer(res.base, np.float32).reshape(*res.shape)
        self.assertEqual(np_res.shape, clang_res.shape)
        np.testing.assert_allclose(np_res, clang_res)

    def test_permute_strided_2(self):
        res = self.l2.reshape(*self.l2.shape, 1).expand(2, 4, 3, 5).permute(3, 0, 2, 1)
        linearize(res.schedule())()

        np_res = np.tile(np.frombuffer(self.l2.base, np.float32).reshape(2, 4, 3, 1), (1, 1, 1, 5)).transpose(3, 0, 2, 1)
        clang_res = np.frombuffer(res.base, np.float32).reshape(*res.shape)
        self.assertEqual(np_res.shape, clang_res.shape)
        np.testing.assert_allclose(np_res, clang_res)

        

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
