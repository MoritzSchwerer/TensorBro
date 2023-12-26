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
        np_res = np.ones((100, 20)) * 10
        clang_res = np.frombuffer(res.base, np.float32).reshape(*res.shape)
        np.testing.assert_allclose(np_res, clang_res)
        
    # def test_permute(self):
    #     res = self.l1.movement(MovementOps.RESHAPE, (20, 5))
    #     linearize(res.schedule())()
    #     np_res = np.ones((20, 5)) * 10
    #     clang_res = np.frombuffer(res.base, np.float32).reshape(*res.shape)
    #     np.testing.assert_allclose(np_res, clang_res)

    # def test_pad(self):
    #     res = self.l1.movement(MovementOps.RESHAPE, (20, 5))
    #     linearize(res.schedule())()
    #     np_res = np.ones((20, 5)) * 10
    #     clang_res = np.frombuffer(res.base, np.float32).reshape(*res.shape)
    #     np.testing.assert_allclose(np_res, clang_res)




class TestLazyOpsUnary(unittest.TestCase):
    def setUp(self):
        self.l1 = LazyBuffer.full(10, (10, 10), device="CLANG")

    def test_unary_neg(self):
        res = self.l1.elementwise(UnaryOps.NEG)
        linearize(res.schedule())()
        np_res = -np.ones((10, 10)) * 10
        clang_res = np.frombuffer(res.base, np.float32).reshape(*res.shape)
        np.testing.assert_allclose(np_res, clang_res)

    def test_unary_sin(self):
        res = self.l1.elementwise(UnaryOps.SIN)
        linearize(res.schedule())()
        np_res = np.sin(np.ones((10, 10)) * 10)
        clang_res = np.frombuffer(res.base, np.float32).reshape(*res.shape)
        np.testing.assert_allclose(np_res, clang_res)

    def test_unary_sqrt(self):
        res = self.l1.elementwise(UnaryOps.SQRT)
        linearize(res.schedule())()
        np_res = np.sqrt(np.ones((10, 10)) * 10)
        clang_res = np.frombuffer(res.base, np.float32).reshape(*res.shape)
        np.testing.assert_allclose(np_res, clang_res)

    def test_unary_exp2(self):
        res = self.l1.elementwise(UnaryOps.EXP2)
        linearize(res.schedule())()
        np_res = np.exp2(np.ones((10, 10)) * 10)
        clang_res = np.frombuffer(res.base, np.float32).reshape(*res.shape)
        np.testing.assert_allclose(np_res, clang_res)

    def test_unary_log2(self):
        res = self.l1.elementwise(UnaryOps.LOG2)
        linearize(res.schedule())()
        np_res = np.log2(np.ones((10, 10)) * 10)
        clang_res = np.frombuffer(res.base, np.float32).reshape(*res.shape)
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

    def test_elemwise_add(self):
        res = self.l1.elementwise(BinaryOps.ADD, self.l2)
        linearize(res.schedule())()
        np_res = (np.ones((10, 10)) * 10) + (np.ones((10, 10)) * 10)
        clang_res = np.frombuffer(res.base, np.float32).reshape(*res.shape)
        np.testing.assert_allclose(np_res, clang_res)

    def test_elemwise_sub(self):
        res = self.l1.elementwise(BinaryOps.SUB, self.l2)
        linearize(res.schedule())()
        np_res = (np.ones((10, 10)) * 10) - (np.ones((10, 10)) * 10)
        clang_res = np.frombuffer(res.base, np.float32).reshape(*res.shape)
        np.testing.assert_allclose(np_res, clang_res)

    def test_elemwise_div(self):
        res = self.l1.elementwise(BinaryOps.DIV, self.l2)
        linearize(res.schedule())()
        np_res = (np.ones((10, 10)) * 10) / (np.ones((10, 10)) * 10)
        clang_res = np.frombuffer(res.base, np.float32).reshape(*res.shape)
        np.testing.assert_allclose(np_res, clang_res)

    def test_elemwise_max(self):
        res = self.l1.elementwise(BinaryOps.MAX, self.l2)
        linearize(res.schedule())()
        np_res = np.maximum(np.ones((10, 10)) * 10, np.ones((10, 10)) * 10)
        clang_res = np.frombuffer(res.base, np.float32).reshape(*res.shape)
        np.testing.assert_allclose(np_res, clang_res)
