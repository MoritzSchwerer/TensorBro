import math
import numpy as np

from typing import Optional, Tuple
from .ops import LazyOp, OpType, BinaryOps


class LazyBuffer:
    def __init__(self, op: Optional[LazyOp], device: str, shape: Tuple[int, ...], base=None):
        self.op: Optional[LazyOp] = op
        self.device: str = device
        self.shape: Tuple[int, ...] = shape
        self.base = base

    def reg_binary(self, op: OpType, *srcs):
        srcs = (self,) + srcs
        lazy_op = LazyOp(op, srcs)
        return LazyBuffer(lazy_op, self.device, self.shape)

    def __mul__(self, other):
        return self.reg_binary(BinaryOps.Mul, other)

    def __add__(self, other):
        return self.reg_binary(BinaryOps.Add, other)

    def __repr__(self):
        return f'LazyBuffer: op={self.op}, device={self.device}, shape={self.shape}, base={self.base}'

    def schedule(self, seen=None):
        seen = seen if seen is not None else set()
        seen.add(self)

        if self.op is not None:
            for src in self.op.srcs:
                src.schedule(seen)

        return seen
