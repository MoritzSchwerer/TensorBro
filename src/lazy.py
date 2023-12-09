import math
import numpy as np

from typing import Optional, Tuple, Union
from .ops import LazyOp, OpType, BinaryOps, UnaryOps, TernaryOps


class ShapeTracker:
    def __init__(self, view, stride):
        self._views = [view]
        self._strides = [stride]

    @property
    def view(self) -> Tuple[int, ...]:
        return self._views[-1]

    def stride(self) -> Tuple[int, ...]:
        return self._strides[-1]

    def __repr__(self):
        return f'ST: views={self._views}, strides={self._strides}.'


class LazyBuffer:
    def __init__(self, op: Optional[LazyOp], device: str, shape_tracker, base=None):
        self.op: Optional[LazyOp] = op
        self.device: str = device
        self.shape_tracker = shape_tracker
        self.base = base

    def __repr__(self):
        return f'LazyBuffer: op={self.op}, device={self.device}, st={self.shape_tracker}'

    def schedule(self, seen=None):
        seen = seen if seen is not None else set()
        seen.add(self)

        if self.op is not None:
            for src in self.op.srcs:
                src.schedule(seen)

        return seen

    # TODO: if we find elemwise, movement elemwise pattern
    # we can push the movement op above the first elementwise op
    def elementwise(self, op: Union[UnaryOps, BinaryOps, TernaryOps], *srcs):
        for src in srcs:
            assert (
                src.shape_tracker.view == self.shape_tracker.view
            ), f'Shapes do not match, broadcasting not implemented yet.'
        srcs = (self,) + srcs
        lazy_op = LazyOp(op, srcs)
        return LazyBuffer(lazy_op, self.device, self.shape_tracker)

    def matmul(self, other):
        # this will be expand, elementwise mul and sum
        raise NotImplementedError('Not implemented yet.')
