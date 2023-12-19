import math
from typing import Optional, Tuple, Union, List, Dict, Any
from .ops import LazyOp, BinaryOps, UnaryOps, TernaryOps, LoadOps, BufferOps, MovementOps

from dataclasses import dataclass

import ctypes as c


class ShapeTracker:
    def __init__(self, view: Tuple[int, ...], stride: Optional[Tuple[int, ...]] = None):
        self._views: List[Tuple[int, ...]] = [view]
        self._strides: List[Tuple[int, ...]] = [stride] if stride is not None else [tuple(1 for _ in range(len(view)))]

    @staticmethod
    def from_shape(shape: Tuple[int, ...]):
        return ShapeTracker(shape)

    @property
    def view(self) -> Tuple[int, ...]:
        return self._views[-1]

    def stride(self) -> Tuple[int, ...]:
        return self._strides[-1]

    def __repr__(self):
        # return f'ST: views={self._views}, strides={self._strides}.'
        return '<ST>'


@dataclass(frozen=True)
class ScheduleItem:
    op: LazyOp
    target: 'LazyBuffer'
    srcs: Tuple[Union['LazyBuffer', LazyOp], ...]


class LazyBuffer:
    def __init__(self, op: Optional[LazyOp], device: str, shape_tracker, base=None):
        self.op: Optional[LazyOp] = op
        self.device: str = device
        self.shape_tracker = shape_tracker
        self._base = base
        self._realized = True if base is not None else False

    @property
    def shape(self):
        return self.shape_tracker.view

    @property
    def is_realized(self) -> bool:
        return self._realized

    @property
    def size(self):
        return math.prod(self.shape_tracker.view)

    @property
    def is_pointer(self) -> bool:
        return isinstance(self.base, c._CData)

    def realize(self, value=None):
        self._realized = True
        if value is not None:
            self._base = value
        else:
            self._base = CAllocator.alloc(c.c_float, math.prod(self.shape_tracker.view))

    @property
    def base(self):
        return self._base

    def __repr__(self):
        return f'<LazyBuffer: op={self.op.op}, realized={self.is_realized}>'

    """
    This should just split the tree at movement ops and
    keep the sub trees
    """

    def schedule(self, seen=None):
        seen = dict() if seen is None else seen
        if self in seen:
            return []

        op = self.op

        ret = []
        for src in op.buffers:
            ret += src.schedule()

        ret.append(ScheduleItem(op, self, tuple(op.buffers)))
        return ret

    @property
    def buffers(self):
        return (self,)

    # this is fake just to test something
    def movement(self, op: MovementOps, new_shape):
        lazy_op = LazyOp(op, (self,))
        return LazyBuffer(lazy_op, self.device, ShapeTracker(new_shape))

    def reshape(self, new_shape):
        return self.movement(MovementOps.RESHAPE, new_shape)

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

    @staticmethod
    def rand(shape, device):
        lazy_op = LazyOp(LoadOps.RAND, (), arg=1)
        return LazyBuffer(lazy_op, device, ShapeTracker(shape))

    @staticmethod
    def full(value, shape, device='CPU'):
        lazy_op = LazyOp(LoadOps.CONST, (), value)
        st = ShapeTracker.from_shape(shape)
        return LazyBuffer(lazy_op, device, st)

    @property
    def shape(self):
        return self.shape_tracker.view
