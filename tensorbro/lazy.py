import math
import ctypes as c

from typing import Optional, Tuple, Union, List

from .ops import LazyOp, BinaryOps, UnaryOps, TernaryOps, LoadOps, MovementOps
from .runners.clang import CAllocator
from .linearizer import ScheduleItem


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

    @property
    def stride(self) -> Tuple[int, ...]:
        return self._strides[-1]

    def __repr__(self):
        return '<ST>'


class LazyBuffer:
    def __init__(self, op: Optional[LazyOp], device: str, shape_tracker, base=None):
        self.op: Optional[LazyOp] = op
        self.device: str = device
        self.shape_tracker: ShapeTracker = shape_tracker
        self._base = base
        self._realized: bool = True if base is not None else False

    @property
    def st(self):
        return self.shape_tracker

    @property
    def shape(self):
        return self.st.view

    @property
    def is_realized(self) -> bool:
        return self._realized

    @property
    def size(self):
        return math.prod(self.shape)

    def realize(self, value=None):
        self._realized = True
        if value is not None:
            self._base = value
        else:
            self._base = CAllocator.alloc(c.c_float, math.prod([sh // st for sh, st in zip(self.shape, self.st.stride)]))

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

    def movement(self, op: MovementOps, new_shape: Tuple[int, ...]):
        if op is MovementOps.EXPAND:
            assert len(self.shape) == len(new_shape), "Shapes must have the same number of dimensions for expand."
            for x, y in zip(self.shape, new_shape):
                assert x == y or x == 1, f"Shape of original buffer must be 1 at the expanded dimensions, not: {x}"
            expand = tuple([(y // x) for x, y in zip(self.shape, new_shape)])
            self.st._strides.append(expand)
        self.st._views.append(new_shape)
        return self

    def reshape(self, *new_shape: int):
        return self.movement(MovementOps.RESHAPE, tuple(new_shape))

    def expand(self, *new_shape: int):
        return self.movement(MovementOps.EXPAND, tuple(new_shape))

    # TODO: if we find elemwise, movement elemwise pattern
    # we can push the movement op above the first elementwise op
    def elementwise(self, op: Union[UnaryOps, BinaryOps, TernaryOps], *srcs):
        for src in srcs:
            assert (
                src.shape == self.shape
            ), 'Shapes do not match, broadcasting not implemented yet.'
        srcs = (self,) + srcs
        lazy_op = LazyOp(op, srcs) # type: ignore
        return LazyBuffer(lazy_op, self.device, self.shape_tracker)


    # utility functions to make life easier
    @staticmethod
    def rand(shape, device, seed=1):
        lazy_op = LazyOp(LoadOps.RAND, (), arg=seed)
        return LazyBuffer(lazy_op, device, ShapeTracker(shape))

    @staticmethod
    def full(value, shape, device='CPU'):
        lazy_op = LazyOp(LoadOps.CONST, (), value)
        st = ShapeTracker.from_shape(shape)
        return LazyBuffer(lazy_op, device, st)
