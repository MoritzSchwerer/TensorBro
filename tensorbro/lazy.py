import math
import ctypes as c

from typing import Optional, Tuple, Union, List

from .ops import LazyOp, BinaryOps, UnaryOps, TernaryOps, LoadOps, MovementOps, ReduceOps
from .runners.clang import CAllocator
from .linearizer import ScheduleItem

def permute_shape(shape, dim):
    return tuple(map(lambda i: shape[i], dim))

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

    def movement(self, op: MovementOps, arg: Tuple[int, ...]):
        if op is MovementOps.PERMUTE:
            src = (self,)
            lazy_op = LazyOp(op, src, arg) # type: ignore
            new_shape = permute_shape(self.shape, arg)
            return LazyBuffer(lazy_op, self.device, ShapeTracker.from_shape(new_shape))
        new_shape = arg

        if op is MovementOps.EXPAND:
            assert len(self.shape) == len(new_shape), "Shapes must have the same number of dimensions for expand."
            for x, y in zip(self.shape, new_shape):
                assert x == y or x == 1, f"Shape of original buffer must be 1 at the expanded dimensions, not: {x}"
            expand = tuple([(y // x) for x, y in zip(self.shape, new_shape)])
            self.st._strides.append(expand)
        self.st._views.append(new_shape)
        return self

    def permute(self, *args: int):
        assert len(self.shape) == len(args), "Length of shape needs to be same as permute inputs"
        return self.movement(MovementOps.PERMUTE, tuple(args))

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
        return LazyBuffer(lazy_op, self.device, self.st)

    def __mul__(self, other):
        return self.elementwise(BinaryOps.MUL, other)

    def __add__(self, other):
        return self.elementwise(BinaryOps.ADD, other)

    def __sub__(self, other):
        return self.elementwise(BinaryOps.SUB, other)

    def __div__(self, other):
        return self.elementwise(BinaryOps.DIV, other)

    def matmul(self, other):
        n1, n2 = len(self.shape), len(other.shape)
        assert n1 > 1 and n2 > 1, "Buffers both need to be at least 2D"
        assert self.shape[-1] == other.shape[-min(n2, 2)], f"Buffers not compatible for matmul: {self.shape} and {other.shape}."
        x = self.reshape(*self.shape[0:-1], *[1]*min(n1-1, n2-1, 1), self.shape[-1])
        y = other.reshape(*other.shape[0:-2], *[1]*min(n1-1, n2-1, 1), *other.shape[-min(n2, 2):]).transpose(-1, -min(n2, 2))

    def reduce(self, op: ReduceOps, dim: int = 0):
        new_shape = tuple([size for i, size in enumerate(self.shape) if i != dim])
        lazy_op = LazyOp(op, (self,), dim) # type: ignore
        return LazyBuffer(lazy_op, self.device, ShapeTracker.from_shape(new_shape))


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
