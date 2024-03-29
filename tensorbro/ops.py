from dataclasses import dataclass
from enum import Enum, auto

from typing import Union, Type, Tuple, Any
from .tensor import Context

class LoadOps(Enum):
    EMPTY = auto()
    CONST = auto()
    RAND = auto()


class UnaryOps(Enum):
    NEG = auto()
    SIN = auto()
    SQRT = auto()
    EXP2 = auto()
    LOG2 = auto()


class BinaryOps(Enum):
    MUL = auto()
    ADD = auto()
    SUB = auto()
    DIV = auto()
    MAX = auto()
    MATMUL = auto()


class TernaryOps(Enum):
    MULACC = auto()
    WHERE = auto()


class ReduceOps(Enum):
    SUM = auto()
    MAX = auto()


class MovementOps(Enum):
    RESHAPE = auto()
    EXPAND = auto()
    PERMUTE = auto()
    PAD = auto()


Op = Union[UnaryOps, BinaryOps, ReduceOps, MovementOps, LoadOps, TernaryOps]
OpType = Union[
    Type[UnaryOps],
    Type[BinaryOps],
    Type[ReduceOps],
    Type[MovementOps],
    Type[LoadOps],
    Type[TernaryOps],
]


@dataclass(frozen=True)
class LazyOp:
    op: OpType
    srcs: Tuple[Union[Any, Any], ...]
    arg: Any = None

    @property
    def buffers(self):
        return sum([s.buffers for s in self.srcs], ())

class Mul(Context):
    def forward(self, x, y):
        self.x, self.y = x, y
        return x * y

    def backward(self, out_grad):
        return self.y * out_grad, self.x * out_grad

    def __repr__(self):
        return f'Mul: x={self.x}, y={self.y}'

class Add(Context):
    def forward(self, x, y):
        return x + y

    def backward(self, out_grad):
        return out_grad, out_grad

class Sub(Context):
    def forward(self, x, y):
        return x - y

    def backward(self, out_grad):
        return out_grad, -out_grad

class Matmul(Context):
    def forward(self, x, y):
        self.x, self.y = x, y
        return x.matmul(y)

    def backward(self, out_grad):
        return out_grad.matmul(self.y), self.x.matmul(out_grad)

    def __repr__(self):
        return f'Mul: x={self.x}, y={self.y}'
