from .tensor import Context

from dataclasses import dataclass
from enum import Enum, auto

from typing import Union, Type, Tuple, Any


class LoadOps(Enum):
    EMPTY = auto()
    CONST = auto()
    RAND = auto()


class UnaryOps(Enum):
    NEG = auto()
    SIN = auto()
    CAST = auto()
    SQRT = auto()
    EXP2 = auto()
    LOG2 = auto()


class BinaryOps(Enum):
    MUL = auto()
    ADD = auto()
    SUB = auto()
    DIV = auto()
    MAX = auto()
    MOD = auto()


class TernaryOps(Enum):
    MULACC = auto()
    WHERE = auto()


class ReduceOps(Enum):
    SUM = auto()
    MAX = auto()


class BufferOps(Enum):
    LOAD = auto()
    CONST = auto()
    STORE = auto()


class MovementOps(Enum):
    RESHAPE = auto()
    EXPAND = auto()
    PERMUTE = auto()
    PAD = auto()


Op = Union[UnaryOps, BinaryOps, ReduceOps, MovementOps, LoadOps, TernaryOps, BufferOps]
OpType = Union[
    Type[UnaryOps],
    Type[BinaryOps],
    Type[ReduceOps],
    Type[MovementOps],
    Type[LoadOps],
    Type[TernaryOps],
    Type[BufferOps],
]


@dataclass(frozen=True)
class LazyOp:
    op: OpType
    srcs: Tuple[Union[Any, Any], ...]
    arg: Any = None


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
