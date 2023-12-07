from .tensor import Context

from dataclasses import dataclass
from enum import Enum, auto

from typing import Union, Type, Tuple, Any


class LoadOps(Enum):
    Empty = auto()
    Const = auto()
    Rand = auto()


class BinaryOps(Enum):
    Mul = auto()
    Add = auto()


OpType = Union[Type[BinaryOps], Type[LoadOps]]


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
