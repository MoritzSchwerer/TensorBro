from typing import Optional, Tuple
import tensorbro.ops as ops

class Context:
    def __init__(self, *inputs: 'Tensor'):
        self.parents: Tuple['Tensor', ...] = inputs

    def forward(self, inputs):
        pass

    def backward(self, out_grad):
        pass

    def __repr__(self):
        return f'{self.__class__.__name__}: parents={self.parents}'

    @classmethod
    def apply(func, *args: 'Tensor'):
        context = func(*args)
        result = Tensor(context.forward(*[t.data for t in args]))
        result.context = context
        return result

class Tensor:
    def __init__(self, data, device="CLANG") -> None:
        self.data = data
        self.context: Optional[Context] = None
        self.device = device

    def __repr__(self):
        return f'Tensor: data={self.data}\n'

    @staticmethod
    def full(value, shape, device="CLANG"):
        from tensorbro import LazyBuffer
        return Tensor(LazyBuffer.full(value, shape, device))

    @staticmethod
    def ones(shape, device="CLANG"):
        return Tensor.full(1, shape, device)

    @staticmethod
    def zeros(shape, device="CLANG"):
        return Tensor.full(0, shape, device)

    @staticmethod
    def rand(shape, device="CLANG"):
        from tensorbro import LazyBuffer
        return Tensor(LazyBuffer.rand(shape, device))

    def __mul__(self, other):
        return ops.Mul.apply(self, other)

    def __add__(self, other):
        return ops.Add.apply(self, other)

    def __sub__(self, other):
        return ops.Sub.apply(self, other)
