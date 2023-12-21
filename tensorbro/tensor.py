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
    def __init__(self, data) -> None:
        self.data = data
        self.context: Optional[Context] = None

    def __repr__(self):
        return f'Tensor: data={self.data}\n'

    def __mul__(self, other):
        return ops.Mul.apply(self, other)

    def __add__(self, other):
        return ops.Add.apply(self, other)
