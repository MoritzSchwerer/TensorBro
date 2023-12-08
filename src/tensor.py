from typing import Optional


class Context:
    def __init__(self, *inputs: 'Tensor'):
        self.parents: 'Tensor' = inputs

    def forward(self, inputs):
        pass

    def backward(self, out_grad):
        pass

    def __repr__(self):
        return f'{self.__class__.__name__}: parents={self.parents}'

    @classmethod
    def apply(func, *args: 'Tensor'):
        # create the context for the op
        context = func(*args)
        # call forward method
        result = Tensor(context.forward(*[t.data for t in args]))
        result.context = context
        return result


import src.ops as ops


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
