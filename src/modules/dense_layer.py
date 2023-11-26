import torch

from .base_module import BaseModule

class DenseLayer(BaseModule):
    def __init__(self, in_dim, out_dim):
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._weights = None
        self._bias = None
        self._deltas = {}
        self._init_params()


    def _init_params(self) -> None:
        # TODO: this initializiaton is just for testing
        self._weights = torch.rand((self._in_dim, self._out_dim))
        self._bias = torch.rand((1, self._out_dim))


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._input = input
        return input @ self._weights + self._bias 


    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        self._deltas['bias'] = grad.mean(0)
        self._deltas['weights'] = self._input.T @ grad
        return grad @ self._weights.T

    @property
    def deltas(self) -> dict[str, torch.Tensor]:
        return self._deltas

    @property
    def params(self) -> dict[str, torch.Tensor]:
        return {
            'weights': self._weights,
            'bias': self._bias
        }
        


