import torch

class BaseModule():
    """
    The base module class of this framework (like torch.nn.Module).
    """

    def __init__(self):
        pass

    def forward(self, input: torch.Tensor):
        raise NotImplementedError(f"Method forward not implemented for {self.__class__.__name__}.")

    def backward(self, output: torch.Tensor):
        raise NotImplementedError(f"Method backward not implemented for {self.__class__.__name__}.")

    def __call__(self, input):
        return self.forward(input)

    def get_params(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError(f"Method get_params not implemented for {self.__class__.__name__}.")

