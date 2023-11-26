
class BaseOptim():
    def __init__(self):
        """
        Set all the parameters for the optimizer.
        ex. learning rate, momentum...
        """
        pass

    def initialize_params(self, params: list[dict]) -> None:
        """
        Initialize the parameters of the model that will be updated
        by this optimizer.
        """
        self._model_params = params

    def update_params(self, deltas: list[dict]) -> None:
        raise NotImplementedError(f"Method initialize_params not implemented for {self.__class__.__name__}.")
