import torch

from src.modules.dense_layer import DenseLayer

if __name__ == "__main__":
    print("Test: this should converge to 0")
    layer = DenseLayer(4, 5)
    print(layer.weights)
    params = layer.get_params()
    params['weights'] += 10
    print(layer.weights)

