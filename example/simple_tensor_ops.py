#################################################################
###       NOTE: this needs to be run with PYTHONPATH='.'      ###
#################################################################

import numpy as np
from tensorbro import Tensor

# initialize random tensors
t1 = Tensor.rand((10, 10))
t2 = Tensor.rand((10, 10))
t3 = Tensor.rand((10, 10))

# perform matmul
res = (t1 @ t2) * t3

# materialize result because by default we are lazy
res.materialize()

# copy data from clang buffer over into numpy arrays
np_res = np.frombuffer(res.data.base, np.float32).reshape(*res.data.shape)
np_t1 = np.frombuffer(t1.data.base, np.float32).reshape(*t1.data.shape)
np_t2 = np.frombuffer(t2.data.base, np.float32).reshape(*t2.data.shape)
np_t3 = np.frombuffer(t3.data.base, np.float32).reshape(*t3.data.shape)

# check if result is correct
if np.allclose(np_res, (np_t1 @ np_t2) * np_t3, rtol=1e-5):
    print("Yay, result is correct!")
else:
    print("There is an error :(")
