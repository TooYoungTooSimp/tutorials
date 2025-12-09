import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import numpy as np


@cute.jit
def load_and_store(res: cute.Tensor, a: cute.Tensor, b: cute.Tensor):
    """
    Load data from memory and store the result to memory.
    :param res: The destination tensor to store the result.
    :param a: The source tensor to be loaded.
    :param b: The source tensor to be loaded.
    """
    a_vec = a.load()
    print(f"a_vec: {a_vec}")
    b_vec = b.load()
    print(f"b_vec: {b_vec}")
    res.store(a_vec + b_vec)
    cute.print_tensor(res)


a = np.ones(1024).reshape((-1, 4)).astype(np.float32)
b = np.ones(1024).reshape((-1, 4)).astype(np.float32)
c = np.zeros(1024).reshape((-1, 4)).astype(np.float32)
load_and_store(from_dlpack(c), from_dlpack(a), from_dlpack(b))
