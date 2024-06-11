import numpy as np
from numba import jit

class MatrixG:
    def __init__(self, dim = 500) -> None:
        self.A = self.init(dim)
        self.A = self.permutate()

    def init(self, dim):
        A = np.eye(dim)
        for i in range(1, dim): A[i][i-1] = -1
        return A

    def permutate(self):
        A_permuted = np.random.permutation(self.A)
        A_permuted = np.random.permutation(A_permuted.T).T
        return A_permuted
    
    def inv(self):
        inv_A = np.linalg.inv(self.A)
        return MatrixG.from_array(inv_A)
    
    def __mul__(self, other):
        if isinstance(other, np.ndarray):
            return np.dot(self.A, other)
        else:
            raise ValueError(f"Not supported combination, {type(self)} and {type(other)}")
    
    @classmethod
    def from_array(cls, array):
        obj = cls(dim=array.shape[0])
        obj.A = array
        return obj