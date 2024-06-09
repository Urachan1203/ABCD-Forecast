from module.ABCDForecast.X.generator_base import XGeneratorBase
from module.ABCDForecast.matrix_g import MatrixG
from numba import jit


class XGeneratorRandom(XGeneratorBase):
    def __init__(self, num_stock = 500) -> None:
        super().__init__()
        self.F = MatrixG(dim=num_stock)
    
    @jit
    def f(self, X):
        for i in range(X.shape[0]):
            X[i] = self.F * X[i]
        return X