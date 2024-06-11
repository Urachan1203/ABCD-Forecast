from module.ABCDForecast.X.generator_base import XGeneratorBase
from module.ABCDForecast.matrix_g import MatrixG
from numba import jit


class XGeneratorRandom(XGeneratorBase):
    """
    Yの変換と同様のMatrixを使ってXも変換する。銘柄iと銘柄jのスプレッドリターンになる。
    """
    def __init__(self, num_stock = 500) -> None:
        super().__init__()
        self.F = MatrixG(dim=num_stock)
    
    def f(self, X):
        for i in range(X.shape[0]):
            X[i] = self.F * X[i]
        return X