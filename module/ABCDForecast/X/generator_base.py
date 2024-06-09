import abc
import numpy as np

class XGeneratorBase(metaclass=abc.ABCMeta):
    """
    raw inputの拡張を行うインターフェース（論文中のfに相当）
    input -> [batch, stock, time]
    """
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def f(self, X) -> np.ndarray:
        raise NotImplementedError()
