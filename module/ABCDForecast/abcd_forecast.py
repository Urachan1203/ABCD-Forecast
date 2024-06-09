from module.ABCDForecast.forecaster import Forecaster
from module.ABCDForecast.X.generator import XGeneratorRandom
import numpy as np

class ABCDForecast:
    def __init__(self, 
                 X,
                 Y=None,
                 num_forecaster = 100,
                 X_generators = None,
                 num_stock = 500,
                 mode='train'
                 ) -> None:
        self.forecasters = []

        for i in range(num_forecaster):
            self.forecasters.append(
                Forecaster(X, Y, X_generator=X_generators[i], num_stock=num_stock, mode=mode)
            )
        
    def train(self):
        for i, forecaster in enumerate(self.forecasters):
            print(f"training forecast {i}") 
            forecaster.train()

    def predict(self, X):
        Y_estimated = []
        for forecaster in self.forecasters : 
            Y_estimated.append(forecaster.predict(X))
        return np.array(Y_estimated)
            
    
    def detransform_y(self, Y):
        Y_detransformed = []
        for forecaster in self.forecasters :
            Y_detransformed.append(forecaster.detransform_y(Y))
        return np.array(Y_detransformed)