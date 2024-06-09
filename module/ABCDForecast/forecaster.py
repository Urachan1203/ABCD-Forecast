from module.ABCDForecast.matrix_g import MatrixG
from module.ABCDForecast.X.generator import XGeneratorRandom
from sklearn.tree import DecisionTreeRegressor
from numba import jit

class Forecaster:
    def __init__(self, 
                 X, 
                 Y = None,
                 X_generator = None,
                 num_stock = 500,
                 mode = "train"
                 ) -> None:
        if mode == "train" and Y == None : raise Exception("Y must be specified in training mode.")

        self.G = MatrixG(dim = num_stock)
        self.X = X
        self.Y = Y
        self.X_generator = X_generator
        self.X_transform = self.X_generator.f(X)
        self.predictor = DecisionTreeRegressor()

        if mode == "train" : self.Y_transform = self.transform_y(Y)
        else : self.Y_transform = None
    
    def train(self):
        self.predictor.fit(self.X_transform, self.Y_transform)
        return
    
    def predict(self, X):
        return self.predictor.predict(self.X_generator.f(X))
    
    @jit
    def transform_y(self, Y):
        for i in range(Y.shape[0]):
            Y[i] = self.G * Y[i]
        return Y
    
    def detransform_y(self, Y):
        for i in range(Y.shape[0]):
            Y[i] = self.G.inv() * Y[i]
        return Y