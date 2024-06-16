from module.ABCDForecast.matrix_g import MatrixG
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from numba import jit
import numpy as np
import copy

class Forecaster:
    def __init__(self, 
                 X, 
                 Y = None,
                 X_generator = None,
                 num_stock = 500,
                 mode = "train"
                 ) -> None:
        if mode == "train" and not isinstance(Y, np.ndarray) : raise Exception("Y must be specified in training mode.")

        self.G = MatrixG(dim = num_stock)
        self.X_generator = X_generator
        self.X_transform = self.X_generator.f(copy.deepcopy(X))
        self.predictor = DecisionTreeRegressor()

        if mode == "train" : self.Y_transform = self.transform_y(copy.deepcopy(Y))
        else : self.Y_transform = None
    
    def train(self):
        self.predictor.fit(
            self.X_transform.reshape(
                (self.X_transform.shape[0], self.X_transform.shape[1] * self.X_transform.shape[2])
                ), 
            self.Y_transform)
        return
    
    def predict(self, X):
        X_transform = self.X_generator.f(X)
        Y_predict = self.predictor.predict(
            X_transform.reshape(
                (X_transform.shape[0], X_transform.shape[1] * X_transform.shape[2])
                )
            )
        return self.detransform_y(Y_predict)

    def transform_y(self, Y):
        for i in range(Y.shape[0]):
            Y[i] = self.G * Y[i]
        return Y
    
    def detransform_y(self, Y):
        for i in range(Y.shape[0]):
            Y[i] = self.G.inv() * Y[i]
        return Y