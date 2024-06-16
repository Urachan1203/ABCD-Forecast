import pickle
import copy
from module.ABCDForecast.abcd_forecast import ABCDForecast

class Evaluator:
    def __init__(self, forecaster_paths, X_eval, Y_true) -> None:
        forecasters = []
        
        for forecaster_path in forecaster_paths:
            with open(forecaster_path, 'rb') as f:
                forecasters.extend(pickle.load(f))
                
        self.abcd_forecast = ABCDForecast(forecasters=forecasters, mode='eval')
        self.X_eval = X_eval
        self.Y_true = Y_true
        
    def run_evaluation(self):
        Y_predicted = self.abcd_forecast.predict(self.X_eval)
        Y_aggregated = self.abcd_forecast.aggregate_by_score(Y_predicted)
        return Y_aggregated, Y_predicted, self.Y_true
            
