from module.ABCDForecast.forecaster import Forecaster
from module.ABCDForecast.X.generator import XGeneratorRandom
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pickle

class ABCDForecast:
    def __init__(self, 
                 X,
                 Y=None,
                 num_forecaster = 100,
                 X_generators = None,
                 forecaster_per_group = 100,
                 num_stock = 500,
                 mode='train'
                 ) -> None:
        self.X = X
        self.Y = Y
        self.num_forecaster = num_forecaster
        self.X_generators = X_generators
        self.forecasters_per_group = forecaster_per_group
        self.num_stock = num_stock
        self.mode = mode
        self.forecasters = []
        
    
    def train_parallel(self):
        """
        forecasterをグループに分け、順番に学習する。
        1グループに含まれるforecasterをforecaster_per_groupで指定する。
        forecaster_per_group の値はマシンスペックと相談。（一括でやろうとするとメモリが足りない・・・）
        """
        q = self.num_forecaster // self.forecasters_per_group
        group_num = q + 1 if self.num_forecaster % self.forecasters_per_group else q
        
        for g in range(group_num):
            print(f"Processng group {g}/{group_num}")
            forecasters = [
                Forecaster(
                    self.X, 
                    self.Y, 
                    X_generator=self.X_generators[i],
                    num_stock=self.num_stock,
                    mode=self.mode
                    ) for i in range(g * self.forecasters_per_group, min((g + 1) * self.forecasters_per_group, self.num_forecaster))
                ]
            with ThreadPoolExecutor(max_workers=os.cpu_count() * 5) as executor:
                futures = [executor.submit(forecaster.train) for forecaster in forecasters]
                for i, future in enumerate(as_completed(futures)):
                    print(f"training forecast {i}")
                    future.result()  # This will re-raise any exception raised by the forecaster's train method
                    
            with open(f'./data/forecasters/forecaster_{g}.pkl', 'wb') as f:
                pickle.dump(forecasters, f)

    # Y_estimated[f][t][s] -> f : forecaster / t : time / s : stock 
    # FIXME : pickleにされたforecasterを読み込んでpredictする流れにする。
    def predict(self, X):
        Y_estimated = []
        for forecaster in self.forecasters : 
            Y_estimated.append(forecaster.predict(X))
        return np.array(Y_estimated)
            
    # FIXME : pickleにされたforecasterを読み込んでpredictする流れにする。
    def detransform_y(self, Y):
        Y_detransformed = []
        for f, forecaster in enumerate(self.forecasters) :
            Y_detransformed.append(forecaster.detransform_y(Y[f]))
        return np.array(Y_detransformed)

    # FIXME : pickleにされたforecasterを読み込んでpredictする流れにする。
    def aggregate_by_score(self, Y):
        Y_aggregated = []   # Y_aggregated[t][s] -> t : time / s : stock
        
        # Y[f][t][s] -> f : forecaster / t : time / s : stock
        for t in range(Y.shape[1]):
            Y_t = []
            for s in range(Y.shape[2]):
                Y_t.append(np.sum(Y[:, t, s]) / Y.shape[0])
            Y_aggregated.append(Y_t)
            
        return np.array(Y_aggregated)
            