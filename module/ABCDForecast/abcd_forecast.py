from module.ABCDForecast.forecaster import Forecaster
from module.ABCDForecast.X.generator import XGeneratorRandom
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pickle
import copy
from tqdm import tqdm

class ABCDForecast:
    def __init__(self, 
                 X=None,
                 Y=None,
                 num_forecaster = None,
                 X_generators = None,
                 forecaster_per_group = None,
                 num_stock = None,
                 mode='train',
                 forecasters = []
                 ) -> None:
        
        if mode == 'train':
            self.X = X
            self.Y = Y
            self.num_forecaster = num_forecaster
            self.X_generators = X_generators
            self.forecasters_per_group = forecaster_per_group
            self.num_stock = num_stock
            self.mode = mode
            self.forecasters = [] 
        elif mode == 'eval':
            self.X = None
            self.Y = None
            self.num_forecaster = len(forecasters)
            self.X_generators = None
            self.forecasters_per_group = None
            self.num_stock = None
            self.mode = mode
            self.forecasters = forecasters
        else:
            raise NotImplementedError(f"Unexpected mode : {mode}")

    
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
            g_start = g * self.forecasters_per_group
            g_end = min((g + 1) * self.forecasters_per_group, self.num_forecaster)
            forecasters = [
                Forecaster(
                    self.X, 
                    self.Y, 
                    X_generator=self.X_generators[i],
                    num_stock=self.num_stock,
                    mode=self.mode
                    ) for i in range(g_start, g_end)
                ]
            # for i, forecaster in enumerate(forecasters): 
            #     forecaster.train()
            #     print(i)
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = [executor.submit(forecaster.train) for forecaster in forecasters]
                for i, future in enumerate(as_completed(futures)):
                    print(f"training forecast {i}")
                    future.result()  # This will re-raise any exception raised by the forecaster's train method
            
            # NOTE : このままpickle保存すると、forecasterあたり45MBぐらいストレージを消費してしまうので、予測に不要なパラメータは削除する。
            for forecaster in forecasters:
                forecaster.X_transform = None
                forecaster.Y_transform = None
            with open(f'./data/forecasters/forecaster_{str(g_start).zfill(5)}_{str(g_end).zfill(5)}.pkl', 'wb') as f:
                pickle.dump(forecasters, f)

    
    def predict(self, X):
        Y_estimated = []    # Y_estimated[f][t][s] -> f : forecaster / t : time / s : stock 
        
        # with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        #     futures = [executor.submit(forecaster.predict, copy.deepcopy(X)) for forecaster in self.forecasters]
        #     for i, future in enumerate(as_completed(futures)):
        #             print(f"Finish : {i}")
        #             Y_estimated.append(future.result())  # This will re-raise any exception raised by the forecaster's train method
        
        for forecaster in tqdm(self.forecasters) : 
            Y_estimated.append(forecaster.predict(copy.deepcopy(X)))
        return np.array(Y_estimated)


    def aggregate_by_score(self, Y):
        """
        input : Y[f][t][s] -> f : forecaster / t : time / s : stock
        output : Y_aggregated[t][s] -> t : time / s : stock
        """
        Y_aggregated = []
        
        for t in range(Y.shape[1]):
            Y_t = []
            for s in range(Y.shape[2]):
                Y_t.append(np.sum(Y[:, t, s]) / Y.shape[0])
            Y_aggregated.append(Y_t)
            
        return np.array(Y_aggregated)
            