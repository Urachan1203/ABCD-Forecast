from module.stock_csv_loader import StockCSVLoader
import numpy as np

class Dataset:
    def __init__(self, stock_csv_paths, window = 30, late = 1, train_rate = 0.8) -> None:
        self.raw_all = StockCSVLoader().load_all_as_ndarray(stock_csv_paths)
        self.test_start_idx = int(self.raw_all.shape[0] * train_rate)

        self.dataset = self.generate(window = window, late = late)
    
    def generate(self, window = 30, late = 1):
        X_train = []
        Y_train = []
        X_test = []
        Y_test = []
        
        target_idx = window + late
        while target_idx < self.test_start_idx:
            end = target_idx - late - 1
            start = end - window + 1
            X_train.append(self.raw_all[:,start:end])
            Y_train.append(self.raw_all[:, target_idx].T)
            target_idx += 1
        
        target_idx = self.test_start_idx + window + late
        while target_idx < self.raw_all.shape[0]:
            end = target_idx - late - 1
            start = end - window + 1
            X_test.append(self.raw_all[:, start:end].astype(np.float32))
            Y_test.append(self.raw_all[:, target_idx].T.astype(np.float32))
        return np.array(X_train), np.array(Y_train)
