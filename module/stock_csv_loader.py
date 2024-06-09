import pandas as pd
import os
import numpy as np

class StockCSVLoader:
    """
    株価情報が入ったCSVをロードし、各銘柄の日次対数利回り
    対数利回り：ln((close - open) / open)
    """
    def __init__(self) -> None:
        pass

    def _load(self, csv_path):
        df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)

        # 欠損値を線形補間する
        # 最上段、最下段の値は線形補間できないので、その場合は直前・直後の値で置き換え        
        df = df.infer_objects(copy=False)
        df = df.interpolate()
        df = df.ffill()
        df = df.bfill()

        return df
    
    def _calculate_ln_return(self, df : pd.DataFrame, col_suffix):
        col = f'{col_suffix}_LogDailyReturn'
        df[col] = np.log(df['AdjustmentClose'] / df['AdjustmentOpen'])
        return df[col]

    def load(self, stock_price_csv_paths):
        dfs = []
        for csv_path in stock_price_csv_paths:
            stock_code = os.path.basename(csv_path).split(".")[0]
            loaded = self._load(csv_path)
            dfs.append(self._calculate_ln_return(loaded, col_suffix = stock_code))
        return pd.concat(dfs, axis=1)
    
    def load_all_as_ndarray(self, stock_price_csv_paths):
        return self.load(stock_price_csv_paths).T.values

