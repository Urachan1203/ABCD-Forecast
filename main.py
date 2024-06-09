import argparse
from module.dataset import Dataset
from module.ABCDForecast.abcd_forecast import ABCDForecast
from module.ABCDForecast.X.generator import XGeneratorRandom
import glob

def option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_forecaster', type=int, default=100)

def main(args):

    print("Generating Dataset")
    paths = glob.glob('/Users/urachan/Desktop/dev/stock-analyze/data/quants/stock_price/*.csv')
    dataset = Dataset(paths, window=30, late=1, train_rate=0.8)
    X_train, Y_train, X_test, Y_test = dataset.generate()

    print("Constructing XGenerators")
    X_generators = [XGeneratorRandom(num_stock=500) for _ in range(args.num_forecaster)]

    # ABCDForecastの宣言。
    # forecasterの数だけXGeneratorを指定する
    abcd_forecast = ABCDForecast(
        X_train, 
        Y_train, 
        num_forecaster=100, 
        X_generators=X_generators,
        num_stock=500,
        model='train'
    )

    abcd_forecast.train()

    Y_estimated = abcd_forecast.predict(X_test)
    Y_estimated_detransform = abcd_forecast.detransform_y(Y_estimated)
    
    return

if __name__=="__main__":
    args = option()
    main(args)