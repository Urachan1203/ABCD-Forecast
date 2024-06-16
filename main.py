import argparse
from module.dataset import Dataset
from module.ABCDForecast.abcd_forecast import ABCDForecast
from module.ABCDForecast.X.generator import XGeneratorRandom
import glob
import pickle
import yaml
import os
from pprint import pprint

def option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_forecaster', type=int, default=50)
    return parser.parse_args()

def main(args):
    
    with open("./config/config.yaml", 'r') as yml:
        config = yaml.safe_load(yml)
        # config['train']['num_forecaster'] = args.num_forecaster   # overwritten by commandline args
    
    pprint(config)        
    
    paths = []
    # for stock_code in config['data']['stock']['topix100']:
    for stock_code in config['data']['stock']['topix30']:
        paths.append(os.path.join('./data/quants/stock_price', f'{stock_code}0.csv'))
    
    stock_num = len(paths)
    print(f'{stock_num} files ware loaded.')
    pprint(paths)
    
    dataset = Dataset(paths, window=config['train']['window'], late=config['train']['late'], train_rate=config['train']['train_rate'])
    X_train, Y_train, _, _ = dataset.generate()

    print("Constructing XGenerators")
    X_generators = [XGeneratorRandom(num_stock=stock_num) for _ in range(config['train']['num_forecaster'])]

    # ABCDForecastの宣言。
    # forecasterの数だけXGeneratorを指定する
    abcd_forecast = ABCDForecast(
        X_train, 
        Y_train, 
        num_forecaster=config['train']['num_forecaster'], 
        X_generators=X_generators,
        forecaster_per_group=config['train']['forecaster_per_group'],
        num_stock=stock_num,
        mode='train'
    )

    print("train")
    abcd_forecast.train_parallel()
    
    return

if __name__=="__main__":
    args = option()
    main(args)