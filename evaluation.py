import glob
import pickle
from module.dataset import Dataset
from module.ABCDForecast.abcd_forecast import ABCDForecast

def main():
    paths = glob.glob('./data/quants/stock_price/*.csv')
    stock_num = len(paths)
    
    dataset = Dataset(paths, window=30, late=1, train_rate=0.8)
    X_train, Y_train, X_test, Y_test = dataset.generate()
    
    with open('./abcd-forecast.pkl', 'rb') as f:
        abcd_forecast = pickle.load(f)
    
    Y_predicted = abcd_forecast.predict(X_test)
    Y_predicted_detransform = abcd_forecast.detransform_y(Y_predicted)
    
    # Y_aggregated[t][s]
    Y_aggregated = abcd_forecast.aggregate_by_score(Y_predicted_detransform)
    print(Y_aggregated)
    
if __name__=='__main__':
    main()