# declaration of library
import pandas as pd;
import numpy as np;
from sklearn.model_selection import train_test_split;
from sklearn.preprocessing import MinMaxScaler;

# define class pre-processing
class PreProcessing():

    # property class
    x = "";
    y = "";

    # method normalized-data
    def normalization(data):

        # normalize features
        scaler = MinMaxScaler(feature_range=(-1,1));
        data = scaler.fit_transform(data);
        
        # return values
        return data;

    # method splitting-data
    def splitting(data, train_size, test_size):
        
        # split data train and test
        train_data, test_data = train_test_split(data, train_size=train_size, test_size=test_size, shuffle=False);
        
        # reteurn values
        return train_data, test_data;

     # method supervised learning
    def supervised_learning(dataset, look_back=1):
        
        # declare variable X and Y
        dataX = []
        dataY = []
        
        # for loop for create supervised learning
        for i in range(len(dataset)-look_back):
            
            # insert value X and Y 
            dataX.append(dataset[i:(i+look_back), 0])
            dataY.append(dataset[i + look_back, 0])
        
        # return value X and Y
        return np.array(dataX), np.array(dataY)