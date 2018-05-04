#################################################
#created the 04/05/2018 09:52 by Alexis Blanchet#
#################################################
#-*- coding: utf-8 -*-
'''

'''

'''
Améliorations possibles:

'''
import warnings
warnings.filterwarnings('ignore')
#################################################
###########        Imports      #################
#################################################
import sys
import numpy as np
import pandas as pd
import scipy.stats
import plotly
import plotly.graph_objs as go
import plotly.offline as offline
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pickle
#################################################
########### Global variables ####################
#################################################
### XGB modeling
params = {'eta': 0.02,
          'max_depth': 20, 
          'subsample': 0.9, 
          'colsample_bytree': 0.9, 
          'colsample_bylevel':0.9,
          'min_child_weight':1,
          'alpha':2,
          'objective': 'binary:logistic',
          'eval_metric': 'logloss',
          'seed': 99,
          'silent': False}
params2 = {'eta': 0.02,
          'max_depth': 15, 
          'subsample': 0.9, 
          'colsample_bytree': 0.9, 
          'colsample_bylevel':0.9,
          'min_child_weight':1,
          'alpha':1,
          'objective': 'binary:logistic',
          'eval_metric': 'logloss',
          'seed': 99,
          'silent': False}

######################################################
class Classifier(BaseEstimator):
    def __init__(self):
        pass
 
    def fit(self, X, y):
        x1, x2, y1, y2 = train_test_split(X, y[:X.shape[0]], test_size=0.2, random_state=99)
        watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
        self.clf1 = xgb.train(params, xgb.DMatrix(x1, y1), 5000,  watchlist, maximize = False,verbose_eval=100, early_stopping_rounds=300)
        self.clf2 = xgb.train(params2, xgb.DMatrix(x1, y1), 5000,  watchlist, maximize = False,verbose_eval=100, early_stopping_rounds=300)
        
       
    def predict(self, X):
        return self.clf.predict(X)
 
    def predict_proba(self, X):
        res1 = self.clf1.predict(xgb.DMatrix(X), ntree_limit=self.clf1.best_ntree_limit)
        res2 = self.clf2.predict(xgb.DMatrix(X), ntree_limit=self.clf2.best_ntree_limit)
        res = [(r1+r2)*0.5 for r1,r2 in zip(res1,res2)]
        return np.array([ [1-c,c] for c in res])

#################################################
########### Important functions #################
#################################################
def load(fileX ='/home/alexis/Bureau/Stage/Time-series/data/processed/sfrdaily_20180430_0_192_0_cleandata-processed.csv' ,fileY = '/home/alexis/Bureau/Stage/Time-series/y_true2.csv'):
    df = pd.read_csv(fileX)
    y = pd.read_csv(fileY)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(1)
    X_train = df.values
    y_train = y['CP'][3:].values.reshape(-1, 1)
    t = df['t']
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    return  X_train,y_train,t

def process(dataset,Y):
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    trainX, testX = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    trainY, testY = Y[0:train_size], Y[train_size:len(dataset)]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    return trainX,testX,trainY,testY


def model_fit(X,y):
    clf = Classifier()
    clf.fit(X,y)
    return clf

def find_index(l,v):
    res = []
    for i, j in enumerate(l):
        if(j == v):
            res.append(i)
    return res    


def plot_res(df,trainPredict,testPredict,y):
    x = df
    t= [i/60 +3 for i in range(len(x))]


    l1 = find_index(trainPredict,1)
    l2 = find_index(testPredict,1)

    x1 = [t[i] for i in l1]
    x2 = [t[i+len(trainPredict)] for i in l2]

    y1 = [x[i] for i in l1]
    y2 = [x[i+len(trainPredict)] for i in l2]

    l3 = find_index(y,1)
    x3 = [t[i] for i in l3]
    y3 = [x[i] for i in l3]


    trace1 = go.Scatter(
            x= t,
            y= x,
            name = 'true',

    )
    trace2 = go.Scatter(
            x =x1,
            y=y1,
            mode = 'markers',
            name ='train',
    )
    trace3 = go.Scatter(
            x=x2,
            y= y2,
            mode = 'markers',
            name = 'test',
    )
    trace4 = go.Scatter(
            x=x3,
            y=y3,
            mode = 'markers',
            name = 'true markers'
    )

    fig = tools.make_subplots(rows=4, cols=1, specs=[[{}], [{}], [{}], [{}]],
                                  shared_xaxes=True, shared_yaxes=True,
                                  vertical_spacing=0.001)
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 1)
    fig.append_trace(trace3, 1, 1)
    fig.append_trace(trace4, 1, 1)

    fig['layout'].update(height=3000, width=2000, title='Annomalie detection')
    plot(fig, filename='CatBoost.html')

def save_model(model):
    pickle.dump(model.clf1, open("XGB1.pickle.dat", "wb"))
    pickle.dump(model.clf2, open("XGB2.pickle.dat", "wb"))
    

#################################################
########### main with options ###################
#################################################


def main(argv):
    X,y,df = load()
    trainX,testX,trainY,testY = process(X,y)
    model = model_fit(trainX,trainY)
    # make predictions
    trainPredict = model.predict_proba(trainX)
    testPredict = model.predict_proba(testX)
    testPredict = list([1 if i[1]>0.15 else 0 for i in testPredict])
    trainPredict = list([1 if i[1]>0.15 else 0 for i in trainPredict])
    # plot results
    plot_res(df,trainPredict,testPredict,y)
    #save model
    save_model(model)
    return ("process achevé sans erreures")


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
