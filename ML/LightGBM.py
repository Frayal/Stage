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
import lightgbm as lgb
from sklearn.metrics import log_loss
#################################################
########### Global variables ####################
#################################################
### LGB modeling
params = {'learning_rate': 0.0015,
          'subsample': 0.9,
          #'subsample_freq': 1,
          'colsample_bytree': 0.9,
          'colsample_bylevel':0.9,
          'reg_alpha': 0.0,
          'reg_lambda': 0,
          'max_depth' : 6,
          'num_leaves': 10,        
          'min_data_in_leaf': 1, 
          'boosting': 'goss',#'rf','dart','goss','gbdt'
          'objective': 'binary',
          'metric': 'binary_logloss',
          'is_training_metric': True,
          'seed': 99,}

params1 = {'learning_rate': 0.0015,
          'subsample': 0.9,
          #'subsample_freq': 1,
          'colsample_bytree': 0.9,
          'colsample_bylevel':0.9,
          'reg_alpha': 0.0,
          'reg_lambda': 0,
          'max_depth' : 7,
          'num_leaves': 10,        
          'min_data_in_leaf': 1, 
          'boosting': 'goss',#'rf','dart','goss','gbdt'
          'objective': 'binary',
          'metric': 'binary_logloss',
          'is_training_metric': True,
          'seed': 99,}
MAX_TREES = 5000

######################################################
class Classifier(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        
        x1, x2, y1, y2 = train_test_split(X, y, test_size=0.2, random_state=99)
        watchlist = [(lgb.Dataset(x1, label=y1), 'train'), (lgb.Dataset(x2, label=y2), 'valid')]
        self.clf2 = lgb.train(params, lgb.Dataset(x1, label=y1), MAX_TREES, lgb.Dataset(x2, label=y2),verbose_eval=100, feval=logloss_lgbm, early_stopping_rounds=300)
        self.clf1 = lgb.train(params1, lgb.Dataset(x1, label=y1), MAX_TREES, lgb.Dataset(x2, label=y2),verbose_eval=100, feval=logloss_lgbm, early_stopping_rounds=300)
    def predict(self, X):
        return self.clf1.predict(X)

    def predict_proba(self, X):
        res1 = self.clf1.predict(X, num_iteration = self.clf1.best_iteration)
        res2 = self.clf2.predict(X,num_iteration = self.clf2.best_iteration)
        return np.array([[1-0.5*(a+b),0.5*(a+b)] for a,b in zip(res1,res2)])

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
    
    pred = trainPredict+testPredict
    tp = np.sum([z*x for z,x in zip(pred,y)])
    fp = np.sum([np.clip(z-x,0,1) for z,x in zip(pred,y)])
    fn = np.sum([np.clip(z-x,0,1) for z,x in zip(y,pred)])
    
    beta = 2
    p = tp/np.sum(pred)
    r = tp/np.sum(y)
    beta_squared = beta ** 2
    f = (beta_squared + 1) * (p * r) / (beta_squared * p + r)
    print("precison: "+str(p)+" recall: "+str(r)+" fbeta: "+str(f))

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
    #plot(fig, filename='CatBoost.html')


def logloss_lgbm(preds, dtrain):
    labels = dtrain.get_label()
    score = 1-log_loss(labels, preds)
    return 'logloss', score,True

#################################################
########### main with options ###################
#################################################


def main(argv):
    X,y,df = load()
    trainX,testX,trainY,testY = process(X,y)
    model = model_fit(trainX,[y[0] for y in trainY])
    # make predictions
    trainPredict = model.predict_proba(trainX)
    testPredict = model.predict_proba(testX)
    testPredict1 = list([1 if i[1]>0.15 else 0 for i in testPredict])
    trainPredict1 = list([1 if i[1]>0.15 else 0 for i in trainPredict])
    # plot results
    #plot_res(df,trainPredict1,testPredict1,y)
    res = pd.DataFrame(np.concatenate((trainPredict,testPredict)))
    res.to_csv('lightGBM.csv',index=False)
    return res

if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
