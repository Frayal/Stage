#################################################
#created the 17/05/2018 14:14 by Alexis Blanchet#
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
import os
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
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import preprocessing
#################################################
########### Global variables ####################
#################################################

#################################################
########### Important functions #################
#################################################
def load(fileX):
    df = pd.read_csv('/home/alexis/Bureau/Project/results/truemerge/'+fileX)
    y = df['labels']
    return df.drop(['labels'],axis=1),y

def load_all():
    X = pd.DataFrame()
    Y = pd.DataFrame()
    files = os.listdir('/home/alexis/Bureau/Project/results/truemerge')
    for file in files:
        df,y = load(file)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        X_train = df
        y_train = y
        X = pd.concat([X,X_train])
        Y = pd.concat([Y,y_train])
    for f in X.columns:
        if X[f].dtype=='object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(X[f].values))
            X[f] = lbl.transform(list(X[f].values))
    return X,Y
######################################################
######################################################
### XGB modeling
params = {'eta': 0.001,
          'max_depth': 20,
          'subsample': 0.9,
          'colsample_bytree': 1,
          'colsample_bylevel':1,
          'min_child_weight':1,
          'alpha':5,
          'objective': 'multi:softprob',
          'eval_metric': 'mlogloss',
          'seed': 99,
          'silent': 1,
         'num_class' : 3,
         }
params2 = {'eta': 0.001,
          'max_depth': 15,
          'subsample': 0.9,
          'colsample_bytree': 0.9,
          'colsample_bylevel':0.9,
          'min_child_weight':0.9,
          'alpha':5,
          'objective': 'multi:softprob',
          'eval_metric': 'mlogloss',
          'seed': 42,
          'silent': 1,
          'num_class' : 3,
         }
######################################################
class Classifier(BaseEstimator):
    def __init__(self):
        pass

    def fit(self,X,y):
        np.random.seed(42)
        x1, x2, y1, y2 = train_test_split(X.values, y.values, test_size=0.2)
        watchlist = [(xgb.DMatrix(x1, y1, weight = [int(y)*2+1 for y in y1]), 'train'), (xgb.DMatrix(x2, y2,weight = [int(y)*2+1 for y in y2]), 'valid')]
        self.clf1 = xgb.train(params, (xgb.DMatrix(x1, y1, weight = [int(y)*2+1 for y in y1])), 50000,  watchlist, maximize = False,verbose_eval=500, early_stopping_rounds=3000)
        self.clf2 = xgb.train(params2, (xgb.DMatrix(x1, y1, weight = [int(y)*2+1 for y in y1])), 50000,  watchlist, maximize = False,verbose_eval=500, early_stopping_rounds=3000)


    def predict(self, X):
        return self.clf.predict(X.values)

    def predict_proba(self, X):
        res1 = self.clf1.predict(xgb.DMatrix(X.values), ntree_limit=self.clf1.best_ntree_limit)
        res2 = self.clf2.predict(xgb.DMatrix(X.values), ntree_limit=self.clf2.best_ntree_limit)
        return np.array([[(a[0]+b[0])/2,(a[1]+b[1])/2,(a[2]+b[2])/2] for a,b in zip(res1,res2)])


#################################################
#################################################
class Classifier2(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X,y):
        x1, x2, y1, y2 = train_test_split(X, y, test_size=0.2, random_state=99)
        self.clf1 = CatBoostClassifier(iterations=5000,learning_rate=0.01, depth=11,metric_period = 50, loss_function='MultiClass', eval_metric='MultiClass', random_seed=99, od_type='Iter', od_wait=500,class_weights = [1,3,5])
        self.clf1.fit(x1,y1,verbose=True,eval_set=(x2,y2),use_best_model=True)
        self.clf2 = CatBoostClassifier(iterations=5000,learning_rate=0.01, depth=12,metric_period = 50, loss_function='MultiClass', eval_metric='MultiClass', random_seed=99, od_type='Iter', od_wait=500,class_weights = [1,3,5])
        self.clf2.fit(x1,y1,verbose=True,eval_set=(x2,y2),use_best_model=True)
    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return np.array(np.array([[(a[0]+b[0])/2,(a[1]+b[1])/2,(a[2]+b[2])/2] for a,b in zip(self.clf2.predict_proba(X),self.clf1.predict_proba(X))]))

#################################################
#################################################
def find_index(l,v):
    res = []
    for i, j in enumerate(l):
        if(j == v):
            res.append(i)
    return res

def get_label(y_score,p1=0.5,p2=0.5):
    res = []
    for i in range(len(y_score)):
        #res.append(np.argmax(y_score[i]))
        if(y_score[i][1]>p1):
            res.append(1)
        elif(y_score[i][0]>p2):
            res.append(0)
        else:
            res.append(2)
    return res

def mesure_class(y_pred,y_true,j):
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(y_pred)):
        if(y_pred[i] == j):
            if(y_true[i] == j):
                TP += 1
            else:
                FP += 1
    for i in range(len(y_true)):
        if(y_true[i] == j):
            if(y_pred[i] == j):
                pass
            else:
                FN += 1
    return TP,FP,FN

def score(tp,fp,fn,epsilon=10**-5):
    beta = 2
    p = tp/(tp+fp+epsilon)
    r = tp/(tp+fn+epsilon)
    beta_squared = beta ** 2
    f = (beta_squared + 1) * (p * r) / (beta_squared * p + r+epsilon)

    print("|| precison: "+str(p)+"|| recall: "+str(r)+"|| fbeta: "+str(f))
    print('--------------------------------------------------')

def mesure(y_score,y_test,p1=0.5,p2=0.5):
    y = get_label(y_score,p1,p2)
    TP1,FP1,FN1 = mesure_class(y,y_test,0)
    TP2,FP2,FN2 = mesure_class(y,y_test,1)
    TP3,FP3,FN3 = mesure_class(y,y_test,2)
    print("pour la classe 0")
    score(TP1,FP1,FN1)
    print("pour la classe 1")
    score(TP2,FP2,FN2)
    print("pour la classe 2")
    score(TP3,FP3,FN3)

def mismatch(y_score,y_test,p1=0.5,p2=0.5):
    y = get_label(y_score,p1,p2)
    FP = 0
    FF = 0
    for i in range(len(y)):
        if(y[i]==1):
            if(y_test[i]==2):
                FP += 1
            else:
                pass
        if(y[i]==2):
            if(y_test[i]==1):
                FF += 1
            else:
                pass
        else:
            pass
    print("fausses publicités")
    print(FP)
    print("fausses fins")
    print(FF)
    return 0









#################################################################
def main(argv):
    X,Y = load_all()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    ##########################################
    clf = Classifier()
    clf.fit(X_train,Y_train)
    y_pred = clf.predict_proba(X_test)
    clf2 = Classifier2()
    clf2.fit(X_train,Y_train)
    y_pred2 = clf2.predict_proba(X_test)
    ##########################################

    print('############XGB##############')
    mesure(y_pred,Y_test)
    mismatch(y_pred,Y_test)
    print('############CatBoost##############')
    mesure(y_pred2,Y_test)
    mismatch(y_pred2,Y_test)


    return ("process achevé sans erreures")

if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
