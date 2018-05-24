#################################################
#created the 23/04/2018 12:57 by Alexis Blanchet#
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
import pickle

#################################################
########### Global variables ####################
#################################################


#################################################
########### Important functions #################
#################################################



#####################################################
#####################################################
###### Definition of the Xgboost classifier #########
#####################################################    
#####################################################
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
######################################################
### XGB modeling
params = {'eta': 0.02,
          'max_depth': 9, 
          'subsample': 0.9, 
          'colsample_bytree': 0.9, 
          'colsample_bylevel':0.9,
          'min_child_weight':5,
          'alpha':2,
          'objective': 'multi:softprob',
          'num_class':3,
          'eval_metric': 'mlogloss',
          'seed': 99,
          'silent': False}
params2 = {'eta': 0.02,
          'max_depth': 8, 
          'subsample': 0.9, 
          'colsample_bytree': 0.9, 
          'colsample_bylevel':0.9,
          'min_child_weight':5,
          'alpha':1,
          'objective': 'multi:softprob',
          'num_class':3,
          'eval_metric': 'mlogloss',
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
        res = [ sum_list(a,b) for a,b in zip(res1,res2)]
        return res



#################################################
########### main with options ###################
#################################################
    
def main(argv):
    #load data to trai
    
    # save model to file
    pickle.dump(model, open("pima.pickle.dat", "wb"))
    
    return ("process achevé sans erreures")


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
