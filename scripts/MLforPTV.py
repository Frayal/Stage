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
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from keras.utils import np_utils
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
import pickle
from tpot import TPOTClassifier
from sklearn import linear_model
#################################################
########### Global variables ####################
#################################################

#################################################
########### Important functions #################
#################################################
def load(fileX,c):
    df = pd.read_csv('/home/alexis/Bureau/Project/results/truemerge/'+str(c)+'/'+fileX)
    if('Heure' in df.columns.values or 'programme' in df.columns.values):
        print('colonnes en trop',fileX)
    if('labels' not in df.columns.values):
        print('pas de labels',fileX)
    y = df['labels']
    return df.drop(['labels'],axis=1),y

def load_all(CHAINE):
    X = pd.DataFrame()
    Y = pd.DataFrame()
    if(CHAINE in ['TF1','all']):
        files = os.listdir('/home/alexis/Bureau/Project/results/truemerge/TF1/')
        for file in files:
            if(file.split('_')[-2] in ['2017-12-09','2017-12-20','2017-12-08','2017-12-14'] or (file.split('_')[-2]).split('-')[0] == '2018'):
                print(file.split('_')[-2])
            else:
                df,y = load(file,'TF1')
                df = df.replace([np.inf, -np.inf], np.nan)
                df = df.fillna(0)
                X_train = df
                y_train = y
                X = pd.concat([X,X_train])
                Y = pd.concat([Y,y_train])
    if(CHAINE in ['M6','all']):
        files = os.listdir('/home/alexis/Bureau/Project/results/truemerge/M6/')
        for file in files:
            if(file.split('_')[-2] in ['2017-12-20','2017-12-25','2017-12-10','2017-12-09','2017-12-29','2017-12-26'] or (file.split('_')[-2]).split('-')[0] == '2018'):
                print(file.split('_')[-2])
            else:
                df,y = load(file,'M6')
                df = df.replace([np.inf, -np.inf], np.nan)
                df = df.fillna(0)
                X_train = df
                y_train = y
                X = pd.concat([X,X_train])
                Y = pd.concat([Y,y_train])
    return X,Y
######################################################
######################################################
### XGB modeling
params = {'eta': 0.001,
          'max_depth': 18,
          'subsample': 0.9,
          'colsample_bytree': 1,
          'colsample_bylevel':1,
          'min_child_weight':1,
          'alpha':1,
          'objective': 'multi:softprob',
          'eval_metric': 'mlogloss',
          'seed': 99,
          'silent': 1,
         'num_class' : 3,
         }
params2 = {'eta': 0.001,
          'max_depth': 19,
          'subsample': 0.9,
          'colsample_bytree': 0.9,
          'colsample_bylevel':0.9,
          'min_child_weight':0.9,
          'alpha':1,
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
        x1, x2, y1, y2 = train_test_split(X.values, y.values, test_size=0.2,random_state = 99)
        watchlist = [(xgb.DMatrix(x1, y1, weight = [int(y)*2+1 for y in y1]), 'train'), (xgb.DMatrix(x2, y2,weight = [int(y)*2+1 for y in y2]), 'valid')]
        self.clf1 = xgb.train(params, (xgb.DMatrix(x1, y1, weight = [int(y)*2+1 for y in y1])), 10000,  watchlist, maximize = False,verbose_eval=500, early_stopping_rounds=300)
        self.clf2 = xgb.train(params2, (xgb.DMatrix(x1, y1, weight = [int(y)*2+1 for y in y1])), 10000,  watchlist, maximize = False,verbose_eval=500, early_stopping_rounds=300)


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
        x1, x2, y1, y2 = train_test_split(X, y, test_size=0.2,random_state = 7)
        self.clf1 = CatBoostClassifier(iterations=7000,learning_rate=0.01, depth=10,metric_period = 500, loss_function='MultiClass', eval_metric='MultiClass', random_seed=99, od_type='Iter', od_wait=300,class_weights = [1,2,3])
        self.clf1.fit(x1,y1,verbose=True,eval_set=(x2,y2),use_best_model=True)
        self.clf2 = CatBoostClassifier(iterations=7000,learning_rate=0.01, depth=11,metric_period = 500, loss_function='MultiClass', eval_metric='MultiClass', random_seed=99, od_type='Iter', od_wait=300,class_weights = [1,2,3])
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
    return [np.argmax(y) for y in y_score]
    if(p1 == 0):
        return [np.argmax(y) for y in y_score]
    else:
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
def acc(y_score,y_test,p1=0.5,p2=0.5):
    res = 0
    y = get_label(y_score,p1,p2)
    for i in range(len(y)):
        if(y[i] == y_test[i]):
            res+=1
        else:
            pass
    print("accuracy: "+str(res/len(y)))

def ROC_curve(y_score,Y_test):
    y_test = label_binarize(Y_test, classes=[0, 1, 2])
    n_classes = y_test.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(15,15))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))



    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig('auc.png')
    plt.show()


def save_model_cat(model):
    model.clf1.save_model("model_PTV/catboostmodel1")
    model.clf2.save_model("model_PTV/catboostmodel2")

def save_model_xgb(model):
    pickle.dump(model.clf1, open("model_PTV/XGB1.pickle.dat", "wb"))
    pickle.dump(model.clf2, open("model_PTV/XGB2.pickle.dat", "wb"))

def save_model(clf,name):
    pickle.dump(clf, open("model_PTV/"+name+".pickle.dat", "wb"))


#################################################################
def main(argv):
    if(len(argv) == 0):
        argv = ['all']
    if(argv[0] == 'test'):
        Y_test = pd.read_csv('results.csv').values
        y_pred = pd.read_csv('y_pred.csv').values
        y_pred2 = pd.read_csv('y_pred2.csv').values
        y_pred4 = pd.read_csv('y_pred4.csv').values
        y_pred5 = pd.read_csv('y_pred5.csv').values
        res = [[(res1[0]+res2[0]+res3[0]+res4[0]+res5[0]+res6[0])/6,(res1[1]+res2[1]+res3[1]+res4[1]+res5[1]+res6[1])/6,(res1[2]+res2[2]+res3[2]+res4[2]+res5[2]+res6[2])/6] for res1,res2,res3,res4,res5,res6 in zip(y_pred,y_pred2,y_pred2,y_pred4,y_pred5,y_pred)]

        for p1 in [0]:
            for p2 in [0]:
                print('################### '+str(p1)+' ### '+str(p2)+'###################')
                print('############XGB##############')
                mesure(y_pred,Y_test,p1,p2)
                mismatch(y_pred,Y_test,p1,p2)
                acc(y_pred,Y_test,p1,p2)
                print('############CatBoost##############')
                mesure(y_pred2,Y_test,p1,p2)
                mismatch(y_pred2,Y_test,p1,p2)
                acc(y_pred2,Y_test,p1,p2)
                print('############GradientBoostingClassifier##############')
                mesure(y_pred4,Y_test,p1,p2)
                mismatch(y_pred4,Y_test,p1,p2)
                acc(y_pred4,Y_test,p1,p2)
                print('############RandomForestClassifier##############')
                mesure(y_pred5,Y_test,p1,p2)
                mismatch(y_pred5,Y_test,p1,p2)
                acc(y_pred5,Y_test,p1,p2)
                print('############Stack##############')
                mesure(res,Y_test,p1,p2)
                mismatch(res,Y_test,p1,p2)
                acc(res,Y_test,p1,p2)

    elif(len(argv) == 1):
        X,Y = load_all(argv[0])
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_train = X_train.fillna(0)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.fillna(0)
        Y_test  = [Y[0] for Y in Y_test.values]
        ##########################################
        np.random.seed(42)
        clf = Classifier()
        clf.fit(X_train,Y_train)
        y_pred = clf.predict_proba(X_test)
        clf2 = Classifier2()
        clf2.fit(X_train,Y_train)
        y_pred2 = clf2.predict_proba(X_test)

        dtree_model = DecisionTreeClassifier(max_depth = 10).fit(X_train,Y_train)
        y_pred3 = dtree_model.predict_proba(X_test)

        tpot = GradientBoostingClassifier(learning_rate=0.05, max_depth=10, max_features=0.75, min_samples_leaf=7, min_samples_split=16, n_estimators=500, subsample=0.9)
        tpot.fit(X_train,Y_train)
        print(tpot.score(X_test, Y_test))
        y_pred4 = tpot.predict_proba(X_test)

        RF_model = RandomForestClassifier(max_depth = 15).fit(X_train,Y_train)
        y_pred5 = RF_model.predict_proba(X_test)
        ##########################################
        save_model_xgb(clf)
        save_model_cat(clf2)
        save_model(dtree_model,"DT")
        save_model(RF_model,"RF")
        pickle.dump(tpot, open("model_PTV/GradientBoostingClassifier.pickle.dat", "wb"))
        pickle.dump(RF_model, open("model_PTV/RandomForestClassifier.pickle.dat", "wb"))
        res = [[(res1[0]+res2[0]+res3[0]+res4[0]+res5[0]+res6[0])/6,(res1[1]+res2[1]+res3[1]+res4[1]+res5[1]+res6[1])/6,(res1[2]+res2[2]+res3[2]+res4[2]+res5[2]+res6[2])/6] for res1,res2,res3,res4,res5,res6 in zip(y_pred,y_pred2,y_pred3,y_pred4,y_pred5,y_pred)]
        for p1,p2 in zip([0],[0]):
            print('############XGB##############')
            mesure(y_pred,Y_test,p1,p2)
            mismatch(y_pred,Y_test,p1,p2)
            acc(y_pred,Y_test,p1,p2)
            print('############CatBoost##############')
            mesure(y_pred2,Y_test,p1,p2)
            mismatch(y_pred2,Y_test,p1,p2)
            acc(y_pred2,Y_test,p1,p2)
            print('############DecisionTreeClassifier##############')
            mesure(y_pred3,Y_test,p1,p2)
            mismatch(y_pred3,Y_test,p1,p2)
            acc(y_pred3,Y_test,p1,p2)
            print('############GradientBoostingClassifier##############')
            mesure(y_pred4,Y_test,p1,p2)
            mismatch(y_pred4,Y_test,p1,p2)
            acc(y_pred4,Y_test,p1,p2)
            print('############RandomForestClassifier##############')
            mesure(y_pred5,Y_test,p1,p2)
            mismatch(y_pred5,Y_test,p1,p2)
            acc(y_pred5,Y_test,p1,p2)
            print('############Stack##############')
            mesure(res,Y_test,p1,p2)
            mismatch(res,Y_test,p1,p2)
            acc(res,Y_test,p1,p2)

        #ROC_curve(y_pred,Y_test)
        #ROC_curve(y_pred2,Y_test)
        pd.DataFrame(Y_test).to_csv('results.csv',index=False)
        pd.DataFrame(y_pred).to_csv('y_pred.csv',index=False)
        pd.DataFrame(y_pred2).to_csv('y_pred2.csv',index=False)
        pd.DataFrame(y_pred4).to_csv('y_pred4.csv',index=False)
        pd.DataFrame(y_pred5).to_csv('y_pred5.csv',index=False)


    return ("process achevé sans erreures")

if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
