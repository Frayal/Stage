#################################################
#created the 20/04/2018 12:57 by Alexis Blanchet#
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
import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
#################################################
########### Global variables ####################
#################################################
fileX_train ='/home/alexis/Bureau/Stage/Time-series/data/processed/sfrdaily_20180430_0_192_0_cleandata-processed.csv'
fileY_train = '/home/alexis/Bureau/historique/label-30-04.csv'

fileX_valid ='/home/alexis/Bureau/Stage/Time-series/data/processed/sfrdaily_20180507_0_192_0_cleandata-processed.csv'
fileY_valid = '/home/alexis/Bureau/historique/label-07-05.csv'

fileX_test ='/home/alexis/Bureau/Stage/Time-series/data/processed/sfrdaily_20180509_0_192_0_cleandata-processed.csv'
fileY_test = '/home/alexis/Bureau/historique/label-09-05.csv'
#################################################
########### Important functions #################
#################################################

def load(fileX):
    df = pd.read_csv(fileX)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(1)
    X_train = df.values
    t = df['t']
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    return  X_train,t

def load_models():
    SVC = []
    SVC.append(joblib.load('model/SVC1.joblib.pkl'))
    SVC.append(joblib.load('model/SVC2.joblib.pkl'))
    SVC.append(joblib.load('model/SVC3.joblib.pkl'))
    SVC.append(joblib.load('model/SVC4.joblib.pkl'))
    
    XGB = []
    XGB.append(pickle.load(open("model/XGB1.pickle.dat", "rb")))
    XGB.append(pickle.load(open("model/XGB2.pickle.dat", "rb")))
    
    CatBoost = []
    CatBoost.append(CatBoostClassifier().load_model(fname="model/catboostmodel1"))
    CatBoost.append(CatBoostClassifier().load_model(fname="model/catboostmodel2"))
    
    KNN = []
    KNN.append(pickle.load(open('model/KNN1.sav', 'rb')))
    KNN.append(pickle.load(open('model/KNN2.sav', 'rb')))
    KNN.append(pickle.load(open('model/KNN3.sav', 'rb')))
    KNN.append(pickle.load(open('model/KNN4.sav', 'rb')))
    
    
    
    LGBM = []
    LGBM.append(joblib.load('model/LGBM1.pkl'))
    LGBM.append(joblib.load('model/LGBM2.pkl'))
    
    
    
    json_file = open("model/NN.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    NN = model_from_json(loaded_model_json)
    # load weights into new model
    NN.compile(loss='binary_crossentropy', optimizer='adamax',metrics=['accuracy'])
    NN.load_weights("model/NN.h5")
    print("Loaded model from disk")
    
    
    logistic = pickle.load(open('model/logistic_regression.sav', 'rb'))
    return(SVC,XGB,CatBoost,KNN,LGBM,NN,logistic)

def makepredictions(X,SVC,XGB,CatBoost,KNN,LGBM,NN):
    #LGBM,CAT,SVC,NN,XGB,KNN
       
    res1 = LGBM[0].predict(X, num_iteration = LGBM[0].best_iteration)
    res2 = LGBM[1].predict(X,num_iteration = LGBM[1].best_iteration)
    l1 = pd.DataFrame([[1-0.5*(a+b),0.5*(a+b)] for a,b in zip(res1,res2)])
    
    
    l2 =  pd.DataFrame([[1-(v[1]+l[1])*0.5,(v[1]+l[1])*0.5] for v,l in zip(CatBoost[0].predict_proba(X),CatBoost[1].predict_proba(X))])
      
    
    res = [SVC[0].predict_proba(X),SVC[1].predict_proba(X),SVC[2].predict_proba(X),SVC[3].predict_proba(X)]
    l3 = []
    for i in range(len(res)):
        l3.append(res[i][:,0])
        l3.append(res[i][:,1])
    l3 = pd.DataFrame(l3).T
    
    
    l4 = pd.DataFrame(NN.predict_proba(X))
    
    res1 = XGB[0].predict(xgb.DMatrix(X), ntree_limit=XGB[0].best_ntree_limit)
    res2 = XGB[1].predict(xgb.DMatrix(X), ntree_limit=XGB[1].best_ntree_limit)
    res = [[1-(r1+r2)*0.5,(r1+r2)*0.5] for r1,r2 in zip(res1,res2)]
    l5 = pd.DataFrame(res)
    
    res = [KNN[0].predict_proba(X),KNN[1].predict_proba(X),KNN[2].predict_proba(X),KNN[3].predict_proba(X)]
    l6 = []
    for i in range(len(res)):
        l6.append(res[i][:,0])
        l6.append(res[i][:,1])
    l6 = pd.DataFrame(l6).T

    
   
    print(l1.sum())
    
    
    return pd.concat([l1,l2,l3,l4,l5,l6], axis=1) #################################""
    
def scoring(predict,y,h = [3,27],threshold=0.5):     
    pred = list([1 if i[-1]>threshold else 0 for i in predict])
    pred = pred[(h[0]-3)*60:(h[1]-3)*60]
    y = y[(h[0]-3)*60:(h[1]-3)*60]
    tp = np.sum([z*x for z,x in zip(pred,y)])
    fp = np.sum([np.clip(z-x,0,1) for z,x in zip(pred,y)])
    fn = np.sum([np.clip(z-x,0,1) for z,x in zip(y,pred)])
    
    beta = 2
    p = tp/np.sum(pred)
    r = tp/np.sum(y)
    beta_squared = beta ** 2
    f = (beta_squared + 1) * (p * r) / (beta_squared * p + r)
    print('----------------------------------------------------------------------------------------------------')
    print("||precison: "+str(p)+"||recall: "+str(r)+"||fbeta: "+str(f))
    tp,fp,fn = mesure(pred,y)
    beta = 2
    p = tp/(tp+fp)
    r = tp/(tp+fn)
    beta_squared = beta ** 2
    f = (beta_squared + 1) * (p * r) / (beta_squared * p + r)
    
    print("||precison: "+str(p)+"||recall: "+str(r)+"||fbeta: "+str(f))
    print('----------------------------------------------------------------------------------------------------')
    return 0
def mesure(y_pred,y_true):
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(y_pred)-1):
        i = i+1
        if(y_pred[i] == 1):
            if(sum(y_true[i-1:i+1])>0):
                TP += 1
            else:
                FP += 1
    for i in range(len(y_true)-1):
        i = i+1
        if(y_true[i] == 1):
            if(sum(y_pred[i-1:i+1])>0):
                pass
            else:
                FN += 1
    return TP,FP,FN
    

#################################################
########### main with options ###################
#################################################


def main(argv):
    DATE = argv[0]
    THRESHOLD = float(argv[1])
    d = list(DATE)
    d = str(d[-2])+str(d[-1])+"-"+str(d[-4])+str(d[-3])
    fileX = '/home/alexis/Bureau/Stage/Time-series/data/processed/sfrdaily_'+str(DATE)+'_0_192_0_cleandata-processed.csv'
    fileY = '/home/alexis/Bureau/historique/label-'+d+'.csv'
    y = pd.read_csv(fileY)
    Y = y['label'][3:].values.reshape(-1, 1)
    X,t = load(fileX)
    SVC,XGB,CatBoost,KNN,LGBM,NN,logistic = load_models()
    df = makepredictions(X,SVC,XGB,CatBoost,KNN,LGBM,NN)
    X_train = df.values
    np.random.seed(7)
    print(X_train.shape)
    np.random.seed(7)
    Predict = logistic.predict_proba(X_train)
    np.random.seed(7)
    
    pred = pd.DataFrame([1 if i[-1]>THRESHOLD else 0 for i in Predict])
    pred.to_csv('pred_'+str(DATE)+'.csv',index=False)
    scoring(Predict,Y,h = [3,27],threshold=THRESHOLD) 

if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
