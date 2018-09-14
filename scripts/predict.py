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
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.externals import joblib
import time
#################################################
########### Global variables ####################
#################################################
#################################################
########### Global variables ####################
#################################################
PATH_IN = '/home/alexis/Bureau/finalproject/DatasIn/RTS/'
PATH_SCRIPT = '/home/alexis/Bureau/finalproject/scripts/'
PATH_OUT = '/home/alexis/Bureau/finalproject/Datas/'
LOG = "log.txt"
#################################################
########### Important functions #################
#################################################

def get_path():
    datas = pd.read_csv('path.csv')
    return datas['PathtoTempDatas'].values[0],datas['PathtoScripts'].values[0],datas['PathtoTempDatas'].values[0]
def Report(error):
    with open(LOG,'a+') as file:
        file.write(str(error)+' \n')
        print(str(error))
def load(fileX):
    df = pd.read_csv(fileX)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(1)
    X_train = df.values
    t = df.index.values

    scaler = MinMaxScaler(feature_range=(0, 1))#StandardScaler()
    s = StandardScaler()
    X_train_minmax = scaler.fit_transform(X_train)
    X_train_meanvar = s.fit_transform(X_train)
    return  X_train_minmax,X_train_meanvar,t

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

    # TODO: add NN,LSTM back
    #json_file = open("model/NN.json", 'r')
    #loaded_model_json = json_file.read()
    #json_file.close()
    #NN = model_from_json(loaded_model_json)
    # load weights into new model
    #NN.compile(loss='binary_crossentropy', optimizer='adamax',metrics=['accuracy'])
    #NN.load_weights("model/NN.h5")
    #Report("Loaded NN from disk")

    #json_file = open("model/LSTM.json", 'r')
    #loaded_model_json = json_file.read()
    #json_file.close()
    #LSTM = model_from_json(loaded_model_json)
    # load weights into new model
    #LSTM.compile(loss='binary_crossentropy', optimizer='adamax',metrics=['accuracy'])
    #LSTM.load_weights("model/LSTM.h5")
    #Report("Loaded LSTM from disk")


    logistic = pickle.load(open('model/logistic_regression.sav', 'rb'))
    return(SVC,XGB,CatBoost,KNN,LGBM,logistic)#NN,LSTM)#TODO: add NN,LSTM back

def makepredictions(X_minmax,X_meanvar,SVC,XGB,CatBoost,KNN,LGBM):#,NN,LSTM):
    #LGBM,CAT,SVC,NN,XGB,KNN

    res1 = LGBM[0].predict(X_meanvar, num_iteration = LGBM[0].best_iteration)
    res2 = LGBM[1].predict(X_meanvar,num_iteration = LGBM[1].best_iteration)
    l1 = pd.DataFrame([[1-0.5*(a+b),0.5*(a+b)] for a,b in zip(res1,res2)])


    l2 =  pd.DataFrame([[1-(v[1]+l[1])*0.5,(v[1]+l[1])*0.5] for v,l in zip(CatBoost[0].predict_proba(X_meanvar),CatBoost[1].predict_proba(X_meanvar))])


    res = [SVC[0].predict_proba(X_minmax),SVC[1].predict_proba(X_minmax),SVC[2].predict_proba(X_minmax),SVC[3].predict_proba(X_minmax)]
    l3 = []
    for i in range(len(res)):
        l3.append(res[i][:,0])
        l3.append(res[i][:,1])
    l3 = pd.DataFrame(l3).T


    #l4 = pd.DataFrame(NN.predict_proba(X_minmax))#TODO: add NN,LSTM back

    res1 = XGB[0].predict(xgb.DMatrix(X_meanvar), ntree_limit=XGB[0].best_ntree_limit)
    res2 = XGB[1].predict(xgb.DMatrix(X_meanvar), ntree_limit=XGB[1].best_ntree_limit)
    res = [[1-(r1+r2)*0.5,(r1+r2)*0.5] for r1,r2 in zip(res1,res2)]
    l5 = pd.DataFrame(res)

    res = [KNN[0].predict_proba(X_minmax),KNN[1].predict_proba(X_minmax),KNN[2].predict_proba(X_minmax),KNN[3].predict_proba(X_minmax)]
    l6 = []
    for i in range(len(res)):
        l6.append(res[i][:,0])
        l6.append(res[i][:,1])
    l6 = pd.DataFrame(l6).T


    #l7 = pd.DataFrame(LSTM.predict_proba(np.reshape(X_minmax,(X_minmax.shape[0],1,X_minmax.shape[1]))))#TODO: add NN,LSTM back


    return pd.concat([l2,l3,l4,l5,l6,l7], axis=1) #TODO: add NN,LSTM back

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
    Report('----------------------------------------------------------------------------------------------------')
    Report("||precison: "+str(p)+"||recall: "+str(r)+"||fbeta: "+str(f))
    tp,fp,fn = mesure(pred,y)
    beta = 2
    p = tp/(tp+fp)
    r = tp/(tp+fn)
    beta_squared = beta ** 2
    f = (beta_squared + 1) * (p * r) / (beta_squared * p + r)

    Report("||precison: "+str(p)+"||recall: "+str(r)+"||fbeta: "+str(f))
    Report('----------------------------------------------------------------------------------------------------')
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
    global PATH_IN,PATH_SCRIPT,PATH_OUT
    PATH_IN,PATH_SCRIPT,PATH_OUT = get_path()
    files = os.listdir(PATH_IN+'RTS/')
    count = 0
    t = time.time()
    for file in files:
        if(file.split('_')[0] == 'sfrdaily'):
            count+=1
            print(count)
            fileX = PATH_IN+'RTS/'+str(file)
            CHAINE = (file.split('_'))[-3]
            DATE = (file.split('_'))[1]
            try:
                hum = pd.read_csv(PATH_OUT+'RTS/pred_proba_'+str(DATE)+'_'+str(CHAINE)+'.csv')
                Report("Fichier déjà prédit: passage au fichier suivant")
            except Exception as e:
                X_minmax,X_meanvar,t = load(fileX)
                SVC,XGB,CatBoost,KNN,LGBM,logistic = load_models() #TODO: add NN,LSTM back
                df = makepredictions(X_minmax,X_meanvar,SVC,XGB,CatBoost,KNN,LGBM)#,NN,LSTM)#TODO: add NN,LSTM back
                X_train = df.values
                np.random.seed(7)
                #Report(X_train.shape)
                np.random.seed(7)
                Predict = logistic.predict_proba(X_train)
                np.random.seed(7)

                pred_proba = pd.DataFrame([i[-1] for i in Predict])
                pred_proba.to_csv(PATH_OUT+'RTS/pred_proba_'+str(DATE)+'_'+str(CHAINE)+'.csv',index=False)
        else:
            pass
    t = time.time()-t
    Report('Temps de calcul pour %s fichiers: %s'%(count,t))


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
