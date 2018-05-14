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
from sklearn import linear_model
import pandas as pd
import numpy as np
#################################################
########### Global variables ####################
#################################################
fileY = '/home/alexis/Bureau/historique/label-09-05.csv'
#################################################
########### Important functions #################
#################################################
def plot_res(trainPredict,testPredict,y):
    testPredict1 = list([1 if i[-1]>0.5 else 0 for i in testPredict])
    trainPredict1 = list([1 if i[-1]>0.5 else 0 for i in trainPredict])
    pred = trainPredict1+testPredict1
    tp = np.sum([z*x for z,x in zip(pred,y)])
    fp = np.sum([np.clip(z-x,0,1) for z,x in zip(pred,y)])
    fn = np.sum([np.clip(z-x,0,1) for z,x in zip(y,pred)])
    
   
    
    beta = 2
    p = tp/np.sum(pred)
    r = tp/np.sum(y)
    beta_squared = beta ** 2
    f = (beta_squared + 1) * (p * r) / (beta_squared * p + r)
    print("precison: "+str(p)+" recall: "+str(r)+" fbeta: "+str(f))
    
    tp,fp,fn = mesure(pred,y)
    beta = 2
    p = tp/(tp+fp)
    r = tp/(tp+fn)
    beta_squared = beta ** 2
    f = (beta_squared + 1) * (p * r) / (beta_squared * p + r)
    
    
    print("precison: "+str(p)+" recall: "+str(r)+" fbeta: "+str(f))

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

#################################################
########### main with options ###################
#################################################


def main(argv):
    y = pd.read_csv(fileY)
    Y = y['label'][3:].values.reshape(-1, 1)
    if(len(argv)==0):
        argv = [0]
    if(str(argv[0]) == 'train'):
        print("LGBM")
        l1 = os.system("python /home/alexis/Bureau/Stage/ML/LightGBM.py 0.316")
        print("catboost")
        l2 = os.system("python /home/alexis/Bureau/Stage/ML/CatBoost.py 0.2")
        print("classic")
        l3 = os.system("python /home/alexis/Bureau/Stage/ML/SVC.py 0.15")
        print("NN")
        l4 = os.system("python /home/alexis/Bureau/Stage/ML/NeuralNetwork.py 0.35")
        print("XGB")
        l5 = os.system("python /home/alexis/Bureau/Stage/ML/XgBoost.py 0.04")
        os.system("python /home/alexis/Bureau/Stage/ML/Stack.py")
        return 0
    else:
        l1 = pd.read_csv("lightGBM.csv")
        l2 = pd.read_csv("catboost.csv")
        l3 = pd.read_csv("SVC.csv")
        l4 = pd.read_csv("NN.csv")
        l5 = pd.read_csv("xgb.csv")
    
    X = pd.concat([l1,l2,l3,l4,l5], axis=1).values
    train_size = int(len(X) * 0.67)
    test_size = len(X) - train_size
    trainX, testX = X[0:train_size:,], X[train_size:len(X):,]
    trainY, testY = Y[0:train_size], Y[train_size:len(X[0])]
    for i in range(10):
        logistic = linear_model.LogisticRegression(C=0.5+0.1*i,class_weight='balanced',penalty='l2')
        logistic.fit(trainX, trainY)
        testPredict = logistic.predict_proba(testX)
        trainPredict = logistic.predict_proba(trainX)
        print("C="+str(0.5+0.1*i))
        plot_res(trainPredict,testPredict,Y)
    return ("process achevé sans erreures")


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
