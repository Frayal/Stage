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
fileY = '/home/alexis/Bureau/Stage/Time-series/y_true2.csv'
#################################################
########### Important functions #################
#################################################
def plot_res(trainPredict,testPredict,y):
    testPredict1 = list([1 if i[1]>0.15 else 0 for i in testPredict])
    trainPredict1 = list([1 if i[1]>0.15 else 0 for i in trainPredict])
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
    logistic = linear_model.LogisticRegression(C=100.0,class_weight='balanced')
    y = pd.read_csv(fileY)
    Y = y['CP'][3:].values.reshape(-1, 1)
    if(str(argv) == 'train'):
        l1 = os.system("python /home/alexis/Bureau/Stage/ML/LightGBM.py")
        l2 = os.system("python /home/alexis/Bureau/Stage/ML/CatBoost.py")
        l3 = os.system("python /home/alexis/Bureau/Stage/ML/classic.py")
        l4 = os.system("python /home/alexis/Bureau/Stage/ML/NeuralNetwork.py")
        l5 = os.system("python /home/alexis/Bureau/Stage/ML/XgBoost.py")
    else:
        l1 = pd.read_csv("lightGBM.csv")
        l2 = pd.read_csv("catboost.csv")
        l3 = pd.read_csv("SVC.csv")
        l4 = pd.read_csv("NN.csv")
        l5 = pd.read_csv("xgb.csv")

    X = pd.concat([l1, l2,l4,l5], axis=1).values
    train_size = int(len(X) * 0.67)
    test_size = len(X) - train_size
    trainX, testX = X[0:train_size:,], X[train_size:len(X):,]
    trainY, testY = Y[0:train_size], Y[train_size:len(X[0])]
    print(X.shape)
    logistic.fit(trainX, trainY)
    testPredict = logistic.predict_proba(testX)
    trainPredict = logistic.predict_proba(trainX)
    plot_res(trainPredict,testPredict,Y)
    return ("process achevé sans erreures")


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
