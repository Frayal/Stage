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
import plotly
import plotly.graph_objs as go
import plotly.offline as offline
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#################################################
########### Global variables ####################
#################################################
fileY = '/home/alexis/Bureau/historique/label-09-05.csv'
fileX ='/home/alexis/Bureau/Stage/Time-series/data/processed/sfrdaily_20180509_0_192_0_cleandata-processed.csv'
#################################################
########### Important functions #################
#################################################
def plot_res(df,trainPredict,testPredict,y,threshold=0.5):
    x = df
    t= [i/60 +3 for i in range(len(x))]
    
    testPredict1 = list([1 if i[-1]>threshold else 0 for i in testPredict])
    trainPredict1 = list([1 if i[-1]>threshold else 0 for i in trainPredict])
    pred = trainPredict1+testPredict1
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
    
    l1 = find_index(trainPredict1,1)
    l2 = find_index(testPredict1,1)

    x1 = [t[i] for i in l1]
    x2 = [t[i+len(trainPredict1)] for i in l2]

    y1 = [x[i] for i in l1]
    y2 = [x[i+len(trainPredict1)] for i in l2]

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
    #plot(fig, filename='Stack.html')
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
        print("KNN")
        l6 = os.system("python /home/alexis/Bureau/Stage/ML/KNN.py 0.2")
        #print("LSTM")
        #l7 = os.system("python /home/alexis/Bureau/Stage/ML/LSTM.py 0.4")
        print('stacking model and training logistic regression...')
        os.system("python /home/alexis/Bureau/Stage/ML/Stack.py")
        return 0
    else:
        l1 = pd.read_csv("lightGBM.csv")
        l2 = pd.read_csv("catboost.csv")
        l3 = pd.read_csv("SVC.csv")
        l4 = pd.read_csv("NN.csv")
        l5 = pd.read_csv("xgb.csv")
        l6 = pd.read_csv("KNN.csv")
        #l7 = pd.read_csv("LSTM.csv")
        
    
    X = pd.concat([l1,l2,l3,l4,l5,l6], axis=1).values
    train_size = int(len(X) * 0.67)
    test_size = len(X) - train_size
    trainX, testX = X[0:train_size:,], X[train_size:len(X):,]
    trainY, testY = Y[0:train_size], Y[train_size:len(X[0])]
    for i in range(1):
        np.random.seed(7)
        logistic = linear_model.LogisticRegression(C=1+0.1*i,class_weight='balanced',penalty='l2')
        logistic.fit(trainX, trainY)
        testPredict = logistic.predict_proba(testX)
        trainPredict = logistic.predict_proba(trainX)
        print("C="+str(1+0.1*i))
        for j in range(1):
            print(0.4)
            plot_res(pd.read_csv(fileX)['t'],trainPredict,testPredict,Y,threshold = 0.4)
    return ("process achevé sans erreures")


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
