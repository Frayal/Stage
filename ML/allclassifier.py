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
from sklearn.model_selection import train_test_split
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis





#################################################
########### Global variables ####################
#################################################
C = 1.0
h = .02  # step size in the mesh

######################################################


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
    classifiers = [
    KNeighborsClassifier(4),
    GaussianProcessClassifier(1.0 * RBF(1.0),n_restarts_optimizer=5,),
    RandomForestClassifier(max_depth=15, n_estimators=1000, max_features=15),
    MLPClassifier(alpha=0.1),
    AdaBoostClassifier(n_estimators = 100,learning_rate=0.1)
    ]
    classifiers = [clf.fit(X,y) for clf in classifiers]
    return classifiers

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

def save_model(model):
    pickle.dump(model.clf1, open("XGB1.pickle.dat", "wb"))
    pickle.dump(model.clf2, open("XGB2.pickle.dat", "wb"))
    

#################################################
########### main with options ###################
#################################################


def main(argv):
    names = ["Nearest Neighbors","Gaussian Process", "Random Forest", "Neural Net", "AdaBoost"]
    X,y,df = load()
    trainX,testX,trainY,testY = process(X,y)
    model = model_fit(trainX,trainY)
    # make predictions
    
    for clf,name in zip(model,names):
        trainPredict = clf.predict_proba(trainX)
        testPredict = clf.predict_proba(testX)
        testPredict = list([1 if i[1]>0.15 else 0 for i in testPredict])
        trainPredict = list([1 if i[1]>0.15 else 0 for i in trainPredict])
        # plot results
        print(name)
        plot_res(df,trainPredict,testPredict,y)
    return ("process achevé sans erreures")


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
