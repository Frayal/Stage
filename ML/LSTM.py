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
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras import backend as K
#################################################
########### Global variables ####################
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
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    return trainX,testX,trainY,testY

def precision(y_true, y_pred, threshold_shift=0.34):
    beta = 1

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    return precision
    

def recall(y_true, y_pred, threshold_shift=0.33):
    beta = 1

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    recall = tp / (tp + fn)
    return recall


def fbeta(y_true, y_pred, threshold_shift=0.33):
    beta = 1

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall)


def model_fit(X,y):
    np.random.seed(42)
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, 28), dropout=0.4))
    model.add(Dense(1))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=[fbeta,precision,recall])
    model.fit(X, y, epochs=100, batch_size=1, verbose=2)
    return model

def find_index(l,v):
    res = []
    for i, j in enumerate(l):
        if(j == v):
            res.append(i)
    return res    


def plot_res(df,trainPredict,testPredict,y):
    x = df
    t= [i/60 +3 for i in range(len(x))]


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
    plot(fig, filename='LSTM.html')

def save_model(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    
#################################################
########### Important functions #################
#################################################



#################################################
########### main with options ###################
#################################################


def main(argv):
    np.random.seed(42)
    X,y,df = load()
    trainX,testX,trainY,testY = process(X,y)
    model = model_fit(trainX,trainY)
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    testPredict = list([1 if i[0]>0.15 else 0 for i in testPredict])
    trainPredict = list([1 if i[0]>0.15 else 0 for i in trainPredict])
    # plot results
    plot_res(df,trainPredict,testPredict,y)
    #save model
    save_model(model)
    return ("process achevé sans erreures")


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
