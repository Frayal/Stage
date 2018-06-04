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
import pickle
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from ast import literal_eval
#################################################
########### Global variables ####################
#################################################
fileY = '/home/alexis/Bureau/historique/label-09-05.csv'
fileX ='/home/alexis/Bureau/Stage/Time-series/data/processed/sfrdaily_20180509_0_192_0_cleandata-processed.csv'
fileY_valid = '/home/alexis/Bureau/historique/label-07-05.csv'
#################################################
########### Important functions #################
#################################################
def plot_res(df,predict,y,h = [3,27],threshold=0.5):
    x = df.values
    x = x[(h[0]-3)*60:(h[1]-3)*60]
    t= [(i+3)/60+h[0] for i in range(len(x))]
    
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
    
    l1 = find_index(pred,1)
    x1 = [t[i] for i in l1]
    y1 = [x[i]+10000 for i in l1]

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
            x=0,
            y=0,
            mode = 'markers',
            name = 'test',
    )
    trace4 = go.Scatter(
            x=x3,
            y=y3,
            mode = 'markers',
            name = 'true markers'
    )

    #fig = tools.make_subplots(rows=4, cols=1, specs=[[{}], [{}], [{}], [{}]],
                                  #shared_xaxes=True, shared_yaxes=True,
                                  #vertical_spacing=0.001)
    #fig.append_trace(trace1, 1, 1)
    #fig.append_trace(trace2, 1, 1)
    #fig.append_trace(trace3, 1, 1)
    #fig.append_trace(trace4, 1, 1)

    #fig['layout'].update(height=3000, width=2000, title='Annomalie detection '+str(h[0])+'h-'+str(h[1])+'h')
    #plot(fig, filename='Stack'+str(h[0])+'h-'+str(h[1])+'h.html')
    
def ROC_curve(Y_test,y_score):
    y_test = np.array([[0,1] if i==1 else [1,0] for i in Y_test])
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
    #### get files names ###
    names = pd.read_csv('files.csv')
    fileX_train = literal_eval(names['fileX_train'][0])
    fileY_train = literal_eval(names['fileY_train'][0])

    fileX_valid =literal_eval(names['fileX_valid'][0])
    fileY_valid = literal_eval(names['fileY_valid'][0])
    fileX =literal_eval(names['fileX_test'][0])
    fileY = literal_eval(names['fileY_test'][0])
    y = pd.DataFrame()
    for filex,filey in zip(fileX,fileY  ):
        y_ = pd.read_csv(filey)
        y_train = y_['label'][3:]
        y = pd.concat([y,y_train])
    Y = y.values.reshape(-1, 1)
    
    y_valid = pd.DataFrame()
    for filex,filey in zip(fileX_valid,fileY_valid):
        y_ = pd.read_csv(filey)
        y_train = y_['label'][3:]
        y_valid = pd.concat([y_valid,y_train])
    Y  _valid = y_valid.values.reshape(-1, 1)
    if(len(argv)==0):
        argv = [0.5]
    if(str(argv[0]) == 'trainclf'):
        print('training models ...')
        print("LGBM")
        l1 = os.system("python /home/alexis/Bureau/Stage/ML/LightGBM.py 0.5")
        print("catboost")
        l2 = os.system("python /home/alexis/Bureau/Stage/ML/CatBoost.py 0.5")
        print("classic")
        l3 = os.system("python /home/alexis/Bureau/Stage/ML/SVC.py 0.5")
        print("NN")
        l4 = os.system("python /home/alexis/Bureau/Stage/ML/NeuralNetwork.py 0.5")
        print("XGB")
        l5 = os.system("python /home/alexis/Bureau/Stage/ML/XgBoost.py 0.5")
        print("KNN")
        l6 = os.system("python /home/alexis/Bureau/Stage/ML/KNN.py 0.5")
        print("LSTM")
        l7 = os.system("python /home/alexis/Bureau/Stage/ML/LSTM.py 0.5")
        return 0
    if(str(argv[0]) == 'trainlogreg'):
        print('training logistic regression...')
        l1 = pd.read_csv("lightGBM_valid.csv")
        l2 = pd.read_csv("catboost_valid.csv")
        l3 = pd.read_csv("SVC_valid.csv")
        l4 = pd.read_csv("NN_valid.csv")
        l5 = pd.read_csv("xgb_valid.csv")
        l6 = pd.read_csv("KNN_valid.csv")
        l7 = pd.read_csv("LSTM_valid.csv")
        
        
        X_valid = pd.concat([l2,l3,l4,l5,l6,l7], axis=1).values #"****************"
        for i in [0.001]:
            print("C="+str(i))
            np.random.seed(7)
            logistic = linear_model.LogisticRegression(C=i,class_weight='balanced',penalty='l2')
            logistic.fit(X_valid, Y_valid)
            pickle.dump(logistic, open('model/logistic_regression.sav', 'wb'))
            os.system("python /home/alexis/Bureau/Stage/ML/Stack.py")
        return 0
    if(str(argv[0]) == 'trainall'):
        os.system("python /home/alexis/Bureau/Stage/ML/Stack.py trainclf")
        os.system("python /home/alexis/Bureau/Stage/ML/Stack.py trainlogreg")
    
    else:
        T = argv
        print('Scoring...')
        l1 = pd.read_csv("lightGBM.csv")
        l2 = pd.read_csv("catboost.csv")
        l3 = pd.read_csv("SVC.csv")
        l4 = pd.read_csv("NN.csv")
        l5 = pd.read_csv("xgb.csv")
        l6 = pd.read_csv("KNN.csv")
        l7 = pd.read_csv("LSTM.csv")
            
        X = pd.concat([l2,l3,l4,l5,l6,l7], axis=1).values #"********************"
        np.random.seed(7)
        logistic = pickle.load(open('model/logistic_regression.sav', 'rb'))
        np.random.seed(7)
        Predict = logistic.predict_proba(X)
        for j in T:
            print("Threshold="+str(j))
            #for h in [[3,27],[6,13],[13,20],[20,27],[6,24],[10,13],[12,15],[6,11],[13,16],[14,18],[16,19],[19,22],[20,23],[23,27],[10,18]]:
                #print(h)
                #plot_res(pd.read_csv(fileX[0])['t'],Predict,Y,h,threshold = float(j))
                #pred = list([1 if i[-1]>float(j) else 0 for i in Predict])

            plot_res(pd.read_csv(fileX[0])['t'],Predict,Y,threshold = float(j))
        #ROC_curve(Y,Predict)
    return ("process achevé sans erreures")


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
