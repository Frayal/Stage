#################################################
#created the 19/04/2018 12:57 by Alexis Blanchet#
#################################################
#-*- coding: utf-8 -*-
'''
exploration des données des chaînes
recherche de distribution
visualisation des données
récupération des événements notables dans l'historique
'''

'''
Améliorations possibles:
- meilleur crible des données pour trouver des sous catégories donnant
des fonnées plus utilisable (et utiliser ces spécificités)
- une fois l'historique des pubs trouvées, enlever le thresholding et le remplacer
par un xgboost afin d'optimiser les poids de chaque détécteur
- écrire le main pour que le fichier execute l'ensemble des actions automatiquement
(ce serait pas mal)
'''
import warnings
warnings.filterwarnings('ignore')
#################################################
###########        Imports      #################
#################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series
from pandas import DataFrame
from pandas import concat
import random
import os
import sys
import scipy.stats
import matplotlib
import pickle
import plotly
import plotly.graph_objs as go
import plotly.offline as offline
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import random
import pickle
import xgboost as xgb
from catboost import CatBoostClassifier
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
#################################################
########### Global variables ####################
#################################################

PATH = '/home/alexis/Bureau/Stage/Time-series/clean data/'
NOISE_LEVEL = 0
THRESHOLD = 2e7

#################################################
########### Important functions #################
#################################################

def load_timeserie(file):
    '''
    prend en entrée le nom d'un fichier se trouvant dans le dossier
    des times Series
    renvoie en sortie un DataFrame contenant les valeurs d'audition
    ATTENTION: le processus d'écriture peut se faire en même temps
    que la lecture ==> Tester l'option os.O_NONBLOCK qui doit régler
    le problème.
    '''
    temp = file.split('.')
    if(temp[-1] == 'txt'):
        data_file = open(PATH+str(file), 'r',os.O_NONBLOCK)
        data_file = data_file.read()
        datas = data_file.split('\n')
        data = pd.DataFrame([float(x) for x in datas[:-1]])
        return data
    if(temp[-1] == 'csv'):
        data = pd.read_csv(PATH+str(file))
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.fillna(1)
        return data
    else:
        print("mauvais format: veuillez fournir un .txt ou un .csv")
        return 0

def add_noise(l):
    '''
    prend en entrée une liste
    rajout après chaque point un point légèrement bruité
    renvoie la nouvelle liste avec l'information bruitée
    '''
    res = []
    for i in range(len(l)):
        res.append(l[i])
        res.append(np.random.normal(res[-1],max([(res[-1]/(10**3)),1])))
    return res

def shift_preprocess(data,name,noise_level=NOISE_LEVEL):
    '''
    prend en entrée un dataframe contenant les valeurs d'audition
    renvoie un dataframe de 4 colonnes contenant les valeurs bruitées
    et un shift permettant d'utiliser l'historique (fixé a 3 ici)
    '''
    temps = data['values'].values
    for i in range(noise_level-1):
        temps = add_noise(temps)
    temps = DataFrame(temps)
    dataframe = concat([temps.shift(3), temps.shift(2), temps.shift(1), temps], axis=1)
    dataframe.columns = ['t-3', 't-2', 't-1', 't']
    dataframe['minutes'] = dataframe.index
    dataframe = dataframe.drop(dataframe.index[[0,1,2]])
    '''
    plt.plot(dataframe['t'].values)
    plt.savefig('data/png/'+name+'-0.png')
    '''
    return dataframe



def KFDR(a,b):
    GAMMA = 10
    sigma = 0.5*a[1] + 0.5*b[1]
    return (((4**2/8)*(a[0]-b[0])**2)/(sigma + GAMMA))



def processing(dataframe,name):
    '''
    prend en entrée un dataframe contenant les shifts
    renvoie en sortie un dataframe de features
    permet le plot de certains features intéressants
    a compléter en fonction des features que l'on souhaite
    considérer ou non.

    '''
    SIGMA = 1
    L = 2000000
    
    THRESHOLD = max(dataframe['t'].values)/100
    dataframe["diff t t-1"]=dataframe.apply(lambda x:max(-THRESHOLD,min(THRESHOLD,x["t"]-x["t-1"])),axis = 1)
    dataframe["diff t t-2"]=dataframe.apply(lambda x:max(-THRESHOLD**2,min(THRESHOLD**2,x["t"]-x["t-2"])),axis = 1)
    dataframe["diff t t-3"]=dataframe.apply(lambda x:max(-THRESHOLD**2,min(THRESHOLD**2,x["t"]-x["t-3"])),axis = 1)
    dataframe["diff t-1 t-2"]=dataframe.apply(lambda x:max(-THRESHOLD,min(THRESHOLD**2,x["t-1"]-x["t-2"])),axis = 1)
    dataframe["diff t-1 t-3"]=dataframe.apply(lambda x:max(-THRESHOLD**2,min(THRESHOLD**2,x["t-1"]-x["t-3"])),axis = 1)
    dataframe["diff t-2 t-3"]=dataframe.apply(lambda x:max(-THRESHOLD,min(THRESHOLD,x["t-2"]-x["t-3"])),axis = 1)
    
    dataframe["mean"]=(dataframe["t"]+dataframe["t-1"]+dataframe["t-2"]+dataframe["t-3"])/4
    dataframe["distance to mean"] = dataframe.apply(lambda x:max(-THRESHOLD,min(10**7,(x["t"]-x["mean"])/x["t"])),axis = 1)
    dataframe["pente t t-2"] = dataframe.apply(lambda x:max(-THRESHOLD,min(THRESHOLD,x["diff t t-2"]/2)),axis = 1)
    dataframe["pente t t-3"] = dataframe.apply(lambda x:max(-THRESHOLD,min(THRESHOLD,x["diff t t-3"]/3)),axis = 1)
    dataframe["pente t-1 t-3"] = dataframe.apply(lambda x:max(-THRESHOLD,min(THRESHOLD,x["diff t-1 t-3"]/2)),axis = 1)
    
    
    dataframe["diff pente 1-2"] = (dataframe["diff t t-1"] - dataframe["diff t-1 t-2"])/dataframe["diff t t-1"]
    dataframe["diff pente 1-3"] = (dataframe["diff t t-1"] - dataframe["diff t-2 t-3"])/dataframe["diff t t-1"]
    dataframe['diff pentes'] = dataframe.apply(lambda x:max(-50,min(50,x["diff pente 1-2"]-x["diff pente 1-3"])),axis = 1)
    dataframe["diff pente 2-3"] = (dataframe["diff t-1 t-2"] - dataframe["diff t-2 t-3"])/dataframe["diff t-1 t-2"]
    
    dataframe['GP'] = (SIGMA**2)*(np.exp(-((dataframe['t']-dataframe['t-1'])**2)/dataframe['t']**2))
    dataframe['covariance'] = 0.25*((dataframe['t']-dataframe['mean'])**2 + (dataframe['t-1']-dataframe['mean'])**2 +(dataframe['t-2']-dataframe['mean'])**2 + (dataframe['t-3']-dataframe['mean'])**2)
    dataframe['pics'] = dataframe.apply(lambda row: 1 if((row['t-2']<row['t-1']) & (row['t']<row['t-1'])) else -1 if((row['t-2']>row['t-1']) & (row['t']>row['t-1'])) else 0,axis =1)
    
    h = [0,0]
    
    for index, row in dataframe.iterrows():
        dataframe.set_value(index, 'KDFR', KFDR([row['mean'],row['covariance']],h))
        h = [row['mean'],row['covariance']]
    dataframe.set_value(3, 'KDFR', 1.5)
    
    x = dataframe["diff t t-1"]
    m = np.mean(x)
    sd = sum([(y-m)**2 for y in x])/(len(x)-1)
    #print("moyenne: %s   standard deviation: %s" %(str(m),str(sd)))

    d = scipy.stats.norm(m, sd/1000)
    time = [i/((2**NOISE_LEVEL)*60) for i in range(len(x))]
    prob = [ d.pdf(y) for y in x]
    dist = [ d.cdf(y) for y in x]
    dataframe["probability"] = prob
    dataframe["distribution"] = dist

    dataframe["skewness"]= ((3)**0.5)*(((dataframe['t']-dataframe["mean"])**3)/4+((dataframe['t-1']-dataframe["mean"])**3)/4
                            +((dataframe['t-2']-dataframe["mean"])**3)/4+((dataframe['t-3']-dataframe["mean"])**3)/4)/(sd**3)
    dataframe["kurtosis"]= (((dataframe['t']-dataframe["mean"])**4)/4+((dataframe['t-1']-dataframe["mean"])**4)/4
                            +((dataframe['t-2']-dataframe["mean"])**4)/4+((dataframe['t-3']-dataframe["mean"])**4)/4)/(sd**4)
    '''
    plt.subplot(2, 1, 1)
    plt.plot(time,prob, '-')
    ax=plt.gca()
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    plt.title('probability and distribution of the change in the number of viewers')

    plt.subplot(2, 1, 2)
    plt.plot(time,dist, '.-')
    ax=plt.gca()
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    plt.xlabel('time (arbitrary)')
    plt.savefig('data/png/'+name+'-1.png')
    plt.close()
    '''
    return dataframe



def annomalie_detection(df,clf,automatic_threshold = True):
    '''
    prend en entrée un DataFrame
    renvoie en sortie une liste de points ou il y a annomalie
    les threshold sont a déterminer manuellement pour le moment
    on peut cependant essayer de les fixer de manière automatique
    en fonction de l'historique de la chaîne/departement/regroupement(csp)
    '''
    res = []
    irrelevant = ('t-3', 't-2', 't-1', 't')
    allnames = df.columns
    names = list(set(allnames) - set(irrelevant))
    #print(names)
    temp_df = df[names]
    if(automatic_threshold):
        
        info = df.describe()
        temp_df['signe'] = np.sign(temp_df['pente t t-2']+temp_df['pente t-1 t-3']+temp_df['pente t t-3']+temp_df['diff t-1 t-2'])
        #['diff t-2 t-3', 'distribution', 'skewness', 'diff pente 1-3','covariance', 'KDFR', 'diff pente 1-2', 'pente t-1 t-3', 'diff t-1 t-2','pente t t-2', 'diff pente 2-3', 'distance to mean', 'pente t t-3','GP', 'probability']
        poids = [ 0.01635393,0.03110482,  0.01142569,  0.00263594,  0.02041714,  0.04325364,0.00982603,  0.01245762,  0.02220948,  0.00309737,0.00173761,0.04263638,0.03090817,  0.00556751,  0.01292914,  0.0071749,0.04850749,0.01424759,0.0060423,0.03402776,0.00756308,0.0317241,0.00425174,0.0046664 ]
        for i in range(len(names)-len(poids)):
            poids.append(random.randint(1,4))
        ptot = sum(poids)
        for name,p in zip(names,poids):
            m = info.loc[['mean']][name].values[0]
            sd = info.loc[['std']][name].values[0]
            d = scipy.stats.norm(m, sd)
            temp_df[name] = temp_df[name].apply(lambda x: p*d.cdf(x))
            
            
        temp_df['proba'] = temp_df[names].sum(axis = 1)
        temp_df['proba'] = temp_df['proba'].apply(lambda x: abs((x-ptot*0.5)/ptot))
        #temp_df['r'] = [ 1 if (v[1]+l[1])*0.5>0.05    else 0 for v,l in zip(clf[0].predict_proba(df.values),clf[1].predict_proba(df.values))]
        #temp_df['proba'] =  (temp_df['proba']*0.8+ temp_df['r']*0.2)
        #print(temp_df.head())
            
        temp_df['annomalies'] =  temp_df['proba'].apply(threshold) 
        
        res = temp_df['annomalies']*temp_df['signe']
        #res = df['pics']
        
        

    else:
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        scaler = MinMaxScaler(feature_range=(0, 1))
        trainX = scaler.fit_transform(df)
        #trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        res1 = clf[0].predict_proba(trainX)
        res2 = clf[1].predict_proba(trainX)
        res = [(r1[1]+r2[1])*0.5 for r1,r2 in zip(res1,res2)]
        res = [1 if l>0.17 else 0 for l in res]
        #v = [1 if x[0]>0.1 else 0 for x in clf.predict(trainX)]
            
    return res

def threshold(x):
    if(x>0.17): 
        return 1
    else:
        return 0

def find_index(l,v):
    res = []
    for i, j in enumerate(l):
        if(j == v):
            res.append(i)
    return res    

def plot_annomalies(annomalies,df,name,real_data,file):
    """
    prend en entrée le nombres d'indicateurs qui detectent une annomalie
    ainsi que le datafame contenant tous les features(on a effacé 3 lignes)
    renvoie un graph présentant les zones d'incertitudes quand a la présence
    d'un événement important dans la plage horaire
    en vert une montée d'audience, en orange une baisse d'audiance
    """
    dfx = pd.read_csv(file)['debut']
    annomalies = list(annomalies)
    l1 = find_index(annomalies,0)
    l2 = find_index(annomalies,-1)
    l3 = find_index(annomalies,1)

    x = df['t'].values
    t= [i/(60*2**(max([NOISE_LEVEL-1,0])))+3 for i in range(len(x))]
    t2 = [i/(60) for i in range(len(real_data))]
    x1 = [t[i] for i in l1]
    x2 = [t[i] for i in l2]
    x3 = [t[i] for i in l3]
    y1 = [x[i] for i in l1]
    y2 = [x[i] for i in l2]
    y3 = [x[i] for i in l3]
    
    dfy = [x[(d - 3*60)%1440]+ 10000000  for d in dfx]
    
    trace1 = go.Scatter(
        x=x1,
        y=y1,
        mode = 'markers',
        name = 'regular',
    
    )
    trace2 = go.Scatter(
        x=x2,
        y=y2,
        mode = 'markers',
        name ='anormal loss',
    )
    trace3 = go.Scatter(
        x=x3,
        y=y3,
        mode = 'markers',
        name = 'anormal gain',
    )
    trace4 = go.Scatter(
        x=dfx/60,
        y=dfy,
        mode = 'markers',
        name = 'begin of programmes',
    )
        
    fig = tools.make_subplots(rows=4, cols=1, specs=[[{}], [{}], [{}], [{}]],
                              shared_xaxes=True, shared_yaxes=True,
                              vertical_spacing=0.001)
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 1)
    fig.append_trace(trace3, 1, 1)
    fig.append_trace(trace4, 1, 1)

    fig['layout'].update(height=3000, width=2000, title='Annomalie detection')
    plot(fig, filename='data/html/'+name+'.html')


#################################################
########### main with options ###################
#################################################
import sys

def main(argv):
    df = load_timeserie(argv)
    if(type(df) == int):
        return("wrong imput file")
    real_data = df['values'][3:]
    df = shift_preprocess(df,argv.split('.')[0])
    df = processing(df,argv.split('.')[0])
    #json_file = open('/home/alexis/Bureau/Stage/ML/model.json', 'r')
    #loaded_model_json = json_file.read()
    #json_file.close()
    #loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    #loaded_model.load_weights("/home/alexis/Bureau/Stage/ML/model.h5")
    #loaded_model.compile(loss='binary_crossentropy', optimizer='adam')
    #clf1 = CatBoostClassifier().load_model("model1")
    #clf2 = CatBoostClassifier().load_model("model2")
    #clf1 = pickle.load(open("pima.pickle.dat", "rb"))
    #clf2 = pickle.load(open("pima2.pickle.dat", "rb"))
    clf = []
    annomalies = annomalie_detection(df,clf,True)
    df["label"] = annomalies
    df.to_csv('data/processed/'+argv.split('.')[0]+"-processed.csv",index=False)
    df.to_csv('/home/alexis/Bureau/Stage/historique/RTS/'+argv.split('.')[0]+"-processed.csv",index=False)
    date = list(argv.split('.')[0].split('_')[1])
    date = "".join(date[-2:])
    #plot_annomalies(annomalies,df,argv.split('.')[0],real_data,'/home/alexis/Bureau/Stage/ProgrammesTV/IPTV_0192_2017-12-'+str(date)+'_TF1.csv')
    m = max(annomalies)
    y = [1 if b/(m)>0.5 else 0 for b in annomalies]
    y = DataFrame(y)
    y.to_csv('data/processed/'+argv.split('.')[0]+"-y.csv",index=False)
    y.to_csv('/home/alexis/Bureau/Stage/historique/RTS/'+argv.split('.')[0]+"-y.csv",index=False)
    return ("process achevé sans erreures")


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1])
