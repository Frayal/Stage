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


#################################################
########### Global variables ####################
#################################################

PATH = '/home/alexis/Bureau/Stage/Time-series/clean data/'
NOISE_LEVEL = 2
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
        data = [float(x) for x in datas[:-1]]
        return data
    if(temp[-1] == 'csv'):
        data = pd.read_csv(PATH+str(file),index_col = 0)
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
        res.append(np.random.normal(res[-1],max([res[-1]/10**8,1])))
    return res

def shift_preprocess(data,name,noise_level=NOISE_LEVEL):
    '''
    prend en entrée un dataframe contenant les valeurs d'audition
    renvoie un dataframe de 4 colonnes contenant les valeurs bruitées
    et un shift permettant d'utiliser l'historique (fixé a 3 ici)
    '''
    temps = data['values'].values
    for i in range(noise_level):
        temps = add_noise(temps)
    temps = DataFrame(temps)
    dataframe = concat([temps.shift(3), temps.shift(2), temps.shift(1), temps], axis=1)
    dataframe.columns = ['t-3', 't-2', 't-1', 't']
    dataframe['minutes'] = dataframe.index
    dataframe = dataframe.drop(dataframe.index[[0,1,2]])
    print(dataframe.head())
    plt.plot(dataframe['t'].values)
    plt.savefig('data/png/'+name+'-0.png')
    return dataframe



def processing(dataframe,name):
    '''
    prend en entrée un dataframe contenant les shifts
    renvoie en sortie un dataframe de features
    permet le plot de certains features intéressants
    a compléter en fonction des features que l'on souhaite
    considérer ou non.

    '''
    dataframe["diff t t-1"]=dataframe["t"]-dataframe["t-1"]
    dataframe["diff t t-2"]=dataframe["t"]-dataframe["t-2"]
    dataframe["diff t t-3"]=dataframe["t"]-dataframe["t-3"]
    dataframe["diff t-1 t-2"]=dataframe["t-1"]-dataframe["t-2"]
    dataframe["diff t-1 t-3"]=dataframe["t-1"]-dataframe["t-3"]
    dataframe["diff t-2 t-3"]=dataframe["t-2"]-dataframe["t-3"]
    dataframe["diff t t-1"]=dataframe["t"]-dataframe["t-1"]
    dataframe["mean"]=(dataframe["t"]+dataframe["t-1"]+dataframe["t-2"]+dataframe["t-3"])/4
    dataframe["distance to mean"] = dataframe["t"] - dataframe["mean"]
    dataframe["pente t t-2"] = dataframe["diff t t-2"]/2
    dataframe["pente t t-3"] = dataframe["diff t t-3"]/3
    dataframe["pente t-1 t-3"] = dataframe["diff t-1 t-3"]/2

    x = dataframe["diff t t-1"]
    m = np.mean(x)
    sd = sum([(y-m)**2 for y in x])/(len(x)-1)
    print("moyenne: %s   standard deviation: %s" %(str(m),str(sd)))

    d = scipy.stats.norm(m, sd)
    time = [i/((2**NOISE_LEVEL)*60) for i in range(len(x))]
    prob = [ d.pdf(y) for y in x]
    dist = [ d.cdf(y) for y in x]
    dataframe["probability"] = prob
    dataframe["distribution"] = dist

    dataframe["skewness"]= ((3)**0.5)*(((dataframe['t']-dataframe["mean"])**3)/4+((dataframe['t-1']-dataframe["mean"])**3)/4
                            +((dataframe['t-2']-dataframe["mean"])**3)/4+((dataframe['t-3']-dataframe["mean"])**3)/4)/(sd**3)
    dataframe["skewness"]= (((dataframe['t']-dataframe["mean"])**4)/4+((dataframe['t-1']-dataframe["mean"])**4)/4
                            +((dataframe['t-2']-dataframe["mean"])**4)/4+((dataframe['t-3']-dataframe["mean"])**4)/4)/(sd**4)
   
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
    return dataframe


def annomalie_detection(df,automatic_threshold = True):
    '''
    prend en entrée un DataFrame
    renvoie en sortie une liste de points ou il y a annomalie
    les threshold sont a déterminer manuellement pour le moment
    on peut cependant essayer de les fixer de manière automatique
    en fonction de l'historique de la chaîne/departement/regroupement(csp)
    '''
    if(automatic_threshold):
        l = []
        res = []
        names = df.columns
        for n in names:
            if any(c==n for c in ("t", "t-1", "t-2","t-3","diff t-1 t-2","diff t-1 t-3","diff t t-1","diff t t-2","diff t t-3","diff t-2 t-3")):
                pass
            else:
                df[n]=df[n]/(abs(df[n]).max())
        for n in names:
            if any(c==n for c in ("t", "t-1", "t-2","t-3","diff t-1 t-2","diff t-1 t-3","diff t t-1","diff t t-2","diff t t-3","diff t-2 t-3")):
                pass
            else:
                df_temp = df.where(df[n]>0.8)
                l.append(df_temp['t'].values)
        for i in df['t'].values:
            if(i%1000 == 0): print(i)
            count = 0
            for j in range(len(l)):
                if any(c == i for c in l[j]):
                    count+=1
            res.append(count)
    else:
        pass
        #TODO:implement a xgboost to determine annomalies
        # load model from file
        loaded_model = pickle.load(open("pima.pickle.dat", "rb"))
        # make predictions for test data
        y_pred = loaded_model.predict_proba(df.values)
        
    
    return res


def plot_annomalies(annomalies,df,name):
    """
    prend en entrée le nombres d'indicateurs qui detectent une annomalie
    ainsi que le datafame contenant tous les features(on a effacé 3 lignes)
    renvoie un graph présentant les zones d'incertitudes quand a la présence
    d'un événement important dans la plage horaire
    """
    c = np.cos([0.3*b for b in annomalies])
    x = df['t'].values
    fig, ax = plt.subplots()
    ax.scatter([i/60 for i in range(len(x))],x,c=c)
    ax=plt.gca()
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    plt.savefig('data/png/'+name+'-2.png')
    print("quel jolis graphes!")



#################################################
########### main with options ###################
#################################################
import sys

def main(argv):
    df = load_timeserie(argv)
    df = shift_preprocess(df,argv.split('.')[0])
    df = processing(df,argv.split('.')[0])
    df.to_csv('data/processed/'+argv.split('.')[0]+"-processed.csv",index=False)

    annomalies = annomalie_detection(df)
    plot_annomalies(annomalies,df,argv)
    return ("process achevé sans erreures")


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1])
