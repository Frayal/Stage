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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import scipy.stats
import matplotlib
import plotly.offline as offline
import plotly.graph_objs as go
import datetime
#################################################
########### Global variables ####################
#################################################


#################################################
########### Important functions #################
#################################################
def get_info(file):
    pd.read_csv(file)
    date = list(file.split('-')[1])
    year = ''.join((date[0],date[1],date[2],date[3]))
    month = ''.join((date[4],date[5]))
    day = ''.join((date[6],date[7]))
    heure = ''.join((date[8],date[9]))
    minutes = ''.join((date[10],date[11]))
    return([year,month,day,heure,minutes])


def load_file(file,chaine):
    l = get_info(file)
    df = pd.read_csv(file,sep=';')
    df = df.rename(index=str, columns={"Unnamed: 0": "tf", df.columns.values[1]: "count"})
    df["chaine d'origine"]=df['tf'].apply(get_origin)
    df["chaine d'arrivée"]=df['tf'].apply(get_destination)
    gain = df[df["chaine d'arrivée"]==int(chaine)]['count'].sum()
    loss = df[df["chaine d'origine"]==int(chaine)]['count'].sum()
    temp_df = pd.DataFrame()
    temp_df['année'] = [l[0]]
    temp_df['mois'] = l[1]
    temp_df['jour'] = l[2]
    temp_df['heure'] = l[3]
    temp_df['minute'] = l[4]
    temp_df['loss'] = loss
    temp_df['gain'] = gain
    
    return temp_df


def get_origin(x):
    x = int(x) & ((1<<24)-1)
    return x

def get_destination(x):
    x = int(x)>>24 & ((1<<24)-1)
    return x
#################################################
########### main with options ###################
#################################################


def main(argv):
    gain = pd.DataFrame()
    loss = pd.DataFrame()
    files = os.listdir('/home/alexis/Bureau/Stage/Vectors')
    for file in files:
        if(file.split('.')[1] == 'py' or file.split('.')[1] == 'ipynb' or file.split('.')[1] == 'csv' or file.split('.')[1] == 'ipynb_checkpoints'
):
            pass
        else:
            print(file)
            df = load_file(file,argv)
            gain = gain.append(df.drop(['loss'],axis=1))
            loss = loss.append(df.drop(['gain'],axis=1))
        gain.to_csv('gain.csv',index=False)
        loss.to_csv('loss.csv',index=False)
    
    return ("process achevé sans erreures")


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1])
