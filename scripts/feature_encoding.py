#################################################
#created the 114/06/2018 16:24 by Alexis Blanchet#
#################################################
#-*- coding: utf-8 -*-
'''
L'objectif est un encodage manuel des categoricals features afin de ne pas être surpris
en direct par une catégorie non encore rencontrée.
On encodera de plus de différente manière les features afin de ne pas être spécifique à la chaîne que l'on traite
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
import pandas as pd
import numpy as np
import os

#################################################
########### Global variables ####################
#################################################
names = ['minute','partie de la journée','Change Point','pourcentage','partie du programme','programme','duree','nombre de pub potentiel',
 'lastCP','lastPub','lastend','currentduree','Pubinhour','probability of CP','nb de pubs encore possible','chaine','CLE-FORMAT','CLE-GENRE','Heure','labels']
irrelevant = ['']

#################################################
########### Important functions #################
#################################################
def encoding_partoftheday(x):
    if(x == 'nuit'):
        return 1
    if(x == 'fin de nuit'):
        return 2
    if(x == 'début de matinée'):
        return 3
    if(x == 'matinée'):
        return 4
    if(x == 'midi'):
        return 5
    if(x == 'après-midi'):
        return 6
    if(x == "fin d'après-midi"):
        return 7
    if(x == 'prime time'):
        return 8
    elif(type(x) == str):
        return 0
    else:
        return x

def encoding_partofprogramme(x):
    if(x == 'début de programme'):
        return 1
    if(x == 'milieu de programme'):
        return 2
    if(x == 'début de fin de programme'):
        return 3
    if(x == 'fin de programme'):
        return 4
    if(x == 'en dehors du programme'):
        return 5
    elif(type(x) == str):
        return 0
    else:
        return x


def encoding_duree(x):
    if(x == 'très court'):
        return 1
    if(x == 'court'):
        return 2
    if(x == 'moyen'):
        return 3
    if(x == 'long'):
        return 4
    if(x == 'très long'):
        return 5
    if(x == 'super long'):
        return 6
    elif(type(x) == str):
        return 0
    else:
        return x

def encoding_chaine(x):
    if(x == 'TF1'):
        return 1
    if(x == 'M6'):
        return 2
    elif(type(x) == str):
        return 0
    else:
        return x


def process(df):
    df['partie de la journée'] = df['partie de la journée'].apply(lambda x: encoding_partoftheday(x))
    df['partie du programme'] = df['partie du programme'].apply(lambda x: encoding_partofprogramme(x))
    df["duree"] = df['duree'].apply(lambda x: encoding_duree(x))
    df['chaine'] = df['chaine'].apply(lamdba x: encoding_chaine(x))
    if('Heure' in df.columns.values):
        df['Time-h'] = df['Heure'].apply(lambda x: (x.split(':'))[0])
        df['Time-m'] = df['Heure'].apply(lambda x: (x.split(':'))[1])
        df = df.drop(['Heure'],axis = 1)
    if('programme' in df.columns.values):
        df = df.drop(['programme'],axis = 1)
    return df


#################################################
########### main with options ###################
#################################################


def main(argv):
    chaine = argv[0]
    files = os.listdir('/home/alexis/Bureau/Project/results/truemerge/'+chaine)
    for file in files:
        print(file)
        datas = pd.read_csv('/home/alexis/Bureau/Project/results/truemerge/'+chaine+'/'+file)
        datas = process(datas)
        datas.to_csv('/home/alexis/Bureau/Project/results/truemerge/'+chaine+'/'+file,index=False)


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
