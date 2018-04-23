#################################################
#created the 20/04/2018 12:57 by Alexis Blanchet#
#################################################
#-*- coding: utf-8 -*-
'''
pre-processing des datas
fabrication des Times Series
selection et agrégation de flux
prend en entrée:
    un fichier .csv
    un département(si 0 tous les départements, 99 étranger)
    une chaîne
    une catégorie csp (si 0 agrégation de toutes les csp)

renvoie:
    un fichier csp contenant la colonne d'index correspondant
    aux minutes de la journée et la colonne des valeurs d'audition
'''

'''
Améliorations possibles:
mettre en entrée des listes pour permettre l'agrégation de plusieures
csp ou départements
'''
import warnings
warnings.filterwarnings('ignore')
#################################################
###########        Imports      #################
#################################################
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series
from pandas import DataFrame
from pandas import concat
import random
import os

#################################################
########### Global variables ####################
#################################################
PATH = ''

#################################################
########### Important functions #################
#################################################

#################################################
########### main with options ###################
#################################################
import sys

def main(argv):
    file = argv[0]
    departement = argv[1]
    chaine = argv[2]
    csp = argv[3]

    f = file.split(".")[0]
    f = f.split('_')

    df = pd.read_csv(PATH+str(file),sep=';')
    df = df.loc[df["IDCST"]== int(chaine)]
    df = df.loc[df["DPT"]== int(departement)]
    if(int(csp) == 0):
        df = df.sum(axis = 0)
    else:
        df = df.loc[df["IDCIBLE"] == int(csp)]
    df = df.drop(["IDCST","DPT","IDCIBLE"])
    df.to_csv(PATH+"clean data/"+str(f[0])+"_"+str(f[1])+"_"+str(departement)+"_"+str(chaine)+"_"+str(csp)+"_cleandata.csv",header=['values'])
    return ("process achevé sans erreures")


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
