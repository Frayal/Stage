#################################################
#created the 25/04/2018 15:41 by Alexis Blanchet#
#################################################
#-*- coding: utf-8 -*-
'''

'''

'''
Améliorations possibles:
ATTENTION dans le nom du fichier figure l'équivalence chaine-identification
il faut donc les deux pour lire le bon fichier mais une simple lecture des
noms peut donner l'équivalence des tables
'''
import warnings
warnings.filterwarnings('ignore')
#################################################
###########        Imports      #################
#################################################
import sys
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import xmltodict, json
from collections import OrderedDict
import os
#################################################
########### Global variables ####################
#################################################
PATH ='/home/alexis/Bureau/Project/Datas/PTV/'

#################################################
########### Important functions #################
#################################################


def processPTV(file):
    irrelevant = ['LESPERSONNES','LESTEXTES','PHOTO','MAGAZINEXY','SERIE','NATIONALITE','FILMTELEFILM','GENRE','CS']
    tree = ET.parse(file)
    root = tree.getroot()


    #print(root.tag)
    l1 = [child for child in root]
    #print(l1)
    l2 = [[child for child in root] for root in l1 ]
    #print(l2[1])

    data = pd.DataFrame()
    for i,elem in enumerate(l2[1]):
        xmlstr = ET.tostring(elem)
        o = xmltodict.parse(xmlstr)
        df2 = pd.DataFrame.from_dict(o["DIFFUSION"],orient = 'index').rename(columns={0: i})
        data = [data,df2.T]
        data = pd.concat(data)


    temp_data = pd.DataFrame()
    for i in range(data.shape[0]):
        df = data['ATTRIBUTS']
        l = (df[[i]].values)[0]
        df2 = pd.DataFrame.from_dict(l,orient = 'index')
        df2 = df2.rename(columns = {0: i})
        #print(df2.head())
        temp_data = [temp_data,df2]
        temp_data = pd.concat(temp_data,axis = 1)
    temp_data.head()
    data = data.join(temp_data.T)
    temp_data = pd.DataFrame()
    names = data.columns.values
    n = list(set(names) - set(irrelevant))
    data = data[n]
    for i in range(data.shape[0]):
        df = data['EMISSION']
        l = (df[[i]].values)[0]
        df2 = pd.DataFrame.from_dict(l,orient = 'index')
        df2 = df2.rename(columns = {0: i})
        #print(df2.head())
        temp_data = [temp_data,df2]
        temp_data = pd.concat(temp_data,axis = 1)
        temp_data = temp_data.drop(['@DATEMODIF'])

    temp_data.head()

    data = data.join(temp_data.T)
    temp_data = pd.DataFrame()
    data = data.drop(['ATTRIBUTS','EMISSION'],axis = 1)

    for i in range(data.shape[0]):
        df = data['FORMAT']
        l = (df[[i]].values)[0]
        df2 = pd.DataFrame.from_dict(l,orient = 'index')
        df2 = df2.rename(columns = {0: i})
        #print(df2.head())
        temp_data = [temp_data,df2]
        temp_data = pd.concat(temp_data,axis = 1)
    temp_data = temp_data.drop(['#text'])
    temp_data = temp_data.T.rename(columns={"@CLE": "CLE-FORMAT"})

    temp_data.head()
    data = data.join(temp_data)
    temp_data = pd.DataFrame()
    data = data.drop(['FORMAT'],axis=1)

    for i in range(data.shape[0]):
        df = data['GENRE']
        l = (df[[i]].values)[0]
        df2 = pd.DataFrame.from_dict(l,orient = 'index')
        df2 = df2.rename(columns = {0: i})
        #print(df2.head())
        temp_data = [temp_data,df2]
        temp_data = pd.concat(temp_data,axis = 1)
        temp_data = temp_data.drop(['@GENRESIMPLE'])

    temp_data = temp_data.T.rename(columns={"@CLE": "CLE-GENRE","#text":"description programme"})
    temp_data.head()
    data = data.join(temp_data)
    temp_data = pd.DataFrame()

    names = data.columns.values
    n = list(set(names) - set(irrelevant))
    data = data[n]

    data['DATE'] = data['DATEHEURE'].apply(lambda x: x.split('T')[0])
    data['HEURE'] = data['DATEHEURE'].apply(lambda x: x.split('T')[1])
    data['debut'] = data['HEURE'].apply(lambda x: int(x.split(':')[0])*60 + int(x.split(':')[1]))
    return data




#################################################
########### main with options ###################
#################################################


def main(argv):
    chaine = argv
    c = list(chaine)
    n = 4-len(c)
    c = ['0']*n
    a = "".join(c)
    chaine = a+chaine

    files = os.listdir(PATH+'/pluri_201712/')
    for file in files:
        print(file)
        date = file.split('_')[0]
        f = str(PATH)+'/pluri_201712/'+file+'/IPTV_'+chaine+'_'+str(date)+'_TF1.xml'
        data = processPTV(f)
        data.to_csv(PATH+'extracted/IPTV_0192_'+str(date)+'_TF1.csv')


    return ("process achevé sans erreures")


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1])
