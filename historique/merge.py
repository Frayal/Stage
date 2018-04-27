#################################################
#created the 27/04/2018 15:32 by Alexis Blanchet#
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

#################################################
########### Global variables ####################
#################################################

IRRELEVANT = ['@DATEMODIF','@CLEDIF','DATEHEURE','RATIO','HEURE','@CLEEMI']

#################################################
########### Important functions #################
#################################################


def updateargs(CHAINE,DATE):
    JOINDATE = "".join(list(DATE.split('-')))
    c = list(CHAINE)
    n = 4-len(c)
    c = ['0']*n
    a = "".join(c)
    CHAINE2 = a+CHAINE
    return(JOINDATE,CHAINE2)

def dfjoin(df,tdf):
    for index, row in df.iterrows():
    for index2, row2 in tdf.iterrows():
        if(int(row2['debut']) < int(row['minutes']%1440) < int(row2['fin'])):
            df.set_value(index, 'isinprogramme', 1)
            df.set_value(index, 'fin', row2['fin'])
            df.set_value(index, 'debut', int(row2['debut']))
            df.ix[index,'TITRE']=row2['titre']

        if(int(row2['debut']) < int(row['minutes']) < int(row2['fin'])):
            df.set_value(index, 'isinprogramme', 1)
            df.set_value(index, 'fin', row2['fin'])
            df.set_value(index, 'debut', int(row2['debut']))
            df.ix[index,'TITRE']=row2['titre']

        else:
            pass
    df[df['isinprogramme']==0]['titre'] = 'en dehors de programmes'
    return df




def get_features_from_join(df):
    df['temps depuis debut'] = 0
    df['temps avant fin'] = 0
    df['pourcentage déjà vu'] = 0

    df['temps depuis debut'] = df[df['isinprogramme'] == 1].apply(lambda row: min(abs(row['minutes'] - row['debut']),abs(row['minutes']%1440 - row['debut'])),axis = 1)
    df['temps avant fin'] = df[df['isinprogramme'] == 1].apply(lambda row: min(abs(row['fin'] - row['minutes']),abs(row['fin'] - row['minutes']%1440)),axis = 1)
    df = df.fillna(1)
    df['pourcentage déjà vu'] = df['temps depuis debut']/df['DUREE']
    return df





def processing(X_RTS,X_PTV):
    X_RTS['minutes'] = X_RTS['minutes']+180
    # Creating temp DataFrame to make the join possible
    tdf = pd.DataFrame()
    tdf['debut'] = X_PTV['debut']
    tdf['fin'] = tdf['debut']+X_PTV['DUREE']
    tdf['titre'] = X_PTV['TITRE']
    # creating the final dataframe and fill it with values from both Dataframe
    df = pd.DataFrame()
    df['minutes'] = (X_RTS['minutes'])
    df['isinprogramme'] = 0
    df['debut'] = 0
    df['fin'] = 0
    df = dfjoin(df,tdf)
    df = pd.merge(df, X_RTS, on='minutes')
    df = pd.merge(df, X_PTV, on=['debut','TITRE'], how='left')
    df = df.drop(IRRELEVANT,axis=1)
    df = df = df.fillna(-1)
    df = get_features_from_join(df)

#################################################
########### main with options ###################
#################################################


def main(argv):
    CHAINE = argv[0]
    DATE = argv[1]
    JOINDATE,CHAINE2 = updateargs(CHAINE,DATE)

    X_PTV = pd.read_csv('PTV/IPTV_'+CHAINE2+'_'+DATE+'_TF1.csv',index_col=False).drop(['Unnamed: 0'],axis=1)
    X_RTS = pd.read_csv('RTS/sfrdaily_'+JOINDATE+'_0_'+CHAINE+'_0_cleandata-processed.csv',index_col=False)

    df = processing(X_RTS,X_PTV)
    df.to_csv('joined data/'+CHAINE+'_'+DATE+'.csv',index = False)


    return ("process achevé sans erreures")


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
