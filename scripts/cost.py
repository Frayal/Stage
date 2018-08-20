#######################################################################
#created the 16/08/2018 14:48 by Alexis Blanchet#
#################################################
#-*- coding: utf-8 -*-
'''
Calcul de la fonction de coût sur l'ensemble des propositions de programmes TV
afin de renvoyer le plus probable. N'est appelé qu'après calcul de toutes les
possibilités.
/!\ WIP /!\ A NE PAS UTILISER. A MODIFIER. A TESTER.
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
import pandas as pd
import numpy as np
import datetime
import def_context
import time
#################################################
########### Global variables ####################
#################################################
def find_cost(date,numero,nom_chaine,i):
    '''
    création et remplissage d'un DataFrame pour calculer simplement le coûte
    '''
    ####
    file = PATH_IN+'PTV/IPTV_'+numero+'_'+date+'_'+nom_chaine+'.csv'
    otherfile = PATH_OUT+'T'+str(i)+'/new_ptv/new_PTV_'+date+'_'+nom_chaine+'.csv'
    ####
    try:
        ptv = pd.read_csv(file)
        new_ptv = pd.read_csv(otherfile)
        df = pd.DataFrame()
    except Exception as e:
        def_context.Report('petit problème: '+str(e))
        return [0,0,0,0]

    df['titre'] = ptv['TITRE']
    df['debut'] = ptv['debut']%1440
    df['duree'] = ptv['DUREE']%1440
    df['fin'] = (ptv['debut']+ptv['DUREE'])%1440
    df['vrai fin'] = 0
    df['ND'] = 0
    df['pourcentage vu'] = 0
    j = 0
    for i in range(new_ptv.shape[0]):
        if(new_ptv['TITRE'].iloc[i] == df['titre'].iloc[j]):
            df['vrai fin'].iloc[j] = new_ptv['minute'].iloc[j]
            df['pourcentage vu'].iloc[j] = new_ptv['pourcentage vu'].iloc[j]
            if(new_ptv['Évenement'].iloc[j] == "fin d'un programme" ):
                df['ND'].iloc[j] = 0
            else:
                df['ND'].iloc[j] = 1
            j +=1
            j = j%df.shape[0]
        else:
            pass
    df['vrai debut'] = (df['vrai fin'] - df['duree']*df['pourcentage vu'])%1440
    df = df[df['pourcentage vu']>0]
    df['cout'] = 0
    for index, row in df.iterrows():
        df['cout'].iloc[index-1] = (min(abs(row['debut']-row['vrai debut'])%1440,abs(row['debut']%1440-row['vrai debut']%1440)) + min(abs(row['fin']-row['vrai fin'])%1440,abs(row['fin']%1440-row['vrai fin']%1440))) + abs(1-row['pourcentage vu'])*row['duree']+50*row['ND']
    cout = np.sum(df['cout'])
    cout_matin = np.sum(df[df['fin']<13*60]['cout'])
    cout_aprem = np.sum(df[(df['fin']<21*60) & (df['fin']>13*60)]['cout'])
    cout_soir = np.sum(df[df['fin']>21*60]['cout'])

    f = 0
    for x in new_ptv['Évenement'].values:
        if 'HARD RESET OF ALGORITHM' in x:
            f += 1
    df.to_csv('onlyfortest.csv',index=False)
    return([cout + f*400,cout_matin,cout_aprem,cout_soir])


#################################################
########### Important functions #################
#################################################
def main(argv):
    global PATH_IN,PATH_SCRIPT,PATH_OUT
    PATH_IN,PATH_SCRIPT,PATH_OUT = def_context.get_path()
    if(len(argv) == 2):
        chaine = argv[0]
        date = argv[1]
        numero,nom_chaine = def_context.get_tuple(chaine)

        try:
            couts = pd.read_csv('cout.csv')
        except Exception as e:
            couts = pd.DataFrame()
        res = []
        files = os.listdir(PATH_OUT)
        for i in range(len(files)):
            res.append(find_cost(date,numero,nom_chaine,i))
        def_context.Report(res)
        try:
            couts[date+'_'+nom_chaine+'_tout'] = [i[0] for i in res]
            couts[date+'_'+nom_chaine+'_matinee'] = [i[1] for i in res]
            couts[date+'_'+nom_chaine+'_apresmidi'] =[i[2] for i in res]
            couts[date+'_'+nom_chaine+'_soiree'] = [i[3] for i in res]
        except Exception as e:
            def_context.Report('humm: '+str(e))
        couts.to_csv('cout.csv',index=False)
    else:
        files = os.listdir(PATH_IN+'PTV/')
        for file in files:
            time.sleep(1)
            date = file.split('_')[2]
            chaine = file.split('_')[-1].split('.')[0]
            c = os.system('python cost.py '+chaine+' '+date)
            def_context.Report('calcul des coûts pour la journée du %s sur la chaîne %s'%(date,chaine))


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
