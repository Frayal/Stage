#!/usr/bin/env python -W ignore::DeprecationWarning
#-*- coding: utf-8 -*-
#######################################################################
#created the 16/08/2018 14:48 by Alexis Blanchet#
#################################################
'''
Fichier final qui sera le seul fichier appellé par l'utilisateur.
Work In Progress. Entrée sortie à définir.
Appel à l'utilisateur pour avoir les paramètres.
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
from datetime import timedelta, date
import def_context
import time
from subprocess import Popen
import filelock
import random
#################################################
########### Global variables ####################
#################################################
'''
DataFrame.idxmin(axis=0, skipna=True)
Return index of first occurrence of minimum over requested axis. NA/null values are excluded.
'''
MAX_PROCESSES = 40
PATH_IN = '../Datas/'
PATH_SCRIPT = '../scripts/'
PATH_OUT = '../DatasOut/'
LOG = "log.txt"

#################################################
########### Some Functions ######################
#################################################
def get_temp_path():
    datas = pd.read_csv('path.csv')
    return datas['temp_path'].values[0]

def pred(file,numb_folder):
    try:
        relecture = True
        EPSILON = 1e-15
        f = ((file.split('.'))[0].split('_'))[2]
        c = ((file.split('.'))[0].split('_'))[-1]
        PTV,proba = def_context.load_file(str(f),str(c))
        if(len(PTV) == 0):
            return 0
        index_PTV = PTV.index[(PTV['debut'] <= 3*60) & (PTV['debut']+PTV['DUREE'] > 3*60+5)].tolist()[0]
        def_context.Report('Starting with: %s'%(PTV['TITRE'].iloc[index_PTV]))
        lastend = PTV['debut'].loc[index_PTV]
        currentduree = PTV['DUREE'].loc[index_PTV]
        newPTV = def_context.init_newPTV(PTV,str(c))
        historyofpoints = def_context.init_history(str(c),PTV,lastend,currentduree)
        temp_context = historyofpoints.iloc[0]
        importantpts = def_context.get_important_points(c,PTV,index_PTV)
        for i in range(3):
            def_context.Report(str(i)+' '+str(c)+' '+str(f))
            from predictPTV import main as pred1
            l1,temp_newPTV1,temp_history1,index_PTV1,temp_context1 = pred1([str(c),str(f),i,newPTV.iloc[newPTV.shape[0]-1],temp_context,index_PTV,importantpts,PATH_OUT+'T'+str(numb_folder)+'/'])
            if(l1>0 and relecture):
                def_context.Report("Utilisation de la relecture "+str(i)+' '+str(c)+' '+str(f))
                from RLPTV import main as RL
                l2,temp_newPTV2,temp_history2,index_PTV2,temp_context2 = RL([str(c),str(f),i,newPTV.iloc[newPTV.shape[0]-1],temp_context,index_PTV,importantpts,PATH_OUT+'T'+str(numb_folder)+'/'])
                if(l2>5):
                    def_context.Report("Utilisation de l'arbre de décision",f,c,i)
                    if(chaine == 'TF1'):
                        from PTVTF1 import main as arbre1
                        l3,temp_newPTV3,temp_history3,index_PTV3,temp_context3 = arbre1([str(c),str(f),i,newPTV.loc[newPTV.shape[0]-1],temp_context,index_PTV,importantpts])
                    elif(chaine == 'M6'):
                        from PTVM6 import main as arbre2
                        l3,temp_newPTV3,temp_history3,index_PTV3,temp_context3 = arbre2([str(c),str(f),i,newPTV.loc[newPTV.shape[0]-1],temp_context,index_PTV,importantpts])
                    else:
                        l3>5
                    if(l3>0):
                        def_context.Report("AUCUNE DÉCISION NE CONVIENT",f,c)
                        l,temp_newPTV,temp_history,index_PTV,temp_context = l2,temp_newPTV2,temp_history2,index_PTV2,temp_context2
                    else:
                        l,temp_newPTV,temp_history,index_PTV,temp_context = l3,temp_newPTV3,temp_history3,index_PTV3,temp_context3
                else:
                    l,temp_newPTV,temp_history,index_PTV,temp_context = l2,temp_newPTV2,temp_history2,index_PTV2,temp_context2
            else:
                l,temp_newPTV,temp_history,index_PTV,temp_context = l1,temp_newPTV1,temp_history1,index_PTV1,temp_context1
            if(l == 4):
                pass
            else:
                newPTV = pd.concat([newPTV,temp_newPTV.iloc[1:]])
                historyofpoints = pd.concat([historyofpoints,temp_history])

        newPTV['Heure'] = newPTV['minute'].apply(lambda x: str(int(x/60))+':'+str(x%60))
        historyofpoints['Heure'] = historyofpoints['minute'].apply(lambda x: str(int(x/60))+':'+str(x%60))
        newPTV.to_html(PATH_IN+'new_ptv/new_PTV_'+str(f)+'_'+str(c)+'.html')
        newPTV.to_csv(PATH_IN+'new_ptv/new_PTV_'+str(f)+'_'+str(c)+'.csv',index=False)
        historyofpoints.to_html(PATH_IN+'hop/historyofpoints_'+str(f)+'_'+str(c)+'.html')
        historyofpoints.to_csv(PATH_IN+'hop/historyofpoints_'+str(f)+'_'+str(c)+'.csv',index=False)
        newPTV.to_html(PATH_OUT+'T'+str(numb_folder)+'/new_ptv/new_PTV_'+str(f)+'_'+str(c)+'.html')
        newPTV.to_csv(PATH_OUT+'T'+str(numb_folder)+'/new_ptv/new_PTV_'+str(f)+'_'+str(c)+'.csv',index=False)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        def_context.Report("Failed to process {0} at line {2} in {3}: {1}".format(str(file), str(e),sys.exc_info()[-1].tb_lineno,fname))

def update_temp_path(i):
    datas = pd.read_csv('path.csv')
    datas['temp_path'] = datas['PathtoDatasOut']+'T'+str(i)+"/"
    datas.to_csv('path.csv',index=False)

def create_res_file():
    df = pd.DataFrame()
    df['nombre_fichier'] = [0]
    df.to_csv('res_out.csv',index=False)

def exit_file(date,numero,nom_chaine,num_folder,part):
    '''
    création et remplissage d'un DataFrame pour calculer simplement le coûte
    '''
    ####
    file = PATH_IN+'PTV/IPTV_'+numero+'_'+date+'_'+nom_chaine+'.csv'
    otherfile = PATH_OUT+'T'+str(num_folder)+'/new_ptv/new_PTV_'+date+'_'+nom_chaine+'.csv'
    ####
    try:
        ptv = pd.read_csv(file)
        new_ptv = pd.read_csv(otherfile)
        df = pd.DataFrame()
    except Exception as e:
        def_context.Report('petit problème: '+str(e))
        return [0,0,0,0]

    df['titre'] = ptv['TITRE']
    df['clediff'] = ptv['@CLEDIF']
    df['debut'] = ptv['debut']%1440
    df['duree'] = ptv['DUREE']%1440
    df['fin'] = (ptv['debut']+ptv['DUREE'])%1440
    df['vrai fin'] = 0
    df['ND'] = 0
    df['pourcentage vu'] = 0
    new_ptv_ = new_ptv[new_ptv['Évenement'].apply(lambda x: x.split(' ')[0]) == 'fin']
    current = 0
    for j in range(df.shape[0]):
        for i in range(current,new_ptv_.shape[0]):
            if(new_ptv_['TITRE'].iloc[i] == df['titre'].iloc[j]):
                if(abs(df['fin'].iloc[j] - new_ptv_['minute'].iloc[i])<40 or df[df['titre'] == df['titre'].iloc[j]].shape[0] == 1):
                    df['vrai fin'].iloc[j] = new_ptv_['minute'].iloc[i]
                    df['pourcentage vu'].iloc[j] = new_ptv_['pourcentage vu'].iloc[i]
                    if(new_ptv_['Évenement'].iloc[i] == "fin d'un programme" ):
                        df['ND'].iloc[j] = 0
                    else:
                        df['ND'].iloc[j] = 1
                    current = i
                    break
                else:
                    pass

            else:
                pass

    df['vrai debut'] = (df['vrai fin'] - df['duree']*df['pourcentage vu'])%1440
    df['vrai fin'].iloc[df.shape[0]-1] = df['vrai fin'].iloc[df.shape[0]-2] + df['duree'].iloc[df.shape[0]-1]
    df['chaine'] = nom_chaine
    df['date'] = date
    temp_df = pd.DataFrame()
    temp_df[['titre','vrai debut']] = new_ptv[new_ptv['TITRE'] == 'publicité'][['TITRE','minute']]
    for v in df.columns.values:
        if v not in ['titre','vrai debut']:
            temp_df[v] = 0
    temp_df['chaine'] = nom_chaine
    temp_df['date'] = date
    temp_df['vrai fin'] = temp_df['vrai debut'].apply(lambda x: x+6)
    temp_df['fin'] = temp_df['vrai fin']
    temp_df['debut'] = temp_df['vrai debut']
    df = df.append(temp_df).reset_index(drop=True)
    r = pd.read_csv('res_out.csv')
    month = '-'.join(date.split('-')[:-1])
    if('count_'+nom_chaine+'_'+month+'_'+part in r.columns):
        r['count_'+nom_chaine+'_'+month+'_'+part] = r['count_'+nom_chaine+'_'+month+'_'+part]+1
    else:
        r['count_'+nom_chaine+'_'+month+'_'+part] = 1
    if('count_'+nom_chaine+'_'+month in r.columns):
        r['count_'+nom_chaine+'_'+month] = r['count_'+nom_chaine+'_'+month]+1
    else:
        r['count_'+nom_chaine+'_'+month] = 1
    if('count_'+part in r.columns):
        r['count_'+part] = r['count_'+part]+1
    else:
        r['count_'+part] = 1
    if('count_'+month in r.columns):
        r['count_'+month] = r['count_'+month]+1
    else:
        r['count_'+month] = 1
    if('count_'+nom_chaine in r.columns):
        r['count_'+nom_chaine] = r['count_'+nom_chaine]+1
    else:
        r['count_'+nom_chaine] = 1
    for index,x in new_ptv[['Évenement','minute']].iterrows():
        if 'HARD RESET OF ALGORITHM' in x['Évenement']:
            if(x['minute']<=13*60+40 and part == 'matinee' and x['minute']>180):
                if(nom_chaine+'_'+month+'_'+part in r.columns):
                    r[nom_chaine+'_'+month+'_'+part] = r[nom_chaine+'_'+month+'_'+part]+1
                else:
                    r[nom_chaine+'_'+month+'_'+part] = 1
                if(nom_chaine+'_'+month in r.columns):
                    r[nom_chaine+'_'+month] = r[nom_chaine+'_'+month]+1
                else:
                    r[nom_chaine+'_'+month] = 1
                if(part in r.columns):
                    r[part] = r[part]+1
                else:
                    r[part] = 1
                if(month in r.columns):
                    r[month] = r[month]+1
                else:
                    r[month] = 1
                if(nom_chaine in r.columns):
                    r[nom_chaine] = r[nom_chaine]+1
                else:
                    r[nom_chaine] = 1
                def_context.Report('message: %s à la minute %s de la journée %s pour la chaîne %s pour la partie %s. Recherche dans le folder %s ' %(x['Évenement'],x['minute'],date,nom_chaine,part,num_folder))

            elif(x['minute']<=20*60+35 and part == 'apresmidi' and x['minute']>13*60+40):
                if(nom_chaine+'_'+month+'_'+part in r.columns):
                    r[nom_chaine+'_'+month+'_'+part] = r[nom_chaine+'_'+month+'_'+part]+1
                else:
                    r[nom_chaine+'_'+month+'_'+part] = 1
                if(nom_chaine+'_'+month in r.columns):
                    r[nom_chaine+'_'+month] = r[nom_chaine+'_'+month]+1
                else:
                    r[nom_chaine+'_'+month] = 1
                if(part in r.columns):
                    r[part] = r[part]+1
                else:
                    r[part] = 1
                if(month in r.columns):
                    r[month] = r[month]+1
                else:
                    r[month] = 1
                if(nom_chaine in r.columns):
                    r[nom_chaine] = r[nom_chaine]+1
                else:
                    r[nom_chaine] = 1
                def_context.Report('message: %s à la minute %s de la journée %s pour la chaîne %s pour la partie %s. Recherche dans le folder %s ' %(x['Évenement'],x['minute'],date,nom_chaine,part,num_folder))
            elif(part == 'soiree' and (x['minute']>20*60+35 or x['minute']<180) ):
                if(nom_chaine+'_'+month+'_'+part in r.columns):
                    r[nom_chaine+'_'+month+'_'+part] = r[nom_chaine+'_'+month+'_'+part]+1
                else:
                    r[nom_chaine+'_'+month+'_'+part] = 1
                if(nom_chaine+'_'+month in r.columns):
                    r[nom_chaine+'_'+month] = r[nom_chaine+'_'+month]+1
                else:
                    r[nom_chaine+'_'+month] = 1
                if(part in r.columns):
                    r[part] = r[part]+1
                else:
                    r[part] = 1
                if(month in r.columns):
                    r[month] = r[month]+1
                else:
                    r[month] = 1
                if(nom_chaine in r.columns):
                    r[nom_chaine] = r[nom_chaine]+1
                else:
                    r[nom_chaine] = 1
                def_context.Report('message: %s à la minute %s de la journée %s pour la chaîne %s pour la partie %s. Recherche dans le folder %s ' %(x['Évenement'],x['minute'],date,nom_chaine,part,num_folder))
    r['nombre_fichier'] = r['nombre_fichier']+1
    if str(num_folder) in r:
        r[str(num_folder)]+=1
    else:
        r[str(num_folder)] = 1
    r.to_csv('res_out.csv',index=False)

    if(part == 'matinee'):
        return df[(df['fin']<=13*60+40) & (df['fin']>180)]
    elif(part == 'apresmidi'):
        return df[(df['fin']<=20*60+35) & (df['fin']>13*60+40)]
    elif(part == 'soiree'):
        return df[df['fin']>20*60+35].append(df[df['fin']<=180])





def find_best(col):
    l = np.bincount(col)
    col = col.apply(lambda x: x*(0.95**l[x]) if x<1 else x)
    if(col.idxmin()>=16):
        return 0
    else:
        return col.idxmin()
#################################################
########### Main callable #######################
#################################################
def main(argv):
    start = 10
    if(len(argv) == 0):
        print('bonjour')
        start = int(input("A quelle partie voulez vous commencer?"))
        if(start<2):
            chaines = str(input("Quelle Chaînes devont nous traiter?(separez les par un '-'"))
            chaines = chaines.split('-')
            C = [[def_context.get_tuple(chaine)] for chaine in chaines]
        Processes = []
    if(len(argv) == 2):
        pred(argv[0],argv[1])

    ##### Première partie #####
    if(start<=1):
        for chaine in chaines:
            while(len(Processes)>= MAX_PROCESSES):
                        time.sleep(5)
                        for p in range(len(Processes)): # Check the processes in reverse order
                            if Processes[len(Processes)-1-p].poll() is not None: # If the process hasn't finished will return None
                                del Processes[len(Processes)-1-p] # Remove from list - this is why we needed reverse order

            Processes.append(Popen(['python','cleaningRTSfiles.py','0',chaine,'0']))
        for chaine in chaines:
            while(len(Processes)>= MAX_PROCESSES):
                        time.sleep(5)
                        for p in range(len(Processes)): # Check the processes in reverse order
                            if Processes[len(Processes)-1-p].poll() is not None: # If the process hasn't finished will return None
                                del Processes[len(Processes)-1-p] # Remove from list - this is why we needed reverse order

            Processes.append(Popen(['python','extractdatafromPTV.py',chaine]))
        ##### emptying the process queue ######
        while(len(Processes)):
            time.sleep(5)
            for p in range(len(Processes)): # Check the processes in reverse order
                if Processes[len(Processes)-1-p].poll() is not None: # If the process hasn't finished will return None
                    del Processes[len(Processes)-1-p] # Remove from list - this is why we needed reverse order
        Processes.append(Popen(['python','processingdata.py']))

        while(len(Processes)):
            time.sleep(5)
            for p in range(len(Processes)): # Check the processes in reverse order
                if Processes[len(Processes)-1-p].poll() is not None: # If the process hasn't finished will return None
                    del Processes[len(Processes)-1-p] # Remove from list - this is why we needed reverse order
        Processes.append(Popen(['python','predict.py']))
        while(len(Processes)):
            time.sleep(5)
            for p in range(len(Processes)): # Check the processes in reverse order
                if Processes[len(Processes)-1-p].poll() is not None: # If the process hasn't finished will return None
                    del Processes[len(Processes)-1-p] # Remove from list - this is why we needed reverse order
    if (start <= 2):
        Processes = []
        pass_files = []
        for i in range(1,31):
            update_temp_path(i)
            files = os.listdir(PATH_IN+'PTV/')
            nb_files = len(files)
            for file in files:
                if(file in pass_files):
                    pass
                elif(i<5):
                    while (len(Processes)>= MAX_PROCESSES):
                        for p in range(len(Processes)):  # Check the processes in reverse order
                            if Processes[len(Processes)-1-p].poll() is not None:  # If the process hasn't finished will return None
                                del Processes[len(Processes)-1-p]  # Remove from list - this is why we needed reverse order
                        time.sleep(5)
                        def_context.Report('process launch for %s at turn %s'%(file,i))
                    Processes.append(Popen(['python', 'out.py', file,str(i)]))
                else:
                    date = file.split('_')[2]
                    chaine = file.split('_')[-1].split('.')[0]
                    numero,nom_chaine = def_context.get_tuple(chaine)
                    os.system('python cost.py '+str(chaine)+' '+str(date))
                    couts = pd.read_csv('cout.csv')
                    l = np.bincount(couts[date+'_'+nom_chaine+'_tout'])
                    if(min(couts[date+'_'+nom_chaine+'_tout'])<1 and max(l)>=4):
                        pass_files.append(file)
                    else:
                        while(len(Processes)>= MAX_PROCESSES):
                                for p in range(len(Processes)): # Check the processes in reverse order
                                    if Processes[len(Processes)-1-p].poll() is not None: # If the process hasn't finished will return None
                                        del Processes[len(Processes)-1-p] # Remove from list - this is why we needed reverse order
                                time.sleep(5)
                        Processes.append(Popen(['python','out.py',file,str(i)]))
######### Toute les prédictions on été faites ########
    if(start<=3):
        os.system("python cost.py")
        time.sleep(10)
######################################################
    if(start <=4):
        create_res_file()
        df = pd.read_csv('cout.csv').apply(lambda x: x.apply(lambda c: int(c)))
        for col in df.columns.values:
            if('tout' not in col):
                pass
            else:
                df_final=[]
                index_of_best = []
                col = ''.join(list(col)[:-4])
                for mm in ['matinee','apresmidi','soiree']:
                    i = find_best(df[col+mm])
                    a,b = def_context.get_tuple(col.split('_')[1])
                    df_final.append(exit_file(col.split('_')[0],a,b,i,mm))
                    def_context.Report('Best Prediction for %s %s %s occured at %s'%(col.split('_')[1],col.split('_')[0],mm,str(i)))
                #def_context.Report(str(df_final[0].shape)+' '+str(df_final[1].shape)+' '+str(df_final[2].shape))
                df_final = df_final[0].append(df_final[1].append(df_final[2]))
                try:
                    df_final.to_csv('../DatasOut/out/new_PTV_'+col.split('_')[0]+'_'+b+'.csv',index=False)
                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    def_context.Report("Failed to process {0} at line {2} in {3}: {1}".format(str(argv), str(e),sys.exc_info()[-1].tb_lineno,fname))
    if(start<=5):
        res = pd.read_csv('res_out.csv')
        res.loc[1] = res.loc[0]*0
        for col in res:
            if('count' in col):
                pass
            elif(col == 'nombre_fichier'):
                pass
            elif(col in [str(i) for i in range(31)]):
                pass
            else:
                res[col].loc[1] = 1-(res[col].loc[0])/res['count_'+col].loc[0]
                def_context.Report('pour %s : %s erreurs soit %s '%(col,res[col][0],res[col][1]))
        res.to_csv('res_out.csv',index=False)


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
