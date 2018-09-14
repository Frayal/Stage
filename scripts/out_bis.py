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
MAX_PROCESSES = 5
PATH_IN = '../Datas/'
PATH_SCRIPT = '../scripts/'
PATH_OUT = '../DatasOut/'
LOG = "log.txt"

#################################################
########### Some Functions ######################
#################################################
def main(argv):
    try:
        relecture = True
        file = argv[0]
        f = ((file.split('.'))[0].split('_'))[2]
        c = ((file.split('.'))[0].split('_'))[-1]
        if(f=='2017-12-20' or (f in ['2017-12-09','2017-12-06','2018-02-22'] and c=='TF1') or (f in ['2018-02-22'] and c=='M6') or (f.split('-')[0] != str(argv[0].split('-')[0])) or f.split('-')[1] != argv[0].split('-')[1]):
        #or (f in ['2018-02-22'] and c=='M6')
            pass
        elif(c ==''):
            pass
        else:
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
                from predictPTV import main as pred
                l,temp_newPTV,temp_history,index_PTV,temp_context = pred([str(c),str(f),i,newPTV.iloc[newPTV.shape[0]-1],temp_context,index_PTV,importantpts])
                if(l>0 and relecture):
                    def_context.Report("Utilisation de la relecture "+str(i)+' '+str(c)+' '+str(f))
                    from RLPTV import main as RL
                    l,temp_newPTV,temp_history,index_PTV,temp_context = RL([str(c),str(f),i,newPTV.iloc[newPTV.shape[0]-1],temp_context,index_PTV,importantpts])
                if(l == 4):
                    pass
                else:
                    newPTV = pd.concat([newPTV,temp_newPTV.iloc[1:]])
                    historyofpoints = pd.concat([historyofpoints,temp_history])

            newPTV['Heure'] = newPTV['minute'].apply(lambda x: str(int(x/60))+':'+str(x%60))
            historyofpoints['Heure'] = historyofpoints['minute'].apply(lambda x: str(int(x/60))+':'+str(x%60))
            newPTV.to_html(PATH_OUT+'T'+str(argv[1])+'/'+'new_ptv/new_PTV_'+str(f)+'_'+str(c)+'.html')
            newPTV.to_csv(PATH_OUT+'T'+str(argv[1])+'/'+'new_ptv/new_PTV_'+str(f)+'_'+str(c)+'.csv')
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        def_context.Report("Failed to process {0} at line {2} in {3}: {1}".format(str(file), str(e),sys.exc_info()[-1].tb_lineno,fname))


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])