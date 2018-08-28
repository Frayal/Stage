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
def get_temp_path():
    datas = pd.read_csv('path.csv')
    return datas['temp_path'].values[0]



def update_temp_path(i):
    datas = pd.read_csv('path.csv')
    datas['temp_path'] = datas['PathtoDatasOut']+'T'+str(i)+"/"
    datas.to_csv('path.csv',index=False)


def exit_file(date,numero,nom_chaine,i):
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
	df['vrai fin'] = df['fin']
	df['vrai debut'] = df['debut']
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
	df['chaine'] = nom_chaine
	df['date'] = date

	temp_df = pd.DataFrame()
	temp_df[['titre','vrai debut']] = df[df['TITRE'] == 'publicité']['TITRE','minute']
	for v in df.columns.values:
		if v not in ['titre','debut']:
			temp_df[v] = 0
	temp_df['chaine'] = nom_chaine
	temp_df['date'] = date
	temp_df['vrai fin'] = temp_df['vrai début'].apply(lambda x: x+6)
	df.append(temp_df)
	df.to_csv('../DatasOut/out/new_PTV_'+date+'_'+nom_chaine+'.csv',index=False) 




def find_best(col):
	l = np.bincount(col)
	col = col.apply(lambda x: x/l[x])
	return col.idxmin()
#################################################
########### Main callable #######################
#################################################
def main(argv):
	print('bonjour')
	chaines = str(imput("Quelle Chaînes devont nous traiter?(separez les par un '-'"))
	chaines = chaînes.split('-')
	C = [[def_context.get_tuple(chaine)] for chaine in chaines]
	Processes = []
	##### Première partie #####
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
	##### Second Part #####
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
	####### Debut Troisème Partie #####
	for i in range(30):
		update_temp_path(i):
		files = os.listdir(PATH_IN+'PTV/')
        nb_files = len(files)
        for file in files:
        	while(len(Processes)>= MAX_PROCESSES):
					time.sleep(5)
					for p in range(len(Processes)): # Check the processes in reverse order
						if Processes[len(Processes)-1-p].poll() is not None: # If the process hasn't finished will return None
							del Processes[len(Processes)-1-p] # Remove from list - this is why we needed reverse order
				
        	Processes.append(Popen(['python','out_bis.py',file,'T'+str(i)+'/']))
        while(len(Processes)):
		time.sleep(5)
		for p in range(len(Processes)): # Check the processes in reverse order
			if Processes[len(Processes)-1-p].poll() is not None: # If the process hasn't finished will return None
				del Processes[len(Processes)-1-p] # Remove from list - this is why we needed reverse order
	######### Toute les prédictions on été faites ########
	os.system("python cost.py")
	time.sleep(10)
    ######################################################
 	df = pd.read_csv('cout.csv').apply(lambda x: x.apply(lambda c: int(c)))
 	for col in df.columns.values:
 		if('tout' not in col):
 			pass
 		else:
 			i = find_best(df[col])
 			value = df[col][i]
			a,b = def_context.get_tuple(col.split('_')[1])
 			exit_file(col.split('_')[1],a,b,i)
 			def_context.Report('Best Prediction for %s %s occured at %s with an error of %s',%(col.split('_')[1],col.split('_')[0],str(i),str(value)))






if __name__ == "__main__":
	# execute only if run as a script
	main(sys.argv[1:])