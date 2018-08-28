#!/usr/bin/env python -W ignore::DeprecationWarning
#-*- coding: utf-8 -*-
#######################################################################
#created the 16/08/2018 14:48 by Alexis Blanchet#
#################################################
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
from subprocess import Popen
import filelock
#################################################
########### Global variables ####################
#################################################
MAX_PROCESSES = 5
LOCK = filelock.FileLock(file_name = 'cout.csv')
EXTENDED = True
#################################################
########## Important functions ##################
#################################################
def coef(x):
	if(180<x<=9*60+30):
		return 0.25
	if(9*60+30<x<=11*60):
		return 0.5
	if(11*60<x<=13*60):
		return 0.5
	if(13*60<x<=14*60):
		return 0.5
	if(13*60<x<=19*60):
		return 0.5
	if(19*60<x<=21*60):
		return 0.5
	if(21*60<x<=23*60):
		return 0.5
	else:
		return 0.25



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
	df['coef'] = df['fin'].apply(lambda x:coef(x))
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
		df['cout'].iloc[index-1] = (min(abs(row['debut']-row['vrai debut'])%1440,abs(row['debut']%1440-row['vrai debut']%1440)) + min(abs(row['fin']-row['vrai fin'])%1440,abs(row['fin']%1440-row['vrai fin']%1440)))*row['coef']
	cout = np.sum(df['cout'])
	cout_matin = np.sum(df[(df['fin']<=13*60+40) & (df['fin']>180)]['cout'])
	cout_aprem = np.sum(df[(df['fin']<=20*60+35) & (df['fin']>13*60+40)]['cout'])
	cout_soir = np.sum(df[df['fin']>20*60+35]['cout'])+np.sum(df[df['fin']<=180]['cout'])

	f = 0
	for index,x in new_ptv[['Évenement','minute']].iterrows():
		if 'HARD RESET OF ALGORITHM' in x['Évenement']:
			f += 1
			if(180<x['minute']<=13*60+15):
				cout_matin += 10000
			elif(x['minute']<20*60+35):
				cout_aprem += 10000
			else:
				cout_soir += 10000

	df.to_html('test/'+date+'_'+nom_chaine+'_'+str(i)+'.html')
	return([cout + f*10000,cout_matin,cout_aprem,cout_soir])


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
		res = []
		files_ = os.listdir(PATH_OUT)
		for i in range(len(files_)):
			res.append(find_cost(date,numero,nom_chaine,i))
		def_context.Report(res)
		try:
			LOCK.acquire()
			couts = pd.read_csv('cout.csv')
		except Exception as e:
			couts = pd.DataFrame()
		try:
			couts[date+'_'+nom_chaine+'_tout'] = [i[0] for i in res]
			couts[date+'_'+nom_chaine+'_matinee'] = [i[1] for i in res]
			couts[date+'_'+nom_chaine+'_apresmidi'] =[i[2] for i in res]
			couts[date+'_'+nom_chaine+'_soiree'] = [i[3] for i in res]
		except Exception as e:
			def_context.Report('humm: '+str(e))
		couts.to_csv('cout.csv',index=False)
		LOCK.release()
	elif(len(argv) == 1):
		Processes = []
		files = os.listdir(PATH_IN+'PTV/')
		for file in files:
			print(len(Processes))
			date = file.split('_')[2]
			chaine = file.split('_')[-1].split('.')[0]
			print(type(date),type(chaine),date,chaine)
			if( str(argv[0]) in [str(date),str(chaine)]):
				print(argv[0])
				while(len(Processes)>= MAX_PROCESSES):
					time.sleep(5)
					for p in range(len(Processes)): # Check the processes in reverse order
						if Processes[len(Processes)-1-p].poll() is not None: # If the process hasn't finished will return None
							del Processes[len(Processes)-1-p] # Remove from list - this is why we needed reverse order
				if(len(Processes)<MAX_PROCESSES):
					Processes.append(Popen(['python','cost.py',str(chaine),str(date)]))
				else:
					Processes[0].wait()
					Processes.pop(0)
					Processes.append(Popen(['python','cost.py',str(chaine),str(date)]))
				def_context.Report('calcul des coûts pour la journée du %s sur la chaîne %s'%(date,chaine))
				time.sleep(2)
			else:
				continue
	else:
		Processes = []
		files = os.listdir(PATH_IN+'PTV/')
		for file in files:

			print(len(Processes))
			date = file.split('_')[2]
			chaine = file.split('_')[-1].split('.')[0]
			while(len(Processes)>=5):
				time.sleep(5)
				for p in range(len(Processes)): # Check the processes in reverse order
					if Processes[len(Processes)-1-p].poll() is not None: # If the process hasn't finished will return None
						del Processes[len(Processes)-1-p] # Remove from list - this is why we needed reverse order
			if(len(Processes)<MAX_PROCESSES):
				Processes.append(Popen(['python','cost.py',str(chaine),str(date)]))
			else:
				Processes[0].wait()
				Processes.pop(0)
				Processes.append(Popen(['python','cost.py',str(chaine),str(date)]))
			def_context.Report('calcul des coûts pour la journée du %s sur la chaîne %s'%(date,chaine))
			time.sleep(2)

if __name__ == "__main__":
	# execute only if run as a script
	main(sys.argv[1:])
