#################################################
#created the 30/05/2018 11:56 by Alexis Blanchet#
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
import pandas as pd
import numpy as np

#################################################
########### Global variables ####################
#################################################
date = "2018-05-23"

#################################################
########### Important functions #################
#################################################
def load_file(date):
    PTV = pd.read_csv('IPTV_'+date+'_TF1.csv')[['TITRE','DUREE','description programme','HEURE','debut']]
    PTV['fin'] = PTV['debut']+PTV['DUREE']
    Points = pd.read_csv('merged_'+date+'.csv')[['minutes']]
    return PTV,Points

def init_newPTV(PTV):
    begin = (pd.DataFrame(PTV.loc[(PTV.shape[0]-1)]).T).reset_index().drop(['index'],axis=1)
    #Initialisation du NewPTV
    OnlinePTV = pd.DataFrame()
    OnlinePTV['minute'] = [180]
    OnlinePTV['TITRE'] = 'Programmes de la nuit'
    OnlinePTV['Change Point'] = 'Non'
    OnlinePTV['pourcentage de la durée'] = 0
    OnlinePTV['Évenement'] = 'Début de Détection'

def init_history():
    h = pd.DataFrame()
    h['minute'] = [180]
    h['partie de la journée'] = 'nuit'
    h['Change Point'] = 0
    h['pourcentage'] = 0
    h['partie du programme'] = 0
    h['programme'] = "programme de nuit"

    h['']

def find_position(seen_percentage):
    if(seen_percentage<=0.25):
        return "début de programme"
    if(0.25<seen_percentage<=0.75):
        return "milieu de programme"
    if(0.75<seen_percentage<=0.95):
        return "début de fin de programme"
    if(0.95<seen_percentage<=1.05):
        return "fin de programme"
    if(seen_percentage>1.05):
        return "en dehors du programme"

def find_partofday(i):
    part = i/60
    if(3<=part<=7):
        return("fin de nuit")
    elif(7<part<=9):
        return("début de matinée")
    elif(9<part<=12):
        return("matinée")
    elif(12<part<=14):
        return("midi")
    elif(14<part<=17):
        return("après-midi")
    elif(17<part<=20):
        return("fin d'après-midi")
    elif(20<part<=24):
        return("prime time")
    else:
        return('nuit')

def find_ifChangePoint(i,cp):
    if(i==cp):
        return 1
    else:
        return 0





def get_context(i,programme,Points,index_CP,lastCP,lastPub,lastend,lastduree,planifiedend):
    #we create a list with different notes to understand the context
    # minute of the point and its situation in the day
    context = [i]
    context.append(find_partofday(i))
    # Is the Point a Change Point
    context.append(i,Points['minutes'].iloc[index_CP])
    # Where is the Point in the programme:
    seen_percentage = (i-lastend)/lastduree
    context.append(seen_percentage)
    context.append(find_position(seen_percentage))
    # which type of programme we are watching
    context.append(programme['description programme'].split(' ')[0])

    return context




def make_modification():
    #TODO: réécrire le programme quand on sait a l'avance certaines chose (début des journaux par exemple)


def make_newPTV(PTV,Points):
    #Initialisation des Variables
    verbose = False
    index_CP = 0
    index_PTV = PTV.shape[0]-1
    ##########################
    Predictiontimer = 200
    Pubinhour = 0
    lastCP = 200
    lastPub= 0
    lastend = 130
    lastduree = 255
    planifiedend = 385
    begin = True
    ######################
    newPTV = init_newPTV(PTV)
    historyofpoints = init_history()
    ######################
    for i in range(180,1620):
        #Update time of commercials
        if(i%60 == 0):
            Pubinhour = 0
        #Update timmers
        lastPub+=1
        lastCP+=1
        #Update Prediction timer
        if(Predictiontimer<30):
            Predictiontimer-=1
        if(index_CP==Points.shape[0]):
            index_CP -=1
        #let's get the context:
        context = get_context(i,PTV.iloc[index_PTV],Points,index_CP,lastCP,lastPub,lastend,lastduree,planifiedend)



















    return newPTV





#################################################
########### main with options ###################
#################################################


def main(argv):
    date = argv[0]
    PTV,Points = load_file(date)
    new_PTV = make_newPTV(PTV,Points)
    return ("process achevé sans erreures")


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
