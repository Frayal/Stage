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
    OnlinePTV['pourcentage vu'] = 0
    OnlinePTV['Évenement'] = 'Début de Détection'

def init_history():
    h = pd.DataFrame()
    h['minute'] = [179]
    h['partie de la journée'] = 'nuit'
    h['Change Point'] = 0
    h['pourcentage'] = 0
    h['partie du programme'] = 0
    h['programme'] = "programme de nuit"
    h['duree'] = 0
    h['nombre de pub potentiel'] =  0

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

def categorize_duree(duree):
    if(duree<10):
        return("très court")
    elif(10<=duree<30):
        return("court")
    elif(30<=duree<60):
        return("moyen")
    elif(60<=duree<=100):
        return("long")
    else:
        return("très long")

def categorize_type(description):
    mots = description.split(" ")
    if(mots[0] == "Série"):
        return "Série"
    if(mots[0] == "Téléfilm"):
        return "film"
    if(description == 'Magazine jeunesse'):
        return 'dessins animés'
    if(mots[0] == "Magazine"):
        return "magazine"
    if(mots[0]=="Feuilleton"):
        return "Feuilleton"
    else:
        return description

def categorize_pub(name,debut):
    if(name in["Météo,Journal,Magazine"]):
        return 0
    if(name == 'dessins animés'):
        return 3
    if(name in ['Jeu']):
        return 1
    if(name in ['Feuilleton','film','Téléréalité']):
        return 2
    if(name == 'Série' and 180<debut<12*60):
        return 1
    if(name == 'Série'):
        return 2
    else:
        return 10


def categorize_programme(programme):
    p = []
    p.append(categorize_type(programme['description programme']))
    p.append(categorize_duree(programme['DUREE']))
    p.append(categorize_pub(p[0]),programme['debut'])






def get_context(i,programme,Points,index_CP,lastCP,lastPub,lastend,lastduree,planifiedend):
    #we create a list with different notes to understand the context
    # minute of the point and its situation in the day
    context = [i]
    context.append(find_partofday(i))
    # Is the Point a Change Point
    context.append(find_ifChangePoint(i,Points['minutes'].iloc[index_CP]))
    # Where is the Point in the programme:
    seen_percentage = (i-lastend)/lastduree
    context.append(seen_percentage)
    context.append(find_position(seen_percentage))
    # which type of programme we are watching
    p = categorize_programme(programme)
    for i in range(len(p)):
        context.append(p[i])

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
        #Update time of commercials (Reset)
        if(i%60 == 0):
            Pubinhour = 0
        #Update timmers
        lastPub+=1
        lastCP+=1
        if(index_CP==Points.shape[0]):
            index_CP -=1
        #let's get the context:
        context = get_context(i,PTV.iloc[index_PTV],Points,index_CP,lastCP,lastPub,lastend,lastduree,planifiedend)
        historyofpoints.iloc[historyofpoints.shape[0]] = context
        if(historyofpoints['Change Point'].iloc[(historyofpoints.shape[0]-1)]):
            #Change Point ==> Decide what to do with it



        else:
            #Not a Change Point, we'll just check that nothing is wrong in the PTV at this time
            if(Predictiontimer == 0):
                OnlinePTV.loc[OnlinePTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',percentage,"fin d'un programme court"]
                lastend = i
                lastCP=0
                index_PTV += 1
                index_PTV = index_PTV%(PTV.shape[0])
                lastduree = PTV['DUREE'].iloc[index_PTV]
                planifiedend = (lastend + lastduree)
            if(historyofpoints['pourcentage'].iloc[(historyofpoints.shape[0]-1)] == 1):
                elif(60<=duree<=100):
                    return("long")
                else:
                    return("très long")
                if(context[-2] == "très court")
                    Predictiontimer = 2
                if(context[-2] == "court")
                    Predictiontimer = 6
                if(context[-2] == "moyen")
                    Predictiontimer = 12
                else:
                    Predictiontimer = 15
            if(historyofpoints['pourcentage'].iloc[(historyofpoints.shape[0]-1)]>1):
                Predictiontimer -= 1
























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
