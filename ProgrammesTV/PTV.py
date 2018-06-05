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
import os
import pandas as pd
import numpy as np

#################################################
########### Global variables ####################
#################################################


#################################################
########### Important functions #################
#################################################
def load_file(date):
    PTV = pd.read_csv('IPTV_'+date+'_TF1.csv')[['TITRE','DUREE','description programme','HEURE','debut']]
    PTV['fin'] = PTV['debut']+PTV['DUREE']
    Points = pd.read_csv('merged_'+date+'.csv')[['minutes']]
    Points = Points.sort_values('minutes')
    return PTV,Points

def init_newPTV(PTV):
    #Initialisation du NewPTV
    newPTV = pd.DataFrame()
    newPTV['minute'] = [180]
    newPTV['TITRE'] = 'Programmes de la nuit'
    newPTV['Change Point'] = 'Non'
    newPTV['pourcentage vu'] = 0
    newPTV['Évenement'] = 'Début de Détection'
    return newPTV

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
    return h

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

def categorize_pub(name,debut,duree):
    if(duree in ["très court","court"]):
        return 0
    if(name in["Météo","Journal","Magazine","magazine"]):
        return 0
    elif(name in ['dessins animés']):
        return 3
    elif(name in ['Jeu']):
        return 1
    elif(name in ['Feuilleton','film']):
        return 2
    elif(name == 'Série' and 180<debut<12*60):
        return 1
    elif(name == 'Série'and 12*60<debut<21*60):
        return 0
    elif(name == 'Série'):
        return 2
    elif(name in ['Téléréalité'] and debut < 20.5*60):
        return 2
    elif(name in ['Téléréalité'] and debut > 20.5*60):
        return 3
    else:
        return 4


def categorize_programme(programme):
    p = []
    p.append(categorize_type(programme['description programme']))
    p.append(categorize_duree(programme['DUREE']))
    p.append(categorize_pub(p[0],programme['debut'],p[-1]))
    return p






def get_context(i,programme,Points,index_CP,lastCP,lastPub,lastend,currentduree,planifiedend):
    #we create a list with different notes to understand the context
    # minute of the point and its situation in the day
    context = [i]
    context.append(find_partofday(i))
    # Is the Point a Change Point
    context.append(find_ifChangePoint(i,Points['minutes'].iloc[index_CP]))
    # Where is the Point in the programme:
    seen_percentage = (i-lastend)/currentduree
    context.append(seen_percentage)
    context.append(find_position(seen_percentage))
    # which type of programme we are watching
    p = categorize_programme(programme)
    for i in range(len(p)):#3
        context.append(p[i])
    p.append(lastCP)
    p.append(lastPub)
    p.append(lastend)
    p.append(currentduree)
    return context




def make_modification():
    #TODO: réécrire le programme quand on sait a l'avance certaines chose (début des journaux par exemple)
    return 0

def make_newPTV(PTV,Points):
    #Initialisation des Variables
    verbose = False
    index_CP = 0
    index_PTV = PTV.shape[0]-1
    ##########################
    Predictiontimer = 200
    Pubinhour = 0
    lastCP = 200
    lastPub= 500
    lastend = 130
    currentduree = 255
    planifiedend = 385
    begin = True
    nbpub = 0
    Recall = -1
    importantpts = [[13*60,"Journal"],[20*60,"Journal"]]
    ######################
    newPTV = init_newPTV(PTV)
    historyofpoints = init_history()
    labels = [0]
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
        context = get_context(i,PTV.iloc[index_PTV],Points,index_CP,lastCP,lastPub,lastend,currentduree,planifiedend)
        ###### Let's verify that the algo is not doing a crappy predicitions and if this the case, clean his historic #####
        if(i in [j[0] for j in importantpts]):
            p = [j[0] for j in importantpts].index(i)
            #### we are at an important point, let's now see what the algo has predict
            if(PTV['TITRE'].iloc[index_PTV] == importantpts[p][1]):
                #Well he doesn't have the programme wrong, that's a good start
                #let's now find out if we are at a logical point of the programme
                if(context[3]>0.8):
                    Predictiontimer = 200
                    Pubinhour = 0
                    lastCP = 0
                    lastPub = 0
                    lastend = i
                    currentduree = PTV['DUREE'].iloc[index_PTV]
                    planifiedend = (lastend + currentduree)
                    nbpub = 0
                    #we can now keep going throw the process like before
                    #we just add a line to the history to say that a reset occured
                    newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'non',context[3],"--HARD RESET OF ALGORITHM--(in programme)"]
                else:
                    Predictiontimer = 200
                    Pubinhour = 0
                    lastCP = 0
                    lastPub= 0
                    lastend = i
                    currentduree = PTV['DUREE'].iloc[index_PTV]
                    planifiedend = (lastend + currentduree)
                    nbpub = 0
                    #we can now keep going throw the process like before
                    #we just add a line to the history to say that a reset occured
                    newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'non',context[3],"--soft reset to avoid any error--"]




            else:
                #well the programme is wrong, let's reset the algo
                l = PTV.index[PTV['TITRE']=="Journal"].tolist()
                if(i == 13*60):
                    index_PTV = int(l[0])
                if(i == 20*60):
                    index_PTV = int(l[1])
                ##########################
                Predictiontimer = 200
                Pubinhour = 0
                lastCP = 0
                lastPub= 0
                lastend = i
                currentduree = PTV['DUREE'].iloc[index_PTV]
                planifiedend = (lastend + currentduree)
                nbpub = 0
                #we can now keep going throw the process like before
                #we just add a line to the history to say that a reset occured
                newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'non',context[3],"--HARD RESET OF ALGORITHM--(out of programme)"]

        if(context[2]):
            historyofpoints.loc[historyofpoints.shape[0]] = context
            if(lastCP < min(currentduree,4)):
                labels.append(0)
                index_CP+=1
                continue
            #Change Point ==> Decide what to do with it
            if(nbpub>=context[7] or Pubinhour >= 12):
                # La pub n'est plus possible dans le programme ==> Soit il s'agit de la fin dans le programme, Soit c'est un faux Change Points
                if(historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] == "très court"):
                    newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"fin d'un programme"]
                    lastend = i
                    lastCP=0
                    index_PTV += 1
                    index_PTV = index_PTV%(PTV.shape[0])
                    currentduree = PTV['DUREE'].iloc[index_PTV]
                    planifiedend = (lastend + currentduree)
                    Predictiontimer = 200
                    nbpub = 0
                    labels.append(2)

                elif(historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] == "court"):
                    if(context[3]>0.5):
                        newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"fin d'un programme"]
                        lastend = i
                        lastCP=0
                        index_PTV += 1
                        index_PTV = index_PTV%(PTV.shape[0])
                        currentduree = PTV['DUREE'].iloc[index_PTV]
                        planifiedend = (lastend + currentduree)
                        Predictiontimer = 200
                        nbpub = 0
                        labels.append(2)
                    else:
                        labels.append(0)



                elif(historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] == "moyen"):
                    if(context[3]>0.70 ):
                        newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"fin d'un programme"]
                        lastend = i
                        lastCP=0
                        index_PTV += 1
                        index_PTV = index_PTV%(PTV.shape[0])
                        currentduree = PTV['DUREE'].iloc[index_PTV]
                        planifiedend = (lastend + currentduree)
                        Predictiontimer = 200
                        nbpub = 0
                        labels.append(2)
                    elif(context[3]>0.5 and context[5] != 'magazine'):
                        newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"fin d'un programme"]
                        lastend = i
                        lastCP=0
                        index_PTV += 1
                        index_PTV = index_PTV%(PTV.shape[0])
                        currentduree = PTV['DUREE'].iloc[index_PTV]
                        planifiedend = (lastend + currentduree)
                        Predictiontimer = 200
                        nbpub = 0
                        labels.append(2)
                    else:
                        labels.append(0)

                elif(historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] == "long"):
                    if(context[3]>0.70):
                        newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"fin d'un programme"]
                        lastend = i
                        lastCP=0
                        index_PTV += 1
                        index_PTV = index_PTV%(PTV.shape[0])
                        currentduree = PTV['DUREE'].iloc[index_PTV]
                        planifiedend = (lastend + currentduree)
                        Predictiontimer = 200
                        nbpub = 0
                        labels.append(2)
                    else:
                        labels.append(0)


                elif(historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] == "très long"):
                    if(context[3]>0.90):
                        newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"fin d'un programme"]
                        lastend = i
                        lastCP=0
                        index_PTV += 1
                        index_PTV = index_PTV%(PTV.shape[0])
                        currentduree = PTV['DUREE'].iloc[index_PTV]
                        planifiedend = (lastend + currentduree)
                        Predictiontimer = 200
                        nbpub = 0
                        labels.append(2)
                    else:
                        labels.append(0)

            else:
                #la pub est encore possible dans le programme mais pas certaine
                if(context[3]<=0.95 and lastPub>=20 and (i-lastend)>=10 and historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] != "très long" and context[5] != "film"):
                    newPTV.loc[newPTV.shape[0]] = [i%1440,"publicité",'oui',context[3],"publicité dans un programme"]
                    lastCP=0
                    lastPub = 0
                    Pubinhour+=4
                    nbpub+=1
                    labels.append(1)

                elif(context[3]<=0.95 and lastPub>=20 and (i-lastend)>=20 and context[5] == "film"):
                    newPTV.loc[newPTV.shape[0]] = [i%1440,"publicité",'oui',context[3],"publicité dans un programme"]
                    lastCP=0
                    lastPub = 0
                    Pubinhour+=4
                    nbpub+=1
                    labels.append(1)
                elif(context[3]<=0.95  and (historyofpoints['programme'].loc[(historyofpoints.shape[0]-1)] in ["Téléréalité","Feuilleton"]) ):
                    newPTV.loc[newPTV.shape[0]] = [i%1440,"publicité",'oui',context[3],"publicité dans un programme"]
                    lastCP=0
                    lastPub = 0
                    Pubinhour+=4
                    nbpub+=1
                    labels.append(1)
                elif(context[3]<=0.95 and lastPub>=20 and (i-lastend)>=10 and historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] == "très long" ):
                    newPTV.loc[newPTV.shape[0]] = [i%1440,"publicité",'oui',context[3],"publicité dans un programme"]
                    lastCP=0
                    lastPub = 0
                    Pubinhour+=4
                    nbpub+=1
                    labels.append(1)
                elif(historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] == "très court"):
                    newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"fin d'un programme"]
                    lastend = i
                    lastCP=0
                    index_PTV += 1
                    index_PTV = index_PTV%(PTV.shape[0])
                    currentduree = PTV['DUREE'].iloc[index_PTV]
                    planifiedend = (lastend + currentduree)
                    Predictiontimer = 200
                    nbpub = 0
                    labels.append(2)

                elif(historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] == "court"):
                    if(context[3]>0.5):
                        newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"fin d'un programme"]
                        lastend = i
                        lastCP=0
                        index_PTV += 1
                        index_PTV = index_PTV%(PTV.shape[0])
                        currentduree = PTV['DUREE'].iloc[index_PTV]
                        planifiedend = (lastend + currentduree)
                        Predictiontimer = 200
                        nbpub = 0
                        labels.append(2)
                    else:
                        labels.append(0)



                elif(historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] == "moyen"):
                    if(context[3]>0.5):
                        newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"fin d'un programme"]
                        lastend = i
                        lastCP=0
                        index_PTV += 1
                        index_PTV = index_PTV%(PTV.shape[0])
                        currentduree = PTV['DUREE'].iloc[index_PTV]
                        planifiedend = (lastend + currentduree)
                        Predictiontimer = 200
                        nbpub = 0
                        labels.append(2)
                    else:
                        labels.append(0)

                elif(historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] == "long"):
                    if(context[3]>0.75):
                        newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"fin d'un programme"]
                        lastend = i
                        lastCP=0
                        index_PTV += 1
                        index_PTV = index_PTV%(PTV.shape[0])
                        currentduree = PTV['DUREE'].iloc[index_PTV]
                        planifiedend = (lastend + currentduree)
                        Predictiontimer = 200
                        nbpub = 0
                        labels.append(2)
                    else:
                        labels.append(0)
                elif(historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] == "très long"):
                    if(context[3]>0.91):
                        newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"fin d'un programme"]
                        lastend = i
                        lastCP=0
                        index_PTV += 1
                        index_PTV = index_PTV%(PTV.shape[0])
                        currentduree = PTV['DUREE'].iloc[index_PTV]
                        planifiedend = (lastend + currentduree)
                        Predictiontimer = 200
                        nbpub = 0
                        labels.append(2)
                    else:
                        labels.append(0)
                else:
                    label.append(0)
            index_CP+=1
        elif(i in [12*60+50]):
            historyofpoints.loc[historyofpoints.shape[0]] = context
            if(lastCP < min(currentduree,4)):
                labels.append(0)
                index_CP+=1
                continue
            #Change Point ==> Decide what to do with it
            if(nbpub>=context[7] or Pubinhour >= 12):
                # La pub n'est plus possible dans le programme ==> Soit il s'agit de la fin dans le programme, Soit c'est un faux Change Points
                if(historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] == "très court"):
                    newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"fin d'un programme"]
                    lastend = i
                    lastCP=0
                    index_PTV += 1
                    index_PTV = index_PTV%(PTV.shape[0])
                    currentduree = PTV['DUREE'].iloc[index_PTV]
                    planifiedend = (lastend + currentduree)
                    Predictiontimer = 200
                    nbpub = 0
                    labels.append(2)

                elif(historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] == "court"):
                    if(context[3]>=0.5):
                        newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"fin d'un programme"]
                        lastend = i
                        lastCP=0
                        index_PTV += 1
                        index_PTV = index_PTV%(PTV.shape[0])
                        currentduree = PTV['DUREE'].iloc[index_PTV]
                        planifiedend = (lastend + currentduree)
                        Predictiontimer = 200
                        nbpub = 0
                        labels.append(2)
                    else:
                        labels.append(0)



                elif(historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] == "moyen"):
                    if(context[3]>=0.5):
                        newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"fin d'un programme"]
                        lastend = i
                        lastCP=0
                        index_PTV += 1
                        index_PTV = index_PTV%(PTV.shape[0])
                        currentduree = PTV['DUREE'].iloc[index_PTV]
                        planifiedend = (lastend + currentduree)
                        Predictiontimer = 200
                        nbpub = 0
                        labels.append(2)
                    else:
                        labels.append(0)

                elif(historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] == "long"):
                    if(context[3]>0.90):
                        newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"fin d'un programme"]
                        lastend = i
                        lastCP=0
                        index_PTV += 1
                        index_PTV = index_PTV%(PTV.shape[0])
                        currentduree = PTV['DUREE'].iloc[index_PTV]
                        planifiedend = (lastend + currentduree)
                        Predictiontimer = 200
                        nbpub = 0
                        labels.append(2)
                    else:
                        labels.append(0)


                elif(historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] == "très long"):
                    if(context[3]>0.90):
                        newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"fin d'un programme"]
                        lastend = i
                        lastCP=0
                        index_PTV += 1
                        index_PTV = index_PTV%(PTV.shape[0])
                        currentduree = PTV['DUREE'].iloc[index_PTV]
                        planifiedend = (lastend + currentduree)
                        Predictiontimer = 200
                        nbpub = 0
                        labels.append(2)
                    else:
                        labels.append(0)

            else:
                #la pub est encore possible dans le programme mais pas certaine
                if(context[3]<=0.95 and lastPub>=20 and (i-lastend)>=10 and historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] != "très long"):
                    newPTV.loc[newPTV.shape[0]] = [i%1440,"publicité",'oui',context[3],"publicité dans un programme"]
                    lastCP=0
                    lastPub = 0
                    Pubinhour+=4
                    nbpub+=1
                    labels.append(1)

                elif(context[3]<=0.95 and lastPub>=20 and (i-lastend)>=15):
                    newPTV.loc[newPTV.shape[0]] = [i%1440,"publicité",'oui',context[3],"publicité dans un programme"]
                    lastCP=0
                    lastPub = 0
                    Pubinhour+=4
                    nbpub+=1
                    labels.append(1)
                elif(context[3]<=0.95  and (historyofpoints['programme'].loc[(historyofpoints.shape[0]-1)] in ["Téléréalité","Feuilleton"]) ):
                    newPTV.loc[newPTV.shape[0]] = [i%1440,"publicité",'oui',context[3],"publicité dans un programme"]
                    lastCP=0
                    lastPub = 0
                    Pubinhour+=4
                    nbpub+=1
                    labels.append(1)
                elif(historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] == "très court"):
                    newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"fin d'un programme"]
                    lastend = i
                    lastCP=0
                    index_PTV += 1
                    index_PTV = index_PTV%(PTV.shape[0])
                    currentduree = PTV['DUREE'].iloc[index_PTV]
                    planifiedend = (lastend + currentduree)
                    Predictiontimer = 200
                    nbpub = 0
                    labels.append(2)

                elif(historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] == "court"):
                    if(context[3]>0.5):
                        newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"fin d'un programme"]
                        lastend = i
                        lastCP=0
                        index_PTV += 1
                        index_PTV = index_PTV%(PTV.shape[0])
                        currentduree = PTV['DUREE'].iloc[index_PTV]
                        planifiedend = (lastend + currentduree)
                        Predictiontimer = 200
                        nbpub = 0
                        labels.append(2)
                    else:
                        labels.append(0)



                elif(historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] == "moyen"):
                    if(context[3]>0.5):
                        newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"fin d'un programme"]
                        lastend = i
                        lastCP=0
                        index_PTV += 1
                        index_PTV = index_PTV%(PTV.shape[0])
                        currentduree = PTV['DUREE'].iloc[index_PTV]
                        planifiedend = (lastend + currentduree)
                        Predictiontimer = 200
                        nbpub = 0
                        labels.append(2)
                    else:
                        labels.append(0)

                elif(historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] == "long"):
                    if(context[3]>0.75):
                        newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"fin d'un programme"]
                        lastend = i
                        lastCP=0
                        index_PTV += 1
                        index_PTV = index_PTV%(PTV.shape[0])
                        currentduree = PTV['DUREE'].iloc[index_PTV]
                        planifiedend = (lastend + currentduree)
                        Predictiontimer = 200
                        nbpub = 0
                        labels.append(2)
                    else:
                        labels.append(0)
                elif(historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] == "très long"):
                    if(context[3]>0.91):
                        newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"fin d'un programme"]
                        lastend = i
                        lastCP=0
                        index_PTV += 1
                        index_PTV = index_PTV%(PTV.shape[0])
                        currentduree = PTV['DUREE'].iloc[index_PTV]
                        planifiedend = (lastend + currentduree)
                        Predictiontimer = 200
                        nbpub = 0
                        labels.append(2)
                    else:
                        labels.append(0)
                else:
                    labels.append(0)


        else:
            #labels.append(0)
            #Not a Change Point, we'll just check that nothing is wrong in the PTV at this time
            if(Predictiontimer == 0):
                newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'non',context[3],"fin non détectée d'un programme"]
                lastend = i
                lastCP=0
                index_PTV += 1
                index_PTV = index_PTV%(PTV.shape[0])
                currentduree = PTV['DUREE'].iloc[index_PTV]
                planifiedend = (lastend + currentduree)
                Predictiontimer = 200
                nbpub = 0
            elif(context[3] == 1):
                #Dépassement autorisé: Modulable en fonction de la position dans la journée si besoin
                if(context[6] == "très court"):
                    Predictiontimer = 0
                elif(PTV['TITRE'].iloc[index_PTV] == 'Téléshopping'):
                    Predictiontimer = 15
                elif(context[6] == "court"):
                    Predictiontimer = 5
                elif(context[6] == "moyen"):
                    Predictiontimer = 11
                else:
                    Predictiontimer = 15
                if(11.5*60<i<14*60 or 19.5*60<i<21*60):
                    Predictiontimer = 1
            elif(context[3]>1):
                Predictiontimer -= 1
            else:
                pass


    return newPTV,historyofpoints,labels





#################################################
########### main with options ###################
#################################################
dates = ['2018-04-30','2018-05-07','2018-05-09','2018-05-18','2018-05-23','2018-05-28']

def main(argv):
    if(len(argv) == 0):
        for date in dates:
            print(date)
            os.system(' python /home/alexis/Bureau/Stage/ProgrammesTV/PTV.py '+str(date))
    else:
        date = argv[0]
        PTV,Points = load_file(date)
        new_PTV,historyofpoints,labels = make_newPTV(PTV,Points)
        new_PTV['Heure'] = new_PTV['minute'].apply(lambda x: str(int(x/60))+':'+str(x%60))
        historyofpoints['Heure'] = historyofpoints['minute'].apply(lambda x: str(int(x/60))+':'+str(x%60))
        new_PTV.to_html('new_PTV-'+date+'.html')
        historyofpoints.to_html('historyofpoints-'+date+'.html')
        print(historyofpoints.shape,len(labels))
        historyofpoints['labels'] = labels
        historyofpoints.to_csv('true_merge_'+str(date)+'.csv',index=False)

    return ("process achevé sans erreures")


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
