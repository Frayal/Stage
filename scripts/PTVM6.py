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
import datetime
from tqdm import tqdm
#################################################
########### Global variables ####################
#################################################


#################################################
########### Important functions #################
#################################################
def load_file(date):
    try:
        PTV = pd.read_csv('/home/alexis/Bureau/Project/Datas/PTV/extracted/IPTV_0118_'+date+'_M6.csv')
        PTV['fin'] = PTV['debut']+PTV['DUREE']
        JOINDATE = "".join(date.split('-'))
        Points = pd.read_csv('/home/alexis/Bureau/Project/results/pred/pred_'+str(JOINDATE)+'_118.csv').values
        proba = pd.read_csv('/home/alexis/Bureau/Project/results/pred/pred_proba_'+str(JOINDATE)+'_118.csv').values
        return PTV,Points,proba
    except:
        print("Fichier Non existant")
        return [],[],[]

def init_newPTV(PTV):
    #Initialisation du NewPTV
    newPTV = pd.DataFrame()
    newPTV['minute'] = [180]
    newPTV['TITRE'] = 'M6 Music'
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
    h['lastCP'] =  0
    h['lastPub'] =  0
    h['lastend'] =  0
    h['currentduree'] =  0
    h['Pubinhour'] =  0
    h['probability of CP'] = 0
    h['nb de pubs encore possible'] = 0
    h["chaine"]= 'M6'
    h['CLE-FORMAT'] = 0
    h['CLE-GENRE'] = 0
    h['day'] = 0
    #h['per'] = 1
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
    if(cp[i-183]):
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
    elif(100<=duree<=180):
        return("très long")
    else:
        return('super long')

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

def categorize_pub(name,debut,duree,titre,PTV,index_PTV,chaine='M6'):
    if(chaine == 'TF1' and (debut>20*60 or debut<180)):
        if(titre in ['Nos chers voisins'] and debut>20*60):
            return 0
        elif(titre in ['Nos chers voisins','Reportages découverte']):
            return 2
        elif(titre in ['Journal','Téléshopping','Tirage du Loto']):
            return 0
        elif(titre in ['50mn Inside','Téléfoot']):
            if(duree in ['moyen']):
                return 1
            elif(duree in ["long"]):
                return 2
            elif(duree in ["court"]):
                return 1
            else:
                return 4
        elif(name in["Météo","Magazine","magazine"] and duree in ["court","très court"]):
            return 0
        elif(name in ['dessins animés'] and duree != "super long"):
            return 3
        elif(name in ['Jeu']):
            return 1
        elif(name in ['Feuilleton','film','Drame','Thriller']):
            return 2
        elif(name == 'Série' and (debut>21*60) and titre == PTV['TITRE'].loc[index_PTV-1]):
            return 2
        elif(name == 'Série' and (debut>21*60 or debut<180)):
            return 1
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
        elif(duree in ["très court","court"]):
            return 0
        else:
            return 4
    if(chaine == 'TF1' and 180<=debut<=20*60):
        if(titre in ['Nos chers voisins']):
            return 2
        elif(titre in ['Journal']):
            return 0
        elif(titre == '50mn Inside'):
            if(duree in ["moyen","long"]):
                return 2
            if(duree in ["court"]):
                return 1
            else:
                return 4
        elif(name in["Météo","Magazine","magazine"] and duree in ["court","très court"]):
            return 0
        elif(name in ['dessins animés'] and duree != "super long"):
            return 3
        elif(name in ['Jeu']):
            return 1
        elif(name in ['Feuilleton','film','Drame','Thriller']):
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
        elif(duree in ["très court","court"]):
            return 0
        else:
            return 4

    if(chaine == 'M6' and (debut>20*60 or debut <180)):
        if(titre in ['Nos chers voisins','Les reines du shopping']):
            return 2
        elif(titre in ['Une superstar pour Noël']):
            return 3
        elif(titre in ['En famille']):
            if(duree in ["court","moyen",'très court','long']):
                return 0
            else:
                return 2
        elif(titre in ["Chasseurs d'appart'"]):
            if(duree in ["très long","super long"] ):
                return 12
            else:
                return 1
        elif(titre in ['Les aventures de Tintin','Absolument stars','Martine','Les Sisters','M6 boutique','Scènes de ménages','M6 Music']):
            return 0
        elif(titre in ['66 minutes : grand format']):
            return 1
        elif(titre =='Turbo' and debut<11*60):
            return 2
        elif(titre in ['66 minutes']):
            return 4
        elif(name in ['Dessin animé','dessins animés']):
            return 0
        elif(name in ['Journal']):
            return 0
        elif(titre == '50mn Inside'):
            if(duree in ["moyen","long"]):
                return 2
            else:
                return 4
        elif(name in ['magazine'] and duree not in ["court","très court"]):
            return 2
        elif(name in["Météo","Magazine","magazine"] and duree in ["court","très court"]):
            return 0
        elif(name in ['dessins animés'] and duree != "super long"):
            return 3
        elif(name in ['Jeu'] and duree in ['court','très court','moyen']):
            return 2
        elif(name in ['Feuilleton','film','Drame','Thriller']):
            return 2
        elif(name == 'Série' and (debut<13*60 or 25*60>debut>22*60+20)):
            return 1
        elif(name == 'Série'):
            return 2
        elif(name in ['Téléréalité'] and debut < 20.5*60):
            if( duree in ["court","moyen"]):
                return 1
            else:
                return 2
        elif(duree in ["très court","court"]):
            return 0
        else:
            return 10
    if(chaine == 'M6' and 180<=debut<=20*60):
        if(titre in ['Nos chers voisins','Les reines du shopping']):
            return 2
        elif(titre in ['Une superstar pour Noël']):
            return 3
        elif(titre in ['En famille']):
            if(duree in ["court","moyen",'très court','long']):
                return 0
            else:
                return 2
        elif(titre in ["Chasseurs d'appart'"]):
            if(duree in ["très long","super long"] ):
                return 12
            else:
                return 1
        elif(titre in ['Les aventures de Tintin','Absolument stars','Martine','Les Sisters','M6 boutique','Scènes de ménages']):
            return 0
        elif(titre in ['66 minutes : grand format']):
            return 1
        elif(titre =='Turbo' and debut<11*60):
            return 1
        elif(titre in ['66 minutes','M6 Music']):
            return 4
        elif(name in ['Dessin animé','dessins animés']):
            return 0
        elif(name in ['Journal']):
            return 0
        elif(titre == '50mn Inside'):
            if(duree in ["moyen","long"]):
                return 2
            else:
                return 4
        elif(name in ['magazine'] and duree not in ["court","très court"]):
            return 2
        elif(name in["Météo","Magazine","magazine"] and duree in ["court","très court"]):
            return 0
        elif(name in ['dessins animés'] and duree != "super long"):
            return 3
        elif(name in ['Jeu'] and duree in ['court','très court','moyen']):
            return 2
        elif(name in ['Feuilleton','film','Drame','Thriller']):
            return 2
        elif(name == 'Série' and 180<debut<13*60):
            return 1
        elif(name == 'Série'):
            return 2
        elif(name in ['Téléréalité'] and debut < 20.5*60):
            if( duree in ["court","moyen"]):
                return 1
            else:
                return 2
        elif(name in ['Téléréalité'] and debut > 20.5*60):
            return 3
        elif(duree in ["très court","court"]):
            return 0
        else:
            return 10


def categorize_programme(programme,PTV,index_PTV):
    p = []
    p.append(categorize_type(programme['description programme']))
    p.append(categorize_duree(programme['DUREE']))
    p.append(categorize_pub(p[0],programme['debut'],p[-1],programme['TITRE'],PTV,index_PTV))
    return p






def get_context(i,programme,Points,lastCP,lastPub,lastend,currentduree,planifiedend,Pubinhour,probas,nbpub,per,PTV,index_PTV,date):
    #we create a list with different notes to understand the context
    # minute of the point and its situation in the day
    context = [i]
    context.append(find_partofday(i))
    # Is the Point a Change Point
    context.append(find_ifChangePoint(i,Points))
    # Where is the Point in the programme:
    seen_percentage = (i-lastend)/currentduree
    context.append(seen_percentage)
    context.append(find_position(seen_percentage))
    # which type of programme we are watching
    p = categorize_programme(programme,PTV,index_PTV)
    for j in range(len(p)):#3
        context.append(p[j])
    context.append(lastCP)
    context.append(lastPub)
    context.append(lastend)
    context.append(currentduree)
    context.append(Pubinhour)
    context.append(probas[i-183][0])
    context.append(context[7]-nbpub)
    context.append('M6')
    context.append(programme['CLE-FORMAT'])
    context.append(programme['CLE-GENRE'])
    day = datetime.datetime(int(date.split('-')[0]), int(date.split('-')[1]), int(date.split('-')[2]))
    context.append(day.weekday())
    #context.append(per)
    return context


def make_newPTV(PTV,Points,proba,date):
    #Initialisation des Variables
    verbose = False
    index_CP = 0
    index_PTV = PTV.shape[0]-1
    ##########################
    Predictiontimer = 200
    Pubinhour = 0
    lastCP = 200
    lastPub= 500
    lastend = 180
    currentduree = PTV['debut'].loc[0]-180
    planifiedend = PTV['debut'].loc[0]
    begin = True
    nbpub = 0
    Recall = 1
    importantpts = [[12*60+45,"Le 12.45"],[19*60+45,"Le 19.45"],[(int(PTV['HEURE'].loc[PTV.shape[0]-1].split(':')[0])+24)*60+int(PTV['HEURE'].loc[PTV.shape[0]-1].split(':')[1]),PTV['TITRE'].loc[PTV.shape[0]-1]]]
    index_ipts = 0
    error = 0
    chaine = 'M6'
    ######################
    newPTV = init_newPTV(PTV)
    historyofpoints = init_history()
    labels = [0]
    per = 0
    ######################
    for i in tqdm(range(183,1620)):
        #Update time of commercials (Reset)
        if(i%60 == 0):
            Pubinhour = 0
        #Update timmers
        lastPub+=1
        lastCP+=1
        if(index_ipts==len(importantpts)):
            index_ipts-=1
        #let's get the context:
        context = get_context(i,PTV.iloc[index_PTV],Points,lastCP,lastPub,lastend,currentduree,planifiedend,Pubinhour,proba,nbpub,per,PTV,index_PTV,date)

        #Sur M6 il y a 16 minutes de pub entre deux films!!!!!!!!!!!!.....!!!!!!!....!!.!.!.!.!....!.!...!..!.!.!.!
        if(PTV['GENRESIMPLE'].iloc[index_PTV].split(' ')[0] == PTV['GENRESIMPLE'].iloc[index_PTV-1].split(' ')[0] and PTV['GENRESIMPLE'].iloc[index_PTV].split(' ')[0] == 'Téléfilm'
            and (i-lastend)<2 and Recall > 0 and per<0.97 and chaine == 'M6'):
            print(i,PTV['GENRESIMPLE'].iloc[index_PTV])
            lastend = i+5
            lastPub = -25
            Recall -= 0.5
        elif((i-lastend)<2 and Recall > 0 and per<0.97 and chaine == 'M6' and 15*60<i<16*60):
            print(i,PTV['GENRESIMPLE'].iloc[index_PTV],PTV['GENRESIMPLE'].iloc[index_PTV-1])
            lastend = i+5
            lastPub = -25
            Recall -= 0.5

        ###### Let's verify that 'M6 boutique',the algo is not doing a crappy predicitions and if this the case, clean his historic #####
        elif(i == importantpts[index_ipts][0]):
            #### we are at an important point, let's now see what the algo has predict
            if(PTV['TITRE'].iloc[index_PTV] == importantpts[index_ipts][1]):
                #Well he doesn't have the programme wrong, that's a good start
                #let's now find out if we are at a logical point of the programme
                if(i-lastend>13):
                    #Wellllll, the programme began way too early...something went wrong before...Let's rest for now, we'll correct the algo later
                    Predictiontimer = 200
                    Pubinhour = 0
                    lastCP = 0
                    lastPub = 0
                    lastend = i
                    currentduree = PTV['DUREE'].iloc[index_PTV]
                    planifiedend = (lastend + currentduree)
                    nbpub = 0
                    if(index_ipts == 0):
                        print("erreur sur l'après midi")
                    elif(index_ipts == 1):
                        print("erreur sur la matinée")
                    else:
                        print("erreur sur la soirée")
                    error+=1*10**index_ipts
                    #we can now keep going throw the process like before
                    #we just add a line to the history to say that a reset occured
                    newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'non',context[3],"--HARD RESET OF ALGORITHM--(in programme)"]
                    index_ipts+=1
                else:
                    # OMG the ALGO IS RIGHT...here is a candy, let's rest a litle just in case...we never know....
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
                    index_ipts+=1



            else:
                #maybe it's the next programme so calme the fuck down!
                if(PTV['TITRE'].iloc[index_PTV+1] == importantpts[index_ipts][1]):
                    if(planifiedend-i<10):
                        #here you go, it's the next one...just terminate this one and we're good to go
                        newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"fin d'un programme"]
                        lastend = i
                        lastCP=0
                        index_PTV += 1
                        index_PTV = index_PTV%(PTV.shape[0])
                        currentduree = PTV['DUREE'].iloc[index_PTV]
                        planifiedend = (lastend + currentduree)
                        Predictiontimer = 200
                        nbpub = 0
                        index_ipts+=1
                    else:
                        #here you go, it's the next one...just terminate this one and we're good to go
                        newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"--HARD RESET OF ALGORITHM--(Oups)"]
                        lastend = i
                        lastCP=0
                        index_PTV += 1
                        index_PTV = index_PTV%(PTV.shape[0])
                        currentduree = PTV['DUREE'].iloc[index_PTV]
                        planifiedend = (lastend + currentduree)
                        Predictiontimer = 200
                        nbpub = 0
                        if(index_ipts == 0):
                            print("erreur sur l'après midi")
                        elif(index_ipts == 1):
                            print("erreur sur la matinée")
                        else:
                            print("erreur sur la soirée")
                        error+=1*10**index_ipts
                        index_ipts+=1


                else:
                    #well the programme is wrong, and we are not even close to it, let's terminate this thing before it goes completly south. REBOOT The algo, erase the memory, just like in Westworld.
                    #BUT FIRST LET'S VERIFY THAT THERE IS INDEED AN IMPORTANT PROGRAMME THAT DAY...Don't go fuck everything up for no reason
                    l = PTV.index[(PTV['TITRE']==importantpts[index_ipts][1]) & (PTV['debut'] == i)].tolist()
                    if(len(l)>0):
                        index_PTV = l[0]
                        ##########################
                        Predictiontimer = 200
                        Pubinhour = 0
                        lastCP = 0
                        lastPub= 0
                        lastend = i
                        currentduree = PTV['DUREE'].iloc[index_PTV]
                        planifiedend = (lastend + currentduree)
                        nbpub = 0
                        if(index_ipts == 0):
                            print("erreur sur l'après midi")
                        elif(index_ipts == 1):
                            print("erreur sur la matinée")
                        else:
                            print("erreur sur la soirée")
                        error+=1*10**index_ipts
                        #we can now keep going throw the process like before
                        #we just add a line to the history to say that a reset occured
                        newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'non',context[3],"--HARD RESET OF ALGORITHM--(out of programme)"]
                        index_ipts+=1
                    else:
                        index_ipts+=1

        elif(context[2]):
            historyofpoints.loc[historyofpoints.shape[0]] = context
            if(lastCP < min(max(currentduree/2,4),5)):
                labels.append(0)
                index_CP+=1
                continue
            elif(lastPub<=6):
                labels.append(0)
                index_CP+=1
                continue
            #Change Point ==> Decide what to do with it
            if(nbpub>=context[7] or Pubinhour >= 12):
                # La pub n'est plus possible dans le programme ==> Soit il s'agit de la fin dans le programme, Soit c'est un faux Change Points
                if(PTV['TITRE'].iloc[index_PTV] in ['M6 boutique','M6 Music'] and context[3]<0.98):
                    labels.append(0)
                elif(PTV['TITRE'].iloc[index_PTV] in ['Les reines du shopping','Absolument stars'] and context[3]<0.96):
                    labels.append(0)
                elif(PTV['TITRE'].iloc[index_PTV] in ['Desperate Housewives','Les Sisters'] and context[3]<0.65 and i<12*60):
                    labels.append(0)
                elif(PTV['TITRE'].iloc[index_PTV] in ['Desperate Housewives'] and context[3]<0.81 and i<12*60+30):
                    labels.append(0)
                elif(historyofpoints['programme'].loc[(historyofpoints.shape[0]-1)] == "Journal"):
                    if(context[3]>0.70 and i<15*60):
                        newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"fin d'un programme"]
                        lastend = i
                        lastCP=0
                        index_PTV += 1
                        index_PTV = index_PTV%(PTV.shape[0])
                        currentduree = PTV['DUREE'].iloc[index_PTV]
                        planifiedend = (lastend + currentduree)
                        Predictiontimer = 200
                        nbpub = 0
                        per = context[3]
                        labels.append(2)
                    elif(context[3]>0.90 and i>15*60):
                        newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"fin d'un programme"]
                        lastend = i
                        lastCP=0
                        index_PTV += 1
                        index_PTV = index_PTV%(PTV.shape[0])
                        currentduree = PTV['DUREE'].iloc[index_PTV]
                        planifiedend = (lastend + currentduree)
                        Predictiontimer = 200
                        nbpub = 0
                        per = context[3]
                        labels.append(2)
                    else:
                        labels.append(0)
                elif(historyofpoints['programme'].loc[(historyofpoints.shape[0]-1)] == "film"):
                    if(context[3]>0.75 and context[13]>=0.6):
                        newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"fin d'un programme"]
                        lastend = i
                        lastCP=0
                        index_PTV += 1
                        index_PTV = index_PTV%(PTV.shape[0])
                        currentduree = PTV['DUREE'].iloc[index_PTV]
                        planifiedend = (lastend + currentduree)
                        Predictiontimer = 200
                        nbpub = 0
                        per = context[3]
                        labels.append(2)
                    else:
                        labels.append(0)
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
                    per = context[3]
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
                        per = context[3]
                        labels.append(2)
                    else:
                        labels.append(0)



                elif(historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] == "moyen"):
                    if(context[3]>0.80 ):
                        newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"fin d'un programme"]
                        lastend = i
                        lastCP=0
                        index_PTV += 1
                        index_PTV = index_PTV%(PTV.shape[0])
                        currentduree = PTV['DUREE'].iloc[index_PTV]
                        planifiedend = (lastend + currentduree)
                        Predictiontimer = 200
                        nbpub = 0
                        per = context[3]
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
                        per = context[3]
                        labels.append(2)
                    else:
                        labels.append(0)

                elif(historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] == "long"):
                    if(context[3]>0.84):
                        newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"fin d'un programme"]
                        lastend = i
                        lastCP=0
                        index_PTV += 1
                        index_PTV = index_PTV%(PTV.shape[0])
                        currentduree = PTV['DUREE'].iloc[index_PTV]
                        planifiedend = (lastend + currentduree)
                        Predictiontimer = 200
                        nbpub = 0
                        per = context[3]
                        labels.append(2)
                    else:
                        labels.append(0)

                elif(historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] == "super long"):
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
                        per = context[3]
                        labels.append(2)
                    else:
                        labels.append(0)


                elif(historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] == "très long"):
                    if(context[3]>0.80):
                        newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"fin d'un programme"]
                        lastend = i
                        lastCP=0
                        index_PTV += 1
                        index_PTV = index_PTV%(PTV.shape[0])
                        currentduree = PTV['DUREE'].iloc[index_PTV]
                        planifiedend = (lastend + currentduree)
                        Predictiontimer = 200
                        nbpub = 0
                        per = context[3]
                        labels.append(2)
                    else:
                        labels.append(0)

            else:
                #la pub est encore possible dans le programme mais pas certaine
                if(context[3]<0.95 and lastPub>=20 and (i-lastend)>15 and  context[5] in ["Feuilleton","Série"]):
                    newPTV.loc[newPTV.shape[0]] = [i%1440,"publicité",'oui',context[3],"publicité dans un programme"]
                    lastCP=0
                    lastPub = 0
                    Pubinhour+=4
                    nbpub+=1
                    labels.append(1)
                elif(context[3]<0.88 and lastPub>=20 and (i-lastend)>=10 and  context[5] in ["Feuilleton","Série"] and context[13]>0.6):
                    newPTV.loc[newPTV.shape[0]] = [i%1440,"publicité",'oui',context[3],"publicité dans un programme"]
                    lastCP=0
                    lastPub = 0
                    Pubinhour+=4
                    nbpub+=1
                    labels.append(1)
                elif(context[3]<0.95 and lastPub>=20 and (i-lastend)>=15 and  context[5] == "magazine"):
                    newPTV.loc[newPTV.shape[0]] = [i%1440,"publicité",'oui',context[3],"publicité dans un programme"]
                    lastCP=0
                    lastPub = 0
                    Pubinhour+=4
                    nbpub+=1
                    labels.append(1)
                elif(context[3]<0.90 and lastPub>=20 and (i-lastend)>=15 and historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] not in ["super long"] and context[5] not in ["film","Série"]):
                    newPTV.loc[newPTV.shape[0]] = [i%1440,"publicité",'oui',context[3],"publicité dans un programme"]
                    lastCP=0
                    lastPub = 0
                    Pubinhour+=4
                    nbpub+=1
                    labels.append(1)
                elif(context[3]<=0.95 and lastPub>22 and (i-lastend)>=15 and context[5] == "film"):
                    newPTV.loc[newPTV.shape[0]] = [i%1440,"publicité",'oui',context[3],"publicité dans un programme"]
                    lastCP=0
                    lastPub = 0
                    Pubinhour+=4
                    nbpub+=1
                    labels.append(1)
                elif(context[3]<=0.95 and lastPub>=20 and (i-lastend)>=15 and(historyofpoints['programme'].loc[(historyofpoints.shape[0]-1)] in ["Téléréalité","Feuilleton"]) ):
                    newPTV.loc[newPTV.shape[0]] = [i%1440,"publicité",'oui',context[3],"publicité dans un programme"]
                    lastCP=0
                    lastPub = 0
                    Pubinhour+=4
                    nbpub+=1
                    labels.append(1)

                elif(historyofpoints['programme'].loc[(historyofpoints.shape[0]-1)] == 'magazine' and context[3]<=0.95):
                    labels.append(0)

                elif(context[3]<=0.95 and lastPub>=20 and (i-lastend)>=15 and historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] == "très long" ):
                    newPTV.loc[newPTV.shape[0]] = [i%1440,"publicité",'oui',context[3],"publicité dans un programme"]
                    lastCP=0
                    lastPub = 0
                    Pubinhour+=4
                    nbpub+=1
                    labels.append(1)
                elif(context[3]<=0.95 and lastPub>=20 and (i-lastend)>=15 and historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] == "super long" ):
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
                    per = context[3]
                    labels.append(2)

                elif(historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] == "court"):
                    if(context[3]>0.85):
                        newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"fin d'un programme"]
                        lastend = i
                        lastCP=0
                        index_PTV += 1
                        index_PTV = index_PTV%(PTV.shape[0])
                        currentduree = PTV['DUREE'].iloc[index_PTV]
                        planifiedend = (lastend + currentduree)
                        Predictiontimer = 200
                        nbpub = 0
                        per = context[3]
                        labels.append(2)
                    else:
                        labels.append(0)



                elif(historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] == "moyen"):
                    if(context[3]>0.85):
                        newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"fin d'un programme"]
                        lastend = i
                        lastCP=0
                        index_PTV += 1
                        index_PTV = index_PTV%(PTV.shape[0])
                        currentduree = PTV['DUREE'].iloc[index_PTV]
                        planifiedend = (lastend + currentduree)
                        Predictiontimer = 200
                        nbpub = 0
                        per = context[3]
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
                        per = context[3]
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
                        per = context[3]
                        labels.append(2)
                    else:
                        labels.append(0)
                elif(historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] == "super long"):
                    if(context[3]>0.99):
                        newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"fin d'un programme"]
                        lastend = i
                        lastCP=0
                        index_PTV += 1
                        index_PTV = index_PTV%(PTV.shape[0])
                        currentduree = PTV['DUREE'].iloc[index_PTV]
                        planifiedend = (lastend + currentduree)
                        Predictiontimer = 200
                        nbpub = 0
                        per = context[3]
                        labels.append(2)
                    else:
                        labels.append(0)
                else:
                    labels.append(0)
            index_CP+=1
        elif(i in [6*60,6*60+50,8*60+57,10*60,10*60+20]):
            historyofpoints.loc[historyofpoints.shape[0]] = context
            if(lastCP < min(max(currentduree/2,4),5)):
                labels.append(0)
                continue
            #Change Point ==> Decide what to do with it
            if(nbpub>=context[7] or Pubinhour >= 12):
                if(PTV['TITRE'].iloc[index_PTV] in ['M6 boutique','M6 Music'] and context[3]<0.98):
                    labels.append(0)
                elif(PTV['TITRE'].iloc[index_PTV] in ['Les reines du shopping','Absolument stars','Les Sisters'] and context[3]<0.95):
                    labels.append(0)
                elif(PTV['TITRE'].iloc[index_PTV] in ['Desperate Housewives'] and context[3]<0.65 and i<12*60):
                    labels.append(0)
                elif(PTV['TITRE'].iloc[index_PTV] in ['Desperate Housewives'] and context[3]<0.81 and i<12*60+30):
                    labels.append(0)
                # La pub n'est plus possible dans le programme ==> Soit il s'agit de la fin dans le programme, Soit c'est un faux Change Points
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
                    per = context[3]
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
                        per = context[3]
                        labels.append(2)
                    else:
                        labels.append(0)



                elif(historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] == "moyen"):
                    if(context[3]>0.80 ):
                        newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"fin d'un programme"]
                        lastend = i
                        lastCP=0
                        index_PTV += 1
                        index_PTV = index_PTV%(PTV.shape[0])
                        currentduree = PTV['DUREE'].iloc[index_PTV]
                        planifiedend = (lastend + currentduree)
                        Predictiontimer = 200
                        nbpub = 0
                        per = context[3]
                        labels.append(2)
                    elif(context[3]>0.8 and context[5] != 'magazine'):
                        newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"fin d'un programme"]
                        lastend = i
                        lastCP=0
                        index_PTV += 1
                        index_PTV = index_PTV%(PTV.shape[0])
                        currentduree = PTV['DUREE'].iloc[index_PTV]
                        planifiedend = (lastend + currentduree)
                        Predictiontimer = 200
                        nbpub = 0
                        per = context[3]
                        labels.append(2)
                    else:
                        labels.append(0)

                elif(historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] == "long"):
                    if(context[3]>=0.85):
                        newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"fin d'un programme"]
                        lastend = i
                        lastCP=0
                        index_PTV += 1
                        index_PTV = index_PTV%(PTV.shape[0])
                        currentduree = PTV['DUREE'].iloc[index_PTV]
                        planifiedend = (lastend + currentduree)
                        Predictiontimer = 200
                        nbpub = 0
                        per = context[3]
                        labels.append(2)
                    else:
                        labels.append(0)

                elif(historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] == "super long"):
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
                        per = context[3]
                        labels.append(2)
                    else:
                        labels.append(0)


                elif(historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] == "très long"):
                    if(context[3]>0.80):
                        newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"fin d'un programme"]
                        lastend = i
                        lastCP=0
                        index_PTV += 1
                        index_PTV = index_PTV%(PTV.shape[0])
                        currentduree = PTV['DUREE'].iloc[index_PTV]
                        planifiedend = (lastend + currentduree)
                        Predictiontimer = 200
                        nbpub = 0
                        per = context[3]
                        labels.append(2)
                    else:
                        labels.append(0)

            else:
                #la pub est encore possible dans le programme mais pas certaine
                if(context[3]<=0.95 and lastPub>=20 and (i-lastend)>=max(10,0.2*currentduree) and historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] not in ["très long","super long"] and context[5] != "film"):
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
                elif(context[3]<=0.95  and lastPub>=20 and (historyofpoints['programme'].loc[(historyofpoints.shape[0]-1)] in ["Téléréalité","Feuilleton"]) ):
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
                    per = context[3]
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
                        per = context[3]
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
                        per = context[3]
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
                        per = context[3]
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
                        per = context[3]
                        labels.append(2)
                    else:
                        labels.append(0)
                elif(historyofpoints['duree'].loc[(historyofpoints.shape[0]-1)] == "super long"):
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
                        per = context[3]
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
                per = context[3]
            elif(context[3] == 1):
                chaine = 'M6'
                if(i<20*60+30):
                    if(chaine == 'TF1'):
                        if(11.5*60<=i<=14*60 or 19.5*60<i<21*60):
                            Predictiontimer = 1
                        elif(context[6] == "très court"):
                            Predictiontimer = 0
                        elif(PTV['TITRE'].iloc[index_PTV] == 'Téléshopping'):
                            Predictiontimer = 5
                        elif(context[6] == "court"):
                            Predictiontimer = 5
                        elif(context[6] == "moyen"):
                            Predictiontimer = 5
                        elif(context[6] == "très long" or context[6] == "long"):
                            Predictiontimer = 15
                        else:
                            Predictiontimer = 5
                    elif(chaine =='M6'):
                        #Dépassement autorisé: Modulable en fonction de la position dans la journée si besoin
                        #Dépassement autorisé: Modulable en fonction de la position dans la journée si besoin
                        if(i<8*60+56):
                            Predictiontimer = 0
                        elif(13*60<i<14*60):
                            Predictiontimer = 5
                        elif(PTV['TITRE'].iloc[index_PTV] in ['M6 boutique']):
                            Predictiontimer = 0
                        elif(context[6] == "très court"):
                            Predictiontimer = 0
                        elif(context[6] == "court"):
                            Predictiontimer = 2
                        elif(context[6] == "moyen"):
                            Predictiontimer = 5
                        elif(context[6] == "très long"):
                            Predictiontimer = 5
                        elif(context[6] == 'long'):
                            Predictiontimer = 15
                        else:
                            Predictiontimer = 5
                else:
                    if(chaine == 'TF1'):
                        if(context[5] == 'Journal'):
                            if(i<20*60):
                                Predictiontimer = 10
                            else:
                                Predictiontimer = 0
                        elif(context[6] == "très court"):
                            Predictiontimer = 4
                        elif(context[6] == "court"):
                            Predictiontimer = 5
                        elif(context[6] == "moyen"):
                            Predictiontimer = 5
                        elif(context[6] == "très long"):
                            Predictiontimer = 5
                        elif(context[6] == 'long'):
                            Predictiontimer = 15
                        else:
                            Predictiontimer = 5
                    elif(chaine =='M6'):
                        #Dépassement autorisé: Modulable en fonction de la position dans la journée si besoin
                        #Dépassement autorisé: Modulable en fonction de la position dans la journée si besoin
                        if(context[6] == "très court"):
                            Predictiontimer = 15
                        elif(context[6] == "court"):
                            Predictiontimer = 15
                        elif(context[6] == "moyen"):
                            Predictiontimer = 15
                        elif(context[6] == "très long"):
                            Predictiontimer = 15
                        elif(context[6] == 'long'):
                            Predictiontimer = 15
                        else:
                            Predictiontimer = 5

            elif(context[3]>1):
                Predictiontimer -= 1
            else:
                pass


    return newPTV,historyofpoints,labels,error




#################################################
########### main with options ###################
#################################################



def main(argv):
    if(argv[0] == 'test'):
        files = os.listdir('/home/alexis/Bureau/Project/Datas/PTV/extracted')
        for file in files:
            f = ((file.split('.'))[0].split('_'))[2]
            os.system('python /home/alexis/Bureau/Project/scripts/PTV.py '+str(f))
    else:
        date = argv[0]
        d = "".join(date.split('-'))


        PTV,Points,proba = load_file(date)
        if(len(PTV) == 0):
            sys.exit(4)
            return 0
        new_PTV,historyofpoints,labels,error = make_newPTV(PTV,Points,proba,date)

        new_PTV['Heure'] = new_PTV['minute'].apply(lambda x: str(int(x/60))+':'+str(x%60))
        historyofpoints['Heure'] = historyofpoints['minute'].apply(lambda x: str(int(x/60))+':'+str(x%60))
        new_PTV.to_html('/home/alexis/Bureau/Project/results/newPTV/PTV/M6/new_PTV-'+date+'_M6.html')
        new_PTV.to_csv('/home/alexis/Bureau/Project/results/newPTV/PTV/M6/new_PTV-'+date+'_M6.csv',index=False)
        historyofpoints.to_html('/home/alexis/Bureau/Project/results/newPTV/historyofpts/M6/historyofpoints-'+date+'_M6.html')
        historyofpoints['labels'] = labels
        print(date,historyofpoints.shape,len(labels))

        historyofpoints.to_csv('/home/alexis/Bureau/Project/results/truemerge/M6/true_merge_'+str(date)+'_M6.csv',index=False)
        sys.exit(error)



if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
