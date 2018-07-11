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
import pickle
from catboost import CatBoostClassifier
import xgboost as xgb
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import time

#################################################
########### Global variables ####################
#################################################


#################################################
########### Important functions #################
#################################################
def encoding_partoftheday(x):
    if(x == 'nuit'):
        return 1
    if(x == 'fin de nuit'):
        return 2
    if(x == 'début de matinée'):
        return 3
    if(x == 'matinée'):
        return 4
    if(x == 'midi'):
        return 5
    if(x == 'après-midi'):
        return 6
    if(x == "fin d'après-midi"):
        return 7
    if(x == 'prime time'):
        return 8
    elif(type(x) == str):
        return 0
    else:
        return x

def encoding_partofprogramme(x):
    if(x == 'début de programme'):
        return 1
    if(x == 'milieu de programme'):
        return 2
    if(x == 'début de fin de programme'):
        return 3
    if(x == 'fin de programme'):
        return 4
    if(x == 'en dehors du programme'):
        return 5
    elif(type(x) == str):
        return 0
    else:
        return x
def encoding_per(x):
    if(x>=1):
        return 1
    if(x<1):
        return 0

def encoding_duree(x):
    if(x == 'très court'):
        return 1
    if(x == 'court'):
        return 2
    if(x == 'moyen'):
        return 3
    if(x == 'long'):
        return 4
    if(x == 'très long'):
        return 5
    if(x == 'super long'):
        return 6
    elif(type(x) == str):
        return 0
    else:
        return x

def encoding_chaine(x):
    if(x == 'TF1'):
        return 1
    if(x == 'M6'):
        return 2
    elif(type(x) == str):
        return 0
    else:
        return x


def process(df):
    df['Heure'] = df['minute'].apply(lambda x: str(int(x/60))+':'+str(x%60))
    df['partie de la journée'] = df['partie de la journée'].apply(lambda x: encoding_partoftheday(x))
    df['partie du programme'] = df['partie du programme'].apply(lambda x: encoding_partofprogramme(x))
    df["duree"] = df['duree'].apply(lambda x: encoding_duree(x))
    df['chaine'] = df['chaine'].apply(lambda x: encoding_chaine(x))
    if('per' in df.columns.values):
        df['per'] = df['per'].apply(lambda x: encoding_per(x))
    if('Heure' in df.columns.values):
        df['Time-h'] = df['Heure'].apply(lambda x: int((x.split(':'))[0]))
        df['Time-m'] = df['Heure'].apply(lambda x: int((x.split(':'))[1]))
        df = df.drop(['Heure'],axis = 1)
    if('programme' in df.columns.values):
        df = df.drop(['programme'],axis = 1)
    return df




#################################################
def load_file(date,c):
    try:
        if(c == 'TF1'):
            PTV = pd.read_csv('/home/alexis/Bureau/Project/Datas/PTV/extracted/IPTV_0192_'+date+'_TF1.csv')
            PTV['fin'] = PTV['debut']+PTV['DUREE']
            JOINDATE = "".join(date.split('-'))
            Points = pd.read_csv('/home/alexis/Bureau/Project/results/pred/pred_'+str(JOINDATE)+'_192.csv').values
            proba = pd.read_csv('/home/alexis/Bureau/Project/results/pred/pred_proba_'+str(JOINDATE)+'_192.csv').values
            return PTV,Points,proba
        elif(c== 'M6'):
            PTV = pd.read_csv('/home/alexis/Bureau/Project/Datas/PTV/extracted/IPTV_0118_'+date+'_M6.csv')
            PTV['fin'] = PTV['debut']+PTV['DUREE']
            JOINDATE = "".join(date.split('-'))
            Points = pd.read_csv('/home/alexis/Bureau/Project/results/pred/pred_'+str(JOINDATE)+'_118.csv').values
            proba = pd.read_csv('/home/alexis/Bureau/Project/results/pred/pred_proba_'+str(JOINDATE)+'_118.csv').values
            return PTV,Points,proba
    except:
        print("Fichier Non existant")
        return [],[],[]

def init_newPTV(PTV,chaine):
    if(chaine == 'TF1'):
        #Initialisation du NewPTV
        newPTV = pd.DataFrame()
        newPTV['minute'] = [180]
        newPTV['TITRE'] = 'Programmes de la nuit'
        newPTV['Change Point'] = 'Non'
        newPTV['pourcentage vu'] = 0
        newPTV['Évenement'] = 'Début de Détection'
        return newPTV
    if(chaine == 'M6'):
        newPTV = pd.DataFrame()
        newPTV['minute'] = [180]
        newPTV['TITRE'] = 'M6 Music'
        newPTV['Change Point'] = 'Non'
        newPTV['pourcentage vu'] = 0
        newPTV['Évenement'] = 'Début de Détection'
        return newPTV

def init_history(chaine):
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
    h["chaine"]= chaine
    h['CLE-FORMAT'] = 0
    h['CLE-GENRE'] = 0
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

def categorize_pub(name,debut,duree,titre,chaine):
    if(chaine == 'TF1'):
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

    if(chaine == 'M6'):
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
        elif(duree in ["très court","court"]):
            return 0
        else:
            return 10



def categorize_programme(programme,chaine):
    p = []
    p.append(categorize_type(programme['description programme']))
    p.append(categorize_duree(programme['DUREE']))
    p.append(categorize_pub(p[0],programme['debut'],p[-1],programme['TITRE'],chaine))
    return p






def get_context(i,programme,Points,lastCP,lastPub,lastend,currentduree,planifiedend,Pubinhour,probas,nbpub,chaine,per):
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
    p = categorize_programme(programme,chaine)
    for j in range(len(p)):#3
        context.append(p[j])
    context.append(lastCP)
    context.append(lastPub)
    context.append(lastend)
    context.append(currentduree)
    context.append(Pubinhour)
    context.append(probas[i-183][0])
    context.append(context[7]-nbpub)
    context.append(chaine)
    context.append(programme['CLE-FORMAT'])
    context.append(programme['CLE-GENRE'])
    #context.append(per)
    return context

def load_models():
    XGB = []
    XGB.append(pickle.load(open("model_PTV/XGB1.pickle.dat", "rb")))
    XGB.append(pickle.load(open("model_PTV/XGB2.pickle.dat", "rb")))

    CatBoost = []
    CatBoost.append(CatBoostClassifier().load_model(fname="model_PTV/catboostmodel1"))
    CatBoost.append(CatBoostClassifier().load_model(fname="model_PTV/catboostmodel2"))

    rf = pickle.load(open("model_PTV/RF.pickle.dat", "rb"))
    dt = pickle.load(open("model_PTV/DT.pickle.dat", "rb"))

    return XGB,CatBoost,rf,dt

def make_newPTV(PTV,Points,proba,chaine):
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
    wait = 4
    error = 0
    per = 1
    if(chaine == 'TF1'):
        importantpts = [[13*60,"Journal"],[20*60,"Journal"]]
        help = [6*60+30,8*60+25,8*60+30,10*60+55,12*60+50,19*60+50,11*60+52]
    if(chaine == 'M6'):
        importantpts = [[12*60+45,"Le 12.45"],[19*60+45,"Le 19.45"]]
        help = [6*60,6*60+50,8*60+57,10*60]
    index_ipts = 0
    importantpts.append([(int(PTV['HEURE'].loc[PTV.shape[0]-1].split(':')[0])+24)*60+int(PTV['HEURE'].loc[PTV.shape[0]-1].split(':')[1]),PTV['TITRE'].loc[PTV.shape[0]-1]])
    ######################
    newPTV = init_newPTV(PTV,chaine)
    historyofpoints = init_history(chaine)
    labels = [0]
    #########init Classifier#############
    XGB,CatBoost,rf,dt = load_models()
    ####################################
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
        context = get_context(i,PTV.iloc[index_PTV],Points,lastCP,lastPub,lastend,currentduree,planifiedend,Pubinhour,proba,nbpub,chaine,per)
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


        ###### Let's verify that the algo is not doing a crappy predicitions and if this the case, clean his historic #####
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
                        #here you go, it's the next one...But it's far far away
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
                        #we can now keep going throw the process like before
                        #we just add a line to the history to say that a reset occured
                        if(index_ipts == 0):
                            print("erreur sur l'après midi")
                        elif(index_ipts == 1):
                            print("erreur sur la matinée")
                        else:
                            print("erreur sur la soirée")
                        error+=1*10**index_ipts
                        newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'non',context[3],"--HARD RESET OF ALGORITHM--(out of programme)"]
                        index_ipts+=1
                    else:
                        index_ipts+=1

        if(context[2]):
            historyofpoints.loc[historyofpoints.shape[0]] = context
            if(lastCP < min(4,currentduree)):
                labels.append(0)
                index_CP+=1
                continue
            else:
                X = process(pd.DataFrame([context],index=[0],columns=['minute','partie de la journée','Change Point','pourcentage','partie du programme','programme','duree','nombre de pub potentiel','lastCP','lastPub','lastend','currentduree','Pubinhour','probability of CP','nb de pubs encore possible','chaine','CLE-FORMAT','CLE-GENRE'])).values #,'per'
                res1 = CatBoost[0].predict_proba(X)
                res2 = CatBoost[1].predict_proba(X)
                res3 = XGB[0].predict(xgb.DMatrix(X), ntree_limit= XGB[0].best_ntree_limit)
                res4 = XGB[1].predict(xgb.DMatrix(X), ntree_limit= XGB[1].best_ntree_limit)
                res5 = rf.predict_proba(X)
                res6 = dt.predict_proba(X)
                res = [(res1[0][0]+res2[0][0]+res3[0][0]+res4[0][0]+res5[0][0]+res6[0][0])/6,(res1[0][1]+res2[0][1]+res3[0][1]+res4[0][1]+res5[0][1]+res6[0][1])/6,(res1[0][2]+res2[0][2]+res3[0][2]+res4[0][2]+res5[0][2]+res6[0][2])/6]
                cla = np.argmax(res)

                if(cla == 1 and context[14]==0):
                    cla = 0
                if(cla == 2 and context[3]<0.5 and context[11]>30):
                    cla = 0
                if(cla == 2 and context[3]<0.9 and context[11]>=180):
                    cla = 0
                if( cla == 2 and PTV['TITRE'].loc[index_PTV] == 'Programmes de la nuit' and context[3]<1):
                    cla = 0

                if(cla == 1):
                    newPTV.loc[newPTV.shape[0]] = [i%1440,"publicité",'oui',context[3],"publicité dans un programme"]
                    lastCP=0
                    lastPub = 0
                    Pubinhour+=4
                    nbpub+=1
                    wait = 4
                    labels.append(1)
                elif(cla == 2):
                    newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"fin d'un programme"]
                    lastend = i
                    lastCP=0
                    index_PTV += 1
                    index_PTV = index_PTV%(PTV.shape[0])
                    currentduree = PTV['DUREE'].iloc[index_PTV]
                    planifiedend = (lastend + currentduree)
                    Predictiontimer = 200
                    nbpub = 0
                    wait = 5
                    per = context[3]
                    labels.append(2)
                else:
                    labels.append(0)



            index_CP+=1
        elif(i in help):
            historyofpoints.loc[historyofpoints.shape[0]] = context
            if(lastCP < min(4,currentduree)):
                labels.append(0)
                continue
            else:
                X = process(pd.DataFrame([context],index=[0],columns=['minute','partie de la journée','Change Point','pourcentage','partie du programme','programme','duree','nombre de pub potentiel','lastCP','lastPub','lastend','currentduree','Pubinhour','probability of CP','nb de pubs encore possible','chaine','CLE-FORMAT','CLE-GENRE'])).values #,'per'
                res1 = CatBoost[0].predict_proba(X)
                res2 = CatBoost[1].predict_proba(X)
                res3 = XGB[0].predict(xgb.DMatrix(X), ntree_limit= XGB[0].best_ntree_limit)
                res4 = XGB[1].predict(xgb.DMatrix(X), ntree_limit= XGB[1].best_ntree_limit)
                res5 = rf.predict_proba(X)
                res6 = dt.predict_proba(X)
                res = [(res1[0][0]+res2[0][0]+res3[0][0]+res4[0][0]+res5[0][0]+res6[0][0])/6,(res1[0][1]+res2[0][1]+res3[0][1]+res4[0][1]+res5[0][1]+res6[0][1])/6,(res1[0][2]+res2[0][2]+res3[0][2]+res4[0][2]+res5[0][2]+res6[0][2])/6]
                cla = np.argmax(res)

                if(cla == 1 and context[14]==0):
                    cla = 0
                if(cla == 2 and context[3]<0.5 and context[11]>30):
                    cla = 0
                if(cla == 2 and context[3]<0.9 and context[11]>=180):
                    cla = 0
                if( cla == 2 and PTV['TITRE'].loc[index_PTV] == 'Programmes de la nuit' and context[3]<1):
                    cla = 0

                if(cla == 1):
                    newPTV.loc[newPTV.shape[0]] = [i%1440,"publicité",'oui',context[3],"publicité dans un programme"]
                    lastCP=0
                    lastPub = 0
                    Pubinhour+=4
                    nbpub+=1
                    wait = 4
                    labels.append(1)
                elif(cla == 2):
                    newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'oui',context[3],"fin d'un programme"]
                    lastend = i
                    lastCP=0
                    index_PTV += 1
                    index_PTV = index_PTV%(PTV.shape[0])
                    currentduree = PTV['DUREE'].iloc[index_PTV]
                    planifiedend = (lastend + currentduree)
                    Predictiontimer = 200
                    nbpub = 0
                    wait = 5
                    per = context[3]
                    labels.append(2)
                else:
                    labels.append(0)


        else:
            #labels.append(0)
            #Not a Change Point, we'll just check that nothing is wrong in the PTV at this time
            if(Predictiontimer <= 0):
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
                #Dépassement autorisé: Modulable en fonction de la position dans la journée si besoin
                # C'est sur ces valeurs que l'on va jouer pour avoir le meilleur PTV possible
                # Plus les valeurs sont grandes, plus on fait confiance a l'algo
                # Il est important de bien découper la journée celon les périodes horaires que l'on qualifie
                # de "sous tension" si plusieurs programmes courts se succédent. Bien évidement une telle analyse sera
                #plus tard fait automatiquement.
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
            elif(context[3]>1):
                Predictiontimer -= 1
            else:
                pass
    return newPTV,historyofpoints,labels,error




#################################################
########### main with options ###################
#################################################



def main(argv):
    t = time.time()
    if(len(argv) ==0):
        argv = ['2015']

    if(len(argv) == 1):
        relecture = True
        EPSILON = 1e-15
        err = 0
        m = 0
        err_TF1 = 0
        m_TF1 = 0
        err_M6 = 0
        m_M6 = 0

        files = os.listdir('/home/alexis/Bureau/Project/Datas/PTV/extracted')
        for file in files:
            f = ((file.split('.'))[0].split('_'))[2]
            c = ((file.split('.'))[0].split('_'))[-1]
            if(f=='2017-12-20' or (f in ['2017-12-09','2017-12-06','2018-02-22'] and c=='TF1') or (f in ['2018-02-22'] and c=='M6') or  f.split('-')[0] == str(argv[0])):
                #or (f in ['2018-02-22'] and c=='M6')
                pass
            elif(c ==''):
                pass
            else:

                print(c)
                l = os.system('python /home/alexis/Bureau/Project/scripts/PTVwithTruemerge.py '+str(c)+' '+str(f))
                if(l/256>0 and relecture):
                    print("Utilisation de la relecture",f,c)
                    l = os.system('python /home/alexis/Bureau/Project/scripts/newidea.py '+str(c)+' '+str(f))
                    print(l)
                    if(l/256>=1):
                        print("Utilisation de l'arbre de décision",f,c)
                        p = os.system('python /home/alexis/Bureau/Project/scripts/PTV'+str(c)+'.py '+str(f))
                        if(p/256>0):
                            print("AUCUNE DÉCISION NE CONVIENT",f,c)
                        else:
                            l = p

                else:
                    pass
                if(l/256 == 4):
                    pass
                else:
                    err += l/256
                    m+=3
                    if(c == 'M6'):
                        err_M6 += l/256
                        m_M6 += 3
                    if(c == 'TF1'):
                        err_TF1 += l/256
                        m_TF1 += 3

            print(err)
        print(m)
        err = int(err/100)+int((err%100)/10)+int((err%10))
        print("score Total",1-(err/(m+EPSILON)))
        print("score TF1",1-(err_TF1/(m_TF1+EPSILON)))
        print("score M6",1-(err_M6/(m_M6+EPSILON)))
        print("temps de calcul: ",time.time()-t)
    else:
        chaine = argv[0]
        date = argv[1]
        d = "".join(date.split('-'))
        PTV,Points,proba = load_file(date,chaine)
        if(len(PTV) == 0):
            sys.exit(4)
            return 0
        new_PTV,historyofpoints,labels,error = make_newPTV(PTV,Points,proba,chaine)
        new_PTV['Heure'] = new_PTV['minute'].apply(lambda x: str(int(x/60))+':'+str(x%60))
        historyofpoints['Heure'] = historyofpoints['minute'].apply(lambda x: str(int(x/60))+':'+str(x%60))
        new_PTV.to_html('/home/alexis/Bureau/Project/results/ptvbyml/PTV/new_PTV-'+date+'_'+chaine+'.html')
        new_PTV.to_csv('/home/alexis/Bureau/Project/results/ptvbyml/csv/new_PTV-'+date+'_'+chaine+'.csv',index=False)
        historyofpoints['labels'] = labels
        historyofpoints.to_html('/home/alexis/Bureau/Project/results/ptvbyml/historyofpoints/historyofpoints-'+date+'_'+chaine+'.html')
        #historyofpoints.to_csv('/home/alexis/Bureau/Project/results/truemerge/'+chaine+'/true_merge_'+str(date)+'_'+chaine+'.csv',index=False)
        print(chaine,date,historyofpoints.shape,len(labels))
        sys.exit(error)
if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
