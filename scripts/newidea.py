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
from sklearn import linear_model
import datetime
#################################################
########### Global variables ####################
#################################################
THRESHOLD = 0.46

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

def init_history(chaine,PTV):
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
    if(cp[i-183]>THRESHOLD):
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

def categorize_pub(name,debut,duree,titre,chaine,PTV,index_PTV):
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



def categorize_programme(programme,chaine,PTV,index_PTV):
    p = []
    p.append(categorize_type(programme['description programme']))
    p.append(categorize_duree(programme['DUREE']))
    p.append(categorize_pub(p[0],programme['debut'],p[-1],programme['TITRE'],chaine,PTV,index_PTV))
    return p






def get_context(i,programme,Points,lastCP,lastPub,lastend,currentduree,planifiedend,Pubinhour,probas,nbpub,chaine,per,PTV,index_PTV,date):
    #we create a list with different notes to understand the context
    # minute of the point and its situation in the day
    context = [i]
    context.append(find_partofday(i))
    # Is the Point a Change Point
    context.append(find_ifChangePoint(i,probas))
    # Where is the Point in the programme:
    seen_percentage = (i-lastend)/currentduree
    context.append(seen_percentage)
    context.append(find_position(seen_percentage))
    # which type of programme we are watching
    p = categorize_programme(programme,chaine,PTV,index_PTV)
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
    day = datetime.datetime(int(date.split('-')[0]), int(date.split('-')[1]), int(date.split('-')[2]))
    context.append(day.weekday())
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
    gb = pickle.load(open("model_PTV/GradientBoostingClassifier.pickle.dat", "rb"))
    logistic = pickle.load(open('model_PTV/logistic_regression.sav', "rb"))
    return XGB,CatBoost,rf,dt,gb,logistic



def make_predictedPTV(actu,PTV,Points,chaine,j,index_PTV,index_CP,lastCP,lastPub,lastend,currentduree,planifiedend,Pubinhour,proba,nbpub,XGB,CatBoost,rf,dt,gb,logistic,Predictiontimer,per,context,date):
    #Initialisation des Variables
    starting_index = index_PTV
    verbose = False
    index_CP = index_CP
    index_PTV = index_PTV
    ##########################
    a_0 = 200                                      #faute sur le point de contrôle
    a_1 = 10                                       #Coefficient pour le fait de manquer une publicité
    a_2 = 20                                       #Coeffeicient de la pénalité si le journal commence trop tôt
    a_3 = 2                                        #Coeffeicent de péanlité pour ne pas détecter une fin de programme
    a_4 = 20                                       #Paramètre de régularisation -- utilisation des points les plus probables.
    a_5 = 0                                        #Pénalitasion du la valeur absolue de l'éloignement du point prévu de fin.
    error_type_1 = 0                               # erreur sur les programmes de la demi journée
    error_type_2 = 0                               # erreur sur les Points de Contrôle
    error_type_3 = 0                               # Erreur de régularisation
    error_type_4 = 0                               # Erreur du noeud
    ##########################
    Predictiontimer = Predictiontimer
    Pubinhour = Pubinhour
    lastCP = lastCP
    lastPub= lastPub
    lastend = lastend
    currentduree = currentduree
    planifiedend = planifiedend
    begin = True
    nbpub = nbpub
    Recall = 1
    wait = 4
    error = 0
    per = per
    context = context
    if(chaine == 'TF1'):
        importantpts = [[13*60,"Journal"],[20*60,"Journal"]]
        help = [6*60+30,8*60+25,8*60+30,10*60+55,12*60+50,11*60+52]
    if(chaine == 'M6'):
        importantpts = [[12*60+45,"Le 12.45"],[19*60+45,"Le 19.45"]]
        help = [6*60,6*60+50,8*60+57,10*60]

    if(j> importantpts[0][0]):
        end = importantpts[1][0]
        index_ipts = 1
    else:
        end = importantpts[0][0]
        index_ipts = 0
    #####################################
    if(actu == 0):
        error_type_4+= abs(((context[13]-0.46)*a_4)*context[2])
    elif(actu == 1):
        lastCP=0
        lastPub = 0
        Pubinhour+=4
        nbpub+=1
    elif(actu == 2):
        lastend = j
        lastCP=0
        index_PTV += 1
        index_PTV = index_PTV%(PTV.shape[0])
        currentduree = PTV['DUREE'].iloc[index_PTV]
        planifiedend = (lastend + currentduree)
        Predictiontimer = 200
        nbpub = 0
        error_type_4+= abs(context[14])*a_1
        if(context[14]<0):
            error_type_4+= abs(context[14])*50
        error_type_4 += abs((1-context[3])*context[11])*a_5
        per = context[3]
        if(PTV['TITRE'].iloc[index_PTV] == importantpts[index_ipts][1]):
            if(importantpts[index_ipts][0]-j>13):
                error_type_4+= (abs(importantpts[index_ipts][0]-j))*a_2
            else:
                error_type_4 -= abs(importantpts[index_ipts][0]-j)

    #####################################
    for i in range(j+1,end+1):
        ############Update loss################
        a_1 = 10*(0.9**(index_PTV-starting_index))     #Coefficient pour le fait de manquer une publicité
        a_2 = 20*(0.9**(index_PTV-starting_index))     #Coeffeicient de la pénalité si le journal commence trop tôt
        a_3 = 1 *(0.9**(index_PTV-starting_index))     #Coeffeicent de péanlité pour ne pas détecter une fin de programme
        a_4 = 20*(0.9**(index_PTV-starting_index))     #Paramètre de régularisation -- utilisation des points les plus probables.
        a_5 = 0 *(0.9**(index_PTV-starting_index))     #Pénalitasion du la valeur absolue de l'éloignement du point prévu de fin.
        #Update time of commercials (Reset)
        if(i%60 == 0):
            Pubinhour = 0
        #Update timmers
        lastPub+=1
        lastCP+=1
        if(index_ipts==len(importantpts)):
            index_ipts-=1
        #let's get the context:
        context = get_context(i,PTV.iloc[index_PTV],Points,lastCP,lastPub,lastend,currentduree,planifiedend,Pubinhour,proba,nbpub,chaine,per,PTV,index_PTV,date)
        #Sur M6 il y a 16 minutes de pub entre deux films!!!!!!!!!!!!.....!!!!!!!....!!.!.!.!.!....!.!...!..!.!.!.!
        if(PTV['GENRESIMPLE'].iloc[index_PTV].split(' ')[0] == PTV['GENRESIMPLE'].iloc[index_PTV-1].split(' ')[0] and PTV['GENRESIMPLE'].iloc[index_PTV].split(' ')[0] == 'Téléfilm'
            and (i-lastend)<2 and Recall > 0 and per<0.97 and chaine == 'M6'):
            print(i,PTV['GENRESIMPLE'].iloc[index_PTV])
            lastend = i+5
            lastPub = -25
            Recall -= 0.5
        elif((i-lastend)<2 and Recall > 0 and per<0.95 and chaine == 'M6' and 15*60<i<16*60):
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
                    error_type_2+= a_0

                else:
                    # OMG the ALGO IS RIGHT...here is a candy, let's rest a litle just in case...we never know....
                    nbpub = 0




            else:
                #maybe it's the next programme so calme the fuck down!
                if(PTV['TITRE'].iloc[(index_PTV+1)%(PTV.shape[0])] == importantpts[index_ipts][1]):
                    if(planifiedend-i<10):
                        error_type_2 += abs(planifiedend-i)*5
                        per = context[3]
                        error_type_1+= abs(context[14])*a_1
                        error_type_1 += abs((1-context[3])*context[11])*a_5
                        if(context[14]<0):
                            error_type_1+= abs(context[14])*50

                        index_ipts+=1
                    else:

                        error_type_2+= a_0


                else:
                    #well the programme is wrong, and we are not even close to it, let's terminate this thing before it goes completly south. REBOOT The algo, erase the memory, just like in Westworld.
                    #BUT FIRST LET'S VERIFY THAT THERE IS INDEED AN IMPORTANT PROGRAMME THAT DAY...Don't go fuck everything up for no reason
                    l = PTV.index[(PTV['TITRE']==importantpts[index_ipts][1]) & (PTV['debut'] == i)].tolist()
                    if(len(l)>0):

                        error_type_2+= a_0

                    else:
                        print("PAS DE POINT DE CONTRÔLE POUR CETTE PARTIE DE LA JOURNÉE")
                        index_ipts+=1

        elif(context[2]):
            if(lastCP < min(6,currentduree/2)):
                index_CP+=1
                if(lastCP == 1):
                    continue
                else:
                    error_type_3+= abs((context[13]-0.46)*a_4*(0.95*(4-lastCP)))

            else:
                X = process(pd.DataFrame([context],index=[0],columns=['minute','partie de la journée','Change Point','pourcentage','partie du programme','programme','duree','nombre de pub potentiel','lastCP','lastPub','lastend','currentduree','Pubinhour','probability of CP','nb de pubs encore possible','chaine','CLE-FORMAT','CLE-GENRE','day'])).values #,'per'
                res1 = CatBoost[0].predict_proba(X)
                res2 = CatBoost[1].predict_proba(X)
                res3 = XGB[0].predict(xgb.DMatrix(X), ntree_limit= XGB[0].best_ntree_limit)
                res4 = XGB[1].predict(xgb.DMatrix(X), ntree_limit= XGB[1].best_ntree_limit)
                res5 = rf.predict_proba(X)
                res6 = gb.predict_proba(X)
                res7 = dt.predict_proba(X)
                res = [(res1[0][0]+res2[0][0]+res3[0][0]+res4[0][0]+res5[0][0]+res6[0][0])/6,(res1[0][1]+res2[0][1]+res3[0][1]+res4[0][1]+res5[0][1]+res6[0][1])/6,(res1[0][2]+res2[0][2]+res3[0][2]+res4[0][2]+res5[0][2]+res6[0][2])/6]
                y_pred = [(res1[0][0]+res2[0][0])*0.5,(res1[0][1]+res2[0][1])*0.5,(res1[0][2]+res2[0][2])*0.5]
                y_pred2 = [(res3[0][0]+res4[0][0])*0.5,(res3[0][1]+res4[0][1])*0.5,(res3[0][2]+res4[0][2])*0.5]
                X = pd.concat([pd.DataFrame(y_pred).T,pd.DataFrame(y_pred2).T,pd.DataFrame(res7),pd.DataFrame(res6),pd.DataFrame(res5)],axis = 1)
                X = X.replace([np.inf, -np.inf], np.nan)
                X = X.fillna(1)
                X = X.values
                res = logistic.predict_proba(X)
                cla = np.argmax(res)

                if(cla == 1 and context[14]==0):
                    cla = 1
                if(cla == 2 and context[3]<0.5 and context[11]>=20):
                    cla = 0
                if(cla == 2 and context[3]<0.9 and context[11]>=180):
                    cla = 0
                if(context[3]>1):
                    cla = 2
                if(cla == 1 and context[9]<0):
                    cla = 0
                if(cla == 1):
                    lastCP=0
                    lastPub = 0
                    Pubinhour+=4
                    nbpub+=1
                    wait = 4
                elif(cla == 2):
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
                    error_type_1+= abs(context[14])*a_1
                    error_type_1 += abs((1-context[3])*context[11])*a_5
                    if(context[14]<0):
                        error_type_1+= abs(context[14])*50
                    if(PTV['TITRE'].iloc[index_PTV] == importantpts[index_ipts][1]):
                        if(importantpts[index_ipts][0]-i>13):
                            error_type_2+= (abs(importantpts[index_ipts][0]-i))*a_2
                        else:

                            if(chaine == 'M6'):
                                error_type_2 -= importantpts[index_ipts][0]-i
                            else:
                                pass

                else:
                    error_type_3+= abs((context[13]-0.46)*a_4)



            index_CP+=1
        elif(i in help):
            if(lastCP < min(6,currentduree)):
                error_type_3-= 5
            else:
                X = process(pd.DataFrame([context],index=[0],columns=['minute','partie de la journée','Change Point','pourcentage','partie du programme','programme','duree','nombre de pub potentiel','lastCP','lastPub','lastend','currentduree','Pubinhour','probability of CP','nb de pubs encore possible','chaine','CLE-FORMAT','CLE-GENRE','day'])).values #,'per'
                res1 = CatBoost[0].predict_proba(X)
                res2 = CatBoost[1].predict_proba(X)
                res3 = XGB[0].predict(xgb.DMatrix(X), ntree_limit= XGB[0].best_ntree_limit)
                res4 = XGB[1].predict(xgb.DMatrix(X), ntree_limit= XGB[1].best_ntree_limit)
                res5 = rf.predict_proba(X)
                res6 = gb.predict_proba(X)
                res7 = dt.predict_proba(X)
                res = [(res1[0][0]+res2[0][0]+res3[0][0]+res4[0][0]+res5[0][0]+res6[0][0])/6,(res1[0][1]+res2[0][1]+res3[0][1]+res4[0][1]+res5[0][1]+res6[0][1])/6,(res1[0][2]+res2[0][2]+res3[0][2]+res4[0][2]+res5[0][2]+res6[0][2])/6]
                y_pred = [(res1[0][0]+res2[0][0])*0.5,(res1[0][1]+res2[0][1])*0.5,(res1[0][2]+res2[0][2])*0.5]
                y_pred2 = [(res3[0][0]+res4[0][0])*0.5,(res3[0][1]+res4[0][1])*0.5,(res3[0][2]+res4[0][2])*0.5]
                X = pd.concat([pd.DataFrame(y_pred).T,pd.DataFrame(y_pred2).T,pd.DataFrame(res7),pd.DataFrame(res6),pd.DataFrame(res5)],axis = 1)
                X = X.replace([np.inf, -np.inf], np.nan)
                X = X.fillna(1)
                X = X.values
                res = logistic.predict_proba(X)
                cla = np.argmax(res)

                if(cla == 1 and context[14]==0):
                    cla = 1
                if(cla == 2 and context[3]<0.5 and context[11]>=20):
                    cla = 0
                if(cla == 2 and context[3]<0.9 and context[11]>=180):
                    cla = 0
                if(context[3]>1):
                    cla = 2
                if(cla == 1 and context[9]<0):
                    cla = 0


                if(cla == 1):
                    lastCP=0
                    lastPub = 0
                    Pubinhour+=4
                    nbpub+=1
                    wait = 4
                elif(cla == 2):
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
                    error_type_1+= abs(context[14])*a_1
                    error_type_1 += abs((1-context[3])*context[11])*a_5
                    if(context[14]<0):
                        error_type_1+= abs(context[14])*50
                    if(PTV['TITRE'].iloc[index_PTV] == importantpts[index_ipts][1]):
                        if(importantpts[index_ipts][0]-i>13):
                            error_type_2+= (abs(importantpts[index_ipts][0]-i))*a_2
                        else:
                            if(chaine == 'M6'):
                                error_type_2 -= importantpts[index_ipts][0]-i
                            else:
                                pass

                else:
                    error_type_3-=5


        else:
            #labels.append(0)
            #Not a Change Point, we'll just check that nothing is wrong in the PTV at this time
            if(Predictiontimer <= 0):
                l = currentduree
                lastend = i
                lastCP=0
                index_PTV += 1
                index_PTV = index_PTV%(PTV.shape[0])
                currentduree = PTV['DUREE'].iloc[index_PTV]
                planifiedend = (lastend + currentduree)
                Predictiontimer = 200
                nbpub = 0
                error_type_1+= abs(context[14])*a_1
                if(context[14]<0):
                    error_type_1+= abs(context[14])*50
                error_type_1 += abs((1-context[3])*context[11])*a_5
                error_type_1 += max(currentduree/5,4)*a_3
                per = context[3]
                if(PTV['TITRE'].iloc[index_PTV] == importantpts[index_ipts][1]):
                    if(importantpts[index_ipts][0]-i>13):
                        error_type_2+= (abs(importantpts[index_ipts][0]-i))*a_2
                    else:
                        if(chaine == 'M6'):
                            error_type_2 -= importantpts[index_ipts][0]-i
                        else:
                            pass

            elif(context[3] == 1):
                #Dépassement autorisé: Modulable en fonction de la position dans la journée si besoin
                # C'est sur ces valeurs que l'on va jouer pour avoir le meilleur PTV possible
                # Plus les valeurs sont grandes, plus on fait confiance a l'algo
                # Il est important de bien découper la journée celon les périodes horaires que l'on qualifie
                # de "sous tension" si plusieurs programmes courts se succédent. Bien évidement une telle analyse sera
                #plus tard fait automatiquement.
                if(chaine == 'TF1'):
                    if(context[5] == 'Journal'):
                        if(i<20*60):
                            Predictiontimer = 10
                        else:
                            Predictiontimer = 0
                    elif(11.5*60<=i<=14*60 or 19.5*60<i<21*60):
                        Predictiontimer = 0
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
                    if(17*60<i<19.5*60):
                        Predictiontimer += 5
                elif(chaine =='M6'):
                    #Dépassement autorisé: Modulable en fonction de la position dans la journée si besoin
                    #Dépassement autorisé: Modulable en fonction de la position dans la journée si besoin
                    if(i<9*60+56):
                        Predictiontimer = 2
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
    error+= error_type_1 + error_type_2 + error_type_3 +error_type_4
    return error,[error_type_1, error_type_2,error_type_3,error_type_4]




def make_predictedPTV2(actu,PTV,Points,chaine,j,index_PTV,index_CP,lastCP,lastPub,lastend,currentduree,planifiedend,Pubinhour,proba,nbpub,XGB,CatBoost,rf,dt,gb,logistic,Predictiontimer,per,context,date):
    #Initialisation des Variables
    starting_index = index_PTV
    verbose = False
    index_CP = index_CP
    index_PTV = index_PTV
    ##########################
    a_0 = 150                                      #faute sur le point de contrôle
    a_1 = 10                                       #Coefficient pour le fait de manquer une publicité
    a_2 = 10                                       #Coeffeicient de la pénalité si le journal commence trop tôt
    a_3 = 2                                        #Coeffeicent de péanlité pour ne pas détecter une fin de programme
    a_4 = 40                                       #Paramètre de régularisation -- utilisation des points les plus probables.
    a_5 = 0                                        #Pénalitasion du la valeur absolue de l'éloignement du point prévu de fin.
    error_type_1 = 0                               # erreur sur les programmes de la demi journée
    error_type_2 = 0                               # erreur sur les Points de Contrôle
    error_type_3 = 0                               # Erreur de régularisation
    error_type_4 = 0                               # Erreur du noeud
    ##########################
    Predictiontimer = Predictiontimer
    Pubinhour = Pubinhour
    lastCP = lastCP
    lastPub= lastPub
    lastend = lastend
    currentduree = currentduree
    planifiedend = planifiedend
    begin = True
    nbpub = nbpub
    Recall = 1
    wait = 4
    error = 0
    per = per
    context = context
    if(chaine == 'TF1'):
        importantpts = [[(int(PTV['HEURE'].loc[PTV.shape[0]-1].split(':')[0])+24)*60+int(PTV['HEURE'].loc[PTV.shape[0]-1].split(':')[1]),PTV['TITRE'].loc[PTV.shape[0]-1]]]
        help = [6*60+30,8*60+25,8*60+30,10*60+55,12*60+50,11*60+52]
    if(chaine == 'M6'):
        importantpts = [[(int(PTV['HEURE'].loc[PTV.shape[0]-1].split(':')[0])+24)*60+int(PTV['HEURE'].loc[PTV.shape[0]-1].split(':')[1]),PTV['TITRE'].loc[PTV.shape[0]-1]]]
        help = [6*60,6*60+50,8*60+57,10*60]
    end = importantpts[0][0]

    index_ipts = 0
    #####################################
    if(actu == 0):
        error_type_4+= abs(((context[13]-0.46)*a_4)*context[2])
    elif(actu == 1):
        lastCP=0
        lastPub = 0
        Pubinhour+=4
        nbpub+=1
    elif(actu == 2):
        lastend = j
        lastCP=0
        index_PTV += 1
        index_PTV = index_PTV%(PTV.shape[0])
        currentduree = PTV['DUREE'].iloc[index_PTV]
        planifiedend = (lastend + currentduree)
        Predictiontimer = 200
        nbpub = 0
        error_type_4+= abs(context[14])*a_1
        if(context[14]<0):
            error+= abs(context[14])*a_4
        error_type_4 += abs((1-context[3])*context[11])*a_5
        per = context[3]
        if(PTV['TITRE'].iloc[index_PTV] == importantpts[index_ipts][1]):
            if(importantpts[index_ipts][0]-j>13):
                error_type_4+= (abs(importantpts[index_ipts][0]-j))*a_2
            else:
                error_type_4 -= abs(importantpts[index_ipts][0]-j)

    #####################################
    for i in range(j+1,min(end+13,1620)):
        ############Update loss################
        a_1 = 10*(0.9**(index_PTV-starting_index))     #Coefficient pour le fait de manquer une publicité
        a_2 = 20*(0.9**(index_PTV-starting_index))     #Coeffeicient de la pénalité si le journal commence trop tôt
        a_3 = 1 *(0.9**(index_PTV-starting_index))     #Coeffeicent de péanlité pour ne pas détecter une fin de programme
        a_4 = 20*(0.9**(index_PTV-starting_index))     #Paramètre de régularisation -- utilisation des points les plus probables.
        a_5 = 0 *(0.9**(index_PTV-starting_index))     #Pénalitasion du la valeur absolue de l'éloignement du point prévu de fin.
        #Update time of commercials (Reset)
        if(i%60 == 0):
            Pubinhour = 0
        #Update timmers
        lastPub+=1
        lastCP+=1
        if(index_ipts==len(importantpts)):
            index_ipts-=1
        #let's get the context:
        context = get_context(i,PTV.iloc[index_PTV],Points,lastCP,lastPub,lastend,currentduree,planifiedend,Pubinhour,proba,nbpub,chaine,per,PTV,index_PTV,date)
        ###### Let's verify that the algo is not doing a crappy predicitions and if this the case, clean his historic #####
        if(i == importantpts[index_ipts][0]):
            #### we are at an important point, let's now see what the algo has predict
            if(PTV['TITRE'].iloc[index_PTV] == importantpts[index_ipts][1]):
                #Well he doesn't have the programme wrong, that's a good start
                #let's now find out if we are at a logical point of the programme
                if(i-lastend>13):
                    #Wellllll, the programme began way too early...something went wrong before...Let's rest for now, we'll correct the algo later
                    error_type_2+= a_0

                else:
                    # OMG the ALGO IS RIGHT...here is a candy, let's rest a litle just in case...we never know....
                    nbpub = 0




            else:
                #maybe it's the nexTirage du Lotot programme so calme the fuck down!
                if(PTV['TITRE'].iloc[index_PTV+1] == importantpts[index_ipts][1]):
                    if(planifiedend-i<13):
                        per = context[3]
                        error_type_1+= abs(context[14])*a_1
                        error_type_1 += abs((1-context[3])*context[11])*a_5
                        if(context[14]<0):
                            error_type_1+= abs(context[14])*50
                        index_ipts+=1
                    else:

                        error_type_2+= a_0


                else:
                    #well the programme is wrong, and we are not even close to it, let's terminate this thing before it goes completly south. REBOOT The algo, erase the memory, just like in Westworld.
                    #BUT FIRST LET'S VERIFY THAT THERE IS INDEED AN IMPORTANT PROGRAMME THAT DAY...Don't go fuck everything up for no reason
                    l = [0]
                    if(len(l)>0):

                        error_type_2+= a_0

                    else:
                        print("PAS DE POINT DE CONTRÔLE POUR CETTE PARTIE DE LA JOURNÉE")
                        index_ipts+=1

        elif(context[2]):
            if(lastCP < min(6,currentduree/2)):
                index_CP+=1
                if(lastCP == 1):
                    continue
                else:
                    error_type_3+= abs((context[13]-0.46)*a_4)

            else:
                X = process(pd.DataFrame([context],index=[0],columns=['minute','partie de la journée','Change Point','pourcentage','partie du programme','programme','duree','nombre de pub potentiel','lastCP','lastPub','lastend','currentduree','Pubinhour','probability of CP','nb de pubs encore possible','chaine','CLE-FORMAT','CLE-GENRE','day'])).values #,'per'
                res1 = CatBoost[0].predict_proba(X)
                res2 = CatBoost[1].predict_proba(X)
                res3 = XGB[0].predict(xgb.DMatrix(X), ntree_limit= XGB[0].best_ntree_limit)
                res4 = XGB[1].predict(xgb.DMatrix(X), ntree_limit= XGB[1].best_ntree_limit)
                res5 = rf.predict_proba(X)
                res6 = gb.predict_proba(X)
                res7 = dt.predict_proba(X)
                res = [(res1[0][0]+res2[0][0]+res3[0][0]+res4[0][0]+res5[0][0]+res6[0][0])/6,(res1[0][1]+res2[0][1]+res3[0][1]+res4[0][1]+res5[0][1]+res6[0][1])/6,(res1[0][2]+res2[0][2]+res3[0][2]+res4[0][2]+res5[0][2]+res6[0][2])/6]
                y_pred = [(res1[0][0]+res2[0][0])*0.5,(res1[0][1]+res2[0][1])*0.5,(res1[0][2]+res2[0][2])*0.5]
                y_pred2 = [(res3[0][0]+res4[0][0])*0.5,(res3[0][1]+res4[0][1])*0.5,(res3[0][2]+res4[0][2])*0.5]
                X = pd.concat([pd.DataFrame(y_pred).T,pd.DataFrame(y_pred2).T,pd.DataFrame(res7),pd.DataFrame(res6),pd.DataFrame(res5)],axis = 1)
                X = X.replace([np.inf, -np.inf], np.nan)
                X = X.fillna(1)
                X = X.values
                res = logistic.predict_proba(X)
                cla = np.argmax(res)

                if(cla == 1 and context[14]==0):
                    cla = 0
                if(cla == 2 and context[3]<0.75 and context[11]>=20):
                    cla = 0
                if(cla == 2 and context[3]<0.75 and context[11]>=180):
                    cla = 0
                if(context[3]>1):
                    cla = 2
                if(cla == 1):
                    lastCP=0
                    lastPub = 0
                    Pubinhour+=4
                    nbpub+=1
                    wait = 4
                elif(cla == 2):
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
                    error_type_1+= abs(context[14])*a_1
                    error_type_1 += abs((1-context[3])*context[11])*a_5
                    if(context[14]<0):
                        error_type_1+= abs(context[14])*a_4
                    if(PTV['TITRE'].iloc[index_PTV] == importantpts[index_ipts][1]):
                        if(importantpts[index_ipts][0]-i>13):
                            error_type_2+= (abs(importantpts[index_ipts][0]-i))*a_2
                        else:

                            if(chaine == 'M6'):
                                error_type_2 -= importantpts[index_ipts][0]-i
                            else:
                                pass

                else:
                    error_type_3+= abs((context[13]-0.46)*a_4)



            index_CP+=1
        elif(i in help):
            if(lastCP < min(6,currentduree)):
                error_type_3-= 5
            else:
                X = process(pd.DataFrame([context],index=[0],columns=['minute','partie de la journée','Change Point','pourcentage','partie du programme','programme','duree','nombre de pub potentiel','lastCP','lastPub','lastend','currentduree','Pubinhour','probability of CP','nb de pubs encore possible','chaine','CLE-FORMAT','CLE-GENRE','day'])).values #,'per'
                res1 = CatBoost[0].predict_proba(X)
                res2 = CatBoost[1].predict_proba(X)
                res3 = XGB[0].predict(xgb.DMatrix(X), ntree_limit= XGB[0].best_ntree_limit)
                res4 = XGB[1].predict(xgb.DMatrix(X), ntree_limit= XGB[1].best_ntree_limit)
                res5 = rf.predict_proba(X)
                res6 = gb.predict_proba(X)
                res7 = dt.predict_proba(X)
                res = [(res1[0][0]+res2[0][0]+res3[0][0]+res4[0][0]+res5[0][0]+res6[0][0])/6,(res1[0][1]+res2[0][1]+res3[0][1]+res4[0][1]+res5[0][1]+res6[0][1])/6,(res1[0][2]+res2[0][2]+res3[0][2]+res4[0][2]+res5[0][2]+res6[0][2])/6]
                y_pred = [(res1[0][0]+res2[0][0])*0.5,(res1[0][1]+res2[0][1])*0.5,(res1[0][2]+res2[0][2])*0.5]
                y_pred2 = [(res3[0][0]+res4[0][0])*0.5,(res3[0][1]+res4[0][1])*0.5,(res3[0][2]+res4[0][2])*0.5]
                X = pd.concat([pd.DataFrame(y_pred).T,pd.DataFrame(y_pred2).T,pd.DataFrame(res7),pd.DataFrame(res6),pd.DataFrame(res5)],axis = 1)
                X = X.replace([np.inf, -np.inf], np.nan)
                X = X.fillna(1)
                X = X.values
                res = logistic.predict_proba(X)
                cla = np.argmax(res)

                if(cla == 1 and context[14]==0):
                    cla = 0
                if(cla == 2 and context[3]<0.75 and context[11]>=20):
                    cla = 0
                if(cla == 2 and context[3]<0.75 and context[11]>=180):
                    cla = 0
                if(context[3]>1):
                    cla = 2


                if(cla == 1):
                    lastCP=0
                    lastPub = 0
                    Pubinhour+=4
                    nbpub+=1
                    wait = 4
                elif(cla == 2):
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
                    error_type_1+= abs(context[14])*a_1
                    error_type_1 += abs((1-context[3])*context[11])*a_5
                    if(context[14]<0):
                        error_type_1+= abs(context[14])*a_4
                    if(PTV['TITRE'].iloc[index_PTV] == importantpts[index_ipts][1]):
                        if(importantpts[index_ipts][0]-i>13):
                            error_type_2+= (abs(importantpts[index_ipts][0]-i))*a_2
                        else:
                            if(chaine == 'M6'):
                                error_type_2 -= importantpts[index_ipts][0]-i
                            else:
                                pass

                else:
                    error_type_3-=5


        else:
            #labels.append(0)
            #Not a Change Point, we'll just check that nothing is wrong in the PTV at this time
            if(Predictiontimer <= 0):
                l = currentduree
                lastend = i
                lastCP=0
                index_PTV += 1
                index_PTV = index_PTV%(PTV.shape[0])
                currentduree = PTV['DUREE'].iloc[index_PTV]
                planifiedend = (lastend + currentduree)
                Predictiontimer = 200
                nbpub = 0
                error_type_1+= abs(context[14])*a_1
                if(context[14]<0):
                    error_type_1+= abs(context[14])*a_4
                error_type_1 += abs((1-context[3])*context[11])*a_5
                error_type_1 += max(currentduree/5,1)*a_3
                per = context[3]
                if(PTV['TITRE'].iloc[index_PTV] == importantpts[index_ipts][1]):
                    if(importantpts[index_ipts][0]-i>13):
                        error_type_2+= (abs(importantpts[index_ipts][0]-i))*a_2
                    else:
                        if(chaine == 'M6'):
                            error_type_2 -= importantpts[index_ipts][0]-i
                        else:
                            pass

            elif(context[3] == 1 or (context[3]>1 and Predictiontimer>20)):
                #Dépassement autorisé: Modulable en fonction de la position dans la journée si besoin
                # C'est sur ces valeurs que l'on va jouer pour avoir le meilleur PTV possible
                # Plus les valeurs sont grandes, plus on fait confiance a l'algo
                # Il est important de bien découper la journée celon les périodes horaires que l'on qualifie
                # de "sous tension" si plusieurs programmes courts se succédent. Bien évidement une telle analyse sera
                #plus tard fait automatiquement.
                if(chaine == 'TF1'):
                    if(context[5] == 'Journal'):
                        if(i<20*60):
                            Predictiontimer = 10
                        else:
                            Predictiontimer = 0

                    elif(context[6] == "très court"):
                        Predictiontimer = 4
                    elif(PTV['TITRE'].iloc[index_PTV] == 'Téléshopping'):
                        Predictiontimer = 2
                    elif(context[6] == "court"):
                        Predictiontimer = 5
                    elif(context[6] == "moyen"):
                        Predictiontimer = 5
                    elif(context[6] == "très long" or context[6] == "long"):
                        Predictiontimer = 15
                    else:
                        Predictiontimer = 15
                    if(i>25*60+30):
                        Predictiontimer =2
                elif(chaine =='M6'):
                    #Dépassement autorisé: Modulable en fonction de la position dans la journée si besoin
                    #Dépassement autorisé: Modulable en fonction de la position dans la journée si besoin
                    if(i>25*60+30):
                        Predictiontimer = 2
                    elif(13*60<i<14*60):
                        Predictiontimer = 5
                    elif(PTV['TITRE'].iloc[index_PTV] in ['M6 boutique']):
                        Predictiontimer = 0
                    elif(context[6] == "très court"):
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
    error+= error_type_1 + error_type_2 + error_type_3+error_type_4
    return error,[error_type_1, error_type_2,error_type_3,error_type_4]

def most_pobable_path(error_0,error_1,error_2,XGB,CatBoost,rf,dt,gb,logistic,context):
    g = np.argmin([error_0,error_1,error_2])
    min = ([error_0,error_1,error_2])[g]
    res = [error_0,error_1,error_2]
    res.pop(g)
    if(error_0 != error_1 and error_0 != error_2 and error_1 != error_2):
        cla = g
    elif(min not in res):
        cla = g
    elif(g == 0 and context[13]<0.65):
        cla = 0
    else:
        possibles = []
        if(error_0 <= min):
            possibles.append(0)
        if(error_1 <= min):
            possibles.append(1)
        if(error_2 <= min):
            possibles.append(2)
        if(min >= 100):
            print("pas de bonne solution",str(int(context[0]/60))+':'+str(context[0]%60))
        if(len(possibles) == 0):
            print("pas de bonne solution",str(int(context[0]/60))+':'+str(context[0]%60))
            possibles = [0,1,2]

        X = process(pd.DataFrame([context],index=[0],columns=['minute','partie de la journée','Change Point','pourcentage','partie du programme','programme','duree','nombre de pub potentiel','lastCP','lastPub','lastend','currentduree','Pubinhour','probability of CP','nb de pubs encore possible','chaine','CLE-FORMAT','CLE-GENRE','day'])).values #,'per'
        res1 = CatBoost[0].predict_proba(X)
        res2 = CatBoost[1].predict_proba(X)
        res3 = XGB[0].predict(xgb.DMatrix(X), ntree_limit= XGB[0].best_ntree_limit)
        res4 = XGB[1].predict(xgb.DMatrix(X), ntree_limit= XGB[1].best_ntree_limit)
        res5 = rf.predict_proba(X)
        res6 = gb.predict_proba(X)
        res7 = dt.predict_proba(X)
        res = [(res1[0][0]+res2[0][0]+res3[0][0]+res4[0][0]+res5[0][0]+res6[0][0])/6,(res1[0][1]+res2[0][1]+res3[0][1]+res4[0][1]+res5[0][1]+res6[0][1])/6,(res1[0][2]+res2[0][2]+res3[0][2]+res4[0][2]+res5[0][2]+res6[0][2])/6]
        y_pred = [(res1[0][0]+res2[0][0])*0.5,(res1[0][1]+res2[0][1])*0.5,(res1[0][2]+res2[0][2])*0.5]
        y_pred2 = [(res3[0][0]+res4[0][0])*0.5,(res3[0][1]+res4[0][1])*0.5,(res3[0][2]+res4[0][2])*0.5]
        X = pd.concat([pd.DataFrame(y_pred).T,pd.DataFrame(y_pred2).T,pd.DataFrame(res7),pd.DataFrame(res6),pd.DataFrame(res5)],axis = 1)
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(1)
        X = X.values
        res = logistic.predict_proba(X)

        cl = [[res[0][i],i] for i in possibles]
        cla = cl[np.argmax([l[0] for l in cl])][1]
    if(cla == 1 and context[14]==0 and context[0]>20*60):
        cla = most_pobable_path(error_0,5000,error_2,XGB,CatBoost,rf,dt,gb,logistic,context)
    if(cla == 2 and context[3]<0.5 and context[11]>=20):
        cla = most_pobable_path(error_0,error_1,5000,XGB,CatBoost,rf,dt,gb,logistic,context)
    if(cla == 2 and context[3]<0.75 and context[11]>=180):
        cla = most_pobable_path(error_0,error_1,5000,XGB,CatBoost,rf,dt,gb,logistic,context)
    if(cla == 1 and context[9]<0):
        cla = most_pobable_path(error_0,5000,error_2,XGB,CatBoost,rf,dt,gb,logistic,context)
    if(cla == 2 and context[3]<0.75 and context[11]>=20 and context[0]>19*60+50):
        cla = most_pobable_path(error_0,error_1,5000,XGB,CatBoost,rf,dt,gb,logistic,context)
    return cla





def make_newPTV(PTV,Points,proba,chaine,index,lastPTV,lastcontext,index_CP,index_PTV,date):
    #Initialisation des Variables
    verbose = False
    index_CP = index_CP
    index_PTV = index_PTV
    ##########################
    Predictiontimer = 200
    Pubinhour = lastcontext[12]
    lastCP = lastcontext[8]
    lastPub= lastcontext[9]
    lastend = lastcontext[10]
    currentduree = lastcontext[11]
    planifiedend = lastcontext[10]+lastcontext[11]
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
    index_ipts = index
    importantpts.append([(int(PTV['HEURE'].loc[PTV.shape[0]-1].split(':')[0])+24)*60+int(PTV['HEURE'].loc[PTV.shape[0]-1].split(':')[1]),PTV['TITRE'].loc[PTV.shape[0]-1]])
    newPTV = init_newPTV(PTV,chaine)
    historyofpoints = init_history(chaine,PTV)
    ######################
    historyofpoints.loc[0] = lastcontext
    labels = [0]
    start = lastcontext[0]+1
    end = importantpts[index][0]
    print(start,end)
    #########init Classifier#############
    XGB,CatBoost,rf,dt,gb,logistic = load_models()
    ####################################
    for i in tqdm(range(start,end+5)):
        #Update time of commercials (Reset)
        if(i%60 == 0):
            Pubinhour = 0
        #Update timmers
        lastPub+=1
        lastCP+=1
        if(index_ipts==len(importantpts)):
            index_ipts-=1
        #let's get the context:
        context = get_context(i,PTV.iloc[index_PTV],Points,lastCP,lastPub,lastend,currentduree,planifiedend,Pubinhour,proba,nbpub,chaine,per,PTV,index_PTV,date)
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
                        print("erreur sur la matinée")
                    elif(index_ipts == 1):
                        print("erreur sur l'après midi")
                    else:
                        print("erreur sur la soirée")
                    error+=1
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
                            print("erreur sur la matinée")
                        elif(index_ipts == 1):
                            print("erreur sur l'après midi")
                        else:
                            print("erreur sur la soirée")
                        error+=1
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
                            print("erreur sur la matinée")
                        elif(index_ipts == 1):
                            print("erreur sur l'après midi")
                        else:
                            print("erreur sur la soirée")
                        error+=1
                        #we can now keep going throw the process like before
                        #we just add a line to the history to say that a reset occured
                        newPTV.loc[newPTV.shape[0]] = [i%1440,PTV['TITRE'].iloc[index_PTV],'non',context[3],"--HARD RESET OF ALGORITHM--(out of programme)"]
                        index_ipts+=1
                    else:
                        index_ipts+=1
        elif(context[2]):
            historyofpoints.loc[historyofpoints.shape[0]] = context
            if(lastCP < min(6,currentduree/2)):
                labels.append(0)
                index_CP+=1
                continue
            else:
                if(i<importantpts[1][0]):
                    # Ne fonctionne qu'avec un Point de Contrôle et donc qu'avec des points avant 20h
                    error_0,sub_error_0 = make_predictedPTV(0,PTV,Points,chaine,i,index_PTV,index_CP,lastCP,lastPub,lastend,currentduree,planifiedend,Pubinhour,proba,nbpub,XGB,CatBoost,rf,dt,gb,logistic,Predictiontimer,per,context,date)
                    error_1,sub_error_1 = make_predictedPTV(1,PTV,Points,chaine,i,index_PTV,index_CP,lastCP,lastPub,lastend,currentduree,planifiedend,Pubinhour,proba,nbpub,XGB,CatBoost,rf,dt,gb,logistic,Predictiontimer,per,context,date)
                    error_2,sub_error_2 = make_predictedPTV(2,PTV,Points,chaine,i,index_PTV,index_CP,lastCP,lastPub,lastend,currentduree,planifiedend,Pubinhour,proba,nbpub,XGB,CatBoost,rf,dt,gb,logistic,Predictiontimer,per,context,date)
                    cla = most_pobable_path(error_0,error_1,error_2,XGB,CatBoost,rf,dt,gb,logistic,context)
                    print()
                    print(error_0,sub_error_0,'|',error_1,sub_error_1,'|',error_2,sub_error_2,'|',str(int(context[0]/60))+':'+str(context[0]%60),cla)


                else:
                    error_0,sub_error_0 = make_predictedPTV2(0,PTV,Points,chaine,i,index_PTV,index_CP,lastCP,lastPub,lastend,currentduree,planifiedend,Pubinhour,proba,nbpub,XGB,CatBoost,rf,dt,gb,logistic,Predictiontimer,per,context,date)
                    error_1,sub_error_1 = make_predictedPTV2(1,PTV,Points,chaine,i,index_PTV,index_CP,lastCP,lastPub,lastend,currentduree,planifiedend,Pubinhour,proba,nbpub,XGB,CatBoost,rf,dt,gb,logistic,Predictiontimer,per,context,date)
                    error_2,sub_error_2 = make_predictedPTV2(2,PTV,Points,chaine,i,index_PTV,index_CP,lastCP,lastPub,lastend,currentduree,planifiedend,Pubinhour,proba,nbpub,XGB,CatBoost,rf,dt,gb,logistic,Predictiontimer,per,context,date)
                    cla = most_pobable_path(error_0,error_1,error_2,XGB,CatBoost,rf,dt,gb,logistic,context)
                    print()
                    print(error_0,sub_error_0,'|',error_1,sub_error_1,'|',error_2,sub_error_2,'|',str(int(context[0]/60))+':'+str(context[0]%60),cla)


                if(cla == 1 and context[14]==0):
                    cla = 1
                if(cla == 2 and context[3]<0.5 and context[11]>30):
                    cla = 0
                if(cla == 2 and context[3]<0.9 and context[11]>=180):
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
            if(lastCP < min(6,currentduree)):
                labels.append(0)
                index_CP+=1
                continue
            else:
                if(i<importantpts[1][0]):
                    #print("utilisation de l'algo 1")
                    error_0,sub_error_0 = make_predictedPTV(0,PTV,Points,chaine,i,index_PTV,index_CP,lastCP,lastPub,lastend,currentduree,planifiedend,Pubinhour,proba,nbpub,XGB,CatBoost,rf,dt,gb,logistic,Predictiontimer,per,context,date)
                    error_1,sub_error_1 = make_predictedPTV(1,PTV,Points,chaine,i,index_PTV,index_CP,lastCP,lastPub,lastend,currentduree,planifiedend,Pubinhour,proba,nbpub,XGB,CatBoost,rf,dt,gb,logistic,Predictiontimer,per,context,date)
                    error_2,sub_error_2 = make_predictedPTV(2,PTV,Points,chaine,i,index_PTV,index_CP,lastCP,lastPub,lastend,currentduree,planifiedend,Pubinhour,proba,nbpub,XGB,CatBoost,rf,dt,gb,logistic,Predictiontimer,per,context,date)
                    cla = most_pobable_path(error_0,error_1,error_2,XGB,CatBoost,rf,dt,gb,logistic,context)
                    print()
                    print(error_0,sub_error_0,'|',error_1,sub_error_1,'|',error_2,sub_error_2,'|',str(int(context[0]/60))+':'+str(context[0]%60),cla)
                else:
                    #print("utilisation de l'algo 2")
                    error_0,sub_error_0 = make_predictedPTV2(0,PTV,Points,chaine,i,index_PTV,index_CP,lastCP,lastPub,lastend,currentduree,planifiedend,Pubinhour,proba,nbpub,XGB,CatBoost,rf,dt,gb,logistic,Predictiontimer,per,context,date)
                    error_1,sub_error_1 = make_predictedPTV2(1,PTV,Points,chaine,i,index_PTV,index_CP,lastCP,lastPub,lastend,currentduree,planifiedend,Pubinhour,proba,nbpub,XGB,CatBoost,rf,dt,gb,logistic,Predictiontimer,per,context,date)
                    error_2,sub_error_2 = make_predictedPTV2(2,PTV,Points,chaine,i,index_PTV,index_CP,lastCP,lastPub,lastend,currentduree,planifiedend,Pubinhour,proba,nbpub,XGB,CatBoost,rf,dt,gb,logistic,Predictiontimer,per,context,date)
                    cla = most_pobable_path(error_0,error_1,error_2,XGB,CatBoost,rf,dt,gb,logistic,context)
                    print()
                    print(error_0,sub_error_0,'|',error_1,sub_error_1,'|',error_2,sub_error_2,'|',str(int(context[0]/60))+':'+str(context[0]%60),cla)


                if(cla == 1 and context[14]==0):
                    cla = 1
                if(cla == 2 and context[3]<0.5 and context[11]>30):
                    cla = 0
                if(cla == 2 and context[3]<0.9 and context[11]>=180):
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
            elif(context[3] == 1 or (context[3]>1 and Predictiontimer>20)):
                #Dépassement autorisé: Modulable en fonction de la position dans la journée si besoin
                # C'est sur ces valeurs que l'on va jouer pour avoir le meilleur PTV possible
                # Plus les valeurs sont grandes, plus on fait confiance a l'algo
                # Il est important de bien découper	15	215	577	30	0	0.490790	1	TF1	3	1026	9:52	2 la journée celon les périodes horaires que l'on qualifie
                # de "sous tension" si plusieurs programmes courts se succédent. Bien évidement une telle analyse sera
                #plus tard fait automatiquement.
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
    return newPTV,historyofpoints,labels,error,index_CP,index_PTV,context


#################################################
########### main with options ###################
#################################################

def main(argv):
    import warnings
    warnings.filterwarnings("ignore")
    t = time.time()
    if(len(argv) ==0):
        argv = ['2015']

    if(len(argv) == 1):

        EPSILON = 1e-15
        err = 0
        m = 0
        err_TF1 = 0
        m_TF1 = 0
        err_M6 = 0
        m_M6 = 0
        err_type_1 = 0
        err_type_2 = 0
        err_type_3 = 0

        files = os.listdir('/home/alexis/Bureau/Project/Datas/PTV/extracted')
        for file in files:
            f = ((file.split('.'))[0].split('_'))[2]
            c = ((file.split('.'))[0].split('_'))[-1]
            if(f=='2017-12-20' or (f in ['2017-12-09','2017-12-06','2018-02-22'] and c=='TF1') or (f in ['2018-02-22'] and c=='M6') or  f.split('-')[0] == str(argv[0])):
                pass
            elif(c ==''):
                pass
            else:
                if(f=='2017-12-20' or (f in ['2017-12-09','2017-12-06','2018-02-22'] and c=='TF1') or (f in ['2018-02-22'] and c=='M6') or  f.split('-')[0] == str(argv[0])):
                    #or (f in ['2018-02-22'] and c=='M6')
                    pass
                elif(c ==''):
                    pass
                else:


                    PTV,Points,proba = load_file(str(f),str(c))
                    if(len(PTV) == 0):
                        continue
                    newPTV = init_newPTV(PTV,str(c))
                    historyofpoints = init_history(str(c),PTV)
                    index_CP = 0
                    index_PTV = PTV.shape[0]-1
                    temp_context = historyofpoints.iloc[0]

                    for i in range(3):
                        l,temp_newPTV,temp_history,index_CP,index_PTV,temp_context = main([str(c),str(f),i,newPTV.loc[newPTV.shape[0]-1],temp_context,index_CP,index_PTV])
                        if(l == 4):
                            pass
                        else:
                            newPTV = pd.concat([newPTV,temp_newPTV.iloc[1:]])
                            historyofpoints = pd.concat([historyofpoints,temp_history])
                            err += l
                            if(i == 0):
                                err_type_1 += 1
                            if(i == 1):
                                err_type_2 += 1
                            if(i == 2):
                                err_type_3 += 1
                            m+=1
                            if(c == 'M6'):
                                err_M6 += l
                                m_M6 += 1
                            if(c == 'TF1'):
                                err_TF1 += l
                                m_TF1 += 1
                    new_PTV.to_html('/home/alexis/Bureau/Project/results/ptvbyml/PTV/new_PTV-'+date+'_'+chaine+'.html')
                    new_PTV.to_csv('/home/alexis/Bureau/Project/results/ptvbyml/csv/new_PTV-'+date+'_'+chaine+'.csv',index=False)
                    historyofpoints.to_html('/home/alexis/Bureau/Project/results/ptvbyml/historyofpoints/historyofpoints-'+date+'_'+chaine+'.html')
                    historyofpoints.to_csv('/home/alexis/Bureau/Project/results/truemerge/'+chaine+'/true_merge_'+str(date)+'_'+chaine+'.csv',index=False)

            print(err)
        print(m)
        print("score Total",1-(err/(m+EPSILON)))
        print("score TF1",1-(err_TF1/(m_TF1+EPSILON)))
        print("score M6",1-(err_M6/(m_M6+EPSILON)))
        print("score sur la matinée",1-((err_type_1*3)/(m+EPSILON)))
        print("score sur l'après midi",1-((err_type_2*3)/(m+EPSILON)))
        print("score sur la soirée",1-((err_type_3*3)/(m+EPSILON)))
        print("temps de calcul: ",time.time()-t)
    elif(len(argv) == 2):
        import pandas as pd
        c = argv[0]
        f = argv[1]
        PTV,Points,proba = load_file(str(f),str(c))
        if(len(PTV) == 0):
            return('Fichier Manquant')
        newPTV = init_newPTV(PTV,str(c))
        historyofpoints = init_history(str(c),PTV)
        index_CP = 0
        index_PTV = PTV.shape[0]-1
        temp_context = historyofpoints.iloc[0]
        for i in range(3):
            l,temp_newPTV,temp_history,index_CP,index_PTV,temp_context = main([str(c),str(f),i,newPTV.iloc[newPTV.shape[0]-1],temp_context,index_CP,index_PTV])
            if(l == 4):
                pass
            else:
                if(l>0):
                    if(i == 0):
                        print("erreur sur la matinée")
                    elif(i == 1):
                        print("erreur sur l'après midi")
                    else:
                        print("erreur sur la soirée")
                newPTV = pd.concat([newPTV,temp_newPTV.iloc[1:]])
                historyofpoints = pd.concat([historyofpoints,temp_history])
        new_PTV['Heure'] = new_PTV['minute'].apply(lambda x: str(int(x/60))+':'+str(x%60))
        historyofpoints['Heure'] = historyofpoints['minute'].apply(lambda x: str(int(x/60))+':'+str(x%60))
        new_PTV.to_html('/home/alexis/Bureau/Project/results/ptvbyml/PTV/new_PTV-'+date+'_'+chaine+'.html')
        new_PTV.to_csv('/home/alexis/Bureau/Project/results/ptvbyml/csv/new_PTV-'+date+'_'+chaine+'.csv',index=False)
        historyofpoints.to_html('/home/alexis/Bureau/Project/results/ptvbyml/historyofpoints/historyofpoints-'+date+'_'+chaine+'.html')
        historyofpoints.to_csv('/home/alexis/Bureau/Project/results/truemerge/'+chaine+'/true_merge_'+str(date)+'_'+chaine+'.csv',index=False)
        print(chaine,date,historyofpoints.shape,len(labels))
        sys.exit(error)
    else:
        chaine = argv[0]
        date = argv[1]
        index = argv[2]
        d = "".join(date.split('-'))
        PTV,Points,proba = load_file(date,chaine)
        if(len(PTV) == 0):
            return 4,0,0,0,0,0
        new_PTV,historyofpoints,labels,error,index_CP,index_PTV,temp_context = make_newPTV(PTV,Points,proba,chaine,index,argv[3],argv[4],argv[5],argv[6],date)
        print(len(labels),historyofpoints.shape)
        historyofpoints['labels'] = labels
        print(chaine,date,historyofpoints.shape,len(labels))
        return error,new_PTV,historyofpoints,index_CP,index_PTV,temp_context


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
