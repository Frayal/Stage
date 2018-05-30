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

def init_newPTV(PTV=PTV):
    begin = (pd.DataFrame(PTV.loc[(PTV.shape[0]-1)]).T).reset_index().drop(['index'],axis=1)
    #Initialisation du NewPTV
    OnlinePTV = pd.DataFrame()
    OnlinePTV['minute'] = [180]
    OnlinePTV['TITRE'] = 'Programmes de la nuit'
    OnlinePTV['Change Point'] = 'Non'
    OnlinePTV['pourcentage de la durée'] = 0
    OnlinePTV['Évenement'] = 'Début de Détection'


def get_context(i,programme,index_CP=index_CP,lastCP=lastCP,lastPub=lastPub,lastend=lastend,lastduree=lastduree,planifiedend=planifiedend):
    #we create a list with different notes to understand the context
    context = []
    # Is the Point a Change Point
    if(i==Points['minutes'][index_CP]):
        context.append(1)
    else:
        context.append(0)
    # Where is the Point in the programme:
    seen_percentage = (i-lastend)/lastduree
    context.append(seen_percentage)
    # which type of programme we are watching
    context.append(programme['description programme'].split(' ')[0]))

    return context


def make_newPTV(PTV=PTV,Points=Points):
    #Initialisation des Variables
    verbose = False
    index_CP = 0
    index_PTV = PTV.shape[0]-1
    ##########################
    lastCP = 200
    lastPub= 0
    lastend = 130
    lastduree = 255
    planifiedend = 385
    begin = True
    ######################
    newPTV = init_newPTV()
    ######################
    for i in range(180,1620):
        #incrémentation
        lastPub+=1
        lastCP+=1
        if(index_CP==Points.shape[0]):
            index_CP -=1
        #let's get the context:
        context = get_context(i,PTV,)


















    return newPTV





#################################################
########### main with options ###################
#################################################


def main(argv):
    date = argv[0]
    PTV,Points = load_file(date)
    new_PTV = make_newPTV()
    return ("process achevé sans erreures")


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
