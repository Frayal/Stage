#-*- coding: utf-8 -*-
#################################################
#created the 23/04/2018 12:57 by Alexis Blanchet#
#################################################

'''

'''

'''
Ameliorations possibles:

'''
#################################################
###########        Imports      #################
#################################################
import warnings
warnings.filterwarnings('ignore')
import os
import sys


#################################################
########### Global variables ####################
#################################################


#################################################
########### Important functions #################
#################################################



#################################################
########### main with options ###################
#################################################

def main(argv):
    files = os.listdir('/home/alexis/Bureau/Project/Datas/RTS/brut/')
    for file in files:
        if(file.split('.')[-1]=='csv'):
            print(file)
            departement = argv[0]
            chaine = argv[1]
            csp = argv[2]
            os.system("python /home/alexis/Bureau/Project/scripts/cleaning.py "+str(file)+" " +str(departement)+' '+str(chaine)+' '+str(csp))
        else:
            pass

    return ("process fini")


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
