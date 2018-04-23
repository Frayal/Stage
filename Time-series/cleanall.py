#-*- coding: utf-8 -*-
#################################################
#created the 23/04/2018 12:57 by Alexis Blanchet#
#################################################

'''

'''

'''
Ameliorations possibles:

'''
import warnings
warnings.filterwarnings('ignore')
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
    files = os.listdir('/home/alexis/Bureau/Stage/Time-series/')
    for file in files:
        if(file.split('.')[-1]=='csv'):
            departement = argv[0]
            chaine = argv[1]
            csp = argv[2]
            os.system("python /home/alexis/Bureau/Stage/Time-series/cleaning.py "+str(file)+" " +str(departement)+' '+str(chaine)+' '+str(csp))
        else:
            pass
        
    return ("process fini")


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
