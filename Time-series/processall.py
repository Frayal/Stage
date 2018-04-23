#################################################
#created the 20/04/2018 12:57 by Alexis Blanchet#
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
    files = os.listdir('/home/alexis/Bureau/Stage/Time-series/clean data')
    for file in files:
        os.system("python /home/alexis/Bureau/Stage/Time-series/processingdata.py "+str(file))
    return ("process achevé sans erreures")


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
