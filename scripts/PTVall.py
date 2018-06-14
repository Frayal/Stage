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

#################################################
########### main with options ###################
#################################################



def main(argv):
    files = os.listdir('/home/alexis/Bureau/Project/Datas/PTV/extracted')
    for file in files:
        f = ((file.split('.'))[0].split('_'))[2]
        c = ((file.split('.'))[0].split('_'))[-1]
        print(c)
        os.system('python /home/alexis/Bureau/Project/scripts/PTV'+str(c)+'.py '+str(f))



if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
