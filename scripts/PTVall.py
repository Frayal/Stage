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
import subprocess

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
        if(f=='2017-12-20' or f.split('-')[0]=='2017'):
            pass
        elif(c ==''):
            pass
        else:
            print(c)
            l = os.system('python /home/alexis/Bureau/Project/scripts/PTV'+str(c)+'.py '+str(f))
            if(l/256 == 4):
                pass
            else:
                err += l/256
                m+=2
                if(c == 'M6'):
                    err_M6 += l/256
                    m_M6 += 2
                if(c == 'TF1'):
                    err_TF1 += l/256
                    m_TF1 += 2

        print(err)
    print(m)
    print("score Total",1-(err/(m+EPSILON)))
    print("score TF1",1-(err_TF1/(m_TF1+EPSILON)))
    print("score M6",1-(err_M6/(m_M6+EPSILON)))


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
