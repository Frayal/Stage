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
import time
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
    t = time.time()
    if(len(argv)==0):
        argv = ['all']
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
        if(f=='2017-12-20' or (f in ['2017-12-09','2017-12-06'] and c=='TF1') or (f in ['2018-02-22'] and c=='M6') or  f.split('-')[0] == str(argv[0]) and argv[0] != 'all'):
            print(f)
        elif(c ==''):
            pass
        else:
            print(c)
            l = os.system('python /home/alexis/Bureau/Project/scripts/PTV'+str(c)+'.py '+str(f))
            if(l/256 == 4):
                pass
            else:
                l = l/256
                err += int(l/100)+int((l%100)/10)+int((l%10))
                err_type_1 += int(l/100)
                err_type_2 += int((l%100)/10)
                err_type_3 += int((l%10))
                m+=3
                if(c == 'M6'):
                    err_M6 += int(l/100)+int((l%100)/10)+int((l%10))
                    m_M6 += 3
                if(c == 'TF1'):
                    err_TF1 += int(l/100)+int((l%100)/10)+int((l%10))
                    m_TF1 += 3

        print(err)
    print(m)
    print("score Total",1-(err/(m+EPSILON)))
    print("score TF1",1-(err_TF1/(m_TF1+EPSILON)))
    print("score M6",1-(err_M6/(m_M6+EPSILON)))
    print("score sur la matinée",1-((err_type_1*3)/(m+EPSILON)))
    print("score sur l'après midi",1-((err_type_2*3)/(m+EPSILON)))
    print("score sur la soirée",1-((err_type_3*3)/(m+EPSILON)))
    print("temps de calcul: ",time.time()-t)


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
