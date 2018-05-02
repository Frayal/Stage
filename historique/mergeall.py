#################################################
#created the 02/05/2018 13:21 by Alexis Blanchet#
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
    for i in range(int(argv[0])):
        print(i)
        if(i<9): l = "0"+str(i+1)
        else: l = str(i+1)
        os.system("python /home/alexis/Bureau/Stage/historique/merge.py 192 2017-12-"+l)
    return ("process achevé sans erreures")

if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
