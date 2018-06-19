#-*- coding: utf-8 -*-
#################################################
#created the 06/06/2018 11:16 by Alexis Blanchet#
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
    print("extraction")
    os.system("python /home/alexis/Bureau/Project/scripts/extractdata.py 192")
    os.system("python /home/alexis/Bureau/Project/scripts/extractdata.py 118")
    time.sleep(25)
    print("cleaning")
    os.system("python /home/alexis/Bureau/Project/scripts/cleanall.py 0 192 0")
    os.system("python /home/alexis/Bureau/Project/scripts/cleanall.py 0 118 0")
    time.sleep(25)
    print("processing")
    os.system("python /home/alexis/Bureau/Project/scripts/processall.py test")
    time.sleep(25)
    print("predict")
    os.system("python /home/alexis/Bureau/Project/scripts/predict.py test")
    time.sleep(25)
    print("PTV")
    os.system("python /home/alexis/Bureau/Project/scripts/PTVwithTruemerge.py")
    time.sleep(25)




    return ("process fini")


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
