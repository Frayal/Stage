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
    print("cleaning")
    os.system("python /home/alexis/Bureau/Project/scripts/cleanall.py 0 192 0")
    time.sleep(5)
    print("processing")
    os.system("python /home/alexis/Bureau/Project/scripts/processall.py train")
    time.sleep(5)
    print("Stack")
    os.system("python /home/alexis/Bureau/Project/scripts/Stack.py trainall")
    time.sleep(5)
    print("predict")
    os.system("python /home/alexis/Bureau/Project/scripts/predict.py train")
    time.sleep(5)
    print("predict")
    os.system("python /home/alexis/Bureau/Project/scripts/predict.py test")
    time.sleep(5)
    print("PTV")
    os.system("python /home/alexis/Bureau/Project/scripts/PTV.py test")
    time.sleep(5)




    return ("process fini")


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
