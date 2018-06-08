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
    if(argv[0] == "train"):
        files = os.listdir('/home/alexis/Bureau/Project/Datas/train')
        for file in files:
            if((file.split('.'))[-1] == 'csv'):
                if(((file.split('.'))[0].split('-'))[0] == 'label'):
                    pass
                else:
                    print(str(file))
                    os.system("python /home/alexis/Bureau/Project/scripts/processingdata.py "+str(file) +" train")
            else:
                pass
        return ("process achevé sans erreures")
    else:
        files = os.listdir('/home/alexis/Bureau/Project/Datas/RTS/processed/')
        for file in files:
            if((file.split('.'))[-1] == 'csv'):
                print(str(file))
                os.system("python /home/alexis/Bureau/Project/scripts/processingdata.py "+str(file)+" test")
            else:
                pass
        return ("process achevé sans erreures")

if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
