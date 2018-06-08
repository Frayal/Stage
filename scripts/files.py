import pandas as pd
import numpy as np
import sys
from random import shuffle

def make_names(date):
    d = str((date.split('-'))[1])+str((date.split('-'))[0])
    X = '/home/alexis/Bureau/Project/Datas/train/sfrdaily_2018'+d+'_0_192_0_cleandata.csv'
    y = '/home/alexis/Bureau/Project/Datas/train/label-'+date+'.csv'
    return X,y


def make_dataframe(dates):
    train = len(dates)-2
    valid=1
    test=1
    shuffle(dates)
    data = pd.DataFrame()
    fileX_test = []
    fileY_test = []
    fileX_train = []
    fileY_train = []
    fileX_valid = []
    fileY_valid = []

    j = 0
    for i in range(train):
        X,y = make_names(dates[j])
        fileX_train.append(X)
        fileY_train.append(y)
        j+=1
    for i in range(valid):
        X,y = make_names(dates[j])
        fileX_valid.append(X)
        fileY_valid.append(y)
        j+=1
    for i in range(test):
        X,y = make_names(dates[j])
        fileX_test.append(X)
        fileY_test.append(y)
        j+=1

    data['fileX_train'] = [fileX_train]
    data['fileY_train'] = [fileY_train]

    data['fileX_valid'] = [fileX_valid]
    data['fileY_valid'] = [fileY_valid]

    data['fileX_test'] = [fileX_test]
    data['fileY_test'] = [fileY_test]

    data.to_csv('files.csv',index=False)

    return data






def main(argv):
    Dates = ['30-04','07-05','09-05','18-05','23-05','28-05']
    make_dataframe(Dates)

    return "processe has end"

if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
