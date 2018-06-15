import pandas as pd
import numpy as np
import sys
import os
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as offline
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.figure_factory as ff
import random


def end(debut,duree):
    h = str(int((debut+duree)/60))
    m = str(int((debut+duree)%60))
    return h+':'+m+':00'

def plot_res(df,newPTV,PTV):

    x1 = (newPTV['minute'].values-3)/60
    x2 = (PTV['debut'].values)/60
    x1 = [x + 24-3/60 if x<3 else x for x in x1]
    x2 = [x + 24/60 if x<3 else x for x in x2]

    y1 = [df['t'].iloc[l-3] for l in newPTV['minute'].values-180]
    y2 = [df['t'].iloc[l] for l in PTV['debut'].values-180]

    trace1 = go.Scatter(
            x= ((df['minutes']-3)/60+3) ,
            y= df['t'],
            name = 'Audience')
    trace2 = go.Bar(
        x= x1 ,
        y= y1,
        name = 'new PTV')
    trace3 = go.Bar(
        x= x2 ,
        y= y2,
        name = 'PTV')

    data = [trace1,trace2,trace3]
    plot(data, filename='PTVtest.html')




def main(argv):
        date = argv[0]
        newPTV = pd.read_csv('/home/alexis/Bureau/Project/results/newPTV/PTV/TF1/new_PTV-'+date+'_TF1.csv')[['minute']]
        PTV = pd.read_csv('/home/alexis/Bureau/Project/Datas/PTV/extracted/IPTV_0192_'+date+'_TF1.csv')[['debut']]
        df = pd.read_csv('/home/alexis/Bureau/Project/Datas/RTS/processed/sfrdaily_'+"".join(date.split('-'))+'_0_192_0_cleandata.csv')[['t','minutes']]
        plot_res(df,newPTV,PTV)


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
