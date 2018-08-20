# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from datetime import datetime as dt

app = dash.Dash()
df = pd.read_csv('cout.csv')


colors = {
    'background': '#0000',
    'text': '#7FDBFF'
}

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Bonjour Alexis',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    html.Div(children='Visualisation des résultats', style={
        'textAlign': 'center',
        'color': colors['text']
    }),
    html.Div([
    html.Label('Date'),
        dcc.DatePickerSingle(
        id='date-input',
        date='2017-12-02'
        ),
    html.Label('Chaîne'),
    dcc.Dropdown(
        id = 'Chaîne',
        options = [{'label': i, 'value': i} for i in ['TF1','M6','France 2','France 3']],
        value=['TF1'],
        multi=True
    ),
    html.Label('Part of the day'),
        dcc.RadioItems(
        id='toggle1',
        options=[{'label': i, 'value': i} for i in ['tout','matinee','apresmidi','soiree']],
        value='tout'
    ),
    html.Div(id='controls-container1', children=[
    dcc.Graph(
        id='score',
        figure={
            'data': [
                go.Scatter(
                    y = df['2017-12-02_TF1_tout'],
                    mode='lines+markers',
                    name = 'TF1',
                    opacity=0.7,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                ),

            ],
            'layout': go.Layout(
                xaxis={'title': 'Tours de l algorithme'},
                yaxis={'title': 'Score'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
            )
        }
    ),]),

    ], style={'columnCount': 1})])


@app.callback(
    dash.dependencies.Output('score', 'figure'),
    [dash.dependencies.Input('date-input','date'),
    dash.dependencies.Input('Chaîne','value'),
    dash.dependencies.Input('toggle1','value')]
)

def Update(value,value_d,value_c):
    df = pd.read_csv('cout.csv')
    here = []
    not_here = []
    for v in value_d:
        try:
            y = df[value+'_'+v+'_'+value_c]
            here.append(v)
        except Exception as e:
            not_here.append(v)


    return({'data': [
        go.Scatter(
            y=df[value+'_'+v+'_'+value_c],
            mode='lines+markers',
            name = str(v),
            opacity=0.7,
            marker={
                'size': 15,
                'line': {'width': 0.5, 'color': 'white'}
            },
        ) for v in here

    ],
    'layout': go.Layout(
        xaxis={'title': 'Tours de l algorithme'},
        yaxis={'title': 'Score'},
        margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
        legend={'x': 0, 'y': 1},
        hovermode='closest'
    )
    }
    )

if __name__ == '__main__':
    app.run_server(debug=True,port=3004)
