#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 13:11:05 2025

@author: alexanderlarsnas
"""

import dash 
from dash import html, dcc, Input, Output, State
import plotly.express as px
import pandas as pd
import webbrowser
import plotly.io as pio
from threading import Timer

#%%

pio.renderers.default = 'browser'

host = 'localhost'
port=8050


#%%

def open_browser(host='localhost', port=8050):
    webbrowser.open_new(f'http://{host}:{port}')
    
    
#%%

from main import spå_betyg, lin_reg
from main import fig as fig
from main import X_train


colors={
    'background':'white',
    'text':'black',
}



app=dash.Dash(__name__)


app.layout =html.Div(
    style={'backgroundColor':colors['background'],'padding':'20px'},
    
    children=[

    html.H1('Calculate your math grade!', style={'color':colors['text'], 'textAlign':'center'}),
    
    dcc.Graph(id='main-graph', figure=fig),
    
      
    html.Div([
        
        html.H3("Personal Info"),

        html.Label('Sex:'),
        dcc.Dropdown(
            id='sex',
            options=[{'label': 'Female', 'value': 0}, {'label': 'Male', 'value': 1}],
            value=0
        ),
        
    
        html.Label('Age:'),
        dcc.Dropdown(
            id='age',
            options=[{'label': str(i), 'value': i} for i in range(15, 23)], # 15 to 22 years
            value=18
        ),
    
        html.Label('Address (Urban/Rural):'),
        dcc.Dropdown(
            id='address',
            options=[{'label': 'Urban', 'value': 0}, {'label': 'Rural', 'value': 1}],
            value=0
        ),
    
        html.Label('Family Size:'),
        dcc.Dropdown(
            id='famsize',
            options=[{'label': 'Greater than 3', 'value': 0}, {'label': '3 or less', 'value': 1}],
            value=0
        ),
    
        html.Label('Parent Cohabitation Status:'),
        dcc.Dropdown(
            id='Pstatus',
            options=[{'label': 'Living together', 'value': 0}, {'label': 'Living apart', 'value': 1}],
            value=0
        ),
    
    
        html.H3("Parents"),
    
        html.Label("Mother's Education (0=None, 4=Higher):"),
        dcc.Dropdown(
            id='Medu',
            options=[{'label': str(i), 'value': i} for i in range(0, 5)],
            value=4
        ),
    
        html.Label("Father's Education (0=None, 4=Higher):"),
        dcc.Dropdown(
            id='Fedu',
            options=[{'label': str(i), 'value': i} for i in range(0, 5)],
            value=4
        ),
    
        html.Label("Mother's Job:"),
        dcc.Dropdown(
            id='Mjob',
            options=[
                {'label': 'At Home', 'value': 'at_home'},
                {'label': 'Health Care', 'value': 'health'},
                {'label': 'Other', 'value': 'other'},
                {'label': 'Civil Services', 'value': 'services'},
                {'label': 'Teacher', 'value': 'teacher'}
            ],
            value='other'
        ),
    
        html.Label("Father's Job:"),
        dcc.Dropdown(
            id='Fjob',
            options=[
                {'label': 'At Home', 'value': 'at_home'},
                {'label': 'Health Care', 'value': 'health'},
                {'label': 'Other', 'value': 'other'},
                {'label': 'Civil Services', 'value': 'services'},
                {'label': 'Teacher', 'value': 'teacher'}
            ],
            value='other'
        ),
    
        html.H3("Studies & School"),
    
        html.Label('Travel Time to School (1=<15 min, 4=>1h):'),
        dcc.Dropdown(
            id='traveltime',
            options=[{'label': str(i), 'value': i} for i in range(1, 5)],
            value=1
        ),
    
        html.Label('Weekly Study Time (1=<2h, 4=>10h):'),
        dcc.Dropdown(
            id='studytime',
            options=[{'label': str(i), 'value': i} for i in range(1, 5)],
            value=2
        ),
    
        html.Label('Past Class Failures:'),
        dcc.Dropdown(
            id='failures',
            options=[{'label': str(i), 'value': i} for i in range(0, 5)],
            value=0
        ),
    
        html.Label('Extra Educational Support?'),
        dcc.Dropdown(
            id='schoolsup',
            options=[{'label': 'No', 'value': 0}, {'label': 'Yes', 'value': 1}],
            value=0
        ),
        
        html.Label('Extra-curricular Activities?'),
        dcc.Dropdown(
            id='activities',
            options=[{'label': 'No', 'value': 0}, {'label': 'Yes', 'value': 1}],
            value=1
        ),
        
        html.H3("Lifestyle"),
    
        html.Label('In a Romantic Relationship?'),
        dcc.Dropdown(
            id='romantic',
            options=[{'label': 'No', 'value': 0}, {'label': 'Yes', 'value': 1}],
            value=0
        ),
    
        html.Label('Free Time (1=Low, 5=High):'),
        dcc.Dropdown(
            id='freetime',
            options=[{'label': str(i), 'value': i} for i in range(1, 6)],
            value=3
        ),
    
        html.Label('Going Out with Friends (1=Low, 5=High):'),
        dcc.Dropdown(
            id='goout',
            options=[{'label': str(i), 'value': i} for i in range(1, 6)],
            value=3
        ),
    
        html.Label('Workday Alcohol Consumption (1=Low, 5=High):'),
        dcc.Dropdown(
            id='Dalc',
            options=[{'label': str(i), 'value': i} for i in range(1, 6)],
            value=1
        ),
    
        html.Label('Weekend Alcohol Consumption (1=Low, 5=High):'),
        dcc.Dropdown(
            id='Walc',
            options=[{'label': str(i), 'value': i} for i in range(1, 6)],
            value=1
        ),
    
        html.Label('Health Status (1=Bad, 5=Good):'),
        dcc.Dropdown(
            id='health',
            options=[{'label': str(i), 'value': i} for i in range(1, 6)],
            value=5
        ),
    
        html.Label('Absences (Number of times):'),
        dcc.Dropdown(
            id='absences',
            # 0 to 30 selectable options
            options=[{'label': str(i), 'value': i} for i in range(0, 31)], 
            value=2
            )
            ]),

    html.Button(
        'Calculate Grade',
        id='submit-btn',
        n_clicks=0
    ),
    
    html.Div(id='respons'),
    
    ]
    )


@app.callback(
    Output('respons', 'children'),
    Input('submit-btn', 'n_clicks'),
    State('sex', 'value'),
    State('age', 'value'),
    State('address', 'value'),
    State('famsize', 'value'),
    State('Pstatus', 'value'),
    State('Medu', 'value'),
    State('Fedu', 'value'),
    State('Mjob', 'value'),
    State('Fjob', 'value'),
    State('traveltime', 'value'),
    State('studytime', 'value'),
    State('failures', 'value'),
    State('schoolsup', 'value'),
    State('activities', 'value'),
    State('romantic', 'value'),
    State('freetime', 'value'),
    State('goout', 'value'),
    State('Dalc', 'value'),
    State('Walc', 'value'),
    State('health', 'value'),
    State('absences', 'value')
    
    )
def calculate_grade(n_clicks, sex, age, address, famsize, pstatus, 
                                medu, fedu, mjob, fjob, traveltime, studytime, 
                                failures, schoolsup, activities, romantic, 
                                freetime, goout, dalc, walc, health, absences):
    if n_clicks>0:
        input_data = {
            'sex': sex,
            'age': age,
            'address': address,
            'famsize': famsize,
            'Pstatus': pstatus,
            'Medu': medu,
            'Fedu': fedu,
            'Mjob': mjob,
            'Fjob': fjob,
            'traveltime': traveltime,
            'studytime': studytime,
            'failures': failures,
            'schoolsup': schoolsup,
            'activities': activities,
            'romantic': romantic,
            'freetime': freetime,
            'goout': goout,
            'Dalc': dalc,
            'Walc': walc,
            'health': health,
            'absences': absences
        }

  
        grade= spå_betyg(input_data, lin_reg, X_train.columns)
        
    
        return html.H3(f"Your calculated grade is: {grade:.2f} out of 100")
    
    
    return html.H3('Here will your grade be shown')





if __name__=='__main__': #undvika sideeffects när man importerar till andra filer
    Timer(1,open_browser).start()
    app.run(debug=True, host=host, port=port)


