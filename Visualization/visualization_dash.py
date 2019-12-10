# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd

df_uk = pd.read_csv('https://raw.githubusercontent.com/jamobe/DataScienceJobs/master/data/uk_location_lookup.csv')
df_us = pd.read_csv('https://raw.githubusercontent.com/jamobe/DataScienceJobs/master/data/us-states.csv')

country = ['UK','Germany','USA']
regions_uk = df_uk['region'].unique()
regions_us = df_us['region'].unique()


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    children=[html.H1(children='From job descriptions to salary predictions'),

    html.H3(children='by Rachel Lund and Janina Mothes'),

    html.Div(children='''Dash: A web application framework for Python.'''),



    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
            ],
            'layout': {
                'title': 'Dash Data Visualization'
            }
        }
    ),


    html.Div([dcc.Input(id='my-id', placeholder='paste your job description', type='text', required=True),
    html.Div(id='my-div')]),

    html.Div('Select a country:'),
    html.Div([
            dcc.RadioItems(
                id='country',
                options=[{'label': i, 'value': i} for i in country],
            ),
            dcc.Dropdown(
                id='region',
                options=[{'label': i, 'value':i} for i in regions]
            )
        ],
        style={'width': '50%', 'display': 'inline-block'}),

])


@app.callback(
    Output(component_id='my-div', component_property='children'),
    [Input(component_id='my-id', component_property='value')]
)
def update_output_div(input_value):
    return 'job description: "{}"'.format(input_value)


if __name__ == '__main__':
    app.run_server(debug=True)
