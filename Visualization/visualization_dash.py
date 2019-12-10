# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import pickle
import os.path
from UMAP_job import spacy_tokenizer, text_process

df_uk = pd.read_csv('https://raw.githubusercontent.com/jamobe/DataScienceJobs/master/data/uk_location_lookup.csv')
df_us = pd.read_csv('https://raw.githubusercontent.com/jamobe/DataScienceJobs/master/data/us-states.csv')
df_ger = pd.read_csv('https://raw.githubusercontent.com/jamobe/DataScienceJobs/master/data/locations_germany.csv')

all_options = {
    'UK': df_uk['region'].sort_values().unique().tolist(),
    'Germany': df_ger['region'].sort_values().unique().tolist(),
    'USA':  df_us['region'].sort_values().unique().tolist()
}

path = os.getcwd()
with open(path + '/Visualization/umap_jobs.pkl', 'rb') as file:
    rf, cluster_labels = pickle.load(file)


def create_trace(rf, clusters):
    data = []
    for idx, cluster in enumerate(clusters):
        rf_sub = rf.loc[rf['label'] == cluster]
        if cluster == -1:
            data.append(dict(type='scatter', x=rf_sub.x, y=rf_sub.y, mode='markers', marker=dict(color='lightgrey'), text=rf_sub['title'], name=str(cluster)))
        else:
            data.append(dict(type='scatter', x=rf_sub.x, y=rf_sub.y, mode='markers', marker={"color": cluster, "cmid": 0}, text=rf_sub['title'], name=str(cluster)))
    return data



def TFIDF_encoding(text):
    with open(path + '/Pickles/TFIDF_model.pkl', 'rb') as file:
        TFIDF_model = pickle.load(file)
    tfidf_array = TFIDF_model.transform([text]).toarray()
    return tfidf_array


def OHE_encoding(country, region):
    with open(path + '/Pickles/OHE_model.pkl', 'rb') as file:
        OHE_model = pickle.load(file)
    OHE_array = OHE_model.transform(np.array([country, region]).reshape(1,-1)).toarray()
    return OHE_array


def umap_prediction(text):
    with open(path + '/Visualization/umap_encoder.pkl', 'rb') as file:
        mapper = pickle.load(file)
    umap_array = umapper.fit_transform(array)
    return umap_array

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
        html.H1(children='From job descriptions to salary predictions'),
        html.H3(children='by Rachel Lund and Janina Mothes'),
        html.Div(children='''Dash: A web application framework for Python.'''),
        dcc.Graph(
            id='UMAP_jobs',
            figure={'data': data,'layout': dict(title='UMAP visualization for: job descriptions', xaxis=dict(title=''),
                                                yaxis=dict(title=''), width=600, height=600),
                #'config': dict(displaylogo=False,editable=True)
            }
        ),
        html.Hr(),
        html.H4('Predict salary for a job descriptions.'),
        html.Div('Enter your jobs description here (only English language):'),
        dcc.Textarea(id='description', placeholder='paste your job description', cols=100, rows=7, required=True),
        html.Div([
            html.P('Select a country:'),
            dcc.RadioItems(id='country', options=[{'label': i, 'value': i} for i in all_options.keys()], value='UK'),
            html.P('Select a region:'),
            dcc.Dropdown(
                id='region',
                style=dict(width='40%')
            ),
        ]),
        html.Button(id='submit', children='predict salary', n_clicks=0),
        html.Div(id='prediction'),
    ])

@app.callback(
    Output(component_id='region', component_property='options'),
    [Input(component_id='country', component_property='value')]
)
def set_region_options(selected_country):
    return [{'label': i, 'value': i} for i in all_options[selected_country]]

@app.callback(
    Output('region', 'value'),
    [Input('region', 'options')])
def set_region_value(available_options):
    return available_options[0]['value']


@app.callback(
    Output(component_id='UMAP_jobs', component_property='figure'),
    [Input(component_id='description', component_property='value'),
     Input(component_id='country', component_property='value'),
     Input(component_id='region', component_property='value'),
     Input(component_id='submit', component_property='n_clicks')]
)
def predict_salary(description, country, region, n_clicks):
    if n_clicks > 0:
        tfidf_encode = TFIDF_encoding(description)
        ohe_encode = OHE_encoding(country, region)
        x = np.hstack((ohe_encode, tfidf_encode))
        with open(path + '/Pickles/xgb_model.pkl', 'rb') as file:
            xgb_reg = pickle.load(file)
        predicted = np.exp(xgb_reg.predict(x))
    else:
        predicted = n_clicks
    return 'predicted salary: {} €'.format(n_clicks)


@app.callback(
    Output(component_id='prediction', component_property='children'),
    [Input(component_id='description', component_property='value'),
     Input(component_id='country', component_property='value'),
     Input(component_id='region', component_property='value'),
     Input(component_id='submit', component_property='n_clicks')]
)
def predict_salary(description, country, region, n_clicks):
    if n_clicks > 0:
        tfidf_encode = TFIDF_encoding(description)
        ohe_encode = OHE_encoding(country, region)
        x = np.hstack((ohe_encode, tfidf_encode))
        with open(path + '/Pickles/xgb_model.pkl', 'rb') as file:
            xgb_reg = pickle.load(file)
        predicted = np.exp(xgb_reg.predict(x))
    else:
        predicted = n_clicks
    return 'predicted salary: {} €'.format(n_clicks)


if __name__ == '__main__':
    app.run_server(debug=True)
