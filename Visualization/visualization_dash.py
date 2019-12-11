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

def create_trace(rf, clusters):
    data = []
    rf = find_label(rf)
    for idx, cluster in enumerate(clusters):
        rf_sub = rf.loc[rf['label'] == cluster]
        if cluster == -1:
            data.append(dict(type='scatter', x=rf_sub.x, y=rf_sub.y, mode='markers', marker=dict(color='lightgrey'), hovertext=rf_sub['title'], name=str(rf_sub['name'].unique()[0])))
        else:
            data.append(dict(type='scatter', x=rf_sub.x, y=rf_sub.y, mode='markers', marker={"color": cluster, "cauto": 0}, hovertext=rf_sub['title'], name=str(rf_sub['name'].unique()[0])))
    return data


def find_label(rf):
    rf['name'] = 'unknown'
    for cluster in rf.label.unique():
        df = rf.loc[rf['label'] == cluster]
        counts = []
        names = {'Data Scientist':'Scientist', 'Data Engineer':'Engineer', 'Data Analyst':'Analyst', 'Developer':'Develop'}
        for keys, values in names.items():
            total = len(df)
            counts.append(df['title'].str.contains(values).sum()/total)
        most_common = np.array(counts).argmax()
        if np.array(counts).max() >= 0.25:
            rf.loc[rf['label'] == cluster, 'name']= list(names)[most_common]
    return rf


def TFIDF_encoding(df):
    with open(path + '/Pickles/TFIDF_model.pkl', 'rb') as file:
        TFIDF_model = pickle.load(file)
    tfidf_array = TFIDF_model.transform(df['description']).todense()
    return tfidf_array


def OHE_encoding(df):
    with open(path + '/Pickles/OHE_model.pkl', 'rb') as file:
        OHE_model = pickle.load(file)
    OHE_array = OHE_model.transform(df.loc[:,['country', 'region']]).toarray()
    return OHE_array


def umap_prediction(tfidf_array):
    with open(path + '/Visualization/umap_encoder.pkl', 'rb') as file:
        mapper = pickle.load(file)
    mapper._sparse_data = False
    umap_array = mapper.transform(tfidf_array)
    return umap_array


def cluster_prediction(umap_array):
    with open(path + '/Visualization/cluster_labeler.pkl', 'rb') as file:
        clusterer = pickle.load(file)
    cluster_labels = clusterer.fit_predict(umap_array[:, 0:2])
    return cluster_labels


def tech_process(mess, important_terms):
    mess = mess.lower()
    mess = mess.replace("\\n"," ")
    punctuations = '[].:;"()/\\'
    nopunc = [char for char in mess if char not in punctuations]
    nopunc = ''.join(nopunc)
    return [x for x in important_terms if x in nopunc]


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
        html.H1(children='From job descriptions to salary predictions'),
        html.H3(children='by Rachel Lund and Janina Mothes'),
        html.Div(children='''Dash: A web application framework for Python.'''),
        dcc.Graph(id='UMAP_jobs'),
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
     Input(component_id='submit', component_property='n_clicks')]
)
def predict_umap(description, n_clicks):
    with open(path + '/Visualization/umap_jobs.pkl', 'rb') as file:
        rf, cluster_labels = pickle.load(file)
    data = create_trace(rf, np.unique(cluster_labels))
    predict = pd.DataFrame({'description': [description]})
    if n_clicks > 0:
        tfidf_encode = TFIDF_encoding(predict)
        umap_pred = umap_prediction(tfidf_encode)
        data.append(dict(type='scatter', x=umap_pred[:, 0], y=umap_pred[:, 1], mode='markers', marker={'size': 12, "color": 'black', "cmid": 0}))
    return {'data': data,
            'layout': dict(title='UMAP visualization for: job descriptions', hovermode='closest', xaxis=dict(title=''), yaxis=dict(title=''),
                           width=800, height=800)}


@app.callback(
    Output(component_id='prediction', component_property='children'),
    [Input(component_id='description', component_property='value'),
     Input(component_id='country', component_property='value'),
     Input(component_id='region', component_property='value'),
     Input(component_id='submit', component_property='n_clicks')]
)
def predict_salary(description, country, region, n_clicks):
    predict = pd.DataFrame({'description': [description], 'country': [country], 'region': [region]})
    if n_clicks > 0:
        tfidf_encode = TFIDF_encoding(predict)
        ohe_encode = OHE_encoding(predict)
        with open(path + '/Pickles/xgb_model.pkl', 'rb') as file:
            xgb_reg = pickle.load(file)
        with open(path + '/data/OHE.pkl', 'rb') as file:
            _, _, _, feature_names_OHE = pickle.load(file)
        with open(path + '/data/TFIDF.pkl', 'rb') as file:
            _, _, _, feature_names_TFIDF = pickle.load(file)
        X = pd.DataFrame(np.hstack((ohe_encode, tfidf_encode)),columns=list(feature_names_OHE) + list(feature_names_TFIDF))
        predicted = np.exp(xgb_reg.predict(X))
        print(type(predicted))
    else:
        predicted = n_clicks
    return 'predicted salary: {} €'.format(predicted)


if __name__ == '__main__':
    app.run_server(debug=True)
