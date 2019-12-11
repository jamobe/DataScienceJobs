# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import pickle
import os.path
import plotly.graph_objs as go
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


def close_words(df, word, neighbors, model):
    close_words = model.wv.most_similar([word], topn=neighbors)
    word_list = []
    for i in range(neighbors):
        word_list.append(close_words[i][0])
    subdf = df[df['word'].isin(word_list)]
    return subdf


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div([
        html.H1(children='From job descriptions to salary predictions'),
        html.H3(children='by Rachel Lund and Janina Mothes'),
    ], style={'textAlign': 'center'}),
    html.Div([
        html.Div([
            dcc.Graph(id='UMAP_jobs'),
        ], style={'width': '59hh', 'display': 'inline-block', 'vertical-align': 'middle'}),
        html.Div([
            html.H4(children='Predict salary for a job descriptions:'),
            html.Div(children='Enter your jobs description here (in English only):'),
            dcc.Textarea(id='description', value='We need a Python genius', cols=60, rows=20, required=True),
            html.Div([
                html.P(children='Select a country:'),
                dcc.RadioItems(id='country', options=[{'label': i, 'value': i} for i in all_options.keys()], value='UK'),
                html.P(children='Select a region:'),
                dcc.Dropdown(id='region',style=dict(width='40hh')),
            ], style={'padding': 30}),
            html.Button(id='submit', children='predict salary', n_clicks=0, style={'fontSize': 14}),
            html.Div(id='prediction', style={'fontSize': 20, 'padding': 50}),
        ], style={'width': '39hh', 'display': 'inline-block', 'vertical-align': 'middle'})
    ]),

    html.Div([
        html.Div([
            dcc.Graph(id='salary_distribution_jobtitle')
        ], style={'width': '49hh', 'display': 'inline-block', 'vertical-align': 'middle'}),
        html.Div([
            dcc.Graph(id='salary_distribution_country')
        ], style={'width': '49hh', 'display': 'inline-block', 'vertical-align': 'middle'}),

    ]),
    html.Hr(),
    html.Div([
            html.Div([
                dcc.Graph(id='UMAP_words')
            ], style={'width': '59hh', 'display': 'inline-block', 'vertical-align': 'middle'}),
            html.Div([
                html.Div([
                    html.H3(children='Find similar words:'),
                    html.Div(children='Enter your word here (in English only):'),
                    dcc.Input(id='word', value='python', required=True),
                    html.Div([
                        html.Div([
                            daq.Slider(id='number_similar', min=1, max=1000, step=1, value=10,
                                   marks={1: '1',500: '500',1000: '1000'},
                                   handleLabel={"showCurrentValue": True, "label": " "}),
                        ], style={'display':'inline-block', 'padding': 30, 'vertical-align': 'middle'}),
                        html.Div([
                            dcc.Input(id='number_similar_input', value=10, type='number',min=1, max=1000, step=1)
                        ],style={'display':'inline-block', 'padding': 30, 'vertical-align': 'middle'}),
                    ], style={'padding': 30}),
                ], style={'padding': 30}),
                html.Button(id='word_submit', children='Find similar words', n_clicks=0, style={'fontSize': 14}),

            ], style={'width': '39hh', 'display': 'inline-block', 'vertical-align': 'middle'}),
        html.Div(id='similar_words_results', style={'fontSize': 20, 'padding': 50}),
        ]),
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
            'layout': dict(title='UMAP visualization for: job descriptions', legend=dict(orientation="h"),
                           hovermode='closest', xaxis=dict(title=''), yaxis=dict(title=''), width=820, height=700)}


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
        X = pd.DataFrame(np.hstack((ohe_encode, tfidf_encode)), columns=list(feature_names_OHE) + list(feature_names_TFIDF))
        predicted = np.exp(xgb_reg.predict(X))
    else:
        predicted = n_clicks
    return 'predicted salary: {} €'.format(predicted)


@app.callback(
    [Output(component_id='UMAP_words', component_property='figure'),
     Output(component_id='similar_words_results', component_property='children')],
    [Input(component_id='word', component_property='value'),
     Input(component_id='number_similar', component_property='value'),
     Input(component_id='word_submit', component_property='n_clicks')]
)
def predict_umap_word(input_word, closest,  n_clicks):
    with open(path + '/Pickles/word2vec_4.pkl', 'rb') as file:
        w2v_model = pickle.load(file)
    with open(path + '/Visualization/umap_words.pkl', 'rb') as file:
        w2v, word_mapper = pickle.load(file)
    data_word = [dict(type='scatter', x=w2v.x, y=w2v.y, mode='markers', marker=dict(color='lightgrey'),
                     text=w2v['word'], name='whole vocabulary')]
    if n_clicks > 0:
        input_word = input_word.lower()
        if input_word in w2v_model.wv.vocab:
            search_word = w2v.loc[w2v['word'] == input_word]
            similar_words = close_words(w2v, input_word, closest, w2v_model)
            data_word.append(dict(type='scatter', x=similar_words.x, y=similar_words.y, mode='markers',
                                  marker=dict(color='green', size=8), text=similar_words['word'], name='similar words'))
            data_word.append(dict(type='scatter', x=search_word.x, y=search_word.y, mode='markers',
                                  marker=dict(color='red', size=12), text=search_word['word'], name=input_word))
            output = 'Similar words: ' + ', '.join(word for word in similar_words['word'])
        else:
            output = 'word not in vocabulary'
    else:
        output = ' '
    figure = go.Figure(data=data_word,layout=dict(title='UMAP visualization for: vocabulary', legend=dict(orientation="h"),
                           hovermode='closest', xaxis=dict(title=''), yaxis=dict(title=''), width=820, height=700))
    return figure, output


@app.callback(
    Output(component_id='number_similar', component_property='value'),
    [Input(component_id='number_similar_input', component_property='value')]
)
def map_input_slider(value):
    return value


if __name__ == '__main__':
    app.run_server(debug=True)
