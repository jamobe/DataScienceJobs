# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
import dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import pickle
import os.path
import plotly.graph_objs as go
from sqlalchemy import create_engine
from preprocessing_dash import spacy_tokenizer, text_process

import warnings
warnings.filterwarnings("ignore")

def create_trace(rf, cluster_column):
    data = []
    clusters = rf[cluster_column].unique()
    colors_dict = {'unclassified': 'rgb(211,211,211)', 'Data Analyst': 'rgb(255, 127, 14)',
                   'Data Scientist': 'rgb(44, 160, 44)', 'Media & Marketing': 'rgb(214, 39, 40)',
                   'Data Engineer': 'rgb(148, 103, 189)'}
    for idx, cluster in enumerate(clusters):
        colors = colors_dict[cluster]
        df = rf.loc[rf[cluster_column] == cluster]
        if cluster in ['unclassified', 0]:
            data.append(dict(type='scatter', x=df.x, y=df.y, mode='markers', marker={'color': colors},
                             text=df['title'].map(str) + '<br>' + df['salary'].map(str) + '€', name='unclassified'))
        else:
            data.append(dict(type='scatter', x=df.x, y=df.y, mode='markers', marker={'color': colors},
                             text=df['title'].map(str) + '<br>' + df['salary'].map(str) + '€', name=str(cluster)))
    return data


def tfidf_encoding(df):
    with open(path + '/Pickles/TFIDF_model_all.pkl', 'rb') as tfidf_file:
        tfidf_model = pickle.load(tfidf_file)
    tfidf_array = tfidf_model.transform(df['description']).toarray()
    return tfidf_array


def ohe_encoding(df):
    with open(path + '/Pickles/OHE_model_all.pkl', 'rb') as ohe_file:
        ohe_model = pickle.load(ohe_file)
    ohe_array = ohe_model.transform(df.loc[:, ['country', 'region']]).toarray()
    return ohe_array


def umap_prediction(tfidf_array):
    with open(path + '/Visualization/umap_encoder.pkl', 'rb') as umap_file:
        mapper = pickle.load(umap_file)
    umap_array = mapper.transform(tfidf_array)
    return umap_array


def find_close_words(df, word, neighbors, model):
    close_words = model.wv.most_similar([word], topn=neighbors)
    word_list = []
    for i in range(neighbors):
        word_list.append(close_words[i][0])
    subdf = df[df['word'].isin(word_list)]
    return subdf


df_uk = pd.read_csv('https://raw.githubusercontent.com/jamobe/DataScienceJobs/master/data/uk_location_lookup.csv')
df_us = pd.read_csv('https://raw.githubusercontent.com/jamobe/DataScienceJobs/master/data/us-states.csv')
df_ger = pd.read_csv('https://raw.githubusercontent.com/jamobe/DataScienceJobs/master/data/locations_germany.csv')

all_options = {
    'UK': df_uk['region'].sort_values().unique().tolist(),
    'Germany': df_ger['region'].sort_values().unique().tolist(),
    'USA':  df_us['region'].sort_values().unique().tolist()
}

path = os.getcwd()

with open(path + '/Visualization/plots_density.pkl', 'rb') as file:
    fig0, all_fig, cluster_names, df_overview = pickle.load(file)

with open(path + '/Visualization/bar_cluster.pkl', 'rb') as bar_file:
    bar_fig = pickle.load(bar_file)

with open(path + '/Visualization/bar_salary.pkl', 'rb') as top_tech_file:
    top_tech_fig = pickle.load(top_tech_file)

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
                dcc.RadioItems(id='country',
                               options=[{'label': i, 'value': i} for i in all_options.keys()], value='Germany'),
                html.P(children='Select a region:'),
                dcc.Dropdown(id='region', style=dict(width='40hh')),
            ], style={'padding': 30}),
            html.Button(id='submit', children='predict salary', n_clicks=0, style={'fontSize': 14}),
            html.Div(id='prediction', style={'fontSize': 20, 'padding': 50}),
        ], style={'width': '39hh', 'display': 'inline-block', 'vertical-align': 'middle'})
    ]),


    html.Div([
        html.Div([
            dcc.Graph(id='salary_distribution_country', figure=all_fig[5]),
        ], style={'width': '49hh', 'display': 'inline-block', 'vertical-align': 'middle'}),
        html.Div([
            dcc.Graph(id='salary_distribution_jobtitle', figure=fig0)
        ], style={'width': '49hh', 'display': 'inline-block', 'vertical-align': 'middle'}),
        dcc.Dropdown(id='select_job_type', value=5,
                     options=[{'label': name, 'value': idx} for idx, name in enumerate(cluster_names)],
                     style={'width': '40%'}),
        # dash_table.DataTable(id='statistics_table', columns=[{"name": i, "id": i} for i in df_overview.columns],
        #                     data=df_overview.to_dict('records'), style_cell={'maxWidth': 0, 'overflow': 'hidden',
        #                     'textOverflow': 'ellipsis', 'textAlign': 'center'}, style_header={'fontWeight': 'bold'},
        #                    style_data_conditional=[{'if': {'column_id': 'role'},'backgroundColor': '#3D9970',
        #                    'fontWeight': 'bold', 'color': 'white',}],
        #                     style_as_list_view=True)
    ]),
    html.Div([
            html.Div([
                dcc.Graph(id='bar_cluster', figure=bar_fig[5]),
            ], style={'width': '49hh', 'display': 'inline-block', 'vertical-align': 'middle'}),
            html.Div([
                dcc.Graph(id='bar_salary', figure=top_tech_fig)
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
                        html.Div(children='Select number of similar words:'),
                        html.Div([
                            daq.Slider(id='number_similar', min=1, max=100, step=1, value=10,
                                       marks={1: '1', 50: '50', 100: '100'},
                                       handleLabel={"showCurrentValue": True, "label": " "}),
                        ], style={'display': 'inline-block', 'padding': 30, 'vertical-align': 'middle'}),
                        html.Div([
                            dcc.Input(id='number_similar_input', value=10, type='number', min=1, max=100, step=1)
                        ], style={'display': 'inline-block', 'padding': 30, 'vertical-align': 'middle'}),
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
    Output(component_id='prediction', component_property='children'),
    [Input(component_id='submit', component_property='n_clicks')],
    state=[State(component_id='description', component_property='value'),
           State(component_id='country', component_property='value'),
           State(component_id='region', component_property='value')]
)
def predict_salary(n_clicks, description, country, region):
    with open(path + '/Pickles/xgb_model_all.pkl', 'rb') as xgb_file:
        xgb_reg = pickle.load(xgb_file)
    with open(path + '/data/OHE_all.pkl', 'rb') as ohe_file:
        _, _, feature_names_ohe = pickle.load(ohe_file)
    with open(path + '/data/TFIDF_all.pkl', 'rb') as tfidf_file:
        _, _, feature_names_tfidf = pickle.load(tfidf_file)

    if n_clicks > 0:
        predict = pd.DataFrame({'description': [description], 'country': [country], 'region': [region]})
        tfidf_encode = tfidf_encoding(predict)
        ohe_encode = ohe_encoding(predict)
        x = pd.DataFrame(np.hstack((ohe_encode, tfidf_encode)),
                         columns=list(feature_names_ohe) + list(feature_names_tfidf))
        predicted = round(int(np.exp(xgb_reg.predict(x))) / 500) * 500
    else:
        predicted = 0
    return 'predicted salary: %2d€' % predicted


@app.callback(
    Output(component_id='UMAP_jobs', component_property='figure'),
    [Input(component_id='submit', component_property='n_clicks')],
    state=[State(component_id='description', component_property='value')]
)
def update_umap(n_clicks, description):
    with open(path + '/Visualization/umap_jobs.pkl', 'rb') as umap_file:
        rf = pickle.load(umap_file)

    data = create_trace(rf, 'name')
    if n_clicks > 0:
        predict_job = pd.DataFrame({'description': [description]})
        stack_predict = pd.DataFrame(np.repeat(predict_job.values, 60, axis=0))
        stack_predict.columns = predict_job.columns
        tfidf_encode_stack = tfidf_encoding(stack_predict)
        umap_pred = umap_prediction(tfidf_encode_stack)
        x_mean = np.median(umap_pred[:, 0]).reshape(-1)
        y_mean = np.median(umap_pred[:, 1]).reshape(-1)
        data.append(dict(type='scatter', x=x_mean, y=y_mean, mode='markers', name='your job',
                         marker={'size': 10, "color": 'black', "cmid": 0}))

    umap_figure = go.Figure(data=data, layout=dict(title='UMAP visualization for: job descriptions',
                                                   legend=dict(orientation="h"), hovermode='closest',
                                                   xaxis=dict(title=''), yaxis=dict(title=''), width=820, height=700))
    return umap_figure


@app.callback(
    [Output(component_id='salary_distribution_country', component_property='figure'),
     Output(component_id='bar_cluster', component_property='figure')],
    [Input(component_id='select_job_type', component_property='value')]
)
def select_density_plot(selected_job_type):
    return all_fig[selected_job_type], bar_fig[selected_job_type]


@app.callback(
    [Output(component_id='UMAP_words', component_property='figure'),
     Output(component_id='similar_words_results', component_property='children')],
    [Input(component_id='word_submit', component_property='n_clicks')],
    state=[State(component_id='word', component_property='value'),
           State(component_id='number_similar', component_property='value')]
)
def predict_umap_word(n_clicks, input_word, closest):
    with open(path + '/Pickles/word2vec_4.pkl', 'rb') as w2v_file:
        w2v_model = pickle.load(w2v_file)
    with open(path + '/Visualization/umap_words.pkl', 'rb') as umap_w2v_file:
        w2v = pickle.load(umap_w2v_file)
    data_word = [dict(type='scatter', x=w2v.x, y=w2v.y, mode='markers', marker=dict(color='lightgrey'),
                      text=w2v['word'], name='whole vocabulary')]
    if n_clicks > 0:
        input_word = input_word.lower()
        if input_word in w2v_model.wv.vocab:
            search_word = w2v.loc[w2v['word'] == input_word]
            similar_words = find_close_words(w2v, input_word, closest, w2v_model)
            data_word.append(dict(type='scatter', x=similar_words.x, y=similar_words.y, mode='markers',
                                  marker=dict(color='green', size=8), text=similar_words['word'], name='similar words'))
            data_word.append(dict(type='scatter', x=search_word.x, y=search_word.y, mode='markers',
                                  marker=dict(color='red', size=12), text=search_word['word'], name=input_word))
            output = 'Similar words: ' + ', '.join(word for word in similar_words['word'])
        else:
            output = 'word not in vocabulary'
    else:
        output = ' '
    figure = go.Figure(data=data_word, layout=dict(title='UMAP visualization for: vocabulary', title_x=0.5,
                                                   legend=dict(orientation="h"), hovermode='closest',
                                                   xaxis=dict(title=''), yaxis=dict(title=''), width=820, height=700))
    return figure, output


@app.callback(
    Output(component_id='number_similar', component_property='value'),
    [Input(component_id='number_similar_input', component_property='value')]
)
def map_input_slider(value):
    return value


if __name__ == '__main__':
    app.run_server(debug=True)
