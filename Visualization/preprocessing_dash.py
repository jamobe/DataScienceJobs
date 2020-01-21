import pandas as pd
import numpy as np
import pickle
import umap
import re
import os.path
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import spacy
from spacy.lang.en import English
import plotly.figure_factory as ff
import plotly.graph_objs as go
import warnings


def load_data(access_path):
    with open(access_path + '/data/SQL_access.pkl', 'rb') as password_file:
        password = pickle.load(password_file)
    engine = create_engine('postgresql://postgres:' + password +
                           '@dsj-1.c9mo6xd9bf9d.us-west-2.rds.amazonaws.com:5432/')
    loaded_data = pd.read_sql("select * from all_data where language like 'en' and salary_type like 'yearly'", engine)
    loaded_data = loaded_data.dropna(subset=['region', 'country', 'description', 'job_title'], axis=0)
    loaded_data['full_description'] = loaded_data['job_title'] + ' ' + loaded_data['description']
    return loaded_data


def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Lower case of all words
    2. Remove all punctuation
    3. Remove all stopwords
    4. Returns a list of the cleaned text
    """
    mess = mess.lower()
    mess = re.sub(r'[^A-Za-z]+', ' ', mess)  # remove non alphanumeric character [^A-Za-z0-9]
    mess = re.sub(r'https?:/\/\S+', ' ', mess)  # remove links

    # Now just remove any stopwords
    return [word for word in mess.split() if word not in spacy.lang.en.stop_words.STOP_WORDS]


def spacy_tokenizer(doc):
    """
    Tokenizing and lemmatizing the document using SpaCy
    :param doc: text
    :return:
    """
    spacy.load('en_core_web_lg')
    lemmatizer = spacy.lang.en.English()
    tokens = lemmatizer(doc)
    return [token.lemma_ for token in tokens]


def encode_tfidf(df):
    tfidf = TfidfVectorizer(analyzer=text_process, use_idf=True, tokenizer=spacy_tokenizer, min_df=3)
    tfidf_fit = tfidf.fit(df['full_description'])
    tfidf_array = tfidf_fit.transform(df['full_description']).toarray()
    with open(path + '/Visualization/tfidf_encoder_all.pkl', 'wb') as tfidf_file:
        pickle.dump([tfidf_fit, tfidf_array], tfidf_file)
    return tfidf_array


def umap_jobs(array):
    print(array.shape)
    umapper = umap.UMAP(n_neighbors=60, min_dist=0.5, n_components=15, random_state=42)
    # n_neighbors=15, min_dist=0.0, n_components=2
    mapper = umapper.fit(array)
    umapped_array = mapper.transform(array)
    print(umapped_array.shape)
    with open(path + '/Visualization/umap_encoder_2.pkl', 'wb') as umap_file:
        pickle.dump(mapper, umap_file)
    return umapped_array


def occurences(string, word_list):
    counts = 0
    for word in word_list:
        counts += string.count(word)
    return counts


def find_label(df):
    df['name'] = 'unclassified'
    for cluster in df.label.unique():
        df_cluster = df.loc[df['label'] == cluster]
        counts = []
        df_cluster['full_description'] = df_cluster['full_description'].apply(text_process).apply(' '.join)
        df_cluster['title'] = df_cluster['title'].apply(text_process).apply(' '.join)
        names = {'Data Scientist': ['data scientist', 'machine learning', ' ai '],
                 'Data Engineer': ['engineer', 'java', ' aws '],
                 'Data Analyst': ['data analyst', 'business intelligence', 'insight analyst'],
                 'Media & Marketing': ['media', 'marketing', 'digital']}
        for keys, values in names.items():
            counts.append(df_cluster['full_description'].apply(occurences, args=(values,)).sum())
        most_common = np.array(counts).argmax()
        if np.sort(counts)[-1]*0.75 > np.sort(counts)[-2]:
            df.loc[df['label'] == cluster, 'name'] = list(names)[most_common]
    return df


def define_label(string):
    name = 'unclassified'
    cleaned = ' '.join(text_process(string))
    names = {'Data Scientist': ['data scientist', 'machine learning', ' ai '],
             'Data Engineer': ['engineer', 'java', ' aws '],
             'Data Analyst': ['analyst', 'business intelligence', 'insight'],
             'Media & Marketing': ['media', 'marketing', 'digital']}
    counts = []
    for keys, values in names.items():
        counts.append(occurences(cleaned, values))
    most_common = np.array(counts).argmax()
    if np.sort(counts)[-1]*0.75 > np.sort(counts)[-2]:
        name = list(names)[most_common]
    return name


def create_density_plots(df, cluster_names):
    colors_dict = {'unclassified': 'rgb(211, 211, 211)', 'Data Analyst': 'rgb(255, 127, 14)',
                   'Data Scientist': 'rgb(44, 160, 44)', 'Media & Marketing': 'rgb(214, 39, 40)',
                   'Data Engineer': 'rgb(148, 103, 189)'}
    color_list = [color for color in colors_dict.values()]
    data_dist = []
    cluster_overview = pd.DataFrame(columns=['Role', 'Mean Salary', 'Median Salary',
                                             'Minimum Salary', 'Maximum Salary', 'Counts'])
    df = df.dropna(subset=['salary'], axis=0)
    all_figures = []
    for idx, cluster in enumerate(cluster_names):
        df_cluster = df.loc[df['name'] == cluster]
        data_dist.append(df_cluster.salary)
        cluster_overview.loc[idx, 'Role'] = cluster
        cluster_overview.loc[idx, 'Mean Salary'] = round(df_cluster.salary.mean())
        cluster_overview.loc[idx, 'Median Salary'] = round(df_cluster.salary.median())
        cluster_overview.loc[idx, 'Minimum Salary'] = round(df_cluster.salary.min())
        cluster_overview.loc[idx, 'Maximum Salary'] = round(df_cluster.salary.max())
        cluster_overview.loc[idx, 'Counts'] = len(df_cluster.salary)
        country_list = ['UK', 'Germany', 'USA']
        data_dist_country = []
        for country in country_list:
            df_cluster_country = df_cluster.loc[df_cluster['country'] == country]
            data_dist_country.append(df_cluster_country.salary)

        fig = ff.create_distplot(data_dist_country, country_list, bin_size=5000)
        fig.update_layout(title="Salary distribution by country for " + cluster, title_x=0.5,
                          xaxis_title='salary [in €]', yaxis_title='density')
        all_figures.append(fig)

    cluster_overview.loc[len(cluster_names)+1, 'Role'] = 'all'
    cluster_overview.loc[len(cluster_names)+1, 'Mean Salary'] = round(df.salary.mean())
    cluster_overview.loc[len(cluster_names)+1, 'Median Salary'] = round(df.salary.median())
    cluster_overview.loc[len(cluster_names)+1, 'Minimum Salary'] = round(df.salary.min())
    cluster_overview.loc[len(cluster_names)+1, 'Maximum Salary'] = round(df.salary.max())
    cluster_overview.loc[len(cluster_names)+1, 'Counts'] = len(df.salary)

    country_list = ['UK', 'Germany', 'USA']
    data_dist_country = []
    for country in country_list:
        df_country = df.loc[df['country'] == country]
        data_dist_country.append(df_country.salary)

    fig = ff.create_distplot(data_dist_country, country_list, bin_size=5000)
    fig.update_layout(title="Salary distribution by country for all", title_x=0.5, xaxis_title='salary [in €]',
                      yaxis_title='density')
    all_figures.append(fig)
    groups = np.append(cluster_names, 'all')

    figure0 = ff.create_distplot(data_dist, cluster_names, colors=color_list, show_hist=False, bin_size=5000)
    figure0.update_layout(title="Salary distribution by cluster name", title_x=0.5, xaxis_title='salary [in €]',
                          yaxis_title='density')
    return figure0, all_figures, groups, cluster_overview


def create_word_umap(input_path):
    with open(input_path + '/Pickles/word2vec_4.pkl', 'rb') as w2v_file:
        w2v_model = pickle.load(w2v_file)
    words = set(w2v_model.wv.vocab)
    vectors = []
    vocab = []
    for word in words:
        vectors.append(w2v_model.wv.__getitem__([word]))
        vocab.append(word)
    vectors = np.asarray(vectors)
    vectors = vectors.reshape(-1, 300)
    word_umapper = umap.UMAP(n_neighbors=5, min_dist=0.0, n_components=2, random_state=42)
    word_mapped = word_umapper.fit(vectors)
    word_map = word_mapped.transform(vectors)
    word2vector = pd.DataFrame({'x': [x for x in word_map[:, 0]], 'y': [y for y in word_map[:, 1]], 'word': vocab})
    return word2vector


def tech_process(mess, important_terms, abbr):
    mess = mess.lower()
    mess = mess.replace("\\n", " ")
    punctuations = '[].:;"()/\\'
    nopunc = [char for char in mess if char not in punctuations]
    nopunc = ''.join(nopunc)
    return [abbr.get(x, x) for x in important_terms if x in nopunc]


def load_map_tech_dict(tech_path, df):
    tech_dict = pd.read_pickle(tech_path + '/Pickles/broad_tech_dictionary.pickle')
    categories_to_include = ['front_end-technologies', 'databases', 'software-infrastructure-devops', 'data-science',
                             'software_architecture', 'web_design', 'tools', 'cyber_security', 'cloud_computing',
                             'back_end-technologies', 'mobile']
    tech_list = list(set([item.lower() for key in categories_to_include for item in tech_dict[key]]))
    abbr = {' bi ': ' business intelligence ', ' ai ': ' artificial intelligence ', ' databases ': ' database ',
            ' db ': ' database ', ' aws ': ' amazon web services ', 'nlp': 'natural language processing'}
    df['tech_list'] = df['full_description'].apply(tech_process, args=(tech_list, abbr))
    return df, tech_dict


def tech_encoding(df):
    mlb = MultiLabelBinarizer()
    tech_array = pd.DataFrame(mlb.fit_transform(df.pop('tech_list')), columns=mlb.classes_, index=df.index)
    return tech_array


def create_top_technology_cluster_barplot(df, encoded_tech, clusters):
    all_bar_fig = []
    for cluster in clusters:
        sub_df = df.loc[df['name'] == cluster]
        bar_plot = pd.DataFrame(
            encoded_tech.loc[sub_df.index].sum(axis=0, skipna=True, numeric_only=True).sort_values(ascending=False),
            columns=['counts'])
        bar_plot = bar_plot[bar_plot.index != '  ']
        bar_data = go.Bar(x=bar_plot['counts'][0:9], y=bar_plot.index[0:9], orientation='h', name=cluster)
        layout = dict(title='Most important technologies for a ' + cluster, yaxis=dict(autorange="reversed"),
                      xaxis_title='occurrences in ' + str(len(sub_df)) + ' job descriptions')
        bar_fig = go.Figure(data=bar_data, layout=layout)
        all_bar_fig.append(bar_fig)

    bar_plot = pd.DataFrame(encoded_tech.sum(axis=0, skipna=True, numeric_only=True).sort_values(ascending=False),
                            columns=['counts'])
    bar_plot = bar_plot[bar_plot.index != '  ']
    bar_data = go.Bar(x=bar_plot['counts'][0:9], y=bar_plot.index[0:9], orientation='h', name='all')
    layout = dict(title='Most important technologies for a all', yaxis=dict(autorange="reversed"),
                  xaxis_title='occurrences in ' + str(len(rf)) + ' job descriptions')
    bar_fig = go.Figure(data=bar_data, layout=layout)
    all_bar_fig.append(bar_fig)
    return all_bar_fig


def create_technology_salary(df, tech_dict, encoded_tech):
    for key in tech_dict.keys():
        tech_dict[key] = [x.lower() for x in tech_dict[key]]

    df_tech = df[['salary']].join(encoded_tech)
    df_tech_list = pd.DataFrame(columns=df_tech.columns[2:])
    for column in df_tech.columns[2:]:
        df_tech_list.loc[1, column] = df_tech.loc[df_tech[column] == 1, 'salary'].median()
        df_tech_list.loc[2, column] = df_tech.loc[df_tech[column] == 1, 'salary'].count()
    threshold = 10
    df_tech_trans = df_tech_list.T
    df_tech_trans = df_tech_trans[df_tech_trans[2] > threshold]
    top_tech = df_tech_trans.sort_values(by=1, ascending=False)

    violin_fig = go.Figure()
    box_fig = go.Figure()
    for idx, top10 in enumerate(top_tech.index[0:9]):
        tech_data = df_tech.loc[df_tech[top10] == 1, ['salary',top10]]
        tech_data.loc[:,top10] = top10
        violin_fig.add_trace(go.Violin(x=tech_data[top10], y=tech_data.salary,
                                    box_visible=True, meanline_visible=True))
        box_fig.add_trace(go.Box(y=tech_data.salary, name=top10))
    violin_fig.update_layout(dict(title='Advertised salary average (€) of job ads by technologies referenced',
                                  showlegend=False))
    box_fig.update_layout(dict(title='Top 10 technologies ranked by highest median salary',
                                  showlegend=False, yaxis_title='salary [in €]'))
    labels = []
    for idx, name in enumerate(top_tech.index[0:9]):
        tmp = [key for key, value in tech_dict.items() if name in value]
        tmp = ', '.join(tmp)
        labels.append(top_tech.index[idx] + '<sub>(' + tmp + ')</sub>')

    top_tech_bar = go.Bar(x=top_tech[1][0:9], y=labels, orientation='h') #top_tech.index[0:9]
    layout = dict(title='Advertised salary average (€) of job ads by technologies referenced',
                  yaxis=dict(autorange="reversed"), xaxis_title='mean salary')
    top_tech_fig = go.Figure(data=top_tech_bar, layout=layout)
    return box_fig


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    path = os.getcwd()
    data = load_data(path)
    encoded_array = encode_tfidf(data)
    umap_array = umap_jobs(encoded_array)

    rf = pd.DataFrame({'x': [x for x in umap_array[:, 0]],
                       'y': [y for y in umap_array[:, 1]],
                       'company': data['company'],
                       'region': data['region'],
                       'country': data['country'],
                       'title': data['job_title'],
                       'salary': data['salary_average_euros'],
                       'full_description': data['full_description']})

    rf['name'] = rf['full_description'].apply(define_label)

    with open(path + '/Visualization/umap_jobs.pkl', 'wb') as file:
        pickle.dump(rf, file)

    cluster_name = rf.name.unique()
    fig0, all_fig, outputname, df_overview = create_density_plots(rf, cluster_name)
    with open(path + '/Visualization/plots_density.pkl', 'wb') as file:
        pickle.dump([fig0, all_fig, outputname, df_overview], file)

    rf, technical_dict = load_map_tech_dict(path, rf)
    encoded_tech_array = tech_encoding(rf)

    all_bar_figure = create_top_technology_cluster_barplot(rf, encoded_tech_array, cluster_name)
    with open(path + '/Visualization/bar_cluster.pkl', 'wb') as file:
        pickle.dump(all_bar_figure, file)

    top_tech_salary = create_technology_salary(rf, technical_dict, encoded_tech_array)

    with open(path + '/Visualization/bar_salary.pkl', 'wb') as file:
        pickle.dump(top_tech_salary, file)

    w2v = create_word_umap(path)

    with open(path + '/Visualization/umap_words.pkl', 'wb') as file:
        pickle.dump(w2v, file)
