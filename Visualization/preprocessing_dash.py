import pandas as pd
import numpy as np
import pickle
import umap
import re
import os.path
from sqlalchemy import create_engine
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from spacy.lang.en import English
import plotly.figure_factory as ff
import warnings


def load_data(access_path):
    with open(access_path + '/data/SQL_access.pkl', 'rb') as password_file:
        password = pickle.load(password_file)
    engine = create_engine('postgresql://postgres:' + password +
                           '@dsj-1.c9mo6xd9bf9d.us-west-2.rds.amazonaws.com:5432/')
    data = pd.read_sql("select * from all_data where language like 'en' and salary_type like 'yearly'", engine)
    data = data.dropna(subset=['region', 'country', 'description', 'job_title'], axis=0)
    data['full_description'] = data['job_title'] + ' ' + data['description']
    return data


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
    umapper = umap.UMAP(n_neighbors=15, min_dist=0.0, n_components=2, random_state=42)
    mapper = umapper.fit(array)
    umapped_array = mapper.transform(array)
    with open(path + '/Visualization/umap_encoder.pkl', 'wb') as umap_file:
        pickle.dump(mapper, umap_file)
    return umapped_array


def clustering(umap_array):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=120, min_samples=2)
    clustered = clusterer.fit_predict(umap_array[:, 0:2])
    with open(path + '/Visualization/cluster_labeler.pkl', 'wb') as cluster_file:
        pickle.dump(clusterer, cluster_file)
    return clustered


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


def create_density_plots(df):
    colors_dict = {'unclassified': 'rgb(211,211,211)', 'Data Analyst': 'rgb(255, 127, 14)',
                   'Data Scientist': 'rgb(44, 160, 44)', 'Media & Marketing': 'rgb(214, 39, 40)',
                   'Data Engineer': 'rgb(148, 103, 189)'}
    color_list = [color for color in colors_dict.values()]
    data_dist = []
    cluster_overview = pd.DataFrame(columns=['role', 'mean salary','median salary', 'min salary',
                                             'max salary', 'counts'])
    df = df.dropna(subset=['salary', 'title'], axis=0)
    cluster_names = df.name.unique()
    all_figures = []
    for idx, cluster in enumerate(cluster_names):
        df_cluster = df.loc[rf['name'] == cluster]
        data_dist.append(df_cluster.salary)
        cluster_overview.loc[idx, 'role'] = cluster
        cluster_overview.loc[idx, 'mean salary'] = df_cluster.salary.mean()
        cluster_overview.loc[idx, 'median salary'] = df_cluster.salary.median()
        cluster_overview.loc[idx, 'min salary'] = df_cluster.salary.min()
        cluster_overview.loc[idx, 'max salary'] = df_cluster.salary.max()
        cluster_overview.loc[idx, 'counts'] = len(df_cluster.salary)
        country_list = ['UK', 'Germany', 'USA']
        data_dist_country = []
        for country in country_list:
            df_cluster_country = df_cluster.loc[df_cluster['country'] == country]
            data_dist_country.append(df_cluster_country.salary)

        fig = ff.create_distplot(data_dist_country, country_list, bin_size=5000)
        fig.update_layout(title="Salary distribution by country for " + cluster, title_x=0.5,
                          xaxis_title='salary [in €]', yaxis_title='density')
        all_figures.append(fig)

    cluster_overview.loc[len(cluster_names+1), 'role'] = 'all'
    cluster_overview.loc[len(cluster_names+1), 'mean salary'] = df.salary.mean()
    cluster_overview.loc[len(cluster_names+1), 'median salary'] = df.salary.median()
    cluster_overview.loc[len(cluster_names+1), 'min salary'] = df.salary.min()
    cluster_overview.loc[len(cluster_names+1), 'max salary'] = df.salary.max()
    cluster_overview.loc[len(cluster_names+1), 'counts'] = len(df.salary)

    country_list = ['UK', 'Germany', 'USA']
    data_dist_country = []
    for country in country_list:
        df_country = df.loc[df['country'] == country]
        data_dist_country.append(df_country.salary)

    fig = ff.create_distplot(data_dist_country, country_list, bin_size=5000)
    fig.update_layout(title="Salary distribution by country for all", title_x=0.5, xaxis_title='salary [in €]',
                      yaxis_title='density')
    all_figures.append(fig)
    output_name = np.append(cluster_names, 'all')

    figure0 = ff.create_distplot(data_dist, cluster_names, colors=color_list, show_hist=False, bin_size=5000)
    figure0.update_layout(title="Salary distribution by cluster name", title_x=0.5, xaxis_title='salary [in €]',
                          yaxis_title='density')
    return figure0, all_figures, output_name, cluster_overview


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
    return word2vector, word_mapped


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    path = os.getcwd()
    data = load_data(path)
    encoded_array = encode_tfidf(data)
    umap_array = umap_jobs(encoded_array)
    cluster_labels = clustering(umap_array)
    rf = pd.DataFrame({'x': [x for x in umap_array[:, 0]],
                       'y': [y for y in umap_array[:, 1]],
                       'label': [x for x in cluster_labels],
                       'company': data['company'],
                       'region': data['region'],
                       'country': data['country'],
                       'title': data['job_title'],
                       'salary': data['salary_average_euros'],
                       'full_description': data['full_description']})

    rf = find_label(rf)
    with open(path + '/Visualization/umap_jobs.pkl', 'wb') as file:
        pickle.dump(rf, file)

    fig0, all_fig, cluster_name, df_overview = create_density_plots(rf)

    with open(path + '/Visualization/plots_density.pkl', 'wb') as file:
        pickle.dump([fig0, all_fig, cluster_name, df_overview], file)

    w2v, word_mapper = create_word_umap(path)

    with open(path + '/Visualization/umap_words.pkl', 'wb') as file:
        pickle.dump([w2v, word_mapper], file)
