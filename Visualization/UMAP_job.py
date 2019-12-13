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


def load_data(path):
    with open(path + '/data/SQL_access.pkl', 'rb') as file:
        PASSWORD = pickle.load(file)
    engine = create_engine('postgresql://postgres:' + PASSWORD +
                            '@dsj-1.c9mo6xd9bf9d.us-west-2.rds.amazonaws.com:5432/')
    df = pd.read_sql("select * from all_data where language like 'en' and salary_type like 'yearly'", engine)
    df = df.dropna(subset=['region', 'country', 'description', 'job_title'], axis=0)
    df['full_description'] = df['job_title'] + ' ' + df['description']
    return df


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


def encode_TFIDF(df):
    tfidf = TfidfVectorizer(analyzer=text_process, use_idf=True, tokenizer=spacy_tokenizer, min_df=3)
    tfidf_fit = tfidf.fit(df['full_description'])
    tfidf_array = tfidf_fit.transform(df['full_description']).toarray()
    with open(path + '/Visualization/tfidf_encoder_all.pkl', 'wb') as file:
        pickle.dump([tfidf_fit, tfidf_array], file)
    return tfidf_array


def umap_jobs(array):
    umapper = umap.UMAP(n_neighbors=15, min_dist=0.0, n_components=2, random_state=42)
    mapper = umapper.fit(array)
    umap_array = mapper.transform(array)
    with open(path + '/Visualization/umap_encoder.pkl', 'wb') as file:
        pickle.dump(mapper, file)
    return umap_array


def clustering(umap_array):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=120, min_samples=2)
    cluster_labels = clusterer.fit_predict(umap_array[:, 0:2])
    with open(path + '/Visualization/cluster_labeler.pkl', 'wb') as file:
        pickle.dump(clusterer, file)
    return cluster_labels


def occurences(string, word_list):
    counts = 0
    for word in word_list:
        counts += string.count(word)
    return counts


def find_label(rf):
    rf['name'] = 'unclassified'
    for cluster in rf.label.unique():
        df = rf.loc[rf['label'] == cluster]
        counts = []
        df['full_description'] = df['full_description'].apply(text_process).apply(' '.join)
        df['title'] = df['title'].apply(text_process).apply(' '.join)
        names = {'Data Scientist':['data scientist','machine learning', ' ai '], 'Data Engineer':['engineer','java',' aws '],
                 'Data Analyst': ['data analyst','business intelligence','insight analyst'], 'Media & Marketing':['media','marketing','digital']}
        for keys, values in names.items():
            total = len(df)
            counts.append(df['full_description'].apply(occurences, args=(values,)).sum())
        most_common = np.array(counts).argmax()
        if np.sort(counts)[-1]*0.75 > np.sort(counts)[-2]:
            rf.loc[rf['label'] == cluster, 'name'] = list(names)[most_common]
    return rf


def create_density_plots(rf):
    colors_dict = {'unclassified': 'rgb(211,211,211)', 'Data Analyst': 'rgb(255, 127, 14)',
                   'Data Scientist': 'rgb(44, 160, 44)', 'Media & Marketing': 'rgb(214, 39, 40)',
                   'Data Engineer': 'rgb(148, 103, 189)'}
    color_list = [color for color in colors_dict.values()]
    data_dist = []
    rf = rf.dropna(subset=['salary', 'title'], axis=0)
    cluster_name = rf.name.unique()
    all_figures = []
    for cluster in cluster_name:
        rf_sub = rf.loc[rf['name'] == cluster]
        data_dist.append(rf_sub.salary)

        country_list = ['UK', 'Germany', 'USA']
        data_dist_country = []
        for country in country_list:
            rf_sub_country = rf_sub.loc[df['country'] == country]
            data_dist_country.append(rf_sub_country.salary)

        fig = ff.create_distplot(data_dist_country, country_list, bin_size=5000)
        fig.update_layout(title="Salary distribution by country for " + cluster, title_x=0.5, xaxis_title='salary [in €]',
                          yaxis_title='density')
        all_figures.append(fig)

    country_list = ['UK', 'Germany', 'USA']
    data_dist_country = []
    for country in country_list:
        rf_sub_country = rf.loc[df['country'] == country]
        data_dist_country.append(rf_sub_country.salary)

    fig = ff.create_distplot(data_dist_country, country_list, bin_size=5000)
    fig.update_layout(title="Salary distribution by country for all", title_x=0.5, xaxis_title='salary [in €]',
                      yaxis_title='density')
    all_figures.append(fig)
    output_name = np.append(cluster_name,'all')

    fig0 = ff.create_distplot(data_dist, cluster_name, colors=color_list, show_hist=False, bin_size=5000)
    fig0.update_layout(title="Salary distribution by cluster name", title_x=0.5, xaxis_title='salary [in €]',
                       yaxis_title='density')
    return fig0, all_figures, output_name


def create_word_umap(path):
    with open(path + '/Pickles/word2vec_4.pkl', 'rb') as file:
        w2v_model = pickle.load(file)
    words = set(w2v_model.wv.vocab)
    vectors = []
    vocab = []
    for word in words:
        vectors.append(w2v_model.wv.__getitem__([word]))
        vocab.append(word)
    vectors = np.asarray(vectors)
    vectors = vectors.reshape(-1, 300)
    umapper = umap.UMAP(n_neighbors=5, min_dist=0.0, n_components=2, random_state=42)
    word_mapper = umapper.fit(vectors)
    word_map = word_mapper.transform(vectors)
    w2v = pd.DataFrame({'x': [x for x in word_map[:, 0]],
                        'y': [y for y in word_map[:, 1]],
                        'word': vocab})
    return w2v, word_mapper


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    path = os.getcwd()
    df = load_data(path)
    encoded_array = encode_TFIDF(df)
    umap_array = umap_jobs(encoded_array)
    cluster_labels = clustering(umap_array)
    rf = pd.DataFrame({'x': [x for x in umap_array[:, 0]],
                       'y': [y for y in umap_array[:, 1]],
                       'label': [x for x in cluster_labels],
                       'company': df['company'],
                       'region': df['region'],
                       'country': df['country'],
                       'title': df['job_title'],
                       'salary': df['salary_average_euros'],
                       'full_description': df['full_description'],})

    rf = find_label(rf)
    with open(path + '/Visualization/umap_jobs.pkl', 'wb') as file:
        pickle.dump(rf, file)

    fig0, all_fig, cluster_name = create_density_plots(rf)

    with open(path + '/Visualization/plots_density.pkl', 'wb') as file:
        pickle.dump([fig0, all_fig, cluster_name], file)

    w2v, word_mapper = create_word_umap(path)

    with open(path + '/Visualization/umap_words.pkl', 'wb') as file:
        pickle.dump([w2v, word_mapper], file)
