import pandas as pd
import numpy as np
import pickle
import umap
import os.path
from sqlalchemy import create_engine
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from spacy.lang.en import English
import warnings


def load_data(path):
    with open(path + '/data/SQL_access.pkl', 'rb') as file:
        PASSWORD = pickle.load(file)
    engine = create_engine('postgresql://postgres:' + PASSWORD +
                            '@dsj-1.c9mo6xd9bf9d.us-west-2.rds.amazonaws.com:5432/')
    df = pd.read_sql("select * from all_data where language like 'en' and salary_type like 'yearly'", engine)
    df = df.dropna(subset=['region', 'country', 'description', 'job_title'], axis=0)
    df['full_description'] = df['job_title']+ ' ' + df['description']
    return df


def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Lower case of all words
    2. Remove all punctuation
    3. Remove all stopwords
    4. Returns a list of the cleaned text
    """
    punctuations = '!"$%&\'()*,-./:;<=>?@[\\]^_`{|}~'

    # transforms all to lower case words
    mess = mess.lower()

    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in punctuations]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)

    # Now just remove any stopwords
    return [word for word in nopunc.split() if word not in spacy.lang.en.stop_words.STOP_WORDS]


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
    tfidf_transform = tfidf_fit.transform(df['full_description'])
    with open(path + '/Visualization/tfidf_encoder_all.pkl', 'wb') as file:
        pickle.dump(tfidf_fit, file)
    return tfidf_transform


def umap_jobs(array):
    umapper = umap.UMAP(n_neighbors=16, min_dist=0.0, n_components=2, random_state=42)
    #mapper = umapper.fit(array)
    umap_array = umapper.fit_transform(array)
    with open(path + '/Visualization/umap_encoder.pkl', 'wb') as file:
        pickle.dump(umapper, file)
    return umap_array


def clustering(umap_array):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=250, min_samples=5)
    cluster_labels = clusterer.fit_predict(umap_array[:, 0:2])
    with open(path + '/Visualization/cluster_labeler.pkl', 'wb') as file:
        pickle.dump(clusterer, file)
    return cluster_labels


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
                       'title': df['job_title']})

    with open(path + '/Visualization/umap_jobs.pkl', 'wb') as file:
        pickle.dump([rf, cluster_labels], file)
