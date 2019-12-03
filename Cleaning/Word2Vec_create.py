import pandas as pd
import multiprocessing
import os.path
import string
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import pickle
from sqlalchemy import create_engine
import re


def text_process_en(mess):
    """
    Takes in a string of text, then performs the following:
    1. Lower case of all words
    2. Remove all punctuation
    3. Remove all stopwords
    4. Returns a list of the cleaned text
    """

    # transforms all to lower case words
    mess = mess.lower()
    mess = re.sub(r'[^A-Za-z0-9]+', ' ', mess)  # remove non alphanumeric character
    mess = re.sub(r'https?:/\/\S+', ' ', mess)  # remove links
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)

    # Now just remove any stopwords
    return [word for word in mess.split() if word not in stopwords.words('english')]


if __name__ == "__main__":
    path = os.getcwd()
    path_access_file = path + '/data/SQL_access.pkl'
    with open(path_access_file, 'rb') as file:
        access = pickle.load(file)

    engine = create_engine('postgresql://postgres:' + access + '@dsj-1.c9mo6xd9bf9d.us-west-2.rds.amazonaws.com:5432/')
    df_en = pd.read_sql("select * from all_data where language like 'en'", engine) #where train_test_label like 'train'

    df_en.description.replace(regex=r"\\n", value=r" ", inplace=True)
    all_descriptions = df_en['description'].apply(text_process_en).to_list()

    cores = multiprocessing.cpu_count()
    w2v_model = Word2Vec(min_count=3, window=2, size=300, sample=6e-5, alpha=0.03, min_alpha=0.0007, negative=20,
                         workers=cores - 1)
    w2v_model.build_vocab(all_descriptions, progress_per=10000)
    w2v_model.train(all_descriptions, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
    w2v_model.init_sims(replace=True)
    with open(path + '/Pickles/word2vec_2.pkl', 'wb') as file:
        pickle.dump(w2v_model, file)
