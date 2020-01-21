import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle
import os.path
import torch
import torchtext
from torchtext import data
from torchtext import vocab
import re
import spacy
from spacy.lang.en import English


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


def transform_vocab(wordlist, model):
    filtered_wl = [word for word in wordlist if word in model.wv.vocab]
    return model.wv[filtered_wl]


def padding(wordlist, padding_length):
    paddings = padding_length - len(wordlist)
    padded_wordlist = paddings *[len(wordlist[0])*[0]] + wordlist[0:padding_length].tolist()
    return np.array(padded_wordlist)


if __name__ == "__main__":
    path = os.getcwd()
    with open(path + '/Pickles/word2vec.pkl', 'rb') as file:
        w2v_model = pickle.load(file)

    with open(path + '/data/SQL_access.pkl', 'rb') as file:
        PASSWORD = pickle.load(file)
    engine = create_engine('postgresql://postgres:' + PASSWORD +
                           '@dsj-1.c9mo6xd9bf9d.us-west-2.rds.amazonaws.com:5432/')
    df = pd.read_sql("select * from all_data where language like'en'", engine)
    print('Loaded data from SQL database...\n')
    df1 = df.dropna(subset=['salary_average_euros', 'region', 'country', 'train_test_label',
                            'company', 'description'], axis=0)
    df1 = df1.loc[df1.salary_type == 'yearly']
    df1 = df1.reset_index(drop=True)

    torch_df = df1[['description', 'salary_average_euros']]

    y_train = torch.Tensor(np.array(torch_df['salary_average_euros']))
    torch_df['description'] = torch_df['description'].apply(text_process)

    torch_df['description'] = torch_df['description'].apply(transform_vocab, args=(w2v_model,))
    torch_df['lengths'] = torch_df.description.apply(len)

    padding_length = int(round(torch_df.lengths.mean()))
    vocab_size = len(w2v_model.wv.vocab)
    embedding_dim = w2v_model.wv.vector_size
    print(padding_length)
    print(vocab_size)
    print(embedding_dim)

    torch_df['description'] = torch_df['description'].apply(padding, args=(padding_length,))

    X_train = np.zeros([len(torch_df['description']), len(torch_df['description'][0]),
                          len(torch_df['description'][0][0])])

    print(X_train.shape)

    for i in range(len(torch_df['description'])):
        X_train[i, :, :] = torch_df['description'][i]
    print(X_train)