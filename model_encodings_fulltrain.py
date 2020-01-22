# !python -m spacy download en_core_web_lg
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import pickle
import os.path
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import spacy
import re
from spacy.lang.en import English
import re


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


def encode_BOG(df_en, min_df):
    df_en.description.replace(regex=r"\\n", value=r" ", inplace=True)
    print('Performed some basic text cleaning...\n')
    BOG = CountVectorizer(analyzer=text_process, tokenizer=spacy_tokenizer, min_df=min_df)
    BOG_fit = BOG.fit(df_en['description'])
    print('Trained Bag-Of-Words model...\n')
    return BOG_fit


def encode_TFIDF(df_en, min_df):
    df_en.description.replace(regex=r"\\n", value=r" ", inplace=True)
    print('Performed some basic text cleaning...\n')
    TFIDF = TfidfVectorizer(analyzer=text_process, use_idf=True, tokenizer=spacy_tokenizer, min_df=min_df)
    TFIDF_fit = TFIDF.fit(df_en['description'])
    print('Trained TF-IDF model...\n')
    return TFIDF_fit


def transform_vocab(wordlist, model):
    filtered_wl = [word for word in wordlist if word in model.wv.vocab]
    return model.wv[filtered_wl]


def padding(wordlist, padding_length):
    paddings = padding_length - len(wordlist)
    padded_wordlist = paddings * [len(wordlist[0]) * [0]] + wordlist[0:padding_length].tolist()
    return np.array(padded_wordlist)


def w2v_clean_encode(df, model):
    df.loc[:, 'description_list'] = df.loc[:, 'full_description'].apply(text_process)
    df.loc[:, 'description_list'] = df.loc[:, 'description_list'].apply(transform_vocab, args=(model,))
    return df


def padding_transform(df, padding_length):
    df.loc[:, 'description_list'] = df.loc[:, 'description_list'].apply(padding, args=(padding_length,))
    W2V_array = np.zeros(
        [len(df['description_list']), len(df['description_list'].iloc[0]), len(df['description_list'].iloc[0][0])])
    for i in range(len(df['description_list'])):
        W2V_array[i, :, :] = df['description_list'].iloc[i]
    return W2V_array


def tech_process(mess, important_terms):
    mess = mess.lower()
    mess = mess.replace("\\n", " ")
    punctuations = '[].:;"()/\\'
    nopunc = [char for char in mess if char not in punctuations]
    nopunc = ''.join(nopunc)
    return [x for x in important_terms if x in nopunc]


if __name__ == "__main__":
    path = os.getcwd()

    with open(path + '/data/SQL_access.pkl', 'rb') as file:
        PASSWORD = pickle.load(file)
    engine = create_engine('postgresql://postgres:' + PASSWORD +
                           '@dsj-1.c9mo6xd9bf9d.us-west-2.rds.amazonaws.com:5432/')
    df = pd.read_sql("select * from all_data where language like'en' AND salary_average_euros>17000", engine)
    print('Loaded data from SQL database...\n')
    df['full_description'] = df['job_title'] + ' ' + df['description']
    df1 = df.dropna(subset=['salary_average_euros', 'region', 'country', 'train_test_label',
                            'company', 'description'], axis=0)
    df1 = df1.loc[df1.salary_type == 'yearly']
    df1 = df1.reset_index(drop=True)

    # first split the train from the test as denoted in the database
    x_train = df1.loc[df1['train_test_label'] == 'train']
    x_test = df1.loc[df1['train_test_label'] == 'test']
    y_train = x_train['salary_average_euros']
    y_test = x_test['salary_average_euros']


    train_index = x_train['id']
    test_index = x_test['id']

    columns_to_ohe_encode = ['country', 'region']
    train_enc = x_train[columns_to_ohe_encode]
    test_enc = x_test[columns_to_ohe_encode]

    # only train encoding on train data
    enc = preprocessing.OneHotEncoder(categories='auto', handle_unknown='ignore')
    enc.fit(train_enc)

    # get the names of the OHE features
    feature_names_OHE = list(enc.get_feature_names(columns_to_ohe_encode))

    # create encoding
    OHE_train = enc.transform(train_enc).toarray()
    OHE_test = enc.transform(test_enc).toarray()
    print('Performed One-Hot-Encoding for columns: Country, Region...\n')

    BOG_model = encode_BOG(x_train, min_df=3)
    BOG_train = BOG_model.transform(x_train['full_description']).toarray()
    BOG_test = BOG_model.transform(x_test['full_description']).toarray()
    feature_names_BOG = list(BOG_model.get_feature_names())


    TFIDF_model = encode_TFIDF(x_train, min_df=3)
    TFIDF_train = TFIDF_model.transform(x_train['full_description']).toarray()
    TFIDF_test = TFIDF_model.transform(x_test['full_description']).toarray()
    feature_names_TFIDF = list(TFIDF_model.get_feature_names())

    with open(path + '/Pickles/word2vec_2.pkl', 'rb') as file:
        w2v_model = pickle.load(file)

    x_train = w2v_clean_encode(x_train, w2v_model)
    x_test = w2v_clean_encode(x_test, w2v_model)

    x_train.loc[:, 'lengths'] = x_train.loc[:, 'description_list'].apply(len)
    vocab_size = len(w2v_model.wv.vocab)
    embedding_dim = w2v_model.wv.vector_size
    padding_length = int(round(x_train['lengths'].mean()))

    W2V_train = padding_transform(x_train, padding_length)
    W2V_test = padding_transform(x_test, padding_length)

    with open(path + '/data/W2V_all.pkl', 'wb') as file:
        pickle.dump([W2V_train, W2V_test], file)

    tech_dict = pd.read_pickle('Pickles/broad_tech_dictionary.pickle')
    categories_to_include = ['front_end-technologies', 'databases', 'software-infrastructure-devops', 'data-science',
                             'software_architecture', 'web_design', 'tools', 'cyber_security', 'cloud_computing',
                             'back_end-technologies', 'mobile']

    important_terms = list(set([item.lower() for key in categories_to_include for item in tech_dict[key]]))

    tech_terms_train = (x_train['full_description']).apply(tech_process, args=(important_terms,))
    tech_terms_test = (x_test['job_title'] + ' ' + x_test['description']).apply(tech_process, args=(important_terms,))

    feature_names_TECH = important_terms

    mlb = MultiLabelBinarizer(classes=important_terms)
    mlb.fit(tech_terms_train)
    TECH_train = mlb.transform(tech_terms_train)
    TECH_test = mlb.transform(tech_terms_test)
    print('Performed encoding of technical terms...\n')



    # output models
    with open(path + '/Pickles/OHE_model_all.pkl', 'wb') as file:
        pickle.dump(enc, file)
    with open(path + '/Pickles/TFIDF_model_all.pkl', 'wb') as file:
        pickle.dump(TFIDF_model, file)
    with open(path + '/Pickles/BOG_model_all.pkl', 'wb') as file:
        pickle.dump(BOG_model, file)
    with open(path + '/Pickles/TECH_model_all.pkl', 'wb') as file:
        pickle.dump(mlb, file)

    # output different encodings encoded data
    with open(path + '/data/OHE_all.pkl', 'wb') as file:
        pickle.dump([OHE_train, OHE_test, feature_names_OHE], file)
    with open(path + '/data/TFIDF_all.pkl', 'wb') as file:
        pickle.dump([TFIDF_train, TFIDF_test, feature_names_TFIDF], file)
    with open(path + '/data/BOG_all.pkl', 'wb') as file:
        pickle.dump([BOG_train,  BOG_test, feature_names_BOG], file)
    with open(path + '/data/TECH_all.pkl', 'wb') as file:
        pickle.dump([TECH_train, TECH_test, feature_names_TECH], file)
    with open(path + '/data/yTrainValTest_all.pkl', 'wb') as file:
        pickle.dump([y_train,y_test], file)
    with open(path + '/data/IndexTrainValTest_all.pkl', 'wb') as file:
        pickle.dump([train_index, test_index], file)

        # output full data frame
    with open(path + '/data/x_data_all.pkl', 'wb') as file:
        pickle.dump([x_train, x_test], file)

    print('Saved Train, Validation and Test Set in corresponding Pickle Files...\n')
    print(OHE_train.shape)
    print(TFIDF_train.shape)
    print(len(feature_names_TFIDF))