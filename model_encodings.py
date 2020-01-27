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


def load_data(passwordpath):
    with open(passwordpath + '/data/SQL_access.pkl', 'rb') as password_file:
        password = pickle.load(password_file)
    engine = create_engine('postgresql://postgres:' + password +
                           '@dsj-1.c9mo6xd9bf9d.us-west-2.rds.amazonaws.com:5432/')
    dataframe = pd.read_sql("select * from all_data where language like 'en' AND salary_average_euros > 17000 ", engine)
    print('Access data from SQL database...\n')
    print('Loaded ' + str(len(dataframe)) + ' jobs ...\n')
    return dataframe


def filtering(dataframe):
    dataframe['full_description'] = dataframe['job_title'] + ' ' + dataframe['description']
    dataframe = dataframe.dropna(subset=['salary_average_euros', 'region', 'country', 'train_test_label', 'company',
                                         'description'], axis=0)
    dataframe = dataframe.loc[dataframe.salary_type == 'yearly']
    dataframe = dataframe.reset_index(drop=True)
    print('Filtered only jobs with yearly salary...\n')
    print('Remaining ' + str(len(dataframe)) + ' jobs...\n')
    return dataframe


def splitting_test(dataframe):
    x_test_set = dataframe.loc[dataframe['train_test_label'] == 'test']
    y_test_set = x_test_set['salary_average_euros']

    x_train_set = dataframe.loc[dataframe['train_test_label'] == 'train']
    y_train_set = x_train_set['salary_average_euros']

    print('Splitted Train and Test data...\n')

    train_val_idx = x_train_set['id']
    test_idx = x_test_set['id']
    print('Train Set (' + str(len(train_val_idx)) + ') and Test Set (' + str(len(test_idx)) + ')\n')
    return train_val_idx, x_train_set, y_train_set, test_idx, x_test_set, y_test_set


def splitting_train_val(df_train, df_train_y):
    x_train_set, x_val_set, y_train_set, y_val_set = train_test_split(df_train, df_train_y, test_size=0.2,
                                                                      random_state=42)
    print('Splitted Train and Validation data...\n')
    train_idx = x_train_set['id']
    val_idx = x_val_set['id']
    print('Train Set (' + str(len(train_idx)) + ') and Validation Set (' + str(len(val_idx)) + ')\n')
    return train_idx, val_idx, x_train_set, x_val_set, y_train_set, y_val_set


def encode_ohe(columns, dataframe):
    train_encode = dataframe[columns]
    ohe_enc = preprocessing.OneHotEncoder(categories='auto', handle_unknown='ignore')
    enc_fit = ohe_enc.fit(train_encode)
    ohe_train = ohe_enc.transform(train_encode).toarray()
    feature_names_ohe = list(ohe_enc.get_feature_names(columns))
    print('Performed One-Hot-Encoding for columns: Country, Region...\n')
    with open(path + '/Pickles/OHE_model.pkl', 'wb') as ohe_file:
        pickle.dump(enc_fit, ohe_file)
    return enc_fit, feature_names_ohe, ohe_train


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
    mess = re.sub(r'https?://\S+', ' ', mess)  # remove links

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


def encode_bag_of_words(df_en, min_df):
    df_en.description.replace(regex=r"\\n", value=r" ", inplace=True)
    bag_of_words = CountVectorizer(analyzer=text_process, tokenizer=spacy_tokenizer, min_df=min_df)
    bag_of_words_fit = bag_of_words.fit(df_en['description'])
    bag_of_words_train = bag_of_words_fit.transform(x_train['full_description']).toarray()
    print('Trained Bag-Of-Words model...\n')
    feature_names_bag_of_words = list(bag_of_words_fit.get_feature_names())
    with open(path + '/Pickles/BOG_model.pkl', 'wb') as bog_file:
        pickle.dump(bag_of_words_fit, bog_file)
    return bag_of_words_fit, feature_names_bag_of_words, bag_of_words_train


def encode_tfidf(df_en, min_df):
    df_en.description.replace(regex=r"\\n", value=r" ", inplace=True)
    tfidf = TfidfVectorizer(analyzer=text_process, use_idf=True, tokenizer=spacy_tokenizer, min_df=min_df)
    tfidf_fit = tfidf.fit(df_en['description'])
    tfidf_train = tfidf_fit.transform(x_train['full_description']).toarray()
    print('Trained TF-IDF model...\n')
    feature_names_tfidf = list(tfidf_fit.get_feature_names())
    with open(path + '/Pickles/TFIDF_model.pkl', 'wb') as tfidf_file:
        pickle.dump(tfidf_fit, tfidf_file)
    return tfidf_fit, feature_names_tfidf, tfidf_train


def load_tech_terms(techpath):
    tech_dict = pd.read_pickle(techpath + '/Pickles/broad_tech_dictionary.pkl')
    categories_to_include = ['front_end-technologies', 'databases', 'software-infrastructure-devops', 'data-science',
                             'software_architecture', 'web_design', 'tools', 'cyber_security', 'cloud_computing',
                             'back_end-technologies', 'mobile']
    feature_names_tech = list(set([item.lower() for key in categories_to_include for item in tech_dict[key]]))
    return feature_names_tech


def tech_process(mess, importantterms):
    mess = mess.lower()
    mess = mess.replace("\\n", " ")
    punctuations = '[].:;"()/\\'
    nopunc = [char for char in mess if char not in punctuations]
    nopunc = ''.join(nopunc)
    return [x for x in importantterms if x in nopunc]


def encoding_technical_terms(importantterms, df_train):
    mlb_model = MultiLabelBinarizer(classes=importantterms)
    mlb_fit = mlb_model.fit(tech_terms_train)
    tech_train = mlb_fit.transform(df_train)
    print('Performed encoding of technical terms...\n')
    with open(path + '/Pickles/TECH_model.pkl', 'wb') as tech_file:
        pickle.dump(mlb_fit, tech_file)
    return mlb_fit, tech_train


if __name__ == "__main__":
    path = os.getcwd()
    df = load_data(path)
    df_filtered = filtering(df)
    train_val_index, x_train_val, y_train_val, test_index, x_test, y_test = splitting_test(df_filtered)
    train_index, val_index, x_train, x_val, y_train, y_val = splitting_train_val(x_train_val, y_train_val)
    columns_to_ohe_encode = ['country', 'region']
    enc, feature_names_OHE, OHE_train = encode_ohe(columns_to_ohe_encode, x_train)
    OHE_val = enc.transform(x_val[columns_to_ohe_encode]).toarray()
    OHE_test = enc.transform(x_test[columns_to_ohe_encode]).toarray()
    with open(path + '/Pickles/OHE.pkl', 'wb') as file:
        pickle.dump([OHE_train, OHE_val, OHE_test, feature_names_OHE], file)

    BOG_model, feature_names_BOG, BOG_train = encode_bag_of_words(x_train, min_df=3)
    BOG_val = BOG_model.transform(x_val['full_description']).toarray()
    BOG_test = BOG_model.transform(x_test['full_description']).toarray()
    with open(path + '/Pickles/BOG.pkl', 'wb') as file:
        pickle.dump([BOG_train, BOG_val, BOG_test, feature_names_BOG], file)

    TFIDF_model, feature_names_TFIDF, TFIDF_train = encode_tfidf(x_train, min_df=3)
    TFIDF_val = TFIDF_model.transform(x_val['full_description']).toarray()
    TFIDF_test = TFIDF_model.transform(x_test['full_description']).toarray()
    with open(path + '/Pickles/TFIDF.pkl', 'wb') as file:
        pickle.dump([TFIDF_train, TFIDF_val, TFIDF_test, feature_names_TFIDF], file)

    feature_names_TECH = load_tech_terms(path)
    tech_terms_train = (x_train['full_description']).apply(tech_process, args=(feature_names_TECH,))
    tech_terms_val = (x_val['full_description']).apply(tech_process, args=(feature_names_TECH,))
    tech_terms_test = (x_test['full_description']).apply(tech_process, args=(feature_names_TECH,))

    mlb, TECH_train = encoding_technical_terms(feature_names_TECH, tech_terms_train)
    TECH_val = mlb.transform(tech_terms_val)
    TECH_test = mlb.transform(tech_terms_test)
    with open(path + '/Pickles/TECH.pkl', 'wb') as file:
        pickle.dump([TECH_train, TECH_val, TECH_test, feature_names_TECH], file)

    X_Train = pd.DataFrame(np.hstack((OHE_train, TFIDF_train)),
                           columns=list(feature_names_OHE) + list(feature_names_TFIDF))
    X_Val = pd.DataFrame(np.hstack((OHE_val, TFIDF_val)),
                         columns=list(feature_names_OHE) + list(feature_names_TFIDF))
    X_Test = pd.DataFrame(np.hstack((OHE_test, TFIDF_test)),
                          columns=list(feature_names_OHE) + list(feature_names_TFIDF))

    X_Train.to_csv(path + '/Pickles/X_Train.csv')
    X_Val.to_csv(path + '/Pickles/X_Val.csv')
    X_Test.to_csv(path + '/Pickles/X_Test.csv')

    y_train.to_csv(path + '/Pickles/Y_Train.csv', header=True, index=False)
    y_val.to_csv(path + '/Pickles/Y_Val.csv', header=True, index=False)
    y_test.to_csv(path + '/Pickles/Y_Test.csv', header=True, index=False)

    train_index.to_csv(path + '/Pickles/Train_index.csv', header=True, index=False)
    val_index.to_csv(path + '/Pickles/Val_index.csv', header=True, index=False)
    test_index.to_csv(path + '/Pickles/Test_index.csv', header=True, index=False)

    print('Saved Train, Validation and Test Set in corresponding Pickle Files...\n')
