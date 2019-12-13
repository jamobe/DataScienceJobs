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
import pdb


def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Lower case of all words
    2. Remove all punctuation
    3. Remove all stopwords
    4. Returns a list of the cleaned text
    """

    punctuations = '!"$%&\'()*,-./:;<=>?@[\\]^_`{|}~'
    mess = re.sub(r'[^A-Za-z]+', ' ', mess)  # remove non alphanumeric character
    mess = re.sub(r'https?:/\/\S+', ' ', mess)  # remove links

    # punctuations = '!"$%&\'()*,-./:;<=>?@[\\]^_`{|}~'

    mess = mess.lower()
    mess = re.sub(r'[^A-Za-z]+', ' ', mess)  # remove non alphanumeric character [^A-Za-z0-9]
    mess = re.sub(r'https?:/\/\S+', ' ', mess)  # remove links
    # nopunc = [char for char in mess if char not in punctuations]
    # nopunc = ''.join(nopunc)
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
    df = pd.read_sql("select * from all_data where language like'en' and salary_average_euros>=70000", engine)
    print('Loaded data from SQL database...\n')
    df['full_description'] = df['job_title'] + ' ' + df['description']
    df1 = df.dropna(subset=['salary_average_euros', 'region', 'country', 'train_test_label',
                            'company', 'description'], axis=0)
    df1 = df1.loc[df1.salary_type == 'yearly']
    df1 = df1.reset_index(drop=True)

    # first split the train from the test as denoted in the database
    x_train= df1.loc[df1['train_test_label'] == 'train']
    pdb.set_trace()

    # then separate out the validation set based on the all model split
    with open(path + '/data/IndexTrainValTest.pkl', 'rb') as file:
        val_index = pickle.load(file)[1]

    x_train = x_train[~x_train['id'].isin(val_index)]
    train_high_index = x_train['id']
    #test_index = x_test['id']
    y_train = x_train['salary_average_euros']

    #import encodings from all data

    with open(path + '/Pickles/OHE_model.pkl', 'rb') as file:
        enc = pickle.load(file)
    with open(path + '/Pickles/TFIDF_model.pkl', 'rb') as file:
        TFIDF_model= pickle.load(file)
    with open(path + '/Pickles/BOG_model.pkl', 'rb') as file:
        BOG_model= pickle.load(file)
    with open(path + '/Pickles/TECH_model.pkl', 'rb') as file:
        mlb = pickle.load(file)


    columns_to_ohe_encode = ['country', 'region']
    train_enc = x_train[columns_to_ohe_encode]
    #val_enc = x_val[columns_to_ohe_encode]
    #test_enc = x_test[columns_to_ohe_encode]

    # get the names of the OHE features
    #feature_names_OHE = list(enc.get_feature_names(columns_to_ohe_encode))

    # create encoding
    OHE_train = enc.transform(train_enc).toarray()
    #OHE_val = enc.transform(val_enc).toarray()
    #OHE_test = enc.transform(test_enc).toarray()
    print('Performed One-Hot-Encoding for columns: Company, Country, Region...\n')


    BOG_train = BOG_model.transform(x_train['full_description']).toarray()
   # BOG_val = BOG_model.transform(x_val['full_description']).toarray()
    #BOG_test = BOG_model.transform(x_test['full_description']).toarray()
    #feature_names_BOG = list(BOG_model.get_feature_names())


    TFIDF_train = TFIDF_model.transform(x_train['full_description']).toarray()
    #TFIDF_val = TFIDF_model.transform(x_val['full_description']).toarray()
    #TFIDF_test = TFIDF_model.transform(x_test['full_description']).toarray()
    #feature_names_TFIDF = list(TFIDF_model.get_feature_names())


    tech_dict = pd.read_pickle('Pickles/broad_tech_dictionary.pickle')
    categories_to_include = ['front_end-technologies', 'databases', 'software-infrastructure-devops', 'data-science',
                             'software_architecture', 'web_design', 'tools', 'cyber_security', 'cloud_computing',
                             'back_end-technologies', 'mobile']

    important_terms = list(set([item.lower() for key in categories_to_include for item in tech_dict[key]]))

    tech_terms_train = (x_train['full_description']).apply(tech_process, args=(important_terms,))
    #tech_terms_val = (x_val['description']).apply(tech_process, args=(important_terms,))
    #tech_terms_test = (x_test['job_title'] + ' ' + x_test['description']).apply(tech_process, args=(important_terms,))

    #feature_names_TECH = important_terms

    mlb = MultiLabelBinarizer(classes=important_terms)
    mlb.fit(tech_terms_train)
    TECH_train = mlb.transform(tech_terms_train)
    #TECH_val = mlb.transform(tech_terms_val)
    #TECH_test = mlb.transform(tech_terms_test)
    print('Performed encoding of technical terms...\n')

    import pdb; pdb.set_trace()
    # output different encodings encoded data
    with open(path + '/data/OHE_high_salary.pkl', 'wb') as file:
        pickle.dump(OHE_train, file)
    with open(path + '/data/TFIDF_high_salary.pkl', 'wb') as file:
        pickle.dump(TFIDF_train, file)
    with open(path + '/data/BOG_high_salary.pkl', 'wb') as file:
        pickle.dump(BOG_train, file)
    with open(path + '/data/TECH_high_salary.pkl', 'wb') as file:
        pickle.dump(TECH_train, file)
    with open(path + '/data/yTrainValTest_high_salary.pkl', 'wb') as file:
        pickle.dump(y_train, file)
    with open(path + '/data/IndexTrainValTest_high_salary.pkl', 'wb') as file:
        pickle.dump(train_high_index, file)

        # output full data frame
    with open(path + '/data/x_data_high_salary.pkl', 'wb') as file:
        pickle.dump(x_train, file)

    print('Saved Train, Validation and Test Set in corresponding Pickle Files...\n')