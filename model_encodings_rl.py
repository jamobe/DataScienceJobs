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


def tech_process(mess, important_terms):
    mess = mess.lower()
    mess = mess.replace("\\n"," ")
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
    df = pd.read_sql("select * from all_data where language like'en'", engine)
    print('Loaded data from SQL database...\n')

    df1 = df.dropna(subset=['salary_average_euros', 'region', 'country', 'train_test_label',
                            'company', 'description'], axis=0)
    df1 = df1.loc[df1.salary_type == 'yearly']
    df1 = df1.reset_index(drop=True)

    # first split the train from the test as denoted in the database
    df_train = df1.loc[df1['train_test_label'] == 'train']
    x_test = df1.loc[df1['train_test_label'] == 'test']
    df_train_y = df_train['salary_average_euros']
    y_test = x_test['salary_average_euros']

    # then split the train data into train and validation
    x_train, x_val, y_train, y_val = train_test_split(df_train, df_train_y, test_size=0.2, random_state=42)
    print('Splitted Train, Validation and Test data...\n')

    columns_to_ohe_encode = ['company', 'country', 'region']
    train_enc = x_train[columns_to_ohe_encode]
    val_enc = x_val[columns_to_ohe_encode]
    test_enc = x_test[columns_to_ohe_encode]

    # only train encoding on train data
    enc = preprocessing.OneHotEncoder(categories='auto', handle_unknown='ignore')
    enc.fit(train_enc)

    # get the names of the OHE features
    feature_names_OHE = list(enc.get_feature_names(columns_to_ohe_encode))

    # create encoding
    OHE_train = enc.transform(train_enc).toarray()
    OHE_val = enc.transform(val_enc).toarray()
    OHE_test = enc.transform(test_enc).toarray()
    print('Performed One-Hot-Encoding for columns: Company, Country, Region...\n')

    BOG_model = encode_BOG(x_train, min_df=3)
    BOG_train = BOG_model.transform(x_train['description']).toarray()
    BOG_val = BOG_model.transform(x_val['description']).toarray()
    BOG_test = BOG_model.transform(x_test['description']).toarray()
    feature_names_BOG = list(BOG_model.get_feature_names())

    TFIDF_model = encode_TFIDF(x_train, min_df=3)
    TFIDF_train = TFIDF_model.transform(x_train['description']).toarray()
    TFIDF_val = TFIDF_model.transform(x_val['description']).toarray()
    TFIDF_test = TFIDF_model.transform(x_test['description']).toarray()
    feature_names_TFIDF= list(TFIDF_model.get_feature_names())

    tech_dict = pd.read_pickle('Pickles/broad_tech_dictionary.pickle')
    categories_to_include = ['front_end-technologies', 'databases', 'software-infrastructure-devops', 'data-science',
                             'software_architecture', 'web_design', 'tools', 'cyber_security', 'cloud_computing',
                             'back_end-technologies', 'mobile']

    important_terms = list(set([item.lower() for key in categories_to_include for item in tech_dict[key]]))

    tech_terms_train = x_train['description'].apply(tech_process, args=(important_terms,))
    tech_terms_val = x_val['description'].apply(tech_process, args=(important_terms,))
    tech_terms_test = x_test['description'].apply(tech_process, args=(important_terms,))
    
    feature_names_TECH = important_terms

    mlb = MultiLabelBinarizer(classes=important_terms)
    mlb.fit(tech_terms_train)
    TECH_train = mlb.transform(tech_terms_train)
    TECH_val = mlb.transform(tech_terms_val)
    TECH_test = mlb.transform(tech_terms_test)
    print('Performed encoding of technical terms...\n')

#     X_train = np.hstack((OHE_train, TFIDF_train, TECH_train))
#     X_val = np.hstack((OHE_val, TFIDF_val, TECH_val))
#     X_test = np.hstack((OHE_test, TFIDF_test, TECH_test))
#     print('Train Set:' + str(X_train.shape))
#     print('Validation Set:' + str(X_val.shape))
#     print('Test Set:' + str(X_test.shape))
    
#     feature_names = feature_names_OHE + feature_names_TFIDF + feature_names_tech
    
    
#     with open(path + '/data/TrainSetXY.pkl', 'wb') as file:
#         pickle.dump([X_train, y_train], file)
#     with open(path + '/data/ValSetXY.pkl', 'wb') as file:
#         pickle.dump([X_val, y_val], file)
#     with open(path + '/data/TestSetXY.pkl', 'wb') as file:
#         pickle.dump([X_test, y_test], file)
#     print('Saved Train, Validation and Test Set in corresponding Pickle Files...\n')
    
#     with open(path + '/data/feature_names.pkl', 'wb') as file:
#         pickle.dump(feature_names, file)
     
   # output different encodings encoded data
    with open(path + '/data/OHE.pkl', 'wb') as file:
        pickle.dump([OHE_train,OHE_val,OHE_test,feature_names_OHE], file)
    with open(path + '/data/TFIDF.pkl', 'wb') as file:
        pickle.dump([TFIDF_train,TFIDF_val,TFIDF_test,feature_names_TFIDF], file)
    with open(path + '/data/BOG.pkl', 'wb') as file:
        pickle.dump([BOG_train,BOG_val,BOG_test,feature_names_BOG], file)
    with open(path + '/data/TECH.pkl', 'wb') as file:
        pickle.dump([TECH_train,TECH_val,TECH_test,feature_names_TECH], file) 
    with open(path + '/data/yTrainValTest.pkl', 'wb') as file:
        pickle.dump([y_train,y_val,y_test], file) 
    
    print('Saved Train, Validation and Test Set in corresponding Pickle Files...\n')