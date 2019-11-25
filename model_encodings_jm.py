# !python -m spacy download en_core_web_lg
import pandas as pd
import pickle
import os.path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
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


def encode_BOG(df, min_df):
    df_en = df.loc[df.language == 'en']
    print('Selected only English job descriptions...\n')

    df_en.description.replace(regex=r"\\n", value=r" ", inplace=True)
    print('Performed some basic text cleaning...\n')

    BOG = CountVectorizer(analyzer=text_process, tokenizer=spacy_tokenizer, min_df=min_df)
    BOG_fit = BOG.fit(df_en['description'])
    print('Trained Bag-Of-Words model...\n')

    return BOG_fit

def encode_TFIDF(df, min_df):
    df_en = df.loc[df.language == 'en']
    print('Selected only English job descriptions...\n')

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


def get_imp_terms(input_string,tech_list,d, important_terms):
    
    result = [d.get(x,x) for x in important_terms if x in input_string]
    return list(set(result))

def encode_tech_terms(df,tech_list):
    
    #Need to look at inserting spaces where lower case and Capital appear adjacent
#     for x in df['description']:
#         for i in range(len(x)):
#             if x[i].isupper() and x[i-1].islower():
#                 x[i] = x[i].replace(x[i],' '+x[i])
    
    important_terms = list(set([x.lower() for x in tech_list]))
    texts = [x.lower() for x in df['description']]
    
    for i in range(len(texts)):
        a = texts[i].replace("["," ")
        a = a.replace("\n"," ")
        a = a.replace("]"," ")
        a = a.replace("."," ")
        a = a.replace(","," ")
        a = a.replace(":"," ")
        a = a.replace(";"," ")
        a = a.replace('"'," ")
        a = a.replace('('," ")
        a = a.replace(')'," ")
        a = a.replace('\\'," ")
        a = a.replace('/'," ")
        texts[i]=a
    
    d = { ' bi ':' business intelligence ', ' ai ':' artifical intelligence ', ' databases ':' database ',' db ':' database ',' aws ':' amazon web services ','nlp': 'natural language processing'}
    
    dj = pd.DataFrame({'T':texts})
    def get_imp_terms_tech_list(input_string):
        return get_imp_terms(input_string, tech_list,d, important_terms)

    dj['iR']=dj['T'].map(get_imp_terms_tech_list)
    dj['iR'] = [x for x in dj['iR']]

    mlb = MultiLabelBinarizer(classes = important_terms)
    dk = pd.DataFrame(mlb.fit_transform(dj['iR']), columns=important_terms)
    dl = dj.join(dk)
    dl = dl.drop(columns = ['iR','T'])

    return dl

# if __name__ == "__main__":
#     path = os.getcwd()
#     path_access_file = path + '/data/SQL_access.pkl'

#     # connect to the database
#     with open(path_access_file, 'rb') as file:
#         PASSWORD = pickle.load(file)

#     engine = create_engine(
#         'postgresql://postgres:' + PASSWORD + '@dsj-1.c9mo6xd9bf9d.us-west-2.rds.amazonaws.com:5432/')
#     df = pd.read_sql("select * from all_data", engine)
#     print('Loaded data from SQL database...\n')