# !python -m spacy download en_core_web_lg
import pandas as pd
import pickle
import os.path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sqlalchemy import create_engine
from langdetect import detect
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


if __name__ == "__main__":
    path = os.getcwd()
    path_access_file = path + '/data/SQL_access.pkl'

    # connect to the database
    with open(path_access_file, 'rb') as file:
        PASSWORD = pickle.load(file)

    engine = create_engine(
        'postgresql://postgres:' + PASSWORD + '@dsj-1.c9mo6xd9bf9d.us-west-2.rds.amazonaws.com:5432/')
    df = pd.read_sql("select * from all_data", engine)
    print('Loaded data from SQL database...\n')

    df['language'] = df.description.apply(detect)
    df_en = df.loc[df.language == 'en']
    print('Detected languages of each job descriptions...\n')

    df_en.description.replace(regex=r"\\n", value=r" ", inplace=True)
    print('Performed some basic text cleaning...\n')

    BOG = CountVectorizer(analyzer=text_process, tokenizer=spacy_tokenizer, min_df=20)
    BOG_fit = BOG.fit(df_en['description'])
    BOG_transform = BOG_fit.transform(df_en['description'])
    print('Trained Bag-Of-Words model...\n')

    with open(path + '/Pickles/BOG_transform.pkl', 'wb') as file:
        pickle.dump(BOG_transform, file)
    with open(path + '/Pickles/BOG_model.pkl', 'wb') as file:
        pickle.dump([BOG, BOG_fit, BOG_transform], file)

    TFIDF = TfidfVectorizer(analyzer=text_process, use_idf=True, tokenizer=spacy_tokenizer)
    TFIDF_fit = TFIDF.fit(df_en['description'])
    TFIDF_transform = TFIDF_fit.transform(df_en['description'])
    print('Trained TF-IDF model...\n')

    with open(path + '/Pickles/TFIDF_transform.pkl', 'wb') as file:
        pickle.dump(TFIDF_transform, file)
    with open(path + '/Pickles/TFIDF_model.pkl', 'wb') as file:
        pickle.dump([TFIDF, TFIDF_fit, TFIDF_transform], file)
