# from flask import Flask, jsonify, request
# import pandas as pd
# from sklearn.externals import joblib
# import os
#
# app = Flask(__name__)
#
# # stuff we need to load into the app
# #classifier = joblib.load('../model/model.pkl')
# with open ('Form_Page.html') as f:
#     form_page = f.read()
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     basket = request.json['basket']
#     zipCode = request.json['zipCode']
#     totalAmount = request.json['totalAmount']
#     p = probability(basket, zipCode, totalAmount)
#
#     return jsonify({'probability': p}), 201
#
# @app.route('/testing')
#
# def test():
#     return form_page, 201
#
# @app.route('/my-handling-form-page',methods=['POST','GET'] )
# def display():
#     test = request.get_json()
#     print(test)
#     return 'hello', 201
#
# if __name__ == "__main__":
#     app.run()

from flask import Flask, request, render_template,jsonify
#from flask import Flask, render_template, request
import pickle
import spacy
import re
from spacy.lang.en import English
import numpy as np
import pandas as pd

app = Flask(__name__)

def count_char(input_text):
    return len(list(input_text))

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
    mess = mess.lower()
    nopunc = [char for char in mess if char not in punctuations]
    nopunc = ''.join(nopunc)
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

def load_TFIDF_encoding():
    with open('Pickles/TFIDF_model.pkl', 'rb') as file:
        TFIDF_model = pickle.load(file)
    return TFIDF_model

def load_OHE_encoding():
    with open('Pickles/OHE_model.pkl', 'rb') as file:
        OHE_model = pickle.load(file)
    return OHE_model

def load_feature_names_TFIDF():
    with open('data/TFIDF.pkl', 'rb') as file:
        feature_names_TFIDF = pickle.load(file)[3]
    return feature_names_TFIDF

def load_feature_names_OHE():
    with open('data/OHE.pkl', 'rb') as file:
        feature_names_OHE = pickle.load(file)[3]
    return feature_names_OHE

def load_xgb_model():
    with open('Pickles/xgb_model.pkl', 'rb') as file:
        xgb_reg = pickle.load(file)
    return xgb_reg

def encode_TFIDF():
    TFIDF_model = load_TFIDF_encoding()
    original_message = request.form['text']
    clean_message = text_process(original_message)
    clean_string = [' '.join(clean_message)]
    return TFIDF_model.transform(clean_string).toarray()

def encode_OHE():
    company = 0
    region = request.form['region']
    country = request.form['country']
    values = np.array([company, country, region]).reshape(1, 3)
    values = pd.DataFrame(values, columns=('company', 'country', 'region'))
    OHE_model = load_OHE_encoding()
    return OHE_model.transform(values).toarray()

def select_vars():
    feature_names_OHE = load_feature_names_OHE()
    feature_names_TFIDF = load_feature_names_TFIDF()
    TFIDF_encoded = encode_TFIDF()
    OHE_encoded = encode_OHE()
    df = pd.DataFrame(np.hstack((OHE_encoded, TFIDF_encoded)), columns=list(feature_names_OHE) + list(feature_names_TFIDF))
    cols = [c for c in df.columns if 'company_' not in c]
    return df[cols]

def salary_prediction():
    df_model = select_vars()
    xgb_reg = load_xgb_model()
    return np.exp(xgb_reg.predict(df_model))

def do_something(country,region,text):
   country = country.upper()
   region = region.upper()
   text = text.upper()
   combine = country + region+text
   return combine

@app.route('/', methods=['GET', 'POST'])
def form():
    return render_template('Form_Page.html')


@app.route('/join', methods=['GET','POST'])
def my_form_post():
    combine = str(salary_prediction())
    result = {
        "output": combine
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

if __name__ == "__main__":
    app.run()