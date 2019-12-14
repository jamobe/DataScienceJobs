import os.path
import numpy as np
import pickle
import pandas as pd
from sklearn import preprocessing

if __name__ == "__main__":
    path = os.getcwd()

    # read-in encoded files
    with open(path + '/data/OHE_all.pkl', 'rb') as file:
        OHE_train,  OHE_test, feature_names_OHE = pickle.load(file)
    with open(path + '/data/TFIDF_all.pkl', 'rb') as file:
        TFIDF_train, TFIDF_test, feature_names_TFIDF = pickle.load(file)
    with open(path + '/data/BOG_all.pkl', 'rb') as file:
        BOG_train, BOG_test, feature_names_BOG = pickle.load(file)
    with open(path + '/data/TECH_all.pkl', 'rb') as file:
        TECH_train, TECH_test, feature_names_TECH = pickle.load(file)

    # for all data
    X_train = pd.DataFrame(np.hstack((OHE_train, TFIDF_train)),
                           columns=list(feature_names_OHE) + list(feature_names_TFIDF))

    X_test = pd.DataFrame(np.hstack((OHE_test, TFIDF_test)),
                          columns=list(feature_names_OHE) + list(feature_names_TFIDF))

    feature_names = X_train.columns

    with open(path + '/data/x_data_for_models_all.pkl', 'wb') as file:
        pickle.dump([X_train,X_test], file)