import os.path
import numpy as np
import pickle
import pandas as pd

if __name__ == "__main__":
    path = os.getcwd()
    with open(path + '/data/OHE.pkl', 'rb') as file:
        OHE_train,OHE_val,OHE_test,feature_names_OHE = pickle.load(file)
    with open(path + '/data/TFIDF.pkl', 'rb') as file:
        TFIDF_train,TFIDF_val,TFIDF_test,feature_names_TFIDF = pickle.load(file)
    with open(path + '/data/BOG.pkl', 'rb') as file:
        BOG_train,BOG_val,BOG_test,feature_names_BOG = pickle.load(file)
    with open(path + '/data/TECH.pkl', 'rb') as file:    
        TECH_train,TECH_val,TECH_test,feature_names_TECH = pickle.load(file)


    X_train = pd.DataFrame(np.hstack((OHE_train,TFIDF_train)), columns = list(feature_names_OHE)+list(feature_names_TFIDF))
    X_val = pd.DataFrame(np.hstack((OHE_val,TFIDF_val)), columns =list(feature_names_OHE)+list(feature_names_TFIDF))
    X_test = pd.DataFrame(np.hstack((OHE_test,TFIDF_test)), columns = list(feature_names_OHE)+list(feature_names_TFIDF))


    #remove 'company' variables
    cols = [c for c in X_train.columns if 'company_' not in c]

    X_train = X_train[cols]
    X_val = X_val[cols]
    X_test = X_test[cols]
    feature_names = cols
    
    with open(path + '/data/x_data_for_models.pkl', 'wb') as file:
        pickle.dump([X_train,X_val,X_test], file)