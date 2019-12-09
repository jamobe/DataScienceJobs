import os.path
import numpy as np
import pickle
import pandas as pd
from sklearn import preprocessing


if __name__ == "__main__":
    path = os.getcwd()
    
    #read-in encoded files
    with open(path + '/data/OHE.pkl', 'rb') as file:
        OHE_train,OHE_val,OHE_test,feature_names_OHE = pickle.load(file)
    with open(path + '/data/TFIDF.pkl', 'rb') as file:
        TFIDF_train,TFIDF_val,TFIDF_test,feature_names_TFIDF = pickle.load(file)
    with open(path + '/data/BOG.pkl', 'rb') as file:
        BOG_train,BOG_val,BOG_test,feature_names_BOG = pickle.load(file)
    with open(path + '/data/TECH.pkl', 'rb') as file:    
        TECH_train,TECH_val,TECH_test,feature_names_TECH = pickle.load(file)

        
        
    #open full dfs post encoding
#     with open(path + '/data/x_data.pkl', 'rb') as file:
#         x_train,x_val,x_test = pickle.load(file)  
#     x_train = x_train.reset_index(drop=True)
#     x_val= x_train.reset_index(drop=True)
#     x_test = x_train.reset_index(drop=True)
    
    
    X_train = pd.DataFrame(np.hstack((OHE_train,TFIDF_train )), columns = list(feature_names_OHE)+list(feature_names_TFIDF))
    X_val = pd.DataFrame(np.hstack((OHE_val,TFIDF_val)), columns =list(feature_names_OHE)+list(feature_names_TFIDF))
    X_test = pd.DataFrame(np.hstack((OHE_test,TFIDF_test)), columns = list(feature_names_OHE)+list(feature_names_TFIDF))
    
#     #
#     reg_av = x_train.groupby(by = 'region').agg({'salary_average_euros' : ['mean']})   
#     df_reg_av = reg_av[('salary_average_euros', 'mean')].reset_index()

#     x_array = np.array(df_reg_av[('salary_average_euros', 'mean')])
#     df_reg_av[('salary_average_euros', 'mean')] = [i for i in preprocessing.normalize([x_array])[0]]
    
#     d = {}
#     for i in range(len(df_reg_av['region'])):
#         d[df_reg_av['region'][i]] = df_reg_av[('salary_average_euros', 'mean')][i]
    
#     X_train['regional_av'] = x_train['region'].map(d)
#     X_val['regional_av'] = x_val['region'].map(d)    
#     X_test['regional_av'] = x_test['region'].map(d)


    #remove 'company' and region variables
    cols = [c for c in X_train.columns if 'company_' not in c]
#     cols = [c for c in X_train.columns if ('company_' not in c and 'region_' not in c )] 
    X_train2 = X_train[cols]
    X_val2 = X_val[cols]
    X_test2 = X_test[cols]
    feature_names = cols
    
    with open(path + '/data/x_data_for_models.pkl', 'wb') as file:
        pickle.dump([X_train2,X_val2,X_test2], file)