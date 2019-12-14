import numpy as np
import os.path
import pickle
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import pandas as pd


def mean_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_range_percentage_error(y_true, y_pred):
    error = np.abs(y_true- y_pred)-10000
    error[error < 0] = 0
    return np.mean(error/y_true)*100


if __name__ == "__main__":
    path = os.getcwd()
    with open(path + '/data/x_data_for_models.pkl', 'rb') as file:
            X_train,X_val,X_test = pickle.load(file)
    with open(path + '/data/yTrainValTest.pkl', 'rb') as file:
            y_train,y_val,y_test= pickle.load(file) 
    with open(path + '/data/IndexTrainValTest.pkl', 'rb') as file:
        train_index,val_index,test_index= pickle.load(file)    
        
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    scaled_X_train = scaler.transform(X_train)
    scaled_X_val = scaler.transform(X_val)
    scaled_X_test = scaler.transform(X_test)

#     X_trainval = np.concatenate([scaled_X_train, scaled_X_val])
#     y_trainval = np.concatenate([y_train, y_val])

    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}

    svr = svm.SVR(verbose=1, max_iter=10000, gamma='scale')
    gsvr = GridSearchCV(svr, parameters, n_jobs=-1)
    gsvr.fit(X_train, y_train)

    with open(path + '/Pickles/svr_model.pkl', 'wb') as file:
        pickle.dump(gsvr, file)

    y_pred_val = gsvr.predict(scaled_X_val)
    y_pred_train = gsvr.predict(scaled_X_train)
    y_pred_test = gsvr.predict(scaled_X_test)

    print('Train:')
    print('Mean Absolute Error: {0:.0f}'.format(metrics.mean_absolute_error(y_train, y_pred_train)))
    print('Mean Percentage Error: {0:.1f}'.format(mean_percentage_error(y_train, y_pred_train)))
    print('Mean Range Percentage Error: {0:.1f}'.format(mean_range_percentage_error(y_train, y_pred_train)))
    print('R2 Score:{0:.2f}'.format(np.sqrt(metrics.r2_score(y_train, y_pred_train))))

    print('Val:')
    print('Mean Absolute Error: {0:.0f}'.format(metrics.mean_absolute_error(y_train, y_pred_val)))
    print('Mean Percentage Error: {0:.1f}'.format(mean_percentage_error(y_train, y_pred_val)))
    print('Mean Range Percentage Error: {0:.1f}'.format(mean_range_percentage_error(y_train, y_pred_val)))
    print('R2 Score:{0:.2f}'.format(np.sqrt(metrics.r2_score(y_train, y_pred_val))))

    print('Test:')
    print('Mean Absolute Error: {0:.0f}'.format(metrics.mean_absolute_error(y_test, y_pred_test)))
    print('Mean Percentage Error: {0:.1f}'.format(mean_percentage_error(y_test, y_pred_test)))
    print('Mean Range Percentage Error: {0:.1f}'.format(mean_range_percentage_error(y_test, y_pred_test)))
    print('R2 Score:{0:.2f}'.format(np.sqrt(metrics.r2_score(y_test, y_pred_test))))

    svr_preds_val = pd.DataFrame({'id':val_index, 'y_pred_svr': y_pred_val, 'y_true': y_val})
    svr_preds_train = pd.DataFrame({'id':train_index, 'y_pred_svr': y_pred_train, 'y_true':y_train})
    svr_preds_test = pd.DataFrame({'id':test_index, 'y_pred_svr': y_pred_test, 'y_true':y_test})

    with open(path + '/data/SVRpredtrain.pkl', 'wb') as file:
            pickle.dump([svr_preds_train], file)

    with open(path + '/data/SVRpredval.pkl', 'wb') as file:
            pickle.dump([svr_preds_val], file)

    with open(path + '/data/SVRpredtest.pkl', 'wb') as file:
            pickle.dump([svr_preds_test], file)