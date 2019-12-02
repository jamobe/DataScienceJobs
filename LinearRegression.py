import numpy as np
import pickle
import os.path
from sklearn import linear_model
from sklearn import metrics
import pandas as pd


def mean_absolute_percentage_error(y_true, y_pred): 

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_absolute_range_percentage_error(y_true, y_pred): 
    error = np.abs(y_true- y_pred)-10000
    error[error < 0] = 0
    return np.mean(error/y_true)*100#

if __name__ == "__main__":
    path = os.getcwd()
    with open(path + '/data/x_data_for_models.pkl', 'rb') as file:
            X_train,X_val,X_test = pickle.load(file)
    with open(path + '/data/yTrainValTest.pkl', 'rb') as file:
            y_train,y_val,y_test= pickle.load(file) 
    with open(path + '/data/IndexTrainValTest.pkl', 'rb') as file:
        train_index,val_index,test_index= pickle.load(file)

    regr = linear_model.RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])
    regr.fit(X_train, y_train)  # training the algorithm

    y_pred= regr.predict(X_val)
    y_pred_train = regr.predict(X_train)
    
    print('Mean Absolute Error: {0:.0f}'.format( metrics.mean_absolute_error(y_val, y_pred)))
    print('Mean Absolute Percentage Error: {0:.1f}'.format(mean_absolute_percentage_error(y_val, y_pred)))
    print('Mean Absolute Range Percentage Error: {0:.1f}'.format(mean_absolute_range_percentage_error(y_val, y_pred)))

    print('Mean Squared Error: {0:.0f}'.format(metrics.mean_squared_error(y_val, y_pred)))
    print('Root Mean Squared Error:{0:.0f}'.format(np.sqrt(metrics.mean_squared_error(y_val, y_pred))))
    print('R2 Score:{0:.2f}'.format(np.sqrt(metrics.r2_score(y_val, y_pred))))


    print('Mean Absolute Error Train: {0:.0f}'.format( metrics.mean_absolute_error(y_train, y_pred_train)))
    print('Mean Absolute Percentage Error Train: {0:.1f}'.format(mean_absolute_percentage_error(y_train, y_pred_train)))
    print('Mean Absolute Range Percentage Error Train: {0:.1f}'.format(mean_absolute_range_percentage_error(y_train, y_pred_train)))
    print('R2 Score:{0:.2f}'.format(np.sqrt(metrics.r2_score(y_train, y_pred_train))))

    with open(path + '/Pickles/linReg_model.pkl', 'wb') as file:
        pickle.dump(regr, file)

    reg_preds_val = pd.DataFrame({'id':val_index, 'y_pred_reg': y_pred, 'y_true': y_val})
    reg_preds_train= pd.DataFrame({'id':train_index, 'y_pred_reg': y_pred_train, 'y_true':y_train})

    with open(path + '/data/REGpredtrain.pkl', 'wb') as file:
            pickle.dump([reg_preds_train], file)

    with open(path + '/data/REGpredval.pkl', 'wb') as file:
            pickle.dump([reg_preds_val], file)