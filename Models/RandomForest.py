import os.path
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import pandas as pd

# define functions


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
        
    params = {'n_estimators':100,'max_depth': 20, 'n_jobs':-1}
    
    # Create the model with 100 trees
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    
    with open(path + '/Pickles/RF_model.pkl', 'wb') as file:
            pickle.dump([model], file)
            
    y_pred_val = model.predict(X_val)
    y_pred_train = model.predict(X_train)
    y_pred_test= model.predict(X_test)


    print('Train:')
    print('Mean Absolute Error: {0:.0f}'.format(metrics.mean_absolute_error(y_train, y_pred_train)))
    print('Mean Percentage Error: {0:.1f}'.format(mean_percentage_error(y_train, y_pred_train)))
    print('Mean Range Percentage Error: {0:.1f}'.format(mean_range_percentage_error(y_train, y_pred_train)))
    print('R2 Score:{0:.2f}'.format(np.sqrt(metrics.r2_score(y_train, y_pred_train))))

    print('Validation:')
    print('Mean Absolute Error: {0:.0f}'.format(metrics.mean_absolute_error(y_val, y_pred_val)))
    print('Mean Percentage Error: {0:.1f}'.format(mean_percentage_error(y_val, y_pred_val)))
    print('Mean Range Percentage Error: {0:.1f}'.format(mean_range_percentage_error(y_val, y_pred_val)))
    print('R2 Score:{0:.2f}'.format(np.sqrt(metrics.r2_score(y_val, y_pred_val))))

    print('Test:')
    print('Mean Absolute Error: {0:.0f}'.format(metrics.mean_absolute_error(y_test, y_pred_test)))
    print('Mean Percentage Error: {0:.1f}'.format(mean_percentage_error(y_test, y_pred_test)))
    print('Mean Range Percentage Error: {0:.1f}'.format(mean_range_percentage_error(y_test, y_pred_test)))
    print('R2 Score:{0:.2f}'.format(np.sqrt(metrics.r2_score(y_test, y_pred_test))))

    
    rf_preds_val = pd.DataFrame({'id':val_index, 'y_pred_rf': y_pred_val, 'y_true': y_val})
    rf_preds_train = pd.DataFrame({'id':train_index, 'y_pred_rf': y_pred_train, 'y_true':y_train})
    rf_preds_test = pd.DataFrame({'id':test_index, 'y_pred_rf': y_pred_test, 'y_true':y_test})

    with open(path + '/data/RFpredtrain.pkl', 'wb') as file:
            pickle.dump([rf_preds_train], file)

    with open(path + '/data/RFpredval.pkl', 'wb') as file:
            pickle.dump([rf_preds_val], file)

    with open(path + '/data/RFpredtest.pkl', 'wb') as file:
        pickle.dump([rf_preds_test], file)