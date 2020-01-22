import numpy as np
import os.path
import pickle
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import pandas as pd


def mean_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


if __name__ == "__main__":
    path = os.getcwd()
    X_train = pd.read_csv(path + '/Pickles/X_Train.csv')
    X_val = pd.read_csv(path + '/Pickles/X_Val.csv')
    X_test = pd.read_csv(path + '/Pickles/X_Test.csv')

    y_train = pd.read_csv(path + '/Pickles/Y_Train.csv', index_col=False).to_numpy()
    y_val = pd.read_csv(path + '/Pickles/Y_Val.csv', index_col=False).to_numpy()
    y_test = pd.read_csv(path + '/Pickles/Y_Test.csv', index_col=False).to_numpy()

    train_index = pd.read_csv(path + '/Pickles/Train_index.csv', index_col=False).to_numpy()
    val_index = pd.read_csv(path + '/Pickles/Val_index.csv', index_col=False).to_numpy()
    test_index = pd.read_csv(path + '/Pickles/Test_index.csv', index_col=False).to_numpy()
        
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
    print('Mean Percentage Error: {0:.1f}'.format(mean_percentage_error(y_train, y_pred_train)))

    print('Val:')
    print('Mean Percentage Error: {0:.1f}'.format(mean_percentage_error(y_train, y_pred_val)))

    print('Test:')
    print('Mean Percentage Error: {0:.1f}'.format(mean_percentage_error(y_test, y_pred_test)))

    svr_preds_val = pd.DataFrame(data={'y_pred_svr': y_pred_val.reshape(-1,), 'y_true': y_val.reshape(-1,)},
                                 index=val_index.reshape(-1,))
    svr_preds_train = pd.DataFrame(data={'y_pred_svr': y_pred_train.reshape(-1,), 'y_true': y_train.reshape(-1,)},
                                   index=train_index.reshape(-1,))
    svr_preds_test = pd.DataFrame(data={'y_pred_svr': y_pred_test.reshape(-1,), 'y_true': y_test.reshape(-1,)},
                                  index=test_index.reshape(-1,))

    svr_preds_train.to_csv(path + '/Pickles/SVRpredtrain.csv')
    svr_preds_val.to_csv(path + '/Pickles/SVRpredval.csv')
    svr_preds_test.to_csv(path + '/Pickles/SVRpredtest.csv')
