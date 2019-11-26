import numpy as np
import pickle
import os.path
from sklearn import linear_model
from sklearn import metrics

if __name__ == "__main__":
    path = os.getcwd()

    with open(path + '/data/TrainSetXY.pkl', 'rb') as file:
        X_train, y_train = pickle.load(file)
    with open(path + '/data/ValSetXY.pkl', 'rb') as file:
        X_val, y_val = pickle.load(file)
    with open(path + '/data/TestSetXY.pkl', 'rb') as file:
        X_test, y_test = pickle.load(file)

    X_trainval = np.concatenate([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    regr = linear_model.RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])
    regr.fit(X_trainval, y_trainval)  # training the algorithm

    with open(path + '/Pickles/linReg_model.pkl', 'wb') as file:
        pickle.dump(regr, file)

    y_pred_t = regr.predict(X_test)

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_t))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_t))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_t)))
    print('R2 Score:', np.sqrt(metrics.r2_score(y_test, y_pred_t)))

