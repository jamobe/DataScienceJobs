import numpy as np
import os.path
import pickle
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

if __name__ == "__main__":
    path = os.getcwd()

    with open(path + '/data/TrainSetXY.pkl', 'rb') as file:
        X_train, y_train = pickle.load(file)
    with open(path + '/data/ValSetXY.pkl', 'rb') as file:
        X_val, y_val = pickle.load(file)
    with open(path + '/data/TestSetXY.pkl', 'rb') as file:
        X_test, y_test = pickle.load(file)

    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    scaled_X_train = scaler.transform(X_train)
    scaled_X_val = scaler.transform(X_val)
    scaled_X_test = scaler.transform(X_test)

    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}

    svr = svm.SVR(verbose=1, max_iter=10000, gamma='scale')
    clf = GridSearchCV(svr, parameters, n_jobs=-1)
    clf.fit(scaled_X_train, y_train)

    y_pred = clf.predict(scaled_X_test)

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('R2 Score:', np.sqrt(metrics.r2_score(y_test, y_pred)))