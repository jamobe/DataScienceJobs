import os.path
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd


def mean_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


if __name__ == "__main__":
    path = os.getcwd()
    X_train = pd.read_csv(path + '/Pickles/X_Train_all.csv')
    X_test = pd.read_csv(path + '/Pickles/X_Test_all.csv')

    y_train = pd.read_csv(path + '/Pickles/Y_Train_all.csv', index_col=False).to_numpy()
    y_test = pd.read_csv(path + '/Pickles/Y_Test_all.csv', index_col=False).to_numpy()

    train_index = pd.read_csv(path + '/Pickles/Train_index_all.csv', index_col=False).to_numpy()
    test_index = pd.read_csv(path + '/Pickles/Test_index_all.csv', index_col=False).to_numpy()

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(5, 30, num=6)]
    # max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10, 15, 100]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 5, 10]
    # Method of selecting samples for training each tree
    # bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf}

    params = {'n_estimators': 100, 'max_depth': 20, 'n_jobs': -1}

    model = RandomForestRegressor(**params)
    RF_random = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=50,
                                   cv=5, verbose=2, random_state=42, n_jobs=-1)
    RF_random.fit(X_train, y_train.ravel())

    with open(path + '/Pickles/RF_model.pkl', 'wb') as file:
        pickle.dump(RF_random, file)

    y_pred_train = RF_random.predict(X_train)
    y_pred_test = RF_random.predict(X_test)

    print('Train:')
    print('Mean Percentage Error: {0:.1f}'.format(mean_percentage_error(y_train.ravel(), y_pred_train)))

    print('Test:')
    print('Mean Percentage Error: {0:.1f}'.format(mean_percentage_error(y_test.ravel(), y_pred_test)))

    rf_preds_train = pd.DataFrame({'id': train_index.ravel(), 'y_pred_rf': y_pred_train.ravel(),
                                   'y_true': y_train.ravel()})
    rf_preds_test = pd.DataFrame({'id': test_index.ravel(), 'y_pred_rf': y_pred_test.ravel(), 'y_true': y_test.ravel()})

    rf_preds_train.to_csv(path + '/Pickles/RFpredtrain.csv')
    rf_preds_test.to_csv(path + '/Pickles/RFpredtes.csv')
