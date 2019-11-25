import os.path
import numpy as np
import pickle
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn import metrics

if __name__ == "__main__":
    path = os.getcwd()

    with open(path + '/data/TrainSetXY.pkl', 'rb') as file:
        X_train, y_train = pickle.load(file)
    with open(path + '/data/ValSetXY.pkl', 'rb') as file:
        X_val, y_val = pickle.load(file)
    with open(path + '/data/TestSetXY.pkl', 'rb') as file:
        X_test, y_test = pickle.load(file)

    params = {
        # Learning Task Parameters
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',  # Evaluation metrics for validation data
        # Parameters for Tree Booster
        'learning_rate': 0.05,  # Learning Rate: step size shrinkage used to prevent overfitting.
        # Paramters for XGB ScikitLearn API
        'n_jobs': -1,  # Number of parallel threads used to run xgboost
        'n_estimators': 1000,  # number of trees you want to build
        'verbosity': 1,  # degree of verbosity: 0 (silent) - 3 (debug)
        'max_depth': 4,
        'reg_lambda': 1,
        'colsample_bytree': 0.5,
        'subsample': 0.9
    }
    fit_params = {
        'eval_metric': 'rmse',
        'early_stopping_rounds': 3,
        'eval_set': [(X_val, y_val)],
    }
    xgb_reg = xgb.XGBRegressor(**params)

    xgb_reg.fit(X_train, y_train, **fit_params)

    df_DM = xgb.DMatrix(data=X_train, label=y_train)

    def xgb_evaluate(max_depth, reg_lambda, colsample_bytree, subsample, min_child_weight):
        params1 = {
            'colsample_bytree': colsample_bytree,
            'max_depth': int(round(max_depth)),  # Maximum depth of a tree: high value
            # -> prone to overfitting
            'reg_lambda': reg_lambda,  # L2 regularization term on weights
            'subsample': subsample,
            'min_child_weight': min_child_weight
        }
        cv_result = xgb.cv(dtrain=df_DM,
                           params=params1,
                           early_stopping_rounds=3,
                           num_boost_round=1000,
                           metrics='rmse')
        return -cv_result['test-rmse-mean'].iloc[-1]


    optimizer = BayesianOptimization(xgb_evaluate, {'max_depth': (3, 10),
                                                    'reg_lambda': (0, 5),
                                                    'colsample_bytree': (0.3, 1),
                                                    'subsample': (0.5, 1),
                                                    'min_child_weight': (1, 8)})

    optimizer.maximize(init_points=10, n_iter=10)

    params_1 = optimizer.max['params']
    params_1['max_depth'] = int(round(params_1['max_depth']))
    params.update(params_1)

    xgb_reg = xgb.XGBRegressor(**params)
    xgb_reg.fit(X_train, y_train, **fit_params)

    y_pred = xgb_reg.predict(X_test)

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('R2 Score:', np.sqrt(metrics.r2_score(y_test, y_pred)))

    with open(path + '/Pickles/xgb_model_2.pkl', 'wb') as file:
        pickle.dump(xgb_reg, file)