import os.path
import numpy as np
import pickle
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn import metrics


import pandas as pd


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def mean_absolute_range_percentage_error(y_true, y_pred):
    error = np.abs(y_true - y_pred) - 10000
    error[error < 0] = 0
    return np.mean(error / y_true) * 100


if __name__ == "__main__":
    path = os.getcwd()
    with open(path + '/data/x_data_for_models_upsampled.pkl', 'rb') as file:
        X_train, X_val, X_test = pickle.load(file)
    with open(path + '/data/yTrainValTest_upsampled.pkl', 'rb') as file:
        y_train, y_val, y_test = pickle.load(file)
    with open(path + '/data/IndexTrainValTest_upsampled.pkl', 'rb') as file:
        train_index, val_index, test_index = pickle.load(file)

    y_train = np.log(y_train)
    y_val = np.log(y_val)

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

    y_pred = np.exp(xgb_reg.predict(X_val))
    y_pred_train = np.exp(xgb_reg.predict(X_train))
    y_train = np.exp(y_train)
    y_val = np.exp(y_val)

    print('Mean Absolute Error: {0:.0f}'.format(metrics.mean_absolute_error(y_val, y_pred)))
    print('Mean Absolute Percentage Error: {0:.1f}'.format(mean_absolute_percentage_error(y_val, y_pred)))
    print('Mean Absolute Range Percentage Error: {0:.1f}'.format(mean_absolute_range_percentage_error(y_val, y_pred)))

    print('Mean Squared Error: {0:.0f}'.format(metrics.mean_squared_error(y_val, y_pred)))
    print('Root Mean Squared Error:{0:.0f}'.format(np.sqrt(metrics.mean_squared_error(y_val, y_pred))))
    print('R2 Score:{0:.2f}'.format(np.sqrt(metrics.r2_score(y_val, y_pred))))

    print('Mean Absolute Error Train: {0:.0f}'.format(metrics.mean_absolute_error(y_train, y_pred_train)))
    print('Mean Absolute Percentage Error Train: {0:.1f}'.format(mean_absolute_percentage_error(y_train, y_pred_train)))
    print('Mean Absolute Range Percentage Error Train: {0:.1f}'.format(
        mean_absolute_range_percentage_error(y_train, y_pred_train)))
    print('R2 Score Train :{0:.2f}'.format(np.sqrt(metrics.r2_score(y_train, y_pred_train))))

    # save model in pickles file
    with open(path + '/Pickles/xgb_model_upsampled.pkl', 'wb') as file:
        pickle.dump(xgb_reg, file)

    # save predictions in data file
    xgb_preds_val = pd.DataFrame({'id': val_index, 'y_pred_xgb': y_pred, 'y_true': y_val})
    xgb_preds_train = pd.DataFrame({'id': train_index, 'y_pred_xgb': y_pred_train, 'y_true': y_train})

    with open(path + '/data/XGBpredtrain_upsampled.pkl', 'wb') as file:
        pickle.dump([xgb_preds_train], file)

    with open(path + '/data/XGBpredval_upsampled.pkl', 'wb') as file:
        pickle.dump([xgb_preds_val], file)