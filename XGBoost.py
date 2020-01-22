import os.path
import numpy as np
import pickle
import xgboost as xgb
from bayes_opt import BayesianOptimization
import pandas as pd


def mean_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def xgb_evaluate(max_depth, reg_lambda, colsample_bytree, subsample, min_child_weight):
    params1 = {'colsample_bytree': colsample_bytree, 'max_depth': int(round(max_depth)), 'reg_lambda': reg_lambda,
               'subsample': subsample, 'min_child_weight': min_child_weight}
    cv_result = xgb.cv(dtrain=df_DM,
                       params=params1,
                       early_stopping_rounds=3,
                       num_boost_round=1000,
                       metrics='rmse')
    return -cv_result['test-rmse-mean'].iloc[-1]


if __name__ == "__main__":
    path = os.getcwd()

    X_train = pd.read_csv(path + '/Pickles/X_Train.csv')
    X_val = pd.read_csv(path + '/Pickles/X_Val.csv')
    X_test = pd.read_csv(path + '/Pickles/X_Test.csv')

    y_train = pd.read_csv(path + '/Pickles/Y_Train.csv')
    y_val = pd.read_csv(path + '/Pickles/Y_Val.csv')
    y_test = pd.read_csv(path + '/Pickles/Y_Test.csv')

    train_index = pd.read_csv(path + '/Pickles/Train_index.csv', index_col=False).to_numpy()
    val_index = pd.read_csv(path + '/Pickles/Val_index.csv', index_col=False).to_numpy()
    test_index = pd.read_csv(path + '/Pickles/Test_index.csv', index_col=False).to_numpy()
    
    y_train = np.log(y_train)
    y_val = np.log(y_val)
    y_test = np.log(y_test)
        
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.05,
        'n_jobs': -1,
        'n_estimators': 1000,
        'verbosity': 1,
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
    df_DM = xgb.DMatrix(data=X_train, label=y_train)
    optimizer = BayesianOptimization(xgb_evaluate, {'max_depth': (3, 10),
                                                    'reg_lambda': (0, 5),
                                                    'colsample_bytree': (0.3, 1),
                                                    'subsample': (0.5, 1),
                                                    'min_child_weight': (1, 8)})

    optimizer.maximize(init_points=10, n_iter=10)

    params_1 = optimizer.max['params']
    params_1['max_depth'] = int(round(params_1['max_depth']))
    params.update(params_1)
    xgb_reg.fit(X_train, y_train, **fit_params)

    with open(path + '/Pickles/XGBparams.pkl', 'wb') as file:
        pickle.dump(params, file)

    y_pred_val = np.exp(xgb_reg.predict(X_val))
    y_pred_train = np.exp(xgb_reg.predict(X_train))
    y_pred_test = np.exp(xgb_reg.predict(X_test))
    
    y_train = np.exp(y_train)
    y_val = np.exp(y_val)
    y_test = np.exp(y_test)

    print('Train:')
    print('Mean Percentage Error: {0:.1f}'.format(mean_percentage_error(y_train, y_pred_train)))

    print('Validation:')
    print('Mean Percentage Error: {0:.1f}'.format(mean_percentage_error(y_val, y_pred_val)))

    print('Test:')
    print('Mean Percentage Error: {0:.1f}'.format(mean_percentage_error(y_test, y_pred_test)))

    # save model in pickles file
    with open(path + '/Pickles/xgb_model.pkl', 'wb') as file:
        pickle.dump(xgb_reg, file)

    # save predictions in data file
    xgb_preds_val = pd.DataFrame(data={'y_pred_xgb': y_pred_val.reshape(-1,), 'y_true': y_val.reshape(-1,)},
                                 index=val_index.reshape(-1,))
    xgb_preds_train = pd.DataFrame(data={'y_pred_xgb': y_pred_train.reshape(-1,), 'y_true': y_train.reshape(-1,)},
                                   index=train_index.reshape(-1,))
    xgb_preds_test = pd.DataFrame(data={'y_pred_xgb': y_pred_test.reshape(-1,), 'y_true': y_test.reshape(-1,)},
                                  index=test_index.reshape(-1,))

    xgb_preds_train.to_csv(path + '/Pickles/XGBpredtrain.csv')
    xgb_preds_val.to_csv(path + '/Pickles/XGBpredval.csv')
    xgb_preds_test.to_csv(path + '/Pickles/XGBpredtest.csv')
