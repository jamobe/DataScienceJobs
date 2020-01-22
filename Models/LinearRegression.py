import numpy as np
import pickle
import os.path
from sklearn import linear_model
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

    regr = linear_model.RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])
    regr.fit(X_train, y_train)

    y_pred_val = regr.predict(X_val)
    y_pred_train = regr.predict(X_train)
    y_pred_test = regr.predict(X_test)

    print('Train:')
    print('Mean Percentage Error: {0:.1f}'.format(mean_percentage_error(y_train, y_pred_train)))

    print('Validation:')
    print('Mean Percentage Error: {0:.1f}'.format(mean_percentage_error(y_val, y_pred_val)))
    
    print('Test:')
    print('Mean Percentage Error: {0:.1f}'.format(mean_percentage_error(y_test, y_pred_test)))

    with open(path + '/Pickles/linReg_model.pkl', 'wb') as file:
        pickle.dump(regr, file)

    reg_preds_val = pd.DataFrame(data={'y_pred_reg': y_pred_val.reshape(-1,), 'y_true': y_val.reshape(-1,)},
                                 index=val_index.reshape(-1,))
    reg_preds_train = pd.DataFrame(data={'y_pred_reg': y_pred_train.reshape(-1,), 'y_true': y_train.reshape(-1,)},
                                   index=train_index.reshape(-1,))
    reg_preds_test = pd.DataFrame(data={'y_pred_reg': y_pred_test.reshape(-1,), 'y_true': y_test.reshape(-1,)},
                                  index=test_index.reshape(-1,))

    reg_preds_train.to_csv(path + '/Pickles/REGpredtrain.csv')
    reg_preds_val.to_csv(path + '/Pickles/REGpredval.csv')
    reg_preds_test.to_csv(path + '/Pickles/REGpredtes.csv')
