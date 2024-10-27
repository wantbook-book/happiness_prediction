import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, RepeatedKFold
from scipy import sparse
from datetime import datetime
from pathlib import Path
#自定义评价函数
def myFeval(preds, xgbtrain):
    label = xgbtrain.get_label()
    score = mean_squared_error(label,preds)
    return 'myFeval',score

def main():
    preprocess_data_path = Path(__file__).parent / 'preprocess_data'
    if not preprocess_data_path.exists():
        print('preprocess_data does not exist. run preprocess.py first')
        return
    # 加载预处理的数据集
    X_train_ = pd.read_csv(preprocess_data_path / 'X_train.csv')
    X_test_ = pd.read_csv(preprocess_data_path / 'X_test.csv')
    y_train_ = pd.read_csv(preprocess_data_path / 'y_train.csv')
    target_column = 'happiness'
    X_train = np.array(X_train_)
    y_train = np.array(y_train_)
    X_test  = np.array(X_test_)

    ##### xgb

    xgb_params = {"booster":'gbtree','eta': 0.005, 'max_depth': 5, 'subsample': 0.7, 
                'colsample_bytree': 0.8, 'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True, 'nthread': 8}
    folds = KFold(n_splits=5, shuffle=True, random_state=2018)
    oof_xgb = np.zeros(len(X_train_))
    predictions_xgb = np.zeros(len(X_test_))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
        print("fold n°{}".format(fold_+1))
        trn_data = xgb.DMatrix(X_train[trn_idx], y_train[trn_idx])
        val_data = xgb.DMatrix(X_train[val_idx], y_train[val_idx])
        
        watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
        clf = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=100, params=xgb_params,feval = myFeval)
        # oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]), ntree_limit=clf.num_boost_rounds)
        oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]))
        # predictions_xgb += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.num_boost_rounds) / folds.n_splits
        predictions_xgb += clf.predict(xgb.DMatrix(X_test)) / folds.n_splits
        
    print("CV score: {:<8.8f}".format(mean_squared_error(oof_xgb, y_train_)))


if __name__ == '__main__':
    main()