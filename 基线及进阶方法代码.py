!pip install xgboost lightgbm catboost scikit-learn
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

def train_and_evaluate(X_train, y_train, X_test, y_test):
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # XGBoost算法
    xgb_model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, verbosity=0)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb_train = xgb_model.predict(X_train)
    y_pred_xgb_test = xgb_model.predict(X_test)
    score_xgb_train = mean_squared_error(y_train, y_pred_xgb_train)
    score_xgb_test = mean_squared_error(y_test, y_pred_xgb_test)
    print(f'XGBoost Train Score: {score_xgb_train}, Test Score: {score_xgb_test}')

    # LightGBM算法
    lgb_model = lgb.LGBMRegressor(n_estimators=50, num_leaves=15, learning_rate=0.1)
    lgb_model.fit(X_train, y_train)
    y_pred_lgb_train = lgb_model.predict(X_train)
    y_pred_lgb_test = lgb_model.predict(X_test)
    score_lgb_train = mean_squared_error(y_train, y_pred_lgb_train)
    score_lgb_test = mean_squared_error(y_test, y_pred_lgb_test)
    print(f'LightGBM Train Score: {score_lgb_train}, Test Score: {score_lgb_test}')

    # CatBoost算法
    ctb_model = CatBoostRegressor(n_estimators=50, depth=3, learning_rate=0.1, verbose=0)
    ctb_model.fit(X_train, y_train)
    y_pred_ctb_train = ctb_model.predict(X_train)
    y_pred_ctb_test = ctb_model.predict(X_test)
    score_ctb_train = mean_squared_error(y_train, y_pred_ctb_train)
    score_ctb_test = mean_squared_error(y_test, y_pred_ctb_test)
    print(f'CatBoost Train Score: {score_ctb_train}, Test Score: {score_ctb_test}')

    # GradientBoostingRegressor算法
    gbdt_model = GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.1)
    gbdt_model.fit(X_train, y_train)
    y_pred_gbdt_train = gbdt_model.predict(X_train)
    y_pred_gbdt_test = gbdt_model.predict(X_test)
    score_gbdt_train = mean_squared_error(y_train, y_pred_gbdt_train)
    score_gbdt_test = mean_squared_error(y_test, y_pred_gbdt_test)
    print(f'GBDT Train Score: {score_gbdt_train}, Test Score: {score_gbdt_test}')

    # HistGradientBoostingRegressor算法
    hist_gbdt_model = HistGradientBoostingRegressor(max_iter=50, max_depth=3, learning_rate=0.1)
    hist_gbdt_model.fit(X_train, y_train)
    y_pred_hist_gbdt_train = hist_gbdt_model.predict(X_train)
    y_pred_hist_gbdt_test = hist_gbdt_model.predict(X_test)
    score_hist_gbdt_train = mean_squared_error(y_train, y_pred_hist_gbdt_train)
    score_hist_gbdt_test = mean_squared_error(y_test, y_pred_hist_gbdt_test)
    print(f'HistGradientBoosting Train Score: {score_hist_gbdt_train}, Test Score: {score_hist_gbdt_test}')

    # RandomForestRegressor算法
    rf_model = RandomForestRegressor(n_estimators=30, max_depth=5)
    rf_model.fit(X_train, y_train)
    y_pred_rf_train = rf_model.predict(X_train)
    y_pred_rf_test = rf_model.predict(X_test)
    score_rf_train = mean_squared_error(y_train, y_pred_rf_train)
    score_rf_test = mean_squared_error(y_test, y_pred_rf_test)
    print(f'Random Forest Train Score: {score_rf_train}, Test Score: {score_rf_test}')

# 我们准备的一系列处理后的数据集
train_test_pairs = [
    ('/content/drive/MyDrive/Colab Notebooks/split_data/xy_train.csv', '/content/drive/MyDrive/Colab Notebooks/split_data/xy_test.csv'),
    ('/content/drive/MyDrive/Colab Notebooks/split_data/xy_train_filtered_feature.csv', '/content/drive/MyDrive/Colab Notebooks/split_data/xy_test_filtered_feature.csv'),
    ('/content/drive/MyDrive/Colab Notebooks/split_data/xy_train_filtered_feature0.01.csv', '/content/drive/MyDrive/Colab Notebooks/split_data/xy_test_filtered_feature0.01.csv'),
    ('/content/drive/MyDrive/Colab Notebooks/split_data/xy_train_filtered_feature0.005.csv', '/content/drive/MyDrive/Colab Notebooks/split_data/xy_test_filtered_feature0.005.csv'),
    ('/content/drive/MyDrive/Colab Notebooks/split_data/xy_train_resampled_smote.csv', '/content/drive/MyDrive/Colab Notebooks/split_data/xy_test.csv'),
    ('/content/drive/MyDrive/Colab Notebooks/split_data/xy_train_resampled_adasyn.csv', '/content/drive/MyDrive/Colab Notebooks/split_data/xy_test.csv'),
    ('/content/drive/MyDrive/Colab Notebooks/split_data/xy_train_filtered_feature_resampled_smote.csv', '/content/drive/MyDrive/Colab Notebooks/split_data/xy_test_filtered_feature.csv'),
    ('/content/drive/MyDrive/Colab Notebooks/split_data/xy_train_filtered_feature_resampled_adasyn.csv', '/content/drive/MyDrive/Colab Notebooks/split_data/xy_test_filtered_feature.csv'),
    ('/content/drive/MyDrive/Colab Notebooks/split_data/xy_train_filtered_feature_0.005_resampled_smote.csv', '/content/drive/MyDrive/Colab Notebooks/split_data/xy_test_filtered_feature0.005.csv'),
    ('/content/drive/MyDrive/Colab Notebooks/split_data/xy_train_filtered_feature_0.005_resampled_adasyn.csv', '/content/drive/MyDrive/Colab Notebooks/split_data/xy_test_filtered_feature0.005.csv')
]

# 按顺序处理数据集，生成训练数据集和测试数据集，进行训练
for i, (train_path, test_path) in enumerate(train_test_pairs, start=1):
    print(f'\n--- Dataset Pair {i} ---')
    train_data = pd.read_csv(train_path, encoding='ISO-8859-1')
    X_train = train_data.drop(columns=['happiness'])
    y_train = train_data['happiness']

    test_data = pd.read_csv(test_path, encoding='ISO-8859-1')
    X_test = test_data.drop(columns=['happiness'])
    y_test = test_data['happiness']

    train_and_evaluate(X_train, y_train, X_test, y_test)