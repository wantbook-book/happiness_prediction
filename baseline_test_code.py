# 安装必要的库
# !pip install xgboost lightgbm catboost scikit-learn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import joblib

# 加载数据
X = pd.read_csv('dataset/processed_data/xy_train.csv')
# y = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/happiness_prediction/processed_data/y_train.csv')['happiness']
y = X.pop('happiness')

# 数据集划分
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用 SimpleImputer 填充缺失值
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_valid = imputer.transform(X_valid)

# XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, verbosity=0)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_valid)
rmse_xgb = np.sqrt(mean_squared_error(y_valid, y_pred_xgb))
print(f'XGBoost Validation RMSE: {rmse_xgb}')

# LightGBM
lgb_model = lgb.LGBMRegressor(n_estimators=50, num_leaves=15, learning_rate=0.1)
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_valid)
rmse_lgb = np.sqrt(mean_squared_error(y_valid, y_pred_lgb))
print(f'LightGBM Validation RMSE: {rmse_lgb}')

# CatBoost
ctb_model = CatBoostRegressor(n_estimators=50, depth=3, learning_rate=0.1, verbose=0)
ctb_model.fit(X_train, y_train)
y_pred_ctb = ctb_model.predict(X_valid)
rmse_ctb = np.sqrt(mean_squared_error(y_valid, y_pred_ctb))
print(f'CatBoost Validation RMSE: {rmse_ctb}')

# GradientBoostingRegressor
gbdt_model = GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.1)
gbdt_model.fit(X_train, y_train)
y_pred_gbdt = gbdt_model.predict(X_valid)
rmse_gbdt = np.sqrt(mean_squared_error(y_valid, y_pred_gbdt))
print(f'GBDT Validation RMSE: {rmse_gbdt}')

# HistGradientBoostingRegressor
hist_gbdt_model = HistGradientBoostingRegressor(max_iter=50, max_depth=3, learning_rate=0.1)
hist_gbdt_model.fit(X_train, y_train)
y_pred_hist_gbdt = hist_gbdt_model.predict(X_valid)
rmse_hist_gbdt = np.sqrt(mean_squared_error(y_valid, y_pred_hist_gbdt))
print(f'HistGradientBoosting Validation RMSE: {rmse_hist_gbdt}')
