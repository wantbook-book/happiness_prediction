!pip install xgboost lightgbm catboost scikit-learn optuna tqdm
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import optuna
from tqdm import tqdm

def train_and_evaluate(X_train, y_train, X_test, y_test, dataset_name):
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    oof_preds = np.zeros(X_train.shape[0])
    all_test_preds = []

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 定义优化目标函数
    def objective(trial):
        # 定义参数搜索范围
        xgb_param = {
            'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 300),
            'max_depth': trial.suggest_int('xgb_max_depth', 3, 8),
            'learning_rate': trial.suggest_loguniform('xgb_learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_uniform('xgb_subsample', 0.7, 1.0)
        }

        lgb_param = {
            'n_estimators': trial.suggest_int('lgb_n_estimators', 100, 300),
            'num_leaves': trial.suggest_int('lgb_num_leaves', 20, 50),
            'learning_rate': trial.suggest_loguniform('lgb_learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_uniform('lgb_subsample', 0.7, 1.0)
        }

        ctb_param = {
            'iterations': trial.suggest_int('ctb_iterations', 100, 300),
            'depth': trial.suggest_int('ctb_depth', 3, 8),
            'learning_rate': trial.suggest_loguniform('ctb_learning_rate', 0.01, 0.2)
        }

        gbdt_param = {
            'n_estimators': trial.suggest_int('gbdt_n_estimators', 100, 300),
            'max_depth': trial.suggest_int('gbdt_max_depth', 3, 8),
            'learning_rate': trial.suggest_loguniform('gbdt_learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_uniform('gbdt_subsample', 0.7, 1.0)
        }

        hist_gbdt_param = {
            'max_iter': trial.suggest_int('hist_gbdt_max_iter', 100, 300),
            'max_depth': trial.suggest_int('hist_gbdt_max_depth', 3, 8),
            'learning_rate': trial.suggest_loguniform('hist_gbdt_learning_rate', 0.01, 0.2)
        }

        # 创建多种模型
        xgb_model = xgb.XGBRegressor(**xgb_param, verbosity=0)
        lgb_model = lgb.LGBMRegressor(**lgb_param, verbose=-1)
        ctb_model = CatBoostRegressor(**ctb_param, verbose=0)
        gbdt_model = GradientBoostingRegressor(**gbdt_param)
        hist_gbdt_model = HistGradientBoostingRegressor(**hist_gbdt_param)

        fold_rmse = []
        fold_test_preds = np.zeros(X_test.shape[0])

        # 使用KFold方法，交叉验证
        for train_index, valid_index in kf.split(X_train):
            X_tr, X_val = X_train[train_index], X_train[valid_index]
            y_tr, y_val = y_train[train_index], y_train[valid_index]

            models = [xgb_model, lgb_model, ctb_model, gbdt_model, hist_gbdt_model]
            valid_preds = np.zeros(X_val.shape[0])
            test_fold_preds = np.zeros(X_test.shape[0])

            for model in models:
                model.fit(X_tr, y_tr)
                valid_preds += model.predict(X_val) / len(models)
                test_fold_preds += model.predict(X_test) / len(models)

            oof_preds[valid_index] = valid_preds
            fold_test_preds += test_fold_preds / kf.get_n_splits()
            fold_rmse.append(np.sqrt(mean_squared_error(y_val, valid_preds)))
        all_test_preds.append(fold_test_preds)
        return np.mean(fold_rmse)

    # 使用Optuna进行超参数优化
    study = optuna.create_study(direction='minimize')
    for _ in tqdm(range(10), desc=f"Optimizing {dataset_name}"):
        study.optimize(objective, n_trials=1)

    avg_test_preds = np.mean(all_test_preds, axis=0)

    # 使用Ridge回归模型作为最终的学习器
    ridge_model = Ridge()
    ridge_model.fit(oof_preds.reshape(-1, 1), y_train)
    final_preds = ridge_model.predict(avg_test_preds.reshape(-1, 1))

    final_rmse = mean_squared_error(y_test, final_preds, squared=False)
    print(f'{dataset_name} - Ridge Final Test RMSE: {final_rmse}')

# 我们经过处理的10个数据集对
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

# 处理数据集，得到训练集和测试集，开始训练
for i, (train_path, test_path) in enumerate(train_test_pairs, start=1):
    dataset_name = f'Dataset Pair {i}'
    print(f'\n--- {dataset_name} ---')
    train_data = pd.read_csv(train_path, encoding='ISO-8859-1')
    X_train = train_data.drop(columns=['happiness'])
    y_train = train_data['happiness']

    test_data = pd.read_csv(test_path, encoding='ISO-8859-1')
    X_test = test_data.drop(columns=['happiness'])
    y_test = test_data['happiness']

    train_and_evaluate(X_train, y_train, X_test, y_test, dataset_name)