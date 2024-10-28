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

def main():
    preprocessed_data_path = Path(__file__).parent / 'preprocess_data'
    if not preprocessed_data_path.exists():
        print('preprocess_data does not exist. run preprocess.py first')
        return
    results_path = Path(__file__).parent / 'results' / '20241028_143323'
    # 加载测试集
    test_data = pd.read_csv(preprocessed_data_path/'X_test.csv')
    X_test = np.array(test_data)

    # 初始化保存预测结果的数组
    predictions_xgb = np.zeros(len(X_test))

    # 加载每个折的模型并进行预测
    for fold in range(5):
        model_path = results_path / f"xgb_model_{fold}.model"
        clf = xgb.Booster()
        clf.load_model(model_path)
        
        # 使用加载的模型预测测试集
        predictions_xgb += clf.predict(xgb.DMatrix(X_test)) / 5

    # 将最终预测结果保存到 CSV 文件
    happiness_df = pd.DataFrame({'happiness': predictions_xgb})
    happiness_df.to_csv(results_path/'y_test.csv', index=False)
    print(f"Predictions saved to {results_path/'y_test.csv'}")

if __name__ == '__main__':
    main()