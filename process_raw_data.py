import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, RepeatedKFold
from scipy import sparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE, ADASYN
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
from datetime import datetime
def hour_cut(x):
    if 0<=x<6:
        return 0
    elif  6<=x<8:
        return 1
    elif  8<=x<12:
        return 2
    elif  12<=x<14:
        return 3
    elif  14<=x<18:
        return 4
    elif  18<=x<21:
        return 5
    elif  21<=x<24:
        return 6

#出生的年代
def birth_split(x):
    if 1920<=x<=1930:
        return 0
    elif  1930<x<=1940:
        return 1
    elif  1940<x<=1950:
        return 2
    elif  1950<x<=1960:
        return 3
    elif  1960<x<=1970:
        return 4
    elif  1970<x<=1980:
        return 5
    elif  1980<x<=1990:
        return 6
    elif  1990<x<=2000:
        return 7
    else:
        return 8
#收入分组
def income_cut(x):
    if x<0:
        return 0
    elif  0<=x<1200:
        return 1
    elif  1200<x<=10000:
        return 2
    elif  10000<x<24000:
        return 3
    elif  24000<x<40000:
        return 4
    elif  40000<=x:
        return 5
    else:
        return 6
def main(
    drop_unlabled=False,
    filtered_features_thres=0.005,
    raw_data_dir = Path('dataset/raw_data'),
    processed_data_dir = Path('dataset/processed_data'),
    for_change_filtered_features_thres=False
):
    if processed_data_dir.exists() and not for_change_filtered_features_thres:
        print('processed_data_dir exists')
        return
    else:
        processed_data_dir.mkdir(parents=True, exist_ok=True)
    drop_unlabeled = False
    # preprocess_data_dir = Path('preprocess_data.ipynb').parent / 'split_data'
    # preprocess_data_dir.mkdir(parents=True, exist_ok=True)
    #导入数据
    train=pd.read_csv(raw_data_dir/"happiness_train_complete.csv",encoding='ISO-8859-1')
    # test=pd.read_csv(raw_data_dir/"happiness_test_complete.csv",encoding='ISO-8859-1')
    # 拆分train部分数据作为test
    test = train.sample(n=500, random_state=42)
    train = train.drop(test.index)
    if drop_unlabeled:
        # y_train_ = y_train_.loc[y_train_ != -8]
        train = train.loc[train['happiness'] != -8]
    else:
        train['happiness'] = train['happiness'].map(lambda x: 3 if x == -8 else x)
        test['happiness'] = test['happiness'].map(lambda x: 3 if x == -8 else x)

    # 一起预处理
    data = pd.concat([train, test], axis=0, ignore_index=True)
    

    # 填充缺失值
    # 1. 处理时间特征
    data['survey_time'] = pd.to_datetime(data['survey_time'],format='%Y/%m/%d %H:%M')
    data["weekday"]=data["survey_time"].dt.weekday
    data["year"]=data["survey_time"].dt.year
    data["quarter"]=data["survey_time"].dt.quarter
    data["hour"]=data["survey_time"].dt.hour
    data["month"]=data["survey_time"].dt.month
    data['survey_time'] = data['survey_time'].map(lambda x: x.value//10**6)
    # 2. 数据分箱
    # 2.1 一天的hour
    data["hour_cut"]=data["hour"].map(hour_cut)
    # 2.2 出生的年代
    data["birth_s"]=data["birth"].map(birth_split)
    # 2.3 收入水平
    data["income_cut"]=data["income"].map(income_cut)
    # 3. 做问卷时候的年龄
    data["survey_age"]=data["year"]-data["birth"]
    # 4. 去掉过多的缺失值
    data=data.drop(["edu_other"], axis=1)
    # data=data.drop(["happiness"], axis=1)
    data=data.drop(["survey_time"], axis=1)
    # data = data.select_dtypes(include=[np.number])
    # 填充数据
    # for col in data.columns:
    #     positive_median = data[col][data[col] > 0].median()
    #     data[col] = np.where(data[col] < 0, positive_median, data[col])
    # 5. 填充数据
    data["edu_status"]=data["edu_status"].fillna(5)
    data["edu_yr"]=data["edu_yr"].fillna(-2)

    data["property_other"]=data["property_other"].map(lambda x:0 if pd.isnull(x)  else 1)
    data["hukou_loc"]=data["hukou_loc"].fillna(1)
    # data["hukou_loc"]=data["hukou_loc"].fillna(5)

    data["social_neighbor"]=data["social_neighbor"].fillna(8)
    data["social_friend"]=data["social_friend"].fillna(8)

    data["work_status"]=data["work_status"].fillna(0)
    data["work_yr"]=data["work_yr"].fillna(0)
    data["work_type"]=data["work_type"].fillna(0)
    data["work_manage"]=data["work_manage"].fillna(0)
    # family_income_median = data['family_income'][data['family_income'] > 0].median()
    # data["family_income"]=data["family_income"].fillna(family_income_median)
    data["family_income"] = data["family_income"].fillna(-2)
    data["invest_other"]=data["invest_other"].map(lambda x:0 if pd.isnull(x)  else 1)
    data["minor_child"]=data["minor_child"].fillna(0)
    # 存疑
    data["marital_1st"]=data["marital_1st"].fillna(0)
    data["s_birth"]=data["s_birth"].fillna(0)
    data["marital_now"]=data["marital_now"].fillna(0)

    data["s_edu"]=data["s_edu"].fillna(0)
    data["s_political"]=data["s_political"].fillna(0)
    data["s_hukou"]=data["s_hukou"].fillna(0)
    data["s_income"]=data["s_income"].fillna(0)
    data["s_work_exper"]=data["s_work_exper"].fillna(0)
    data["s_work_status"]=data["s_work_status"].fillna(0)
    data["s_work_type"]=data["s_work_type"].fillna(0)
    # 去掉id
    data=data.drop(["id"], axis=1)
    data = data.drop(["survey_age", "property_other", "invest_other", "join_party"], axis=1)


    data = data.select_dtypes(include=[np.number])
    correlation_matrix = data.corr()
    # 特征筛选
    features = correlation_matrix['happiness'][abs(correlation_matrix['happiness']) > filtered_features_thres].index 
    features = features.values.tolist()
    # features.remove('happiness')
    print(len(features))
    xy_train = data[:train.shape[0]][features]
    xy_test = data[train.shape[0]:][features]
    xy_train.to_csv(processed_data_dir/f"xy_train_filtered_feature_{filtered_features_thres}.csv",index=False)
    xy_test.to_csv(processed_data_dir/f"xy_test_filtered_feature_{filtered_features_thres}.csv",index=False)

    # 不筛选特征
    if not for_change_filtered_features_thres:
        xy_train = data[:train.shape[0]]
        xy_test = data[train.shape[0]:]
        xy_train.to_csv(processed_data_dir/f"xy_train.csv",index=False)
        xy_test.to_csv(processed_data_dir/f"xy_test.csv",index=False)

    # 过采样
    # 1. SMOTE 
    # 1.1 过滤特征
    smote = SMOTE(sampling_strategy={1: 1000, 2: 1000})
    y_train = xy_train['happiness']
    x_train = xy_train.drop(columns=['happiness'], axis=1)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
    x_train_resampled['happiness'] = y_train_resampled
    x_train_resampled.to_csv(processed_data_dir/f"xy_train_filtered_feature_{filtered_features_thres}_smote_oversample.csv",index=False)
    # 1.2 不过滤特征
    if not for_change_filtered_features_thres:
        y_train = data[:train.shape[0]]['happiness']
        x_train = data[:train.shape[0]].drop(columns=['happiness'], axis=1)
        x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
        x_train_resampled['happiness'] = y_train_resampled
        x_train_resampled.to_csv(processed_data_dir/f"xy_train_smote_oversample.csv",index=False)
    # 2. ADASYN
    # 2.1 过滤特征
    adasyn = ADASYN(sampling_strategy='minority', random_state=42, n_neighbors=5)
    y_train = xy_train['happiness']
    x_train = xy_train.drop(columns=['happiness'], axis=1)
    x_train_resampled, y_train_resampled = adasyn.fit_resample(x_train, y_train)
    x_train_resampled['happiness'] = y_train_resampled
    x_train_resampled.to_csv(processed_data_dir/f"xy_train_filtered_feature_{filtered_features_thres}_adasyn_oversample.csv",index=False)
    # 2.2 不过滤特征
    if not for_change_filtered_features_thres:
        y_train = data[:train.shape[0]]['happiness']
        x_train = data[:train.shape[0]].drop(columns=['happiness'], axis=1)
        x_train_resampled, y_train_resampled = adasyn.fit_resample(x_train, y_train)
        x_train_resampled['happiness'] = y_train_resampled
        x_train_resampled.to_csv(processed_data_dir/f"xy_train_adasyn_oversample.csv",index=False)


if __name__ == '__main__':
    src_dir = Path(__file__).parent
    main(
        drop_unlabled=False,
        filtered_features_thres=0.005,
        raw_data_dir = src_dir/'dataset/raw_data',
        processed_data_dir = src_dir/'dataset/processed_data',
        for_change_filtered_features_thres=False
    )





