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
    
   
def main():
    preprocess_data_dir = Path(__file__).parent / 'preprocess_data'
    preprocess_data_dir.mkdir(parents=True, exist_ok=True)
    #导入数据
    train_abbr=pd.read_csv("raw_data/happiness_train_abbr.csv",encoding='ISO-8859-1')
    train=pd.read_csv("raw_data/happiness_train_complete.csv",encoding='ISO-8859-1')
    test_abbr=pd.read_csv("raw_data/happiness_test_abbr.csv",encoding='ISO-8859-1')
    test=pd.read_csv("raw_data/happiness_test_complete.csv",encoding='ISO-8859-1')
    test_sub=pd.read_csv("raw_data/happiness_submit.csv",encoding='ISO-8859-1')

    # 绘制缺失指标缺失比图

    y_train_ = train['happiness']
    # 绘制label分布

    # 根据label分布，观察到 -8值， 应该是缺失值，将其替换为中等幸福水平3
    y_train_ = y_train_.map(lambda x: 3 if x == -8 else x)
    # 让label从0开始
    y_train_ = y_train_.map(lambda x: x - 1)

    # 一起预处理
    data = pd.concat([train, test], axis=0, ignore_index=True)

    # 数据处理
    # 1. 处理时间特征
    data['survey_time'] = pd.to_datetime(data['survey_time'],format='%Y/%m/%d %H:%M')
    data["weekday"]=data["survey_time"].dt.weekday
    data["year"]=data["survey_time"].dt.year
    data["quarter"]=data["survey_time"].dt.quarter
    data["hour"]=data["survey_time"].dt.hour
    data["month"]=data["survey_time"].dt.month
    # 2. 数据分段
    # 2.1 一天的hour
    data["hour_cut"]=data["hour"].map(hour_cut)
    # 2.2 出生的年代
    data["birth_s"]=data["birth"].map(birth_split)
    # 2.3 收入水平
    data["income_cut"]=data["income"].map(income_cut)

    # 2. 做问卷时候的年龄
    data["survey_age"]=data["year"]-data["birth"]

    # 3. 去掉过多的缺失值
    data=data.drop(["edu_other"], axis=1)
    data=data.drop(["happiness"], axis=1)
    data=data.drop(["survey_time"], axis=1)
    

    # 4. 填充数据
    data["edu_status"]=data["edu_status"].fillna(5)
    data["edu_yr"]=data["edu_yr"].fillna(-2)
    data["property_other"]=data["property_other"].map(lambda x:0 if pd.isnull(x)  else 1)
    data["hukou_loc"]=data["hukou_loc"].fillna(1)
    data["social_neighbor"]=data["social_neighbor"].fillna(8)
    data["social_friend"]=data["social_friend"].fillna(8)
    data["work_status"]=data["work_status"].fillna(0)
    data["work_yr"]=data["work_yr"].fillna(0)
    data["work_type"]=data["work_type"].fillna(0)
    data["work_manage"]=data["work_manage"].fillna(0)
    data["family_income"]=data["family_income"].fillna(-2)
    data["invest_other"]=data["invest_other"].map(lambda x:0 if pd.isnull(x)  else 1)
    data["minor_child"]=data["minor_child"].fillna(0)
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


    X_train_ = data[:train.shape[0]]
    X_test_  = data[train.shape[0]:]

    # 将训练集和测试集保存为 CSV 文件
    X_train_.to_csv(preprocess_data_dir / "X_train.csv", index=False)
    X_test_.to_csv(preprocess_data_dir / "X_test.csv", index=False)
    y_train_.to_csv(preprocess_data_dir / "y_train.csv", index=False)


if __name__ == '__main__':
    main()


