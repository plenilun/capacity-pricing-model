"""
    此代码使用XGBOOST和最小二乘双层模型，
    使用第一层XGBOOST模型的结果作为新的特征，再使用OSL在这些特征的基础上进行训练
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# 加载数据
def load_and_merge_data():
    teu_df = pd.read_csv('weekly_weighted_teu.csv', usecols=['date', 'weighted_teu'])
    teu_df['date'] = pd.to_datetime(teu_df['date'])
    speed_df = pd.read_csv('weekly_speed.csv', usecols=['date', 'avg_speed'])
    speed_df['date'] = pd.to_datetime(speed_df['date']).dt.tz_localize(None)
    oil_ccfi_df = pd.read_csv('weekly_with_CCFI1.csv',
                              usecols=['week', 'weighted_oil_vix', 'CCFI_weekly_avg',
                                       'weekly_avg_policy_index', 'IPCI'])
    oil_ccfi_df['week'] = pd.to_datetime(oil_ccfi_df['week'])
    oil_ccfi_df['weekly_avg_policy_index'] = oil_ccfi_df['weekly_avg_policy_index'] ** 2

    merged = pd.merge(oil_ccfi_df, teu_df, left_on='week', right_on='date', how='left')
    merged = pd.merge(merged, speed_df, left_on='week', right_on='date', how='left',
                      suffixes=('', '_speed')).drop(columns=['date', 'date_speed']).dropna()
    return {'Combined': merged}

#添加滞后特征
def create_features(df, lag=4):
    features = []
    for l in range(4, lag + 1):
        df[f'CCFI_lag{l}'] = df['CCFI_weekly_avg'].shift(l)
        features.append(f'CCFI_lag{l}')
    for var in ['weighted_teu', 'IPCI', 'weekly_avg_policy_index', 'weighted_oil_vix', 'avg_speed']:
        df[f'{var}_lag{lag}'] = df[var].shift(lag)
        features.append(f'{var}_lag{lag}')
    df['week_of_year'] = df['week'].dt.isocalendar().week
    features.append('week_of_year')
    return df.dropna(), features

# 模型部分
def train_stacking_model(port_data):
    results = {}
    for port, df in port_data.items():
        df, features = create_features(df)
        train_df = df[(df['week'] >= '2020-01-01') & (df['week'] <= '2022-06-30')]
        test_df = df[(df['week'] > '2022-06-30') & (df['week'] <= '2022-12-31')]

        X_train, y_train = train_df[features], train_df['CCFI_weekly_avg']
        X_test, y_test = test_df[features], test_df['CCFI_weekly_avg']

        # 第一层模型（XGBoost）
        base_model = ('xgb', xgb.XGBRegressor(n_estimators=500, learning_rate=0.05,
                                              max_depth=5, subsample=0.8, colsample_bytree=0.8,
                                              random_state=42))

        # 第二层模型（线性回归）
        meta_model = LinearRegression()

        # 构建Stacking
        stacking = StackingRegressor(
            estimators=[base_model],
            final_estimator=meta_model,
            passthrough=True
        )

        stacking.fit(X_train, y_train)
        pred = stacking.predict(X_test)

        # 可视化
        plt.figure(figsize=(12, 6))
        plt.plot(test_df['week'], y_test, label='Actual', color='black')
        plt.plot(test_df['week'], pred, label='Stacking Prediction', linestyle='--', color='blue')
        plt.title('CCFI Prediction - Stacking Model')
        plt.xlabel('Date')
        plt.ylabel('CCFI')
        plt.legend()
        plt.show()

        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        results[port] = {'MAE_Stacking': mae, 'RMSE_Stacking': rmse}
    return results

#执行模型
data = load_and_merge_data()
results = train_stacking_model(data)
print(results)
