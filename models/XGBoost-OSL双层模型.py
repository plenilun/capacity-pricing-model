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
import os

def load_and_merge_data():
    ccfi_df = pd.read_csv('data/processed data/weekly_with_CCFI_full.csv',
                          usecols=['week', 'CCFI_weekly_avg', 'weighted_teu',
                                   'avg_speed', 'weighted_oil_vix',
                                   'weekly_avg_policy_index', 'IPCI'])

    # 转换日期格式
    ccfi_df['week'] = pd.to_datetime(ccfi_df['week'])
    # 确保没有缺失值
    ccfi_df = ccfi_df.dropna()

    return {'Combined': ccfi_df}

# 添加滞后特征
def create_features(df, lag=4):#添加滞后特征
    features = []
    df[f'CCFI_lag{lag}'] = df['CCFI_weekly_avg'].shift(lag)
    features.append(f'CCFI_lag{lag}')
    for var in ['weighted_teu', 'IPCI', 'weekly_avg_policy_index','weighted_oil_vix', 'avg_speed']:
        df[f'{var}_lag{lag}'] = df[var].shift(lag)
        features.append(f'{var}_lag{lag}')

    return df.dropna(), features

# 模型部分
def train_stacking_model(port_data):
    results = {}
    all_predictions = []

    for port, df in port_data.items():
        df, features = create_features(df)
        train_df = df[(df['week'] >= '2020-01-01') & (df['week'] <= '2022-06-30')]
        test_df = df[(df['week'] > '2022-06-30') & (df['week'] <= '2022-12-31')]

        X_train, y_train = train_df[features], train_df['CCFI_weekly_avg']
        X_test, y_test = test_df[features], test_df['CCFI_weekly_avg']

        # 第一层模型（XGBoost）
        base_model = ('xgb', xgb.XGBRegressor(n_estimators=600, learning_rate=0.05,
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

        pred_df = test_df[['week', 'CCFI_weekly_avg']].copy()
        pred_df['Prediction'] = pred
        all_predictions.append(pred_df)

    # 合并并保存所有预测结果
    if all_predictions:
        final_predictions = pd.concat(all_predictions)
        if not os.path.exists('results'):
            os.makedirs('results')
        final_predictions.to_csv('stacking_predictions.csv', index=False)
        print("预测结果已保存到 results/stacking_predictions.csv")

    return results

# 执行模型
data = load_and_merge_data()
results = train_stacking_model(data)
print(results)
