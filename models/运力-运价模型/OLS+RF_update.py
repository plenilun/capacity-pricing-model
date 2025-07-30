"""
此代码用于残差拟合模型，使用最小二乘捕捉回归值，再使用随机森林拟合残差
所有特征使用四周滞后
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import os

def load_and_merge_data():#加载数据
    teu_df = pd.read_csv('weekly_weighted_teu.csv', usecols=['date', 'weighted_teu'])
    teu_df['date'] = pd.to_datetime(teu_df['date'])
    speed_df = pd.read_csv('weekly_speed.csv', usecols=['date', 'avg_speed'])
    speed_df['date'] = pd.to_datetime(speed_df['date']).dt.tz_localize(None)
    oil_ccfi_df = pd.read_csv('weekly_with_CCFI.csv',
                              usecols=['week', 'weighted_oil_vix', 'CCFI_weekly_avg',
                                       'weekly_avg_policy_index', 'IPCI'])
    oil_ccfi_df['week'] = pd.to_datetime(oil_ccfi_df['week'])
    #转换政治指数为平方
    oil_ccfi_df['weekly_avg_policy_index'] = oil_ccfi_df['weekly_avg_policy_index'] ** 2

    # 合并数据
    merged = pd.merge(
        oil_ccfi_df, teu_df, left_on='week', right_on='date', how='left'
    )
    merged = pd.merge(
        merged, speed_df, left_on='week', right_on='date', how='left',
        suffixes=('', '_speed')
    ).drop(columns=['date', 'date_speed']).dropna()

    return {'Combined': merged}

def create_features(df, lag=4):#添加滞后特征
    features = []
    df[f'CCFI_lag{lag}'] = df['CCFI_weekly_avg'].shift(lag)
    features.append(f'CCFI_lag{lag}')
    for var in ['weighted_teu', 'IPCI', 'weekly_avg_policy_index', 'weighted_oil_vix', 'avg_speed']:
        df[f'{var}_lag{lag}'] = df[var].shift(lag)
        features.append(f'{var}_lag{lag}')

    # 周数特征
    df['week_of_year'] = df['week'].dt.isocalendar().week
    features.append('week_of_year')

    return df.dropna(), features

def train_and_evaluate(port_data):#训练模型
    results = {}
    predictions = {}  # 用于存储预测结果

    for port, df in port_data.items():
        processed_df, feature_cols = create_features(df, lag=4)
        print("使用的特征变量：", feature_cols)

        # 划分训练集和测试集
        train_df = processed_df[(processed_df['week'] >= '2020-01-01') & (processed_df['week'] <= '2022-06-30')]
        test_df = processed_df[(processed_df['week'] > '2022-06-30') & (processed_df['week'] <= '2022-12-31')]

        X_train = train_df[feature_cols]
        y_train = train_df['CCFI_weekly_avg']
        X_test = test_df[feature_cols]
        y_test = test_df['CCFI_weekly_avg']

        # OLS模型
        ols = LinearRegression()
        ols.fit(X_train, y_train)
        ols_pred = ols.predict(X_test)
        #计算残差
        residuals_train = y_train - ols.predict(X_train)

        # 使用参数搜索确定随机森林的最佳参数
        param_dist = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        rf = RandomForestRegressor(random_state=42)
        rf_random = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=10,
            cv=5,
            scoring='neg_mean_squared_error',
            random_state=42,
            n_jobs=-1
        )

        print("\n开始RF参数搜索...")
        rf_random.fit(X_train, residuals_train)
        print(f"RF最佳参数: {rf_random.best_params_}")

        best_rf = rf_random.best_estimator_
        final_pred = ols_pred + best_rf.predict(X_test)

        # 保存预测结果
        pred_df = test_df[['week', 'CCFI_weekly_avg']].copy()
        pred_df['OLS_Prediction'] = ols_pred
        pred_df['RF_Residual_Prediction'] = best_rf.predict(X_test)
        pred_df['Hybrid_Prediction'] = final_pred
        predictions[port] = pred_df

        # 可视化结果
        plt.figure(figsize=(12, 6))
        plt.plot(test_df['week'], y_test, label='Actual CCFI', color='black', linewidth=2)
        plt.plot(test_df['week'], final_pred, label='Hybrid (OLS+RF) Prediction', linestyle='--', color='blue',
                 linewidth=2)
        plt.title('CCFI Prediction - Hybrid Model (4-week lag features)', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('CCFI Index', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        # 模型评估
        mae_hybrid = mean_absolute_error(y_test, final_pred)
        rmse_hybrid = np.sqrt(mean_squared_error(y_test, final_pred))
        results[port] = {
            'MAE_OLS+RF': mae_hybrid,
            'RMSE_OLS+RF': rmse_hybrid,
            'Features': feature_cols,
            'Best_RF_Params': rf_random.best_params_
        }
    return results, predictions

# 执行
print("开始加载和合并数据...")
data = load_and_merge_data()
print("数据加载完成，开始训练模型...")
results, predictions = train_and_evaluate(data)
for port, metrics in results.items():
    print(f"\n混合模型(OLS+RF):")
    print(f"- MAE: {metrics['MAE_OLS+RF']:.2f}")
    print(f"- RMSE: {metrics['RMSE_OLS+RF']:.2f}")
    print("\n最佳RF参数:")
    print(metrics['Best_RF_Params'])
    print("\n使用的特征变量:")
    for i, feat in enumerate(metrics['Features'], 1):
        print(f"{i}. {feat}")

# 保存预测结果
if not os.path.exists('results'):
    os.makedirs('results')
for port, pred_df in predictions.items():
    pred_df.to_csv(f'results/{port}_predictions_4weeklag.csv', index=False)
    print(f"\n{port}的预测结果已保存到 results/{port}_predictions_4weeklag.csv")
