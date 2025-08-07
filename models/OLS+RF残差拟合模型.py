"""
此代码用于残差拟合模型，使用最小二乘捕捉回归值，再使用随机森林拟合残差
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
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


def create_features(df, lag=4):#添加滞后特征
    features = []
    df[f'CCFI_lag{lag}'] = df['CCFI_weekly_avg'].shift(lag)
    features.append(f'CCFI_lag{lag}')
    for var in ['weighted_teu', 'IPCI', 'weekly_avg_policy_index','weighted_oil_vix', 'avg_speed']:
        df[f'{var}_lag{lag}'] = df[var].shift(lag)
        features.append(f'{var}_lag{lag}')

    return df.dropna(), features


def train_and_evaluate(port_data):#训练模型
    results = {}
    all_predictions = []
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

        # 绘制模型结果
        plt.figure(figsize=(12, 6))
        plt.plot(test_df['week'], y_test, label='Actual CCFI', color='black', linewidth=2)
        plt.plot(test_df['week'], final_pred, label='Hybrid (OLS+RF) Prediction', linestyle='--', color='blue',
                 linewidth=2)
        plt.title('CCFI Prediction - Hybrid Model', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('CCFI Index', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        # 模型评估
        mae_hybrid = mean_absolute_error(y_test, final_pred)
        rmse_hybrid = np.sqrt(mean_squared_error(y_test, final_pred))

        results[port] = {
            'MAE_OLS+RF': mae_hybrid, 'RMSE_OLS+RF': rmse_hybrid,
            'Features': feature_cols,
            'Best_RF_Params': rf_random.best_params_
        }
        pred_df = test_df[['week', 'CCFI_weekly_avg']].copy()
        pred_df['Prediction'] = final_pred
        all_predictions.append(pred_df)

        # 合并并保存所有预测结果
    if all_predictions:
        final_predictions = pd.concat(all_predictions)
        if not os.path.exists('results'):
            os.makedirs('results')
        final_predictions.to_csv('results/残差_predictions.csv', index=False)
        print("预测结果已保存到 残差_predictions.csv")
    return results


# 执行
print("开始加载和合并数据...")
data = load_and_merge_data()
print("数据加载完成，开始训练模型...")
results = train_and_evaluate(data)

# 输出结果
print("\n=== 模型评估结果 ===")
for port, metrics in results.items():
    print(f"\n混合模型(OLS+RF):")
    print(f"- MAE: {metrics['MAE_OLS+RF']:.2f}")
    print(f"- RMSE: {metrics['RMSE_OLS+RF']:.2f}")
    print("\n最佳RF参数:")
    print(metrics['Best_RF_Params'])
    print("\n使用的特征变量:")
    for i, feat in enumerate(metrics['Features'], 1):
        print(f"{i}. {feat}")
