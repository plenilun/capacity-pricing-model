import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import randint
import matplotlib.pyplot as plt
import seaborn as sns

teu_data = pd.read_csv('weekly_teu_stats.csv', parse_dates=['date'])
ccfi_data = pd.read_csv('weekly_with_CCFI.csv', parse_dates=['week'])
vix_data = pd.read_csv('VIX.csv', parse_dates=['date'])


port_teu = teu_data[teu_data['port_name'] == 'USLSA'].copy()


vix_weekly = vix_data.resample('W-Mon', on='date').mean().reset_index()


merged_data = pd.merge(
    ccfi_data,
    port_teu[['date', 'weekly_teu_sum']],
    left_on='week',
    right_on='date',
    how='left'
)
merged_data = pd.merge(
    merged_data,
    vix_weekly,
    left_on='week',
    right_on='date',
    how='left'
)


merged_data = merged_data[
    (merged_data['week'] >= '2020-01-01') &
    (merged_data['week'] <= '2022-12-31')
    ].dropna().drop(columns=['date_x', 'date_y'])


features = [
    'weekly_teu_sum',
    'Brent_Oil_Avg',
    'VIX',
    'IPCI'
]

# 添加滞后特征
for lag in [1,2,4]:  # 4周滞后
    merged_data[f'CCFI_lag_{lag}'] = merged_data['CCFI_weekly_avg'].shift(lag)
    features.append(f'CCFI_lag_{lag}')

# 添加移动平均特征
for window in [4, 12]:  # 4周和12周移动平均
    merged_data[f'CCFI_ma_{window}'] = merged_data['CCFI_weekly_avg'].rolling(window).mean()
    features.append(f'CCFI_ma_{window}')

merged_data = merged_data.dropna()


# Min-Max归一化
scaler = MinMaxScaler()
X = merged_data[features]
y = merged_data['CCFI_weekly_avg']
X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=features)

#训练模型部分

#时间序列分割
test_size = int(len(X) * 0.2)
X_train, X_test = X_normalized.iloc[:-test_size], X_normalized.iloc[-test_size:]
y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

# 参数搜索空间
param_dist = {
    'criterion': ['squared_error', 'absolute_error'],

    'max_depth': randint(1, 40),
    'min_samples_split': randint(2, 70),
    'min_samples_leaf': randint(1, 30),
    'max_features': randint(1, len(features) + 1),
    'max_leaf_nodes': randint(10, 100),
    'min_weight_fraction_leaf': [0, 0.1, 0.2]
}

# 时间序列交叉验证
tscv = TimeSeriesSplit(n_splits=5)

# 随机搜索得到最佳参数
search = RandomizedSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=200,
    cv=tscv,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
search.fit(X_train, y_train)

best_dt = search.best_estimator_
print("最佳参数组合：", search.best_params_)



def evaluate(y_true, y_pred):
    metrics = {}
    metrics['MSE'] = mean_squared_error(y_true, y_pred)
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    metrics['MAE'] = mean_absolute_error(y_true, y_pred)
    metrics['MAPE'] = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-6)) * 100)
    metrics['SMAPE'] = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-6) * 100)

    metrics['NMSE'] = metrics['MSE'] / np.var(y_true)  # 论文中的Normalized MSE

    return {k: round(v, 4) for k, v in metrics.items()}



train_metrics = evaluate(y_train, best_dt.predict(X_train))
test_metrics = evaluate(y_test, best_dt.predict(X_test))

print("\n训练集评估结果：")
print(pd.DataFrame([train_metrics]))
print("\n测试集评估结果：")
print(pd.DataFrame([test_metrics]))



plt.figure(figsize=(12, 6))


plt.subplot(1, 2, 1)
plt.plot(y_test.values, label='Actual', linewidth=2)
plt.plot(best_dt.predict(X_test), label='Predicted', linestyle='--')
plt.title('CCFI Actual vs Predicted (Test Set)')
plt.xlabel('Week')
plt.ylabel('CCFI')
plt.legend()
plt.grid(True)


plt.subplot(1, 2, 2)
importances = best_dt.feature_importances_
indices = np.argsort(importances)[-10:]  # 取最重要的10个特征
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.show()

results = pd.DataFrame({
    'Week': merged_data['week'].iloc[-test_size:],
    'Actual': y_test,
    'Predicted': best_dt.predict(X_test)
})
results.to_csv('ccfi_prediction_results.csv', index=False)