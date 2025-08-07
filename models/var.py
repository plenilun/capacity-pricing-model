'''
使用VAR模型,假设IPCI(拥堵指数),Brent_Oil_Avg(石油价格),CCFI(运价)相互影响;在此基础上预测CCFI
具体步骤:数据选择与处理(平稳性检验,差分处理);建立VAR模型;使用滑动窗口的方法预测数据
评估模型(MAE,MSE的指标)
'''
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei'] 

# 加载数据并确保日期时间格式
df = pd.read_csv("weekly_with_CCFI_updated.csv")
df['week'] = pd.to_datetime(df['week'])
df.set_index('week', inplace=True)
 
# 选择需要的列并移除空值
df = df[['IPCI', 'Brent_Oil_Avg', 'CCFI_weekly_avg']].dropna()
 
# 确保数据范围正确
df = df.loc['2020-06-07':'2023-05-22']
print(df.head)
#平稳性检验(ADF)
def adf_test(series, significance_level=0.05):
    result = adfuller(series)
    p_value = result[1]
    if p_value < significance_level:
        print(f"P-Value = {p_value} => Reject null hypothesis (data is stationary)")
    else:
        print(f"P-Value = {p_value} => Fail to reject null hypothesis (data is not stationary)")
for column in ['IPCI', 'Brent_Oil_Avg',"CCFI_weekly_avg"]:
    print(f"\nADF Test for {column}:")
    adf_test(df[column]) #三个序列均不平稳，分别对其做一阶差分

# 进行一阶差分处理
df_diff = df.diff().dropna()
df_diff.columns = ['dIPCI', 'dBrent_Oil', 'dCCFI']

for column in ['dIPCI', 'dBrent_Oil',"dCCFI"]:
    print(f"\nADF Test for {column}:")
    adf_test(df_diff[column]) #三个序列均平稳


# 设置参数
window_size = 120 # 窗口大小：120周
lag_order = 4     # VAR模型滞后阶数

# 初始化存储预测结果的数据框
forecast_results = pd.DataFrame(index=df_diff.index[window_size:], 
                               columns=['dCCFI_true', 'dCCFI_pred'])
forecast_results['dCCFI_true'] = df_diff['dCCFI'][window_size:]

# 创建副本
rolling_data = df_diff.copy()

# 滑动窗口预测
for i in range(len(df_diff) - window_size):
    # 获取当前窗口的数据
    window_start = i
    window_end = i + window_size
    forecast_idx = window_end
    
    # 使用当前窗口的数据进行训练
    train_data = rolling_data.iloc[window_start:window_end]
    
    # 训练VAR模型
    model = VAR(train_data)
    fitted_model = model.fit(lag_order, verbose=False)
    
    # 预测下一个时间点
    forecast = fitted_model.forecast(train_data.values, steps=1)
    
    # 保存预测结果
    pred_date = df_diff.index[forecast_idx]
    forecast_results.loc[pred_date, 'dCCFI_pred'] = forecast[0][2]  # dCCFI是第三列
    
    # 更新滚动数据集，将预测值替换到下一个点
    if forecast_idx < len(rolling_data):
        rolling_data.iloc[forecast_idx] = forecast[0]
    
# 计算预测准确性指标
mae = np.mean(np.abs(forecast_results['dCCFI_true'] - forecast_results['dCCFI_pred']))
mse = np.mean((forecast_results['dCCFI_true'] - forecast_results['dCCFI_pred'])**2)
 
print(f"\ndCCFI预测准确性指标：")
print(f"平均绝对误差 (MAE): {mae:.4f}")
print(f"均方误差 (MSE): {mse:.4f}")

# 绘制预测结果
plt.figure(figsize=(12, 6))
plt.plot(forecast_results.index, forecast_results['dCCFI_true'], 
         color='blue', label='真实值')
plt.plot(forecast_results.index, forecast_results['dCCFI_pred'], 
         color='red', linestyle='--', label='预测值')
 
plt.title('dCCFI 滑动窗口VAR(4)预测结果', fontsize=14)
plt.xlabel('时间')
plt.ylabel('dCCFI (差分值)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
 
# 打印首尾几个预测结果作为示例
print("\n首5个预测结果：")
print(forecast_results.head())
print("\n尾5个预测结果：")
print(forecast_results.tail())
 
# 可视化预测误差
plt.figure(figsize=(12, 4))
plt.plot(forecast_results.index, 
         forecast_results['dCCFI_true'] - forecast_results['dCCFI_pred'],
         color='green')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.title('dCCFI 预测误差')
plt.xlabel('时间')
plt.ylabel('误差 (真实值 - 预测值)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 获取CCFI的初始值（训练数据的最后一个真实值）
ccfi_initial = df.loc[df_diff.index[window_size-1], 'CCFI_weekly_avg']
print(f"CCFI初始值（第{window_size}周）: {ccfi_initial:.2f}")

# 创建CCFI预测结果数据框
ccfi_results = pd.DataFrame(index=forecast_results.index)

# 计算CCFI预测值：初始值 + dCCFI的累积和
ccfi_results['CCFI_pred'] = ccfi_initial + forecast_results['dCCFI_pred'].cumsum()

# 获取对应的CCFI真实值
ccfi_results['CCFI_true'] = df.loc[forecast_results.index, 'CCFI_weekly_avg']

# 计算CCFI预测准确性指标
ccfi_mae = np.mean(np.abs(ccfi_results['CCFI_true'] - ccfi_results['CCFI_pred']))
ccfi_mse = np.mean((ccfi_results['CCFI_true'] - ccfi_results['CCFI_pred'])**2)

print(f"\nCCFI预测准确性指标：")
print(f"平均绝对误差 (MAE): {ccfi_mae:.4f}")
print(f"均方误差 (MSE): {ccfi_mse:.4f}")

# 绘制CCFI预测值 vs 真实值对比图
plt.figure(figsize=(12, 6))
plt.plot(ccfi_results.index, ccfi_results['CCFI_true'], 
         color='blue', linewidth=2, label='CCFI真实值')
plt.plot(ccfi_results.index, ccfi_results['CCFI_pred'], 
         color='red', linestyle='--', linewidth=2, label='CCFI预测值')

plt.title('CCFI滑动窗口VAR(4)预测结果', fontsize=14)
plt.xlabel('时间')
plt.ylabel('CCFI')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 显示预测结果样例
print("\n首5个CCFI预测结果：")
print(ccfi_results.head().round(4))
print("\n尾5个CCFI预测结果：")
print(ccfi_results.tail().round(4))

