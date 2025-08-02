'''
对变量weighted_teu做数据分析
分析内容:时间序列图,季节性图(展现年变化模式),滞后图,ACF自相关系数,差分处理,散点图矩阵
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

df = pd.read_csv('weekly_weighted_teu.csv')
df['date'] = pd.to_datetime(df['date'])
start_date = '2020-01-06'
end_date = '2024-12-30'
df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]#确保数据范围为2020-01-06到2024-12-30
print(df)

#时间序列图(Time plots)
def plot_time_series(data, title):
    plt.figure(figsize=(12, 6))
    plt.plot(data['date'], data['weighted_teu'])
    plt.title(title)
    plt.xticks(rotation=45,fontsize=8)
    plt.gca().set_xticks(data['date'][::24])
    plt.xlabel('Week')
    plt.ylabel('weighted_teu')
    plt.grid(True)
    plt.show()

plot_time_series(df, 'Time Plot for weighted teu at both ports')

#展现年变化模式的季节性图(Seasonal plots)
import seaborn as sns
def plot_seasonal_plots(data, title):
    data['end_date'] = data['date']
    data = data.sort_values('end_date')
    data['year'] = data['end_date'].dt.year
    data['week_of_year'] = data['end_date'].dt.isocalendar().week# 提取年份和周数

    plt.figure(figsize=(14, 8))
    sns.lineplot(data=data, x='week_of_year', y='weighted_teu', hue='year', palette='viridis')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.xlabel('Week of Year')
    plt.ylabel('weighted_teu')
    plt.legend(title='Year')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(1, 53)  # 一年有53周
    plt.show()
plot_seasonal_plots(df, 'Seasoanl Plot for weighted teu at both ports by week of Year')

#滞后图(lag plots)
def plot_lag_plots(data, name):
    data_series = data['weighted_teu']
    data_series.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_series.dropna(inplace=True) #确保没有缺失值或无限值
    lags = [4, 8, 12, 16, 20]
    plt.figure(figsize=(15, 5))
    for i, k in enumerate(lags, 1):
        y_t = data_series[k:].reset_index(drop=True)
        y_t_k = data_series[:-k].reset_index(drop=True)# 计算当前值和滞后值
        correlation, p_value = stats.pearsonr(y_t_k, y_t)# 进行Pearson相关检验
        plt.subplot(1, len(lags), i)
        sns.regplot(x=y_t_k, y=y_t, ci=None, scatter_kws={"alpha":0.5})
        plt.title(f'Lag {k} (p={p_value:.3e})')
        plt.xlabel(f'teu (t-{k})')
        plt.ylabel(f'teu (t)')
        plt.grid(True)
        print(f'Variable: {name} -- Lag {k}: Pearson correlation={correlation:.3f}, p-value={p_value:.3e}')# 显示相关系数显著性
    plt.suptitle(f'Lag Scatter Plots for {name} with p-values', fontsize=16)
    plt.tight_layout()
    plt.show()

plot_lag_plots(df, 'Both ports\' weighted_teu')

#ACF自相关系数
from statsmodels.graphics.tsaplots import plot_acf
def plot_ACF(data,name):
    data_series = data['weighted_teu']
    data_series.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_series.dropna(inplace=True)  # 确保没有缺失值或无限值

    plot_acf(data_series, lags=32)  # 设置最大滞后值为32
    plt.title(f'ACF for {name} (Lags 1 to 32)')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.grid(True)
    plt.show()
plot_ACF(df, 'Both ports\' weighted_teu')
df['weighted_teu'].replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=['weighted_teu'], inplace=True)

#差分处理
df['weighted_teu_diff'] = df['weighted_teu'].diff()
df.dropna(subset=['weighted_teu_diff'], inplace=True)# 删除由于差分导致的第一个NaN值
print(df.head())  

plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['weighted_teu_diff'], label='Differenced Weighted TEU', color='green')
plt.xlabel('Week')
plt.ylabel('Weighted TEU Diff')
plt.title('Differenced Weighted TEU Over Time')
plt.grid(True)
plt.legend()
plt.show()


#散点图矩阵(scatterplot matrices)
weekly_with_CCFI_updated = pd.read_csv('weekly_with_CCFI_updated.csv')
weekly_with_CCFI_updated['date'] = pd.to_datetime(weekly_with_CCFI_updated['week']).dt.date
weekly_with_CCFI_updated['date'] = pd.to_datetime(weekly_with_CCFI_updated['date'], format='%Y-%m-%d')
combined_data = weekly_with_CCFI_updated.merge(df, on='date', how='inner')#合并数据集
print(combined_data.head())

columns_to_analyze = ['IPCI', 'Brent_Oil_Avg', 'CCFI_weekly_avg', 'weekly_avg_policy_index', 'weighted_oil_vix']
teu_to_analyse=['weighted_teu']
data_subset = combined_data[columns_to_analyze+teu_to_analyse]
g=sns.pairplot(data_subset)
g.figure.set_size_inches(14, 12) 
g.figure.suptitle('Scatterplot Matrices', fontsize=16, y=0.98)
g.figure.subplots_adjust(top=0.90, bottom=0.10)  
plt.show()#矩阵图

corr_matrix = data_subset.corr()
print("Correlation Coefficient Matrix:")
print(corr_matrix)
plt.figure(figsize=(14, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.yticks(fontsize=8)
plt.xticks(rotation=45, fontsize=6)
plt.title('Correlation Matrix for weighted_teu')
plt.show()#热力图
