'''
对模型的各个自变量做探索性数据分析
自变量:IPCI(拥堵指数), Brent_Oil_Avg(原油价格), CCFI_weekly_avg(运价), weekly_avg_policy_index(政策因素), weighted_oil_vix(加权后的原油价格和股市波动率指数)
分析内容:时间序列图,季节性图(展现年变化模式),散点图矩阵,滞后图,ACF自相关系数,差分处理
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

#绘制各个变量的时间序列图(Time plots) 
data = pd.read_csv('weekly_with_CCFI_updated.csv')
data['week'] = pd.to_datetime(data['week'])
start_date = '2020-01-06'
end_date = '2024-12-30'
data = data[(data['week'] >= start_date) & (data['week'] <= end_date)] #设定数据范围为2020-01-06到2024-12-30
columns_to_plot = ['IPCI', 'Brent_Oil_Avg', 'CCFI_weekly_avg', 'weekly_avg_policy_index', 'weighted_oil_vix']
for column in columns_to_plot:
    plt.figure(figsize=(12, 6))
    plt.plot(data['week'], data[column], label=column)
    plt.title(f'Time Plot of {column}')
    plt.xlabel('Week')
    plt.ylabel(column)
    plt.legend()
    plt.grid(True)
    plt.show()

#绘制展示年变化模式的季节性图(seasonal plots)
data['year'] = data['week'].dt.year
data['week_of_year'] = data['week'].dt.isocalendar().week #提取周数
columns_to_analyze = ['IPCI', 'Brent_Oil_Avg', 'CCFI_weekly_avg', 'weekly_avg_policy_index', 'weighted_oil_vix']
#按周绘制每年的季节性折线图
for column in columns_to_analyze:
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=data, x='week_of_year', y=column, hue='year', palette='viridis')
    plt.title(f'Seasonal Line Plot of {column} by Week of Year')
    plt.xlabel('Week of Year')
    plt.ylabel(column)
    plt.legend(title='Year')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(1, 53)  # 一年有53周
    plt.show()

#绘制散点图矩阵(scatterplot matrices)
columns_to_analyze = ['IPCI', 'Brent_Oil_Avg', 'CCFI_weekly_avg', 'weekly_avg_policy_index', 'weighted_oil_vix']
data_subset = data[columns_to_analyze]
g=sns.pairplot(data_subset) 
g.figure.set_size_inches(14, 12)  # 直接对 PairGrid 设置尺寸
g.figure.suptitle('Scatterplot Matrices', fontsize=16, y=0.98)
g.figure.subplots_adjust(top=0.90, bottom=0.10)  
plt.show()


corr_matrix = data_subset.corr()
print("Correlation Coefficient Matrix:")
print(corr_matrix)#矩阵系数

plt.figure(figsize=(14, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title(f'Correlation Matrix')#热力图
plt.show()

#绘制滞后图(lag plots)
variables = ['IPCI', 'Brent_Oil_Avg', 'CCFI_weekly_avg', 'weekly_avg_policy_index', 'weighted_oil_vix']
for variable in variables:
    data_series = data[variable]
    data_series.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_series.dropna(inplace=True) #确保没有缺失值或无限值
    lags = [4, 8, 12, 16, 20]#规定滞后阶数
    plt.figure(figsize=(15, 5))
    for i, k in enumerate(lags, 1):
        # 计算当前值和滞后值
        y_t = data_series[k:].reset_index(drop=True)
        y_t_k = data_series[:-k].reset_index(drop=True)
        # 进行Pearson相关检验
        correlation, p_value = stats.pearsonr(y_t_k, y_t)
        plt.subplot(1, len(lags), i)
        sns.regplot(x=y_t_k, y=y_t, ci=None, scatter_kws={"alpha":0.5})
        plt.title(f'Lag {k} (p={p_value:.3e})')
        plt.xlabel(f'{variable} (t-{k})')
        plt.ylabel(f'{variable} (t)')
        plt.grid(True)
        # 显示相关系数显著性
        print(f'Variable: {variable} -- Lag {k}: Pearson correlation={correlation:.3f}, p-value={p_value:.3e}')
    plt.suptitle(f'Lag Scatter Plots for {variable} with p-values', fontsize=16)
    plt.tight_layout()
    plt.show()

#绘制ACF(autocorrelation function)图像

variables = ['IPCI', 'Brent_Oil_Avg', 'CCFI_weekly_avg', 'weekly_avg_policy_index', 'weighted_oil_vix']
 
#绘制 ACF 图像
for variable in variables:
    data_series = data[variable]
    data_series.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_series.dropna(inplace=True)  # 确保没有缺失值或无限值
    
    plot_acf(data_series, lags=24)  # 设置最大滞后值为24
    plt.title(f'ACF for {variable} (Lags 1 to 24)')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.grid(True)
    plt.show()


