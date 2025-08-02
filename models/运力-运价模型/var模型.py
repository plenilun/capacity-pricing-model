'''
使用VAR模型,以IPCI(拥堵指数),Brent_Oil_Avg(石油价格),预测CCFI
具体步骤:数据选择与处理(OLS回归分析,平稳性检验,差分处理);建立VAR模型;
评估模型(格兰杰因果性检验,模型稳定性检验,残差自相关检验,脉冲-响应分析);使用模型预测数据
'''

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

df = pd.read_csv("weekly_with_CCFI_updated.csv")
print(df.head)

df['week'] = pd.to_datetime(df['week'])
df.set_index('week', inplace=True)
df = df[['IPCI', 'Brent_Oil_Avg','CCFI_weekly_avg']].dropna()
df = df.iloc[:211, :] #使用2020-01-06至2024-01-15的数据
print(df.head)

#OLS 回归分析

Y = df["CCFI_weekly_avg"]

# Model1：Brent_Oil_Avg=>CCFI
X1 = sm.add_constant(df[["Brent_Oil_Avg"]])
model1 = sm.OLS(Y, X1).fit()

# Model2: IPCI=>CCFI
X2 = sm.add_constant(df[["IPCI"]])
model2 = sm.OLS(Y, X2).fit()

# Model3: IPCI,Brent_Oil_Avg=>CCFI
X3 = sm.add_constant(df[["IPCI", "Brent_Oil_Avg"]])
model3 = sm.OLS(Y, X3).fit()

# Output
print("=== 模型1:仅 Brent_Oil_Avg ===")
print("Model1 - R²:", model1.rsquared)

print("=== 模型2:仅 IPCI ===")
print("Model2 - R²:", model2.rsquared)

print("\n=== 模型3:IPCI + Brent_Oil_Avg ===")
print("Model3 - R²:", model3.rsquared)
print(f'模型1的拟合优度为{model1.rsquared:.3f},模型2的拟合优度是{model2.rsquared:.3f},模型3的拟合优度为{model3.rsquared:.3f};')

if(model1.rsquared < model3.rsquared and model2.rsquared < model3.rsquared):
    print('模型3相对于模型1和2拟合优度有所提升,说明拥堵指数和石油价格中所包含的信息均没有被另一因素所覆盖')
else:
    print('只需选用一个变量')

#运行结果表明，拥堵指数和石油价格中所包含的信息均没有被另一因素所覆盖，即两个变量均要使用
    
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

#差分处理
df_diff = df.diff().dropna()
for col in df_diff.columns:
    print(f"\nADF Test for {col} after diffrence process:")
    adf_test(df_diff[col])#一阶差分后三个序列均平稳

#建立VAR模型
df_diff.columns = ['dIPCI', 'dBrent_Oil', 'dCCFI']
model = VAR(df_diff)
lag_order_selection = model.select_order(maxlags=8)
print(lag_order_selection.summary()) 
optimal_lag = lag_order_selection.selected_orders['aic']  #选择 AIC（lag=4时,AIC值最小，为13.09） 
model_fitted = model.fit(optimal_lag)
print(model_fitted.summary())


#格兰杰因果性检验(Granger Causality Test)
#dIPCI=>dCCFI
granger_result_1 = model_fitted.test_causality(causing='dCCFI', caused='dIPCI', kind='f')
print(granger_result_1.summary())
#dOil=>dCCFI
granger_result_2 = model_fitted.test_causality(causing='dCCFI', caused='dBrent_Oil', kind='f')
print(granger_result_2.summary())
#dIPCI,dOil=>dCCFI
granger_result_3 = model_fitted.test_causality(causing='dCCFI', caused=['dIPCI','dBrent_Oil'], kind='f')
print(granger_result_3.summary()) 
if(granger_result_3.pvalue<0.05):
    print(f'p-value为{granger_result_3.pvalue:.3f},说明IPCI的变化和Oil的变化对CCFI的变化有预测能力')
else:
    print(f'p-value为{granger_result_3.pvalue:.3f},说明IPCI的变化和Oil的变化对CCFI的变化没有预测能力')

#运行结果为p-value=0.044,说明IPCI的变化和Oil的变化对CCFI的变化有预测能力


#模型稳定性检验
is_stable = model_fitted.is_stable(verbose=True)
print("模型稳定性：", "稳定" if is_stable else "不稳定") #模型稳定

#残差自相关检验
residuals = model_fitted.resid
for col in residuals.columns:
    print(f"\n残差自相关检验:{col}")
    lb_test = acorr_ljungbox(residuals[col], lags=[10], return_df=True)
    print(lb_test) 
    if(lb_test['lb_pvalue'].iloc[0]>0.05):
        print('残差为白噪声')
    else:
        print('残差不为白噪声')
#运行结果p-value均大于0.05，残差均为白噪声

#IRF(脉冲-响应分析)
irf = model_fitted.irf(10)
irf.plot(orth=True, impulse='dIPCI', response='dCCFI')
irf.plot(orth=True, impulse='dBrent_Oil', response='dCCFI')
plt.show()

#预测后20天数据
forecast_diff = model_fitted.forecast(y=df_diff.values, steps=20)

last_row = df.iloc[-1].values
forecast_values = [last_row + forecast_diff[0]]

# 每一步累加前一预测值
for i in range(1, 20):
    forecast_values.append(forecast_values[-1] + forecast_diff[i])

forecast_array = np.array(forecast_values)
forecast_df = pd.DataFrame(forecast_array, columns=[ "IPCI", "Brent_Oil_Avg","CCFI"])

# 添加日期索引，从2024-01-22开始，每周递增
forecast_dates = pd.date_range(start="2024-01-22", periods=20, freq="W-MON")
forecast_df.index = forecast_dates
forecast_df.index.name = "week"

print(forecast_df.head)

#可视化预测结果
df_original = pd.read_csv("weekly_with_CCFI_updated.csv")
df_original['week'] = pd.to_datetime(df_original['week'])
plt.figure(figsize=(12, 6))
plt.xlim(pd.to_datetime('2024-01-01'), pd.to_datetime('2024-06-24'))
plt.plot(df_original['week'], df_original['CCFI_weekly_avg'], label='CCFI Weekly Avg', color='blue')
plt.plot(forecast_df.index, forecast_df['CCFI'], label='CCFI predicted', color='green')
plt.xlabel('week')
plt.ylabel('CCFI')
plt.title('Weekly CCFI Avg Over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
