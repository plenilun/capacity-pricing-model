import pandas as pd
from scipy.optimize import minimize

df_ccfi = pd.read_csv('weekly_with_CCFI.csv', usecols=['week', 'Brent_Oil_Avg', 'CCFI_weekly_avg'])
df_vix = pd.read_csv('weekly_VIX.csv', usecols=['date', 'VIX'])

df_ccfi['date'] = pd.to_datetime(df_ccfi['week'])
df_vix['date'] = pd.to_datetime(df_vix['date'])

start_date = '2019-01-01'
end_date = '2023-12-31'

df_ccfi = df_ccfi[(df_ccfi['date'] >= start_date) & (df_ccfi['date'] <= end_date)]
df_vix = df_vix[(df_vix['date'] >= start_date) & (df_vix['date'] <= end_date)]
df_merged = df_ccfi.merge(df_vix, on='date', how='inner')
# weight_oil = 0.518 / (0.518 + 0.072)
# weight_vix = 0.072 / (0.518 + 0.072)
#
# df_merged['combined'] = weight_oil * df_merged['Brent_Oil_Avg'] + weight_vix * df_merged['VIX']
# corr_combined = df_merged['combined'].corr(df_merged['CCFI_weekly_avg'])
# print(f"加权组合与 CCFI 的相关性: {corr_combined:.3f}")
def objective(weights):
    combined = weights[0] * df_merged['Brent_Oil_Avg'] + weights[1] * df_merged['VIX']
    return -combined.corr(df_merged['CCFI_weekly_avg'])  # 最小化负相关 = 最大化正相关

initial_weights = [0.5, 0.5]
bounds = [(0, 1), (0, 1)]
constraints = {'type': 'eq', 'fun': lambda w: w[0] + w[1] - 1}
result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)
optimal_weights = result.x
df_merged['combined_optimal'] = optimal_weights[0] * df_merged['Brent_Oil_Avg'] + optimal_weights[1] * df_merged['VIX']

corr_optimal = df_merged['combined_optimal'].corr(df_merged['CCFI_weekly_avg'])
print(f"优化后的权重: {optimal_weights}")
print(f"优化后的相关性: {corr_optimal:.3f}")
output_df = df_merged[['date', 'combined_optimal']]
output_df.to_csv('weighted_combined_vix_oil.csv', index=False)