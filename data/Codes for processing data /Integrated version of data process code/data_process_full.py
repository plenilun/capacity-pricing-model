"""
    本代码用于所有的数据预处理
    包括所有文件的读取，TEU，油价与股市波动率指数VIX的加权，以及所有特征的整合
    最后生成的weely_with_ccfi_full包含了所有的输入特征
"""
import pandas as pd
from scipy.optimize import minimize

#读取CCFI数据
ccfi = pd.read_csv("CCFI_美西航线.csv")
# 按指定格式解析日期
ccfi['时间'] = pd.to_datetime(ccfi['时间'], format='%Y.%m.%d', errors='coerce')
ccfi.rename(columns={'时间': 'week', 'CCFI': 'CCFI_weekly_avg'}, inplace=True)

#读取政策指数数据
policy = pd.read_csv("Weekly_Policy_Averages.csv")
policy['week_start_date'] = pd.to_datetime(policy['week_start_date'])
policy.rename(columns={'week_start_date': 'week'}, inplace=True)
policy['week'] = policy['week'] + pd.to_timedelta(4 - policy['week'].dt.weekday, unit='D')

#读取 PCI 数据
pci = pd.read_csv("PCI.csv")
pci['week'] = pd.to_datetime(pci['week'])
pci['week'] = pci['week'] + pd.to_timedelta(4 - pci['week'].dt.weekday, unit='D')

#处理到港数据（加权TEU）
arrivals = pd.read_csv("洛杉矶_长滩到港.csv")
def safe_date_parse(date_str):
    try:
        dt = pd.to_datetime(date_str, dayfirst=True, format='mixed')
        if dt.year < 2000 or dt.year > 2030:
            return pd.NaT
        return dt
    except:
        return pd.NaT
arrivals['start_postime'] = arrivals['start_postime'].apply(safe_date_parse)
#arrivals['teu'] = arrivals.groupby('shiptype')['teu'].transform(lambda x: x.fillna(x.mean()))
#arrivals['teu'] = arrivals['teu'].fillna(0)
#按照作业时间加权TEU
container_df = arrivals[arrivals['shiptype'] == '集装箱'].copy()
container_df['date'] = container_df['start_postime'].dt.date
container_df['daily_weight'] = container_df['berth_duration'] / 24
container_df['weighted_teu'] = container_df['teu'] * container_df['daily_weight']
daily_teu = container_df.groupby('date')['weighted_teu'].sum().reset_index()
daily_teu['date'] = pd.to_datetime(daily_teu['date'])
# 聚合成周度数据
weekly_teu = daily_teu.set_index('date').resample('W-FRI')['weighted_teu'].sum().reset_index()
weekly_teu.rename(columns={'date': 'week'}, inplace=True)

#处理油价（布伦特）数据
oil = pd.read_csv("布伦特原油价格.csv")
oil['时间'] = pd.to_datetime(oil['时间'])
oil.rename(columns={'时间': 'date', '期货结算价:布伦特原油(美元/桶)': 'Brent_Oil'}, inplace=True)
# 改为按周五结束的周聚合
oil['week'] = oil['date'] + pd.to_timedelta(4 - oil['date'].dt.weekday, unit='D')
oil_weekly = oil.groupby('week')['Brent_Oil'].mean().reset_index()
oil_weekly.rename(columns={'Brent_Oil': 'Brent_Oil_Avg'}, inplace=True)

# 处理 VIX 数据
vix = pd.read_csv("股市波动率指数VIX（收盘）.csv")
vix['时间'] = pd.to_datetime(vix['时间'])
vix.rename(columns={'时间': 'date', 'CBOE:波动率指数(VIX):收盘()': 'VIX'}, inplace=True)
# 按周聚合
vix['week'] = vix['date'] + pd.to_timedelta(4 - vix['date'].dt.weekday, unit='D')
vix_weekly = vix.groupby('week')['VIX'].mean().reset_index()

oil_vix = pd.merge(oil_weekly, vix_weekly, on='week', how='outer')

#处理航速数据（中国_洛杉矶_长滩）
speed_df = pd.read_csv('中国_洛杉矶_长滩.csv')

speed_df['leg_start_postime'] = speed_df['leg_start_postime'].apply(safe_date_parse)
speed_df['start_postime'] = speed_df['start_postime'].apply(safe_date_parse)

speed_df['voyage_days'] = (speed_df['start_postime'] - speed_df['leg_start_postime']).dt.total_seconds() / 86400
speed_df['speed_per_teu'] = speed_df['speed'] / speed_df['teu']
speed_df['speed_per_abs_days'] = speed_df['speed'] / abs(speed_df['voyage_days'])
speed_df['abs_days_per_speed'] = abs(speed_df['voyage_days']) / speed_df['speed']

# 保留周度平均速度
weekly_speed = speed_df.set_index('start_postime').resample('W-FRI').agg({
    'speed': 'mean'
}).rename(columns={'speed': 'avg_speed'}).reset_index()
weekly_speed.rename(columns={'start_postime': 'week'}, inplace=True)

# 合并所有特征并确保所有数据都按周五结束的周聚合
df_merged = ccfi.merge(policy, on='week', how='outer')
df_merged = df_merged.merge(weekly_teu, on='week', how='outer')
df_merged = df_merged.merge(oil_vix, on='week', how='outer')
df_merged = df_merged.merge(pci, on='week', how='outer')
df_merged['week'] = df_merged['week'].dt.tz_localize(None)
weekly_speed['week'] = weekly_speed['week'].dt.tz_localize(None)
df_merged = df_merged.merge(weekly_speed, on='week', how='outer')

# 优化油价和VIX权重
df_opt = df_merged[(df_merged['week'] >= '2020-01-01') & (df_merged['week'] <= '2022-12-31')].copy()
df_opt = df_opt.dropna(subset=['Brent_Oil_Avg', 'VIX', 'CCFI_weekly_avg'])

def objective(weights):#定义相关性函数
    combined = weights[0] * df_opt['Brent_Oil_Avg'] + weights[1] * df_opt['VIX']
    corr = combined.corr(df_opt['CCFI_weekly_avg'])
    return -corr
initial_weights = [0.5, 0.5]
bounds = [(0, 1), (0, 1)]
constraints = {'type': 'eq', 'fun': lambda w: w[0] + w[1] - 1}
result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints, method='SLSQP')#按照最大相关性进行加权
optimal_weights = result.x
print(f"优化后的权重: [{optimal_weights[0]:.6f}, {optimal_weights[1]:.6f}]")
df_merged['weighted_oil_vix'] = optimal_weights[0] * df_merged['Brent_Oil_Avg'] + optimal_weights[1] * df_merged['VIX']

#筛选时间范围：2020-01-01 ~ 2025-06-30
df_final = df_merged[(df_merged['week'] >= '2020-01-01') & (df_merged['week'] <= '2025-06-30')]

# 保留最终结果
final_cols = ['week', 'CCFI_weekly_avg', 'weekly_avg_policy_index', 'weighted_teu',
              'weighted_oil_vix', 'IPCI', 'avg_speed']
final_df = df_final[final_cols]
final_df = final_df.sort_values('week')
final_df.to_csv('weekly_with_ccfi_full.csv', index=False)
print("✅ weekly_with_ccfi_full.csv 已生成！")
