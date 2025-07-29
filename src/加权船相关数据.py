"""
    此代码用于对TEU的加权处理，将TEU按作业时间进行加权
"""

import pandas as pd

# 读取数据
df = pd.read_csv("洛杉矶_长滩到港.csv")

def safe_date_parse(date_str):#加载正确的日期
    try:
        dt = pd.to_datetime(date_str, dayfirst=True, format='mixed')
        if dt.year < 2000 or dt.year > 2030:
            return pd.NaT
        return dt
    except:
        return pd.NaT

# 2. 时间格式处理
df['start_postime'] = df['start_postime'].apply(safe_date_parse)

# 3. 筛选集装箱船舶
container_df = df[df['shiptype'] == '集装箱'].copy()

# 4. 计算每日权重和加权TEU
container_df["date"] = container_df["start_postime"].dt.date
container_df["daily_weight"] = container_df["berth_duration"] / 24
container_df["weighted_teu"] = container_df["teu"] * container_df["daily_weight"]

# 5. 按日期聚合（合并两个港口）
daily_teu = container_df.groupby("date")["weighted_teu"].sum().reset_index()

# 6. 转换为周度数据（每周一为节点）
daily_teu['date'] = pd.to_datetime(daily_teu['date'])
weekly_teu = daily_teu.set_index('date').resample('W-MON')["weighted_teu"].sum().reset_index()

weekly_teu.columns = ['week_start_date', 'weekly_weighted_teu']

#保存结果
weekly_teu.to_csv("weekly_weighted_teu.csv", index=False)
print("周度TEU数据已保存为 weekly_weighted_teu.csv")
