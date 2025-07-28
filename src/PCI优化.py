import pandas as pd

# 读取数据
df = pd.read_csv("处理后的_洛杉矶_长滩到港.csv")  # 改成本地地址

# 仅保留集装箱船
container_ships = df[df["shiptype"] == "集装箱"].copy()

# 数据预处理 - 转换时间列
container_ships["arrival_time"] = pd.to_datetime(container_ships["arrival_time"], errors="coerce")
container_ships["date"] = container_ships["arrival_time"].dt.date  # 添加日期列用于按日计算平均值

# 计算2023-2025年的平均值作为基准值
#mask = (container_ships["arrival_time"].dt.year >= 2023) & (container_ships["arrival_time"].dt.year <= 2025)
#normal_moor_duration = container_ships.loc[mask, "moor_duration_port"].mean()
#normal_berth_duration = container_ships.loc[mask, "berth_duration"].mean()
normal_moor_duration = 10.0       # 正常进港时长（小时）
normal_berth_duration = 30.0


# 设置权重参数
W_a = 0.6  # 进港时长权重
W_b = 0.4  # 在泊时长权重

# 计算每日集装箱船的平均length、width和height
daily_avg = container_ships.groupby("date").agg({
    "length": "mean",
    "width": "mean",
    "height": "mean"
}).reset_index()
daily_avg.columns = ["date", "daily_avg_length", "daily_avg_width", "daily_avg_height"]

container_ships = pd.merge(container_ships, daily_avg, on="date", how="left")


# 权重函数 - 综合考虑length、width和height三个维度
def assign_weight(row):
    # 计算三个维度的比值
    length_ratio = row["length"] / row["daily_avg_length"] if row["daily_avg_length"] != 0 else 1.0
    width_ratio = row["width"] / row["daily_avg_width"] if row["daily_avg_width"] != 0 else 1.0
    height_ratio = row["height"] / row["daily_avg_height"] if row["daily_avg_height"] != 0 else 1.0

    # 返回三个比值的加权平均
    return 3*(length_ratio + width_ratio + height_ratio)



container_ships["week"] = container_ships["arrival_time"].dt.to_period("W").dt.start_time
container_ships["weight"] = container_ships.apply(assign_weight, axis=1)  # 应用新的权重函数
container_ships["adjusted_moor_duration"] = container_ships["moor_duration_port"].clip(
    lower=normal_moor_duration,
    upper=10 * normal_moor_duration
)
container_ships["adjusted_berth_duration"] = container_ships["berth_duration"].clip(
    lower=normal_berth_duration,
    upper=10 * normal_berth_duration
)


# 按周计算 PCI
def compute_weekly_pci(group):
    weighted_moor = (group["adjusted_moor_duration"] / normal_moor_duration * group["weight"]).sum()
    weighted_berth = (group["adjusted_berth_duration"] / normal_berth_duration * group["weight"]).sum()
    weight_sum = group["weight"].sum()
    if weight_sum == 0:
        return pd.Series({"IPCI_0": 0, "IPCI": 0})
    IPCI_0 = W_a * (weighted_moor / weight_sum) + W_b * (weighted_berth / weight_sum)
    IPCI = 0 if IPCI_0 <= 1 else (IPCI_0 - 1) * 10
    return pd.Series({"IPCI_0": IPCI_0, "IPCI": IPCI})


weekly_pci = container_ships.groupby("week", group_keys=False).apply(compute_weekly_pci).reset_index()

# 导出为 CSV 文件
weekly_pci.to_csv("PCI（优化）.csv", index=False)