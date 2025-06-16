import pandas as pd

# 读取数据
df = pd.read_excel("C://Users//HgHaw//Desktop//洛杉矶_纽约到港效率（含时长）.xlsx") # 改成本地地址

# 仅保留集装箱船
container_ships = df[df["shiptype"] == "集装箱"].copy()

# 设置参数
normal_moor_duration = 10.0       # 正常进港时长（小时）
normal_berth_duration = 30.0      # 正常在泊时长（小时）
W_a = 0.6                          # 进港时长权重
W_b = 0.4                          # 在泊时长权重

# 权重函数
def assign_weight(length):
    if length <= 50:
        return 0.5
    elif length <= 100:
        return 1.0
    elif length <= 150:
        return 1.5
    elif length <= 200:
        return 2.0
    elif length <= 250:
        return 2.5
    elif length <= 300:
        return 3.0
    else:
        return 3.5

# 数据预处理
container_ships["arrival_time"] = pd.to_datetime(container_ships["arrival_time"], errors="coerce")
container_ships["week"] = container_ships["arrival_time"].dt.to_period("W").dt.start_time
container_ships["weight"] = container_ships["length"].apply(assign_weight)
container_ships["adjusted_moor_duration"] = container_ships["moor_duration_port"].clip(lower=normal_moor_duration, upper=10 * normal_moor_duration)
container_ships["adjusted_berth_duration"] = container_ships["berth_duration"].clip(lower=normal_berth_duration, upper=10 * normal_berth_duration)

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

weekly_pci = container_ships.groupby("week").apply(compute_weekly_pci).reset_index()

# 导出为 CSV 文件
weekly_pci.to_csv("C:/Users/HgHaw/Desktop/weekly_pci_container111.csv", index=False, encoding="utf-8-sig")
