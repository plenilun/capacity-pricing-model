import pandas as pd
import numpy as np

df = pd.read_csv("处理后的_洛杉矶_纽约到港效率.csv",
                parse_dates=["arrival_time", "start_postime", "end_postime"])

df['arrival_time'] = pd.to_datetime(df['arrival_time'], format='mixed')
df['start_postime'] = pd.to_datetime(df['start_postime'], format='mixed')
df['end_postime'] = pd.to_datetime(df['end_postime'], format='mixed')
container_df = df[df['shiptype'] == '集装箱'].copy()

container_df["date"] = container_df["start_postime"].dt.date
container_df["daily_weight"] = container_df["berth_duration"] / 24

daily_weighted_teu = container_df.groupby(["leg_end_port_code", "date"]).apply(
    lambda x: pd.Series({
        "weighted_teu": (x["teu"] * x["daily_weight"]).sum(),
        "weighted_length": (x["length"] * x["daily_weight"]).sum(),
        "weighted_width": (x["width"] * x["daily_weight"]).sum(),
        "weighted_draught": (x["draught"] * x["daily_weight"]).sum(),
        "weighted_dwt": (x["dwt"] * x["daily_weight"]).sum(),
        "ship_count": x["daily_weight"].sum(),  # 等效船舶数量
        "avg_berth_duration": x["berth_duration"].mean()
    })
).reset_index()

daily_weighted_teu["teu_proportion"] = daily_weighted_teu.groupby(
    ["leg_end_port_code", pd.to_datetime(daily_weighted_teu["date"]).dt.to_period("M")]
)["weighted_teu"].transform(lambda x: x / x.sum())

container_df["operation_efficiency"] = container_df["teu"] / container_df["berth_duration"]
daily_efficiency = container_df.groupby(["leg_end_port_code", "date"])["operation_efficiency"].mean().reset_index()
daily_weighted_teu = pd.merge(daily_weighted_teu, daily_efficiency, on=["leg_end_port_code", "date"])

daily_weighted_teu.to_csv("daily_weighted_teu.csv", index=False)
