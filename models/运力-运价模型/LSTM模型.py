"""
此代码用于 LSTM 长短期记忆网络模型
所有特征使用四周滞后，并添加长度为四周的滑动窗口
"""
import os
import random
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 固定随机种子
def set_seed(seed: int = 42) -> None:
    """
    设置所有随机数种子，确保结果可重复。
    包含 Python random、NumPy、PyTorch CPU/GPU 及 CuDNN 设置。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 数据加载与预处理
def load_and_merge_data(
    path_main: str,
    path_teu: str,
    path_speed: str,
    start_date: str = '2020-01-01',
    end_date: str = '2022-12-31'
) -> pd.DataFrame:
  
    # 1) 主数据
    df_main = pd.read_excel(path_main)
    df_main.columns = df_main.columns.str.strip()
    df_main = df_main.rename(columns={
        'Date': 'week',
        'CCFI_weekly_avg': 'CCFI',
        'IPCI': 'PCI',
        'Brent_Oil_Avg': 'Brent'
    })
    df_main['week'] = pd.to_datetime(df_main['week'])
    df_main = df_main[['week', 'CCFI', 'PCI', 'Brent',
                       'weekly_avg_policy_index', 'weighted_oil_vix']]

    # 2) TEU 数据
    df_teu = pd.read_csv(path_teu)
    df_teu.columns = df_teu.columns.str.strip()
    df_teu = df_teu.rename(columns={'date': 'week', 'weighted_teu': 'TEU'})
    df_teu['week'] = pd.to_datetime(df_teu['week'], errors='coerce')
    df_teu = df_teu[['week', 'TEU']]

    # 3) 航速数据
    df_speed = pd.read_csv(path_speed)
    df_speed.columns = df_speed.columns.str.strip()
    df_speed = df_speed.rename(columns={'date': 'week'})
    df_speed['week'] = pd.to_datetime(df_speed['week'], errors='coerce')
    df_speed['week'] = df_speed['week'].dt.tz_localize(None)
    df_speed = df_speed[['week', 'avg_speed', 'avg_voyage_days', 'days_per_speed']]

    # 4) 合并并过滤
    df = df_main.merge(df_teu, on='week', how='inner')
    df = df.merge(df_speed, on='week', how='inner')
    df = df.dropna()
    mask = (df['week'] >= start_date) & (df['week'] <= end_date)
    return df.loc[mask].reset_index(drop=True)

# 定义特征滞后项
def add_lag_features(
    df: pd.DataFrame,
    cols: list[str],
    lag: int = 4
) -> pd.DataFrame:
    for col in cols:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[f"{c}_lag{lag}" for c in cols])
    return df.reset_index(drop=True)

# 构造时序序列（滑动窗口）
def create_sequences(
    data: np.ndarray,
    seq_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    构建 LSTM 输入的滑动窗口序列：
      X: (样本数, seq_length, 特征数)
      y: 对应窗口后第一步的目标值（即 CCFI）。
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        window = data[i : i + seq_length]
        target = data[i + seq_length, 0]  # 第 0 维为 CCFI_lag4
        X.append(window)
        y.append(target)
    return np.array(X), np.array(y)
  
# 定义 LSTM 模型
class MultiLSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.4
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        out, _ = self.lstm(x) # x: [batch, seq_len, input_size]
        last = out[:, -1, :] # 取序列中最后一个 timestep 的隐藏状态
        return self.fc(last) # 投射到1维输出上，即“下一周的 CCFI”

# 定义 Training process
def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    lr: float = 0.005,
    num_epochs: int = 500,
    patience: int = 20
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    best_val, wait = float('inf'), 0

    for epoch in range(1, num_epochs + 1): # 迭代训练
        # Training step
        model.train()
        pred = model(X_train).squeeze()
        loss = criterion(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluation step
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val).squeeze()
            val_loss = criterion(val_pred, y_val)

        # early_stopping 逻辑
        if val_loss < best_val:
            best_val, wait = val_loss, 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}, val_loss={val_loss:.4f}")
                break
# 定义 Evaluating process
def evaluate(
    model: nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    scaler: MinMaxScaler
) -> tuple[np.ndarray, float, float]:
    model.eval()
    with torch.no_grad():
        pred = model(X_test).numpy().squeeze()

    # 反归一化，仅针对第 0 维（CCFI）
    min_, max_ = scaler.data_min_[0], scaler.data_max_[0]
    pred_rescaled = pred * (max_ - min_) + min_
    true_rescaled = y_test.numpy() * (max_ - min_) + min_
  
    # 计算 MSE 和 MAE，当做模型评估指标
    mse = mean_squared_error(true_rescaled, pred_rescaled)
    mae = mean_absolute_error(true_rescaled, pred_rescaled)
    return pred_rescaled, mse, mae

#特征重要性分析（可选取多种特征根据结果自行调参）
def permutation_importance(
    model: nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    scaler: MinMaxScaler,
    baseline_mse: float
) -> pd.DataFrame:
    feats = X_test.shape[2] # 代表取“有几个不同特征”的值
    min_, max_ = scaler.data_min_[0], scaler.data_max_[0]
    importances = []
    for i in range(feats): # 依次打乱每个特征
        Xp = X_test.clone()
        perm = torch.randperm(Xp.size(0))
        Xp[:, :, i] = Xp[perm, :, i]
        with torch.no_grad():
            pred = model(Xp).numpy().squeeze()
        pred_rescaled = pred * (max_ - min_) + min_
        true_rescaled = y_test.numpy() * (max_ - min_) + min_
        delta = mean_squared_error(true_rescaled, pred_rescaled) - baseline_mse # 计算重要性指标MSE，如果打乱该特征后MSE上升得越多，说明该特征对模型性能越关键
        importances.append(delta)
      
    # 整理成DataFrame并排序
    df_imp = pd.DataFrame({
        'feature_index': range(feats),
        'mse_increase': importances
    }).sort_values('mse_increase', ascending=False)
    return df_imp

# 模型拟合结果可视化
def plot_fit(
    weeks: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Model Fit: Actual vs. Predicted",
    ylabel: str = "CCFI"
) -> None:
    residuals = y_pred - y_true
    std_dev = np.std(residuals)

    plt.figure(figsize=(14, 7))
    plt.plot(weeks, y_true, label='Actual', linewidth=2.5) # 绘制真实值 CCFI
    plt.plot(weeks, y_pred, label='Predicted', linewidth=2.5) # 绘制预测值 CCFI
    plt.fill_between( # 绘制 ±1σ 区间
        weeks,
        y_pred - std_dev,
        y_pred + std_dev,
        color='orange', alpha=0.3,
        label='±1 Std Dev'
    )
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Week'); plt.ylabel(ylabel)
    plt.xticks(rotation=45); plt.grid(linestyle='--', alpha=0.6)
    plt.legend(); plt.tight_layout()

    # 在图下方显示整体 MSE 和 MAE
    mse = np.mean(residuals**2)
    mae = np.mean(np.abs(residuals))
    plt.figtext(
        0.5, -0.05,
        f'MSE: {mse:.2f}    MAE: {mae:.2f}',
        ha='center', color='gray'
    )
    plt.show()

# 定义 main 函数
def main():
    set_seed(42)
    df = load_and_merge_data(
        path_main='weekly_with_CCFI_updated.csv.xlsx',
        path_teu='weekly_weighted_teu.csv',
        path_speed='weekly_speed.csv'
    )
    df = add_lag_features(
        df,
        cols=[
            'CCFI', 'PCI', 'weekly_avg_policy_index',
            'TEU', 'avg_speed', 'avg_voyage_days',
            'days_per_speed'
        ],
        lag=4
    )
  
    # 特征选择与归一化处理
    feature_cols = [col for col in df.columns if col.endswith('_lag4')][:5]
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[feature_cols])
  
    # 序列化与数据集拆分
    seq_len = 4
    X, y = create_sequences(data_scaled, seq_len)
    n_train = int(0.8 * len(X))
    X_train_full, y_train_full = torch.FloatTensor(X[:n_train]), torch.FloatTensor(y[:n_train])
    X_test, y_test = torch.FloatTensor(X[n_train:]), torch.FloatTensor(y[n_train:])

    # 划分验证集
    n_val = int(0.1 * len(X_train_full))
    X_val, y_val = X_train_full[-n_val:], y_train_full[-n_val:]
    X_train, y_train = X_train_full[:-n_val], y_train_full[:-n_val]

    # 模型训练
    model = MultiLSTMModel(input_size=len(feature_cols))
    train_model(model, X_train, y_train, X_val, y_val)

    # 测试评估
    preds, mse, mae = evaluate(model, X_test, y_test, scaler)
    print(f"Test MSE: {mse:.2f}, MAE: {mae:.2f}")

    # 特征重要性
    imp_df = permutation_importance(model, X_test, y_test, scaler, baseline_mse=mse)
    print(imp_df)

    # 拟合可视化
    weeks = df['week'].iloc[seq_len + n_train:].reset_index(drop=True)
    plot_fit(
        weeks,
        y_true=y_test.numpy(),
        y_pred=preds,
        title='Best Multi-Variable LSTM CCFI Prediction'
    )

if __name__ == '__main__':
    main()
