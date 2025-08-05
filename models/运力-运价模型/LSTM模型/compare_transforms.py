import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
import copy
import numpy as np
import pandas as pd
from scipy import stats

import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 导入模块
from LSTM import (
    load_and_merge_data,
    add_lag_features,
    create_sequences,
    MultiLSTMModel,
    train_model,
    evaluate
)

# 先做一次基本的数据合并
df0 = load_and_merge_data(
    path_main='C://Users//HgHaw//Desktop//weekly_with_CCFI_updated.csv.xlsx',
    path_teu ='C://Users//HgHaw//Desktop//weekly_weighted_teu.csv',
    path_speed='C://Users//HgHaw//Desktop//weekly_speed.csv'
)

# 待对比的变换函数
def identity(x):
    return x

def log1p(x):
    return np.log1p(x.clip(lower=0))

def reciprocal(x):
    eps = 1e-6
    return 1.0 / (x + eps)

def diff(x):
    return x.diff().fillna(0)

def boxcox_transform(x, lmbda):
    # x 必须全正
    return stats.boxcox(x.clip(lower=1e-3), lmbda=lmbda)

transforms = {
    'identity':   identity,
    'log1p':      log1p,
    'reciprocal': reciprocal,
    'diff':       diff,
    'boxcox':     None,  # lambda later
}

# 做 Box–Cox 之前需要先拟合一个 lambda
# 我们直接对 PCI 找到一个 λ，然后对 policy （可调整其他特征）分别用相同 λ
pci = df0['PCI'].clip(lower=1e-3).values
bc_pci, fitted_lambda = stats.boxcox(pci)
transforms['boxcox'] = lambda x: boxcox_transform(x, fitted_lambda)

# 用来存结果
results = []

# 固定随机种子列表，以保证可复现
seeds = [42, 52, 62, 72, 82]

for name, func in transforms.items():
    print(f"\n### Testing transform: {name}")
    # copy 原 df
    df = df0.copy()
    # 应用在两列
    df['t_PCI']    = func(df['PCI'])
    df['t_policy'] = func(df['weekly_avg_policy_index'])
    # 生成滞后项
    df = add_lag_features(
        df,
        cols=['CCFI', 't_PCI', 't_policy', 'TEU',
              'avg_speed','Brent','weighted_oil_vix'], # features 可根据实际情况调整
        lag=4 # 滞后四周
    )
    # 特征列
    feature_cols = [c for c in df.columns if c.endswith('_lag4')]
    
    # 归一化与滑窗
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[feature_cols])
    X, y = create_sequences(data_scaled, seq_length=4)
    
    # 划分
    n = len(X)
    n_train = int(0.8*n)
    
    X_train_full = torch.FloatTensor(X[:n_train])
    y_train_full = torch.FloatTensor(y[:n_train])
    
    X_test = torch.FloatTensor(X[n_train:])
    y_test = torch.FloatTensor(y[n_train:])
    
    # 验证集
    n_val = int(0.1*len(X_train_full))
    X_val, y_val   = X_train_full[-n_val:], y_train_full[-n_val:]
    X_train, y_train = X_train_full[:-n_val], y_train_full[:-n_val]

    best_mse, best_mae = float('inf'), float('inf')
    
    # 多次不同 seed
    for seed in seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model = MultiLSTMModel(input_size=len(feature_cols))
        train_model(model, X_train, y_train, X_val, y_val)
        preds, trues, mse, mae = evaluate(model, X_test, y_test, scaler)
        if mse < best_mse: # 以 MSE 作为指标
            best_mse, best_mae = mse, mae

    print(f"→ Best MSE={best_mse:.2f}, MAE={best_mae:.2f}")
    results.append((name, best_mse, best_mae))

# 汇总成一份 DataFrame
df_res = pd.DataFrame(results, columns=['transform','MSE','MAE'])
print("\n所有变换对比：")
print(df_res.sort_values('MSE').to_string(index=False))
