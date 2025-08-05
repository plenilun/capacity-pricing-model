"""
此代码用于对 LSTM 模型超参数做探索性分析
可在此基础上寻找效果最好的参数
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import random
import json
import numpy as np
import torch
import optuna

# 确保可以导入本目录下的 LSTM 模块
sys.path.append(os.path.expanduser("~/Desktop"))

from LSTM import (
    load_and_merge_data,
    add_lag_features,
    create_sequences,
    MultiLSTMModel,
    train_model,
    evaluate
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. 数据准备（只执行一次） 需换成本地地址
DATA_MAIN  = 'C://Users//HgHaw//Desktop//weekly_with_CCFI_updated.csv.xlsx'
DATA_TEU   = 'C://Users//HgHaw//Desktop//weekly_weighted_teu.csv'
DATA_SPEED = 'C://Users//HgHaw//Desktop//weekly_speed.csv'

df = load_and_merge_data(DATA_MAIN, DATA_TEU, DATA_SPEED)
# 使用 log1p、reciprocal、Box–Cox 等变换后在 LSTM 模块中已生成相应列
# 本处使用 log_PCI、inv_policy、log_weighted_oil_vix

df = add_lag_features(
    df,
    cols=['CCFI', 'TEU','avg_speed', 'Brent',
        'log_PCI', 'inv_policy','bc_oil_vix'],  # 此处可调节 input 哪些特征
    lag=4
)
feature_cols = [c for c in df.columns if c.endswith('_lag4')][:5]

# 缩放与滑动窗口
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[feature_cols])
X, y = create_sequences(data_scaled, seq_length=4)

# 划分：80% 训练+验证，20% 测试
n = len(X)
n_train_val = int(0.8 * n)
X_train_val, y_train_val = X[:n_train_val], y[:n_train_val]
X_test,      y_test      = X[n_train_val:], y[n_train_val:]

# 从 train_val 中切出10%做验证
n_val = int(0.1 * len(X_train_val))
X_val, y_val     = X_train_val[-n_val:], y_train_val[-n_val:]
X_train, y_train = X_train_val[:-n_val], y_train_val[:-n_val]

# 转为 Tensor
X_train, y_train = map(torch.FloatTensor, (X_train, y_train))
X_val,   y_val   = map(torch.FloatTensor, (X_val,   y_val))
X_test,  y_test  = map(torch.FloatTensor, (X_test,  y_test))

# 2. 定义 Optuna 目标：最小化验证集上的最小 val_lose
def objective(trial: optuna.Trial) -> float:
    # 超参数搜索空间（可调节）
    hidden_size = trial.suggest_int('hidden_size', 64, 512, step=64)
    num_layers  = trial.suggest_int('num_layers', 1, 4)
    dropout     = trial.suggest_float('dropout', 0.1, 0.6)
    lr          = trial.suggest_loguniform('lr', 1e-4, 2e-2)

    # 固定随机性
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    model = MultiLSTMModel(
        input_size=len(feature_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )

    # 返回验证集上的最小损失
    best_val = train_model(
        model,
        X_train, y_train,
        X_val,   y_val,
        lr=lr,
        num_epochs=500,
        patience=20,
        return_best_val_loss=True
    )
    return best_val

if __name__ == '__main__':
    # 运行调参
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=300) # 调整运行次数

    # 输出并保存最佳参数
    print("== Best hyperparameters ==")
    print(study.best_trial.params)
    with open('best_params.json', 'w') as f:
        json.dump(study.best_trial.params, f)

    # 3. 最终训练 + 测试评估
    best_p = study.best_trial.params
    final_model = MultiLSTMModel(
        input_size=len(feature_cols),
        hidden_size=best_p['hidden_size'],
        num_layers=best_p['num_layers'],
        dropout=best_p['dropout']
    )
    # 合并训练+验证
    X_full = torch.cat([X_train, X_val], dim=0)
    y_full = torch.cat([y_train, y_val], dim=0)
    # 重训练，不早停
    _ = train_model(
        final_model,
        X_full, y_full,
        X_val=None, y_val=None,
        lr=best_p['lr'],
        num_epochs=200
    )
    # 测试集评估
    preds = final_model(X_test).detach().numpy().squeeze()
    # 反归一化
    min_, max_ = scaler.data_min_[0], scaler.data_max_[0]
    preds = preds * (max_ - min_) + min_
    true  = y_test.numpy() * (max_ - min_) + min_
    mse = mean_squared_error(true, preds)
    mae = mean_absolute_error(true, preds)
    print(f"Final Test ▶ MSE={mse:.2f}, MAE={mae:.2f}")
