"""
此代码用于对 LSTM 模型参数做探索性分析
可在此基础上寻找效果最好的参数
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim

# 固定随机种子seed
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 定义LSTM模型
class MultiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.3):
        super(MultiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# 定义模型训练函数
def train_model(X_train, y_train, X_val, y_val, input_size, hidden_size, dropout,
                num_epochs=500, patience=20, lr=0.005):
    model = MultiLSTMModel(input_size, hidden_size, dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    best_state = None
    best_val_loss = float('inf')
    stop_counter = 0
    for epoch in range(num_epochs):
        model.train()
        output = model(X_train)
        loss = criterion(output.squeeze(), y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
      
        with torch.no_grad():
            val_pred = model(X_val).squeeze()
            val_loss = criterion(val_pred, y_val)
          
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = model.state_dict()
            stop_counter = 0
        else:
            stop_counter += 1
            if stop_counter >= patience:
                break

    model.load_state_dict(best_state)
    return model

# 定义LSTM网格搜索函数，寻找最优hidden_size和dropout_rates
def grid_search_lstm(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor,
                     scaler, input_size, hidden_sizes, dropout_rates, num_trials=3):
    results = []
    val_split = int(0.1 * X_train_tensor.shape[0])
    X_val_tensor = X_train_tensor[-val_split:]
    y_val_tensor = y_train_tensor[-val_split:]
    X_train_sub = X_train_tensor[:-val_split]
    y_train_sub = y_train_tensor[:-val_split]

    for hidden_size in hidden_sizes:
        for dropout in dropout_rates:
            mse_list, mae_list = [], []
            for trial in range(num_trials):
                set_seed(42 + trial)
                model = train_model(X_train_sub, y_train_sub, X_val_tensor, y_val_tensor,
                                    input_size, hidden_size, dropout)
                model.eval()
                with torch.no_grad():
                    pred = model(X_test_tensor).numpy()

                ccfi_min, ccfi_max = scaler.data_min_[0], scaler.data_max_[0]
                pred_rescaled = pred * (ccfi_max - ccfi_min) + ccfi_min
                true_rescaled = y_test_tensor.numpy() * (ccfi_max - ccfi_min) + ccfi_min

                mse = mean_squared_error(true_rescaled, pred_rescaled)
                mae = mean_absolute_error(true_rescaled, pred_rescaled)
                mse_list.append(mse)
                mae_list.append(mae)

            results.append({
                'hidden_size': hidden_size,
                'dropout': dropout,
                'avg_MSE': np.mean(mse_list),
                'avg_MAE': np.mean(mae_list)
            })

    return pd.DataFrame(results).sort_values(by='avg_MSE')

### 用法示例
'''
grid_result_df = grid_search_lstm(
    X_train_tensor=X_train_tensor,
    y_train_tensor=y_train_tensor,
    X_test_tensor=X_test_tensor,
    y_test_tensor=y_test_tensor,
    scaler=scaler,
    input_size=len(features),
    hidden_sizes=[64, 128, 256],
    dropout_rates=[0.1, 0.2, 0.3, 0.4，0.5],
    num_trials=3
)
print(grid_result_df)
'''
