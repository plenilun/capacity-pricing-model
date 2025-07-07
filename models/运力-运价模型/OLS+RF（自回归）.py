import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def load_and_merge_data():
    teu_df = pd.read_csv('weekly_weighted_teu.csv')
    oil_ccfi_df = pd.read_csv('weekly_with_CCFI.csv')
    vix_df = pd.read_csv('VIX.csv')

    teu_df['date'] = pd.to_datetime(teu_df['date'])
    oil_ccfi_df['week'] = pd.to_datetime(oil_ccfi_df['week'])
    vix_df['date'] = pd.to_datetime(vix_df['date'])
    vix_weekly = vix_df.resample('W-Mon', on='date').mean().reset_index()

    port_data = {}
    for port in ['USLSA', 'USNYK']:
        port_teu = teu_df[teu_df['port_name'] == port].copy()
        merged = pd.merge(
            oil_ccfi_df[['week', 'Brent_Oil_Avg', 'CCFI_weekly_avg', 'IPCI']],
            port_teu,
            left_on='week',
            right_on='date',
            how='left'
        ).drop(columns='date')

        merged = pd.merge(merged, vix_weekly, left_on='week', right_on='date', how='left')
        merged = merged.drop(columns='date').dropna()
        port_data[port] = merged

    return port_data

def create_features(df, lag=4):
    features = []
    for l in range(3, lag + 1):
        df[f'CCFI_lag{l}'] = df['CCFI_weekly_avg'].shift(l)
        features.append(f'CCFI_lag{l}')
    for var in ['weighted_teu', 'VIX', 'Brent_Oil_Avg', 'IPCI']:
        df[f'{var}_lag{lag}'] = df[var].shift(lag)
        features.append(f'{var}_lag{lag}')
    df['week_of_year'] = df['week'].dt.isocalendar().week
    features.append('week_of_year')
    return df.dropna(), features

def train_and_evaluate(port_data):
    results = {}
    for port, df in port_data.items():
        processed_df, feature_cols = create_features(df, lag=4)

        train_df = processed_df[(processed_df['week'] >= '2020-01-01') & (processed_df['week'] <= '2022-6-30')]
        test_df = processed_df[(processed_df['week'] > '2022-6-30') & (processed_df['week'] <= '2022-12-31')]

        X_train = train_df[feature_cols]
        y_train = train_df['CCFI_weekly_avg']
        X_test = test_df[feature_cols]
        y_test = test_df['CCFI_weekly_avg']

        # OLS
        ols = LinearRegression()
        ols.fit(X_train, y_train)
        ols_train_pred = ols.predict(X_train)
        ols_test_pred = ols.predict(X_test)
        plt.figure(figsize=(12, 4))
        plt.plot(test_df['week'], y_test, label='Actual CCFI', color='black')
        plt.plot(test_df['week'], ols_test_pred, label='OLS Prediction', linestyle='--', color='orange')
        plt.title(f'{port} - OLS with Autoregressive Features')
        plt.xlabel('Week')
        plt.ylabel('CCFI')
        plt.legend()
        plt.grid()
        plt.show()

        # OLS 误差
        mae_ols = mean_absolute_error(y_test, ols_test_pred)
        rmse_ols = np.sqrt(mean_squared_error(y_test, ols_test_pred))

        # RF 拟合残差
        residuals_train = y_train - ols_train_pred
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, residuals_train)
        rf_correction = rf.predict(X_test)

        final_pred = ols_test_pred + rf_correction
        plt.figure(figsize=(12, 4))
        plt.plot(test_df['week'], y_test, label='Actual CCFI', color='black')
        plt.plot(test_df['week'], final_pred, label='OLS + RF Prediction', linestyle='--', color='blue')
        plt.title(f'{port} - Hybrid Model with Autoregressive Features')
        plt.xlabel('Week')
        plt.ylabel('CCFI')
        plt.legend()
        plt.grid()
        plt.show()

        mae_rf = mean_absolute_error(y_test, final_pred)
        rmse_rf = np.sqrt(mean_squared_error(y_test, final_pred))

        results[port] = {
            'MAE_OLS': mae_ols, 'RMSE_OLS': rmse_ols,
            'MAE_OLS+RF': mae_rf, 'RMSE_OLS+RF': rmse_rf
        }
    return results

port_data = load_and_merge_data()
results = train_and_evaluate(port_data)
for port, metrics in results.items():
    print(f"\n=== {port} ===")
    print(f"OLS:    MAE = {metrics['MAE_OLS']:.2f}, RMSE = {metrics['RMSE_OLS']:.2f}")
    print(f"OLS+RF: MAE = {metrics['MAE_OLS+RF']:.2f}, RMSE = {metrics['RMSE_OLS+RF']:.2f}")