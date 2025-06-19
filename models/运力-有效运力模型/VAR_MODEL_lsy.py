import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, kpss
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import normaltest

def initialize_settings():
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.float_format', '{:.3f}'.format)

initialize_settings()

def load_and_clean_data(teu_path='weekly_teu_stats.csv',
                        pci_path='weekly_pci_container.csv',
                        port_name='USLSA'):

    try:
        teu_df = pd.read_csv(teu_path)
        pci_df = pd.read_csv(pci_path)

        teu_df['date'] = pd.to_datetime(teu_df['date'], errors='coerce')
        pci_df['week'] = pd.to_datetime(pci_df['week'], errors='coerce')

        if teu_df['date'].isnull().any() or pci_df['week'].isnull().any():
            raise ValueError("时间列包含无法解析的日期格式")

        teu_data = teu_df[teu_df['port_name'] == port_name].set_index('date')['weekly_teu_sum']
        pci_data = pci_df.set_index('week')['IPCI']

        merged = pd.merge(teu_data, pci_data, left_index=True, right_index=True, how='inner')
        merged.columns = ['teu', 'pci']

        if len(merged) == 0:
            raise ValueError("合并后数据为空，请检查港口名称或时间范围")

        print(f"\n成功加载数据: {len(merged)}条记录")
        print(f"时间范围: {merged.index.min()} 至 {merged.index.max()}")

        return merged

    except Exception as e:
        print(f"\n数据加载错误: {str(e)}")
        return None

data = load_and_clean_data()
if data is None:
    raise SystemExit("数据加载失败，请检查输入文件")


def check_stationarity(df, max_diff=2):
    print("\n=== 平稳性检验 ===")
    diff_order = 0
    current_data = df.copy()

    for i in range(max_diff + 1):
        if i > 0:
            current_data = current_data.diff().dropna()
            diff_order += 1
            print(f"\n应用 {i} 阶差分后:")

        all_stationary = True
        for col in current_data.columns:
            print(f"\n变量: {col}")
            adf_result = adfuller(current_data[col])
            print(f"ADF统计量: {adf_result[0]:.3f}, p值: {adf_result[1]:.3f}")
            try:
                kpss_result = kpss(current_data[col], regression='c')
                print(f"KPSS统计量: {kpss_result[0]:.3f}, p值: {kpss_result[1]:.3f}")
            except:
                print("KPSS检验失败（可能数据量不足）")
                kpss_result = (None, 1.0)

            if adf_result[1] > 0.05 or (kpss_result[1] is not None and kpss_result[1] < 0.05):
                all_stationary = False

        if all_stationary:
            print("\n所有变量已平稳")
            return current_data, diff_order

    print("\n警告: 已达到最大差分阶数但仍未完全平稳")
    return current_data, diff_order

stationary_data, diff_order = check_stationarity(data)

def build_var_model(df, max_lags=8):
    print("\n=== VAR模型构建 ===")
    model = VAR(df)
    print("\n滞后阶数选择结果:")
    lag_results = model.select_order(max_lags)
    print(lag_results.summary())

    optimal_lag = lag_results.aic
    if pd.isna(optimal_lag):
        optimal_lag = lag_results.bic
        print("使用BIC选择滞后阶数")

    print(f"\n选择的最优滞后阶数: {optimal_lag}")
    var_model = model.fit(optimal_lag)
    print("\nVAR模型结果摘要:")
    print(var_model.summary())

    return var_model, optimal_lag

var_model, optimal_lag = build_var_model(stationary_data)

def model_diagnostics(model, df):
    print("\n=== 模型诊断 ===")
    try:
        resid = model.resid.values if hasattr(model.resid, 'values') else np.array(model.resid)
        if resid.ndim == 1:
            resid = resid.reshape(-1, 1)
    except Exception as e:
        print(f"无法获取残差矩阵: {str(e)}")
        return

    print("\n残差自相关检验 (Durbin-Watson):")
    for i, col in enumerate(df.columns):
        try:
            resid_col = resid[:, i] if resid.shape[1] > 1 else resid.flatten()
            dw_stat = durbin_watson(resid_col)
            interpretation = "无自相关" if 1.5 < dw_stat < 2.5 else "可能存在自相关"
            print(f"{col}: {dw_stat:.3f} - {interpretation}")
        except Exception as e:
            print(f"{col}: 检验失败 - {str(e)}")

    print("\n残差正态性检验:")
    for i, col in enumerate(df.columns):
        try:
            resid_col = resid[:, i] if resid.shape[1] > 1 else resid.flatten()
            stat, p = normaltest(resid_col)
            normal = "正态" if p > 0.05 else "非正态"
            print(f"{col}: p值={p:.3f} - {normal}")
        except Exception as e:
            print(f"{col}: 检验失败 - {str(e)}")

    print("\n模型稳定性检验:")
    try:
        roots = model.roots
        max_root = max(abs(roots))
        print(f"最大特征根倒数: {max_root:.3f}")
        if max_root < 1:
            print("模型稳定 (所有特征根在单位圆内)")
        else:
            print("警告: 模型不稳定!")
    except Exception as e:
        print(f"稳定性检验失败: {str(e)}")

    try:
        plt.figure(figsize=(12, 6))
        if resid.shape[1] == len(df.columns):
            pd.DataFrame(resid, columns=df.columns).plot(subplots=True)
        else:
            plt.plot(resid, label='Residuals')
        plt.suptitle("模型残差")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"残差可视化失败: {str(e)}")


model_diagnostics(var_model, stationary_data)

def granger_causality_test(df, max_lag=4):
    print("\n=== 格兰杰因果检验 ===")
    variables = df.columns
    for caused in variables:
        for causing in variables:
            if caused != causing:
                try:
                    result = grangercausalitytests(df[[caused, causing]], maxlag=max_lag, verbose=False)
                    for lag in range(1, max_lag + 1):
                        p_val = result[lag][0]['ssr_ftest'][1]
                        print(f"  滞后阶 {lag} 的 p 值: {p_val:.4f} -> {'有因果' if p_val < 0.05 else '无因果'}")
                except Exception as e:
                    print(f"  检验失败: {str(e)}")
granger_causality_test(stationary_data, max_lag=optimal_lag)

def impulse_response_analysis(model, periods=12):
    print("\n=== 脉冲响应分析 ===")
    variables = model.names
    irf = model.irf(periods=periods)
    try:
        plt.figure(figsize=(12, 8))
        irf.plot(impulse=None, response=None, subplot_params={'fontsize': 10})
        plt.suptitle(f"脉冲响应函数 (响应期数: {periods})", y=1.02)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"脉冲响应图绘制失败: {str(e)}")

    # 输出脉冲响应数据
    try:
        print("\n脉冲响应数据:")
        for i, impulse_var in enumerate(variables):
            for j, response_var in enumerate(variables):
                print(f"\n{impulse_var} 对 {response_var} 的脉冲响应:")
                print(irf.irfs[:, i, j])
    except Exception as e:
        print(f"脉冲响应数据输出失败: {str(e)}")

    return irf

irf_results = impulse_response_analysis(var_model, periods=12)


def generate_forecasts(model, steps=50):
    print(f"\n=== 生成未来 {steps} 步预测 ===")
    forecast = model.forecast(model.endog, steps=steps)
    forecast_df = pd.DataFrame(forecast, columns=model.names)
    forecast_df.index = pd.date_range(start=data.index[-1] + pd.Timedelta(weeks=1), periods=steps, freq='W')
    print(forecast_df.head())
    return forecast_df

def recover_forecasts(original_data, forecast_df):
    print("\n=== 恢复预测值到原始尺度 ===")
    last_actual = original_data.iloc[-1]
    recovered = forecast_df.cumsum() + last_actual
    recovered.columns = [col + "_restored" for col in forecast_df.columns]
    final_forecast = pd.concat([forecast_df, recovered], axis=1)
    print(final_forecast.head())
    return final_forecast

def save_forecasts_to_csv(df, filename='my_var_results_data.csv'):
    try:
        df.to_csv(filename)
        print(f"\n预测结果已保存到: {filename}")
    except Exception as e:
        print(f"保存失败: {str(e)}")

forecast_steps = 50
forecast_result = generate_forecasts(var_model, steps=forecast_steps)
final_forecast = recover_forecasts(data, forecast_result)
save_forecasts_to_csv(final_forecast)
