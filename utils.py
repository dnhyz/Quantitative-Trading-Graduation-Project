import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ===================== 自动创建文件夹 =====================
def create_folders(base_dir):
    folders = [
        f"{base_dir}01_data/raw_data",
        f"{base_dir}01_data/processed_data",
        f"{base_dir}02_models/tech_model",
        f"{base_dir}02_models/fina_model",
        f"{base_dir}03_results/fit_results",
        f"{base_dir}03_results/backtest",
        f"{base_dir}05_plots/fit_plots"
    ]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"创建文件夹：{folder}")
    return folders

# ===================== 模型评估函数 =====================
def evaluate_model(y_true, y_pred, model_name, save_path, base_dir):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    """计算MSE、R²，保存评估结果（修复base_dir传参问题）"""
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # 保存评估指标
    eval_df = pd.DataFrame({
        "指标": ["MSE", "R²"],
        "值": [mse, r2]
    })
    eval_df.to_csv(f"{save_path}/{model_name}_eval.csv", index=False)
    # 保存真实值vs预测值
    fit_df = pd.DataFrame({
        "真实值": y_true.values if hasattr(y_true, 'values') else y_true,
        "预测值": y_pred,
        "误差": y_true.values - y_pred if hasattr(y_true, 'values') else y_true - y_pred
    })
    fit_df.to_csv(f"{save_path}/{model_name}_fit.csv", index=False)
    # 绘制拟合图并保存
    plt.figure(figsize=(10, 6))
    plt.plot(fit_df["真实值"][:100], label="真实值", color="blue")  # 只画前100个点，更清晰
    plt.plot(fit_df["预测值"][:100], label="预测值", color="red", linestyle="--")
    plt.title(f"{model_name} 真实值vs预测值")
    plt.xlabel("样本序号")
    plt.ylabel("价格")
    plt.legend()
    plt.savefig(f"{base_dir}05_plots/fit_plots/{model_name}_fit_plot.png")
    plt.close()
    return mse, r2

# ===================== 新增：多步预测可视化函数 =====================
def plot_multistep_prediction(historical_data, future_preds, model_name, predict_days, base_dir):
    """绘制历史趋势+未来n天预测"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 取最后100条历史数据（可视化更清晰）
    hist_plot = historical_data[-100:]
    # 构建x轴：历史数据序号 + 未来预测序号
    x_hist = range(len(hist_plot))
    x_future = range(len(hist_plot), len(hist_plot) + predict_days)
    
    # 绘图
    plt.figure(figsize=(12, 7))
    # 历史数据
    plt.plot(x_hist, hist_plot, color="blue", label="历史收盘价", linewidth=1.5)
    # 未来预测
    plt.plot(x_future, future_preds, color="red", label=f"未来{predict_days}天预测", 
             linestyle="--", marker="o", markersize=4)
    # 分割线（历史/未来）
    plt.axvline(x=len(hist_plot)-1, color="gray", linestyle=":", label="历史/未来分割线")
    
    # 图表样式
    plt.title(f"{model_name} - 历史收盘价趋势 + 未来{predict_days}天预测", fontsize=14)
    plt.xlabel("时间序列（最后100条历史数据 + 未来预测）", fontsize=12)
    plt.ylabel("收盘价（元）", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # 保存图片
    plt.savefig(f"{base_dir}05_plots/fit_plots/{model_name}_future_{predict_days}d_plot.png", 
                dpi=300, bbox_inches='tight')
    plt.close()