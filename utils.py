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