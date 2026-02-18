import pandas as pd
import joblib
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
# ===== 关键修复：补全所有需要的导入项 =====
from config import (
    BASE_DIR, TECH_FEATURE_NUM, FINA_FEATURE_NUM, 
    TEST_SIZE, RANDOM_STATE, STOCK_CODE, PREDICT_DAYS,
    END_DATE  # 新增导入END_DATE变量
)
from utils import evaluate_model, plot_multistep_prediction

# ===================== 特征筛选与模型训练 =====================
def train_tech_model(df_tech, tech_feat_cols):
    """训练技术指标模型（升级：拟合趋势+多步预测）"""
    # 特征筛选
    X_tech = df_tech[tech_feat_cols]
    y_tech_close = df_tech["label_close"]
    rfe_tech = RFE(estimator=XGBRegressor(), n_features_to_select=TECH_FEATURE_NUM)
    X_tech_selected = rfe_tech.fit_transform(X_tech, y_tech_close)
    # 筛选后的特征名
    selected_tech_feats = [tech_feat_cols[i] for i in range(len(tech_feat_cols)) if rfe_tech.support_[i]]
    print(f"筛选后的技术特征：{selected_tech_feats}")
    
    # 划分训练集/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_tech_selected, y_tech_close, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # 训练模型（拟合历史趋势）
    model = XGBRegressor(learning_rate=0.1, n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    
    # 预测与评估
    y_pred = model.predict(X_test)
    mse, r2 = evaluate_model(y_test, y_pred, "tech_model", f"{BASE_DIR}03_results/fit_results", BASE_DIR)
    print(f"\n技术模型评估 - MSE: {mse:.2f}, R²: {r2:.2f}")
    
    # 保存模型和特征权重
    joblib.dump(model, f"{BASE_DIR}02_models/tech_model/xgb_tech_model.pkl")
    weight_df = pd.DataFrame({
        "feature": selected_tech_feats,
        "weight": model.feature_importances_
    })
    weight_df.to_csv(f"{BASE_DIR}02_models/tech_model/feature_weights.csv", index=False)
    
    # ========== 新增：多步预测未来n天收盘价 ==========
    print(f"\n开始预测未来 {PREDICT_DAYS} 天收盘价...")
    # 1. 获取最后一组特征（作为预测起点）
    last_features = X_tech_selected[-1].reshape(1, -1)
    # 2. 初始化预测结果列表
    future_predictions = []
    # 3. 迭代预测未来n天（基于前一天预测结果更新特征，简化版趋势外推）
    current_features = last_features.copy()
    for day in range(PREDICT_DAYS):
        # 预测当日收盘价
        day_pred = model.predict(current_features)[0]
        future_predictions.append(day_pred)
        # 模拟更新特征（核心：用预测值替换close特征，保持其他特征趋势）
        # 先找到close在特征列表中的位置
        if 'close' in selected_tech_feats:
            close_idx = selected_tech_feats.index('close')
            current_features[0][close_idx] = day_pred
        # 其他特征按历史趋势小幅波动（模拟市场变化）
        current_features = current_features * (1 + np.random.normal(0, 0.005, current_features.shape))
    
    # 4. 保存未来n天预测结果
    future_dates = pd.date_range(start=pd.to_datetime(END_DATE), periods=PREDICT_DAYS, freq='D')
    future_df = pd.DataFrame({
        "预测日期": future_dates.strftime("%Y%m%d"),
        "预测收盘价": future_predictions,
        "预测天数": [f"未来第{i+1}天" for i in range(PREDICT_DAYS)]
    })
    future_df.to_csv(f"{BASE_DIR}03_results/fit_results/tech_model_future_{PREDICT_DAYS}d.csv", index=False)
    
    # 5. 可视化历史趋势+未来预测
    plot_multistep_prediction(
        historical_data=y_tech_close.values,
        future_preds=future_predictions,
        model_name="tech_model",
        predict_days=PREDICT_DAYS,
        base_dir=BASE_DIR
    )
    
    # 输出预测结果
    print("\n技术模型未来{}天预测收盘价：".format(PREDICT_DAYS))
    for i, pred in enumerate(future_predictions):
        print(f"未来第{i+1}天：{pred:.2f}")
    
    return model

def train_fina_model(df_fina):
    """训练基本面指标模型（升级：拟合趋势+多步预测）"""
    
    # 检查df_fina的列
    print("df_fina的列名：", df_fina.columns.tolist())
    
    # ===== 修复：根据实际存在的列名调整 =====
    # 找出实际存在的列
    available_cols = df_fina.columns.tolist()
    
    # 基础价格特征（这些应该都有）
    price_cols = [col for col in ['open', 'close', 'high', 'low', 'vol'] if col in available_cols]
    
    # 基本面特征（可能存在的列）
    fina_cols = []
    for col in ['pe', 'roe', 'pcf', 'ps']:
        if col in available_cols:
            fina_cols.append(col)
    
    # 如果没有pcf和ps，就用pe和roe代替
    if len(fina_cols) < 2:
        print("提示：基本面特征较少，将使用所有可用特征")
        fina_cols = [col for col in ['pe', 'roe'] if col in available_cols]
    
    # 组合所有特征列
    fina_feat_cols = price_cols + fina_cols
    print(f"使用的特征列：{fina_feat_cols}")
    
    # 过滤缺失值
    df_fina = df_fina.dropna(subset=fina_feat_cols + ['label_close'])
    
    if df_fina.empty:
        print("基本面数据不足，跳过基本面模型训练")
        return None
    
    # 特征筛选
    X_fina = df_fina[fina_feat_cols]
    y_fina_close = df_fina["label_close"]
    
    # 确保n_features_to_select不超过实际特征数
    n_select = min(FINA_FEATURE_NUM, len(fina_feat_cols))
    
    rfe_fina = RFE(estimator=XGBRegressor(), n_features_to_select=n_select)
    X_fina_selected = rfe_fina.fit_transform(X_fina, y_fina_close)
    
    # 筛选后的特征名
    selected_fina_feats = [fina_feat_cols[i] for i in range(len(fina_feat_cols)) if rfe_fina.support_[i]]
    print(f"\n筛选后的基本面特征：{selected_fina_feats}")
    
    # 划分训练集/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_fina_selected, y_fina_close, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # 训练模型
    model = XGBRegressor(learning_rate=0.1, n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    
    # 预测与评估
    y_pred = model.predict(X_test)
    mse, r2 = evaluate_model(y_test, y_pred, "fina_model", f"{BASE_DIR}03_results/fit_results", BASE_DIR)
    print(f"基本面模型评估 - MSE: {mse:.2f}, R²: {r2:.2f}")
    
    # 保存模型和特征权重
    joblib.dump(model, f"{BASE_DIR}02_models/fina_model/xgb_fina_model.pkl")
    weight_df = pd.DataFrame({
        "feature": selected_fina_feats,
        "weight": model.feature_importances_
    })
    weight_df.to_csv(f"{BASE_DIR}02_models/fina_model/feature_weights.csv", index=False)
    
    # ========== 新增：多步预测未来n天收盘价 ==========
    print(f"\n开始预测未来 {PREDICT_DAYS} 天收盘价...")
    # 1. 获取最后一组特征（作为预测起点）
    last_features = X_fina_selected[-1].reshape(1, -1)
    # 2. 初始化预测结果列表
    future_predictions = []
    # 3. 迭代预测未来n天
    current_features = last_features.copy()
    for day in range(PREDICT_DAYS):
        # 预测当日收盘价
        day_pred = model.predict(current_features)[0]
        future_predictions.append(day_pred)
        # 模拟更新特征（用预测值替换close，基本面特征按历史均值波动）
        if 'close' in selected_fina_feats:
            close_idx = selected_fina_feats.index('close')
            current_features[0][close_idx] = day_pred
        # 基本面特征小幅波动（模拟财报数据变化）
        for feat in ['pe', 'roe']:
            if feat in selected_fina_feats:
                feat_idx = selected_fina_feats.index(feat)
                current_features[0][feat_idx] *= (1 + np.random.normal(0, 0.01, 1)[0])
    
    # 4. 保存未来n天预测结果
    future_dates = pd.date_range(start=pd.to_datetime(END_DATE), periods=PREDICT_DAYS, freq='D')
    future_df = pd.DataFrame({
        "预测日期": future_dates.strftime("%Y%m%d"),
        "预测收盘价": future_predictions,
        "预测天数": [f"未来第{i+1}天" for i in range(PREDICT_DAYS)]
    })
    future_df.to_csv(f"{BASE_DIR}03_results/fit_results/fina_model_future_{PREDICT_DAYS}d.csv", index=False)
    
    # 5. 可视化历史趋势+未来预测
    plot_multistep_prediction(
        historical_data=y_fina_close.values,
        future_preds=future_predictions,
        model_name="fina_model",
        predict_days=PREDICT_DAYS,
        base_dir=BASE_DIR
    )
    
    # 输出预测结果
    print("\n基本面模型未来{}天预测收盘价：".format(PREDICT_DAYS))
    for i, pred in enumerate(future_predictions):
        print(f"未来第{i+1}天：{pred:.2f}")
    
    return model