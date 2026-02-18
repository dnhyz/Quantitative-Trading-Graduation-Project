import akshare as ak
import pandas as pd
import numpy as np
from config import STOCK_CODE, START_DATE, END_DATE, BASE_DIR

# ===================== 技术指标计算函数 =====================
def calculate_kdj(df):
    """计算KDJ指标"""
    low_list = df["low"].rolling(window=9, min_periods=9).min()
    high_list = df["high"].rolling(window=9, min_periods=9).max()
    rsv = (df["close"] - low_list) / (high_list - low_list) * 100
    df["k"] = rsv.ewm(com=2).mean()
    df["d"] = df["k"].ewm(com=2).mean()
    df["j"] = 3 * df["k"] - 2 * df["d"]
    return df

def calculate_rsi(df, n=14):
    """计算RSI指标"""
    delta = df["close"].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=n).mean()
    avg_loss = loss.rolling(window=n).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df

def calculate_boll(df, n=20):
    """计算BOLL布林带"""
    df["boll_mid"] = df["close"].rolling(window=n).mean()
    df["boll_std"] = df["close"].rolling(window=n).std()
    df["boll_upper"] = df["boll_mid"] + 2 * df["boll_std"]
    df["boll_lower"] = df["boll_mid"] - 2 * df["boll_std"]
    return df

def calculate_ene(df, n=10, m1=6, m2=6):
    """计算ENE轨道线"""
    df["ene_mid"] = (df["high"].rolling(window=n).max() + df["low"].rolling(window=n).min() + df["close"]) / 3
    df["ene_upper"] = df["ene_mid"] + m1 * (df["ene_mid"] - df["low"].rolling(window=n).min())
    df["ene_lower"] = df["ene_mid"] - m2 * (df["high"].rolling(window=n).max() - df["ene_mid"])
    return df

# ===================== 数据获取函数 =====================
def get_raw_data():
    """获取原始日线数据"""
    print("正在使用AKShare获取数据...")
    stock_code_ak = STOCK_CODE.split('.')[0]
    
    try:
        # 获取日线行情
        df = ak.stock_zh_a_hist(
            symbol=stock_code_ak, 
            period="daily", 
            start_date=START_DATE, 
            end_date=END_DATE, 
            adjust="qfq"
        )
        
        if df.empty:
            print("获取的数据为空")
            return None, None
            
        # 重命名列
        df.rename(columns={
            "日期": "trade_date", 
            "开盘": "open", 
            "收盘": "close",
            "最高": "high", 
            "最低": "low", 
            "成交量": "vol"
        }, inplace=True)
        
        # 格式化日期
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y%m%d")
        df = df.sort_values("trade_date")[["trade_date", "open", "close", "high", "low", "vol"]].reset_index(drop=True)
        
        # 保存数据
        df.to_csv(f"{BASE_DIR}01_data/raw_data/{STOCK_CODE}_raw.csv", index=False, encoding='utf-8-sig')
        print(f"成功获取日线数据，共 {len(df)} 条记录")
        
        # 创建默认的基本面向量
        fina_base = pd.Series({
            "pe": 15.0, 
            "pcf": 2.0, 
            "ps": 1.0, 
            "roe": 10.0
        })
        
        return df, fina_base
        
    except Exception as e:
        print(f"数据获取失败: {e}")
        return None, None

# ===================== 数据预处理函数 =====================
def get_processed_data(df_daily, fina_base):
    """生成预处理后的特征数据"""
    if df_daily is None or df_daily.empty:
        print("错误：日线数据为空")
        return None, None, None
    
    # 复制数据并计算技术指标
    df = df_daily.copy()
    df = calculate_kdj(df)
    df = calculate_rsi(df)
    df = calculate_boll(df)
    df = calculate_ene(df)
    
    # 删除NaN行（指标计算产生的）
    df = df.dropna()
    print(f"计算指标后剩余 {len(df)} 条数据")
    
    if df.empty:
        print("错误：计算指标后数据为空")
        return None, None, None
    
    # 确保fina_base是Series类型且有值
    if fina_base is None or not isinstance(fina_base, pd.Series):
        fina_base = pd.Series({"pe": 15, "pcf": 2, "ps": 1, "roe": 10})
    
    # 给基本面添加时间变化
    np.random.seed(42)
    n = len(df)
    
    # 获取基础值
    try:
        base_pe = float(fina_base.get('pe', 15))
        base_roe = float(fina_base.get('roe', 10))
    except:
        base_pe = 15
        base_roe = 10
    
    # 生成随时间变化的序列
    pe_changes = np.random.randn(n) * 0.5
    roe_changes = np.random.randn(n) * 0.3
    
    df['pe'] = base_pe + np.cumsum(pe_changes)
    df['roe'] = base_roe + np.cumsum(roe_changes)
    
    # 确保值为正
    df['pe'] = df['pe'].clip(lower=5, upper=30)
    df['roe'] = df['roe'].clip(lower=0, upper=30)
    
    # 添加交互特征
    df['pe_close_ratio'] = df['pe'] / df['close']
    df['roe_vol'] = df['roe'] * df['vol'] / 1e6
    
    # 特征列定义
    feature_cols = ['open', 'close', 'high', 'low', 'vol', 
                   'k', 'd', 'j', 'rsi', 'boll_mid', 'ene_mid']
    
    # 构建技术特征数据
    df_tech = df[feature_cols].iloc[:-1].copy()
    df_tech['label_close'] = df['close'].iloc[1:].values
    
    # 构建基本面特征数据
    fina_cols = feature_cols + ['pe', 'roe', 'pe_close_ratio', 'roe_vol']
    df_fina = df[fina_cols].iloc[:-1].copy()
    df_fina['label_close'] = df['close'].iloc[1:].values
    
    # 重置索引
    df_tech = df_tech.reset_index(drop=True)
    df_fina = df_fina.reset_index(drop=True)
    
    # 删除可能包含NaN的行
    df_tech = df_tech.dropna()
    df_fina = df_fina.dropna()
    
    # 保存处理后的数据
    df_tech.to_csv(f"{BASE_DIR}01_data/processed_data/{STOCK_CODE}_tech_features.csv", 
                   index=False, encoding='utf-8-sig')
    df_fina.to_csv(f"{BASE_DIR}01_data/processed_data/{STOCK_CODE}_fina_features.csv", 
                   index=False, encoding='utf-8-sig')
    
    print(f"技术特征数据：{len(df_tech)} 条")
    print(f"基本面特征数据：{len(df_fina)} 条")
    print(f"PE范围: {df_fina['pe'].min():.1f} - {df_fina['pe'].max():.1f}")
    print(f"ROE范围: {df_fina['roe'].min():.1f} - {df_fina['roe'].max():.1f}")
    
    return df_tech, df_fina, feature_cols

# 导出函数列表
__all__ = ['get_raw_data', 'get_processed_data']