import warnings
warnings.filterwarnings('ignore')
from config import BASE_DIR
from utils import create_folders
from data_process import get_raw_data, get_processed_data
from model_train import train_tech_model, train_fina_model

# ===================== 主流程 =====================
if __name__ == "__main__":
    print("===== 开始量化交易模型训练 =====")
    # 1. 创建文件夹
    create_folders(BASE_DIR)
    # 2. 获取原始数据
    print("\nStep 1: 获取原始数据...")
    df_daily, fina_base = get_raw_data()
    # 3. 数据预处理（计算指标、生成特征）
    print("\nStep 2: 数据预处理...")
    df_tech, df_fina, tech_feat_cols = get_processed_data(df_daily, fina_base)
    # 4. 训练技术指标模型
    print("\nStep 3: 训练技术指标模型...")
    tech_model = train_tech_model(df_tech, tech_feat_cols)
    # 5. 训练基本面指标模型
    print("\nStep 4: 训练基本面指标模型...")
    fina_model = train_fina_model(df_fina)
    # 6. 完成
    print("\n===== 模型训练完成！结果已保存到对应文件夹 =====")