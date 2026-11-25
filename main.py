import matplotlib.pyplot as plt

from data_pipeline import get_features
from evaluation import evaluate_and_plot


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


if __name__ == "__main__":

    # ================= 模型选择 =================

    # 可选：
    # LDA
    # SVM
    # Logistic
    # RandomForest
    # XGBoost
    # MLP
    # Riemann
    # EEGNet

    MODEL_TYPE = 'XGBoost'

    # ================= 特征选择 =================

    # 传统机器学习模型使用的特征
    SELECTED_FEATURES = [
        'PSD',
        'DE',
        'Rel',
        'Hjorth',
        'PE',
        'Coherence'
    ]

    print(f"启动系统，当前模型: {MODEL_TYPE}")

    # EEGNet和Riemann使用原始EEG
    if MODEL_TYPE == 'EEGNet':
        feature_list = ['EEGNet']

    elif MODEL_TYPE == 'Riemann':
        feature_list = ['Riemann']

    else:
        feature_list = SELECTED_FEATURES

    print(f"数据处理方式: {feature_list}")

    # 1. 获取数据
    X_data, y = get_features(
        feature_list=feature_list
    )

    # 2. 训练与评估
    evaluate_and_plot(
        X_data,
        y,
        model_type=MODEL_TYPE
    )