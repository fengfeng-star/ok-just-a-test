import mne
import numpy as np
import os
from scipy.io import loadmat
from feature_extractor import extract_selected_features


class RiemannData:
    def __init__(self, X_filt):
        self.X_filt = X_filt
        self.feature_type = 'Riemann'

    @property
    def shape(self):
        return self.X_filt.shape

    def __getitem__(self, index):
        return self.X_filt[index]


class EEGNetData:
    def __init__(self, X_filt):
        self.X_filt = X_filt
        self.feature_type = 'EEGNet'

    @property
    def shape(self):
        return self.X_filt.shape

    def __getitem__(self, index):
        return self.X_filt[index]


def get_features(feature_list, cache_dir=r"D:\bci\11.2 test2"):
    os.makedirs(cache_dir, exist_ok=True)

    cache_x = os.path.join(cache_dir, "filtered_X.npy")
    cache_y = os.path.join(cache_dir, "labels_y.npy")

    if os.path.exists(cache_x) and os.path.exists(cache_y):
        print("找到缓存数据，直接加载...")
        X_filt = np.load(cache_x)
        y = np.load(cache_y)

    else:
        print("未找到缓存，开始加载数据并滤波...")

        file_s1 = r"D:\bci\11.2 test2\注意力\data\Subject1_attention.mat"
        file_s2 = r"D:\bci\11.2 test2\注意力\data\Subject2_attention.mat"

        mat_s1 = loadmat(file_s1)
        mat_s2 = loadmat(file_s2)

        X_raw = np.concatenate([
            mat_s1['data'].transpose(0, 2, 1),
            mat_s2['data'].transpose(0, 2, 1)
        ], axis=0)

        y = np.concatenate([
            mat_s1['label'].ravel(),
            mat_s2['label'].ravel()
        ]).astype(int)

        sfreq = 500

        info = mne.create_info(
            [f"ch{i+1}" for i in range(X_raw.shape[1])],
            sfreq,
            ch_types='eeg'
        )

        epochs = mne.EpochsArray(
            X_raw,
            info,
            verbose=False
        )

        filtered_epochs = epochs.copy().filter(
            0.5,
            45.0,
            method='iir',
            verbose=False
        )

        X_filt = filtered_epochs.get_data()

        np.save(cache_x, X_filt)
        np.save(cache_y, y)

    print(f" 数据就绪: {X_filt.shape[0]} 试次")

    # EEGNet：直接使用原始EEG
    if 'EEGNet' in feature_list:
        if len(feature_list) != 1:
            raise ValueError("EEGNet需要单独使用，请设置为 ['EEGNet']")

        print(" EEGNet直接使用原始EEG数据")
        return EEGNetData(X_filt), y

    # Riemann：每折单独拟合
    if 'Riemann' in feature_list:
        if len(feature_list) != 1:
            raise ValueError(
                "目前 Riemann 特征需要单独使用，"
                "请将 feature_list 设置为 ['Riemann']"
            )

        print("Riemann特征将在每个交叉验证折中单独拟合")
        return RiemannData(X_filt), y

    # 其他传统特征
    print(f"准备提取特征: {feature_list}")

    X_features = extract_selected_features(
        X_filt,
        feature_list,
        sfreq=500
    )

    print(f"特征提取完成！最终矩阵形状: {X_features.shape}")

    return X_features, y