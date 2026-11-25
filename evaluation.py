
import numpy as np
import matplotlib.pyplot as plt

from model_factory import get_model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from riemann_feature import RiemannFeature


def evaluate_and_plot(X_features, y, model_type='LDA'):

    print(f"\n=== 开始 {model_type} 模型训练与 5折交叉验证 ===")

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = []
    all_preds = np.zeros_like(y)

    # 判断数据类型
    is_riemann = getattr(X_features, 'feature_type', None) == 'Riemann'
    is_eegnet = getattr(X_features, 'feature_type', None) == 'EEGNet'

    is_raw_eeg = is_riemann or is_eegnet

    # 检查模型和数据是否匹配
    if model_type == 'EEGNet' and not is_eegnet:
        raise ValueError("EEGNet 必须使用原始 EEG 数据，请设置 feature_list=['EEGNet']")

    if is_eegnet and model_type != 'EEGNet':
        raise ValueError("当前输入是 EEGNet 原始数据，请将 model_type 设置为 'EEGNet'")

    # 原始 EEG 数据
    X_data = X_features.X_filt if is_raw_eeg else X_features

    # EEGNet 标签转换为 0、1、2
    if is_eegnet:
        classes, y_split = np.unique(y, return_inverse=True)
    else:
        classes = None
        y_split = y

    print("数据类型:", type(X_features))
    print("交叉验证数据形状:", X_data.shape)
    print("标签形状:", y.shape)
    print("类别标签:", np.unique(y))

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(X_data, y_split), 1
    ):

        print(f"\n正在进行第 {fold}/5 折...")

        X_train = X_data[train_idx]
        y_train = y_split[train_idx]

        X_test = X_data[test_idx]
        y_test = y[test_idx]

        # Riemann：每折单独拟合
        if is_riemann:

            print("正在拟合 Riemann 切空间...")

            riemann = RiemannFeature()

            X_train = riemann.fit_transform(X_train, y_train)
            X_test = riemann.transform(X_test)

            print("Riemann 特征形状:", X_train.shape)

        # 创建模型：每折重新初始化
        clf = get_model(model_type)

        # LDA 类别先验
        if model_type == 'LDA':

            unique_classes, counts = np.unique(
                y_train,
                return_counts=True
            )

            priors = counts / counts.sum()

            clf.named_steps[
                'lineardiscriminantanalysis'
            ].set_params(priors=priors)

        # 训练
        clf.fit(X_train, y_train)

        # 预测
        pred = clf.predict(X_test)

        # EEGNet 预测结果转换回原始标签
        if is_eegnet:
            pred = classes[np.asarray(pred, dtype=int)]

        score = accuracy_score(y_test, pred)

        scores.append(score)
        all_preds[test_idx] = pred

        print(f"第 {fold} 折准确率: {score * 100:.2f}%")

    mean_acc = np.mean(scores) * 100
    std_acc = np.std(scores) * 100

    print(
        f"\n🎉 平均准确率: "
        f"{mean_acc:.2f}% ± {std_acc:.2f}%"
    )

    plot_confusion_matrix(
        y,
        all_preds,
        model_type,
        mean_acc
    )


def plot_confusion_matrix(y_true, y_pred, model_name, acc):

    cm = confusion_matrix(y_true, y_pred)

    label_names = ['中性', '放松', '专注']

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(
        cm,
        interpolation='nearest',
        cmap=plt.cm.Blues
    )

    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=label_names,
        yticklabels=label_names,
        title=f'{model_name} 混淆矩阵\n平均准确率: {acc:.2f}%',
        ylabel='真实类别',
        xlabel='预测类别'
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right"
    )

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            ax.text(
                j,
                i,
                format(cm[i, j], 'd'),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14
            )

    plt.tight_layout()
    plt.show()
