from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# 导入 MetaBCI 的 EEGNet
from metabci.brainda.algorithms.deep_learning.eegnet import EEGNet


def get_model(model_type='LDA'):
    """
    统一模型接口。
    """
    if model_type == 'LDA':
        return make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis()
        )

    elif model_type == 'SVM':
        return make_pipeline(
            StandardScaler(),
            SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced'
            )
        )

    elif model_type == 'Logistic':
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight='balanced'
            )
        )

    elif model_type == 'RandomForest':
        return RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

    elif model_type == 'XGBoost':
        return XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1
        )

    elif model_type == 'MLP':
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42
            )
        )

    elif model_type == 'EEGNet':
        return EEGNet(
            n_channels=32,
            n_samples=500,
            n_classes=3
        )

    else:
        raise ValueError(f"❌ 未知的模型类型: {model_type}")