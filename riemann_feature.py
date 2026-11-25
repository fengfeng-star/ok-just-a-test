
from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

class RiemannFeature(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # 使用 Ledoit-Wolf 收缩估计，提高协方差矩阵的正定性
        self.cov = Covariances(estimator='lwf')
        covs = self.cov.fit_transform(X)

        # 在当前训练折上拟合切空间参考点
        self.ts = TangentSpace(metric='riemann')
        self.ts.fit(covs)
        return self

    def transform(self, X):
        covs = self.cov.transform(X)
        return self.ts.transform(covs)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
