from sklearn.base import BaseEstimator
import scipy.linalg
from sklearn.decomposition import TruncatedSVD


class PCR(BaseEstimator):
    
    def __init__(self, n_components=1):
        self.n_components = n_components
    
    def fit(self, X, y):
        tr = TruncatedSVD(n_components=self.n_components).fit(X)
        X = tr.transform(X)
        X = tr.inverse_transform(X)
        self.coef_ = scipy.linalg.pinv(X.T @ X) @ X.T @ y
        return self

    def predict(self, X):
        return X @ self.coef_