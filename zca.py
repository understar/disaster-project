# -*- coding: cp936 -*-
"""
Created on Fri Oct 10 09:36:44 2014

@author: shuaiyi
"""
 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import array2d, as_float_array
from scipy.linalg import eigh
import numpy as np
 
class ZCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, bias=.1, copy=True):
        self.n_components = n_components
        self.bias = bias
        self.copy = copy
 
    def fit(self, X, y=None):
        X = array2d(X)
        n_samples, n_features = X.shape
        X = as_float_array(X, copy=self.copy)
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        eigs, eigv = eigh(np.dot(X.T, X) / n_samples + \
                         self.bias * np.identity(n_features))
        components = np.dot(eigv * np.sqrt(1.0 / eigs), eigv.T)
        self.components_ = components
        #Order the explained variance from greatest to least
        self.explained_variance_ = eigs[::-1]
        return self
 
    def transform(self, X):
        X = array2d(X)
        if self.mean_ is not None:
            X -= self.mean_
        X_transformed = np.dot(X, self.components_)
        return X_transformed