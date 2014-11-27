# -*- coding: cp936 -*-
"""
Created on Thu Oct 09 22:22:43 2014

@author: shuaiyi
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import shuffle
from sklearn.svm import LinearSVC # svm
from sklearn.cross_validation import train_test_split # 把训练样本分成训练和测试两部分
from sklearn.externals import joblib # 保存分类器
from sklearn.metrics import confusion_matrix
from zca import ZCA # 白化处理
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

#%% Train target selection
train_hog = True
train_bow = False
train_decaf = False

#%% Load traing data
X = Y = None
if train_hog: # train_hog
    X_neg = np.load("HoG/NEG_HOG.npy")
    X_pos = np.load("HoG/POS_HOG.npy")
    
    Y = np.ones(X_neg.shape[0] + X_pos.shape[0])
    Y[0:X_neg.shape[0]] = 2
    
    X = np.vstack((X_neg, X_pos))
    
    index = np.arange(Y.shape[0])
    shuffle(index)
    
    X = X[index,:]
    Y = Y[index]
elif train_bow == True: # train_bow
    X, Y = np.load("BoW/train_BoW_x.npy"), np.load("BoW/train_BoW_y.npy")
    Y = Y.reshape(Y.shape[0])
    index = np.arange(Y.shape[0])
    shuffle(index)
    
    X = X[index,:]
    Y = Y[index]
elif train_decaf == True:
    X, Y = np.load("420_decaf/420_decaf_X.npy"), np.load("420_decaf/420_decaf_Y.npy")
    Y = Y.reshape(Y.shape[0])
    index = np.arange(Y.shape[0])
    shuffle(index)
    
    X = X[index,:]
    Y = Y[index]

#%% Feature visualization

#PCA show
#pca = PCA(n_components = 2)
#pca.fit(X)
#x_pca = pca.fit_transform(X)
#for i in range(Y.shape[0]):
#    if Y[i] == 1:
#        plt.scatter(x_pca[i,0], x_pca[i,1],marker='o')
#    else:
#        plt.scatter(x_pca[i,0], x_pca[i,1],marker='+')
#
#plt.show()

#%% Prepare training and testing data

x_train, x_test, y_train, y_test = train_test_split(X, Y, 
                                   test_size=0.10, random_state=42)
                                   
# CV
# 关于ZCA 什么时候需要使用？
# zca = ZCA()

# 使用SVC可以获取probability

clf = LinearSVC() #C = 10000, loss='l1', penalty='l2', random_state=42

zca_svm = Pipeline([('clf',clf)]) #('zca',zca)

parameters = {
    #'zca__bias': (0.01, 0.001, 0.0001),
    'clf__C': (1000, 3000, 5000, 7000, 9000),
    'clf__loss': ('l1', 'l2')
    #'clf__penalty':('l1', 'l2')
}



if __name__ == "__main__":
    # step 1: cv (cross validation_ grid search)
    # step 2: train
    gridCV = GridSearchCV(zca_svm, parameters,n_jobs=4,verbose=True)
    print "****************Grid Search******************************"
    gridCV.fit(X,Y)
    
    print "*********************Train******************************"
    # grid_cv results : {'clf__C': 5000, 'zca__bias': 0.01}
    best = gridCV.best_estimator_
    
    best.fit(x_train, y_train)
    y_test_pre = best.predict(x_test)
    cm = confusion_matrix(y_test, y_test_pre)
    print "confusion matrix..."
    print cm
    
    print "*********************Save*******************************"
    joblib.dump(best, "420_decaf/classifier.pkl", compress=3)