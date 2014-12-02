# -*- coding: cp936 -*-
"""
Created on Thu Oct 09 22:22:43 2014

@author: shuaiyi
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import shuffle
from sklearn.svm import LinearSVC, SVC # svm
from sklearn.cross_validation import train_test_split # 把训练样本分成训练和测试两部分
from sklearn.externals import joblib # 保存分类器
from sklearn.metrics import confusion_matrix
from zca import ZCA # 白化处理
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

is_train = False

if len(sys.argv) == 1:
    train_hog = False
    train_bow = True
    train_decaf = False


elif len(sys.argv) < 4:
    print "Usage: train.py train_hog?0:1 train_bow?0:1 train_decaf?0:1"
    sys.exit()
else:
    #%% Train target selection
    train_hog = bool(int(sys.argv[1]))
    train_bow = bool(int(sys.argv[2]))
    train_decaf = bool(int(sys.argv[3]))

prj_name = ''
#%% Load traing data
X = Y = None
if train_hog: # train_hog
    prj_name = 'HoG'
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
    prj_name = 'BoW'
    X, Y = np.load("BoW/train_BoW_x.npy"), np.load("BoW/train_BoW_y.npy")
    Y = Y.reshape(Y.shape[0])
    index = np.arange(Y.shape[0])
    shuffle(index)
    
    X = X[index,:]
    Y = Y[index]
elif train_decaf == True:
    prj_name = '420_decaf'
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
y_train, y_test = y_train -1 , y_test - 1                  

# CV
# 关于ZCA 什么时候需要使用？
# zca = ZCA()

# 使用SVC可以获取probability
clf = SVC(kernel='linear', probability = True, class_weight='auto',random_state=np.random.RandomState())
#clf = LinearSVC() #C = 10000, loss='l1', penalty='l2', random_state=42

zca_svm = Pipeline([('clf',clf)]) #('zca',zca)

parameters = {
    #'zca__bias': (0.01, 0.001, 0.0001),
    'clf__C': (0.01, 0.1, 1, 10, 100)
    #'clf__kernel': ('linear')
    #'clf__C': (1, 10, 20, 30, 40, 50, 60, 70, 80, 90)
    #'clf__C': (100, 200, 300, 400, 500, 600, 700, 800, 900)
    #'clf__C': (1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000)
    #'clf__loss': ('l1', 'l2')
    #'clf__penalty':('l1', 'l2')
}



if __name__ == "__main__":
    # step 1: cv (cross validation_ grid search)
    # step 2: train    
    #
    if is_train:
        gridCV = GridSearchCV(zca_svm, parameters,n_jobs=3,verbose=True)
        print "****************Grid Search******************************"
        gridCV.fit(X,Y)
        
        print "*********************Train******************************"
        # grid_cv results : {'clf__C': 5000, 'zca__bias': 0.01}
        best = gridCV.best_estimator_
        
        best.fit(x_train, y_train)
        
        print "*********************Save*******************************"
        joblib.dump(best, prj_name + "/classifier_svc.pkl", compress=3)
        joblib.dump(gridCV, prj_name + "/grid_cv.pkl", compress=3)
    else:
        best = joblib.load(prj_name + "/classifier_svc.pkl")
        
    print "*********************Test*******************************"
    y_test_pre = best.predict(x_test)
    cm = confusion_matrix(y_test, y_test_pre)
    print "confusion matrix..."
    print cm
    
    
    print "*********************ROC********************************"
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    print best
    y_score = best.predict_proba(x_test)
    
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver operating characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(prj_name + "/ROC.tif", dpi=300)
    plt.show()
    # plt.close()
