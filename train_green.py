# -*- coding: cp936 -*-
"""
Created on Sat Dec 06 20:01:38 2014

@author: shuaiyi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import shuffle
from sklearn.svm import SVC # svm
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib # 保存分类器
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

is_train = False

# import data
df = pd.read_csv("420_decaf/data_set.txt")
data = df.drop(['XCoord','YCoord','FID'], 1)
X = data.iloc[:,0:-1]
X = X.values
Y = data.iloc[:,-1]
Y = Y.values

# 打乱数据
index = np.arange(Y.shape[0])
shuffle(index)

X = X[index,:]
Y = Y[index]

#%% Prepare training and testing data

x_train, x_test, y_train, y_test = train_test_split(X, Y, 
                                   test_size=0.10, random_state=42)             

# 使用SVC可以获取probability
clf = SVC(probability = True, class_weight='auto',random_state=np.random.RandomState())

parameters = {
    'C': (0.001, 0.01, 0.1, 1, 10, 100, 1000),
    'kernel': ('linear', 'rbf')
}


if __name__ == "__main__":
    # step 1: cv (cross validation_ grid search)
    # step 2: train    
    #
    if is_train:
        gridCV = GridSearchCV(clf, parameters,n_jobs=3,verbose=True)
        print "****************Grid Search******************************"
        gridCV.fit(X,Y)
        
        print "*********************Train******************************"
        # grid_cv results : {'clf__C': 5000, 'zca__bias': 0.01}
        best = gridCV.best_estimator_
        
        best.fit(x_train, y_train)
        
        print "*********************Save*******************************"
        joblib.dump(best, "420_decaf/classifier_green.pkl", compress=3)
        joblib.dump(gridCV, "420_decaf/grid_cv_green.pkl", compress=3)
    else:
        best = joblib.load("420_decaf/classifier_green.pkl")
        
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
    plt.savefig("420_decaf/ROC_green.tif", dpi=300)
    plt.show()
    # plt.close()