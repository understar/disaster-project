# -*- coding: cp936 -*-
"""
Created on Sat Oct 11 14:07:58 2014

@author: shuaiyi
"""

import numpy as np
import os
from numpy.random import random
import matplotlib.pyplot as plt
from sklearn.externals import joblib 

from extractHoG import imread
import codebook

def main():
    km = joblib.load("dict_k_means.pkl")
    clf = joblib.load("classifier_bow.pkl")
    f, axarr = plt.subplots(2, 10)
    i = j = 0
    for root, dirs, files in os.walk("samples"):
        for f in files:
            if f[0:8] == "points99":
                if random()>0.5 and i < 10 and codebook.BoW("samples/s_negtive/%s"%f, km) != None and \
                    clf.predict(codebook.BoW("samples/s_negtive/%s"%f, km)) == 1:
                    print f
                    axarr[0,i].imshow(imread("samples/s_negtive/%s"%f))
                    i = i + 1
                    
            elif f[0:8] == "points00":
                if random()>0.5 and j < 10 and codebook.BoW("samples/s_postive/%s"%f, km) != None and \
                    clf.predict(codebook.BoW("samples/s_postive/%s"%f, km)) == 2:
                    print f                    
                    axarr[1,j].imshow(imread("samples/s_postive/%s"%f))
                    j = j + 1
            plt.show()
    
if __name__ == "__main__":
    main()