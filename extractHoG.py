# -*- coding: cp936 -*-
"""
Created on Thu Oct 09 22:07:26 2014

@author: shuaiyi
"""
import os
import numpy as np
from skimage.feature import hog
from skimage import color
from skimage.io import imread

def HoG_arr(arr):
    arr = color.rgb2gray(arr)
    return hog(arr, orientations=8, pixels_per_cell=(32, 32),
               cells_per_block=(2, 2), visualise=False, normalise=True)
               
def HoG(img_path):
    image = imread(img_path)
    image = color.rgb2gray(image)
    
    return hog(image, orientations=8, pixels_per_cell=(32, 32),
               cells_per_block=(2, 2), visualise=False, normalise=True)
               
def main():
    neg_results = []
    pos_results = []
    for root, dirs, files in os.walk("samples"):
        for f in files:
            print f
            if f[0:8] == "points99":
                neg_results.append(HoG("samples/s_negtive/%s"%f))
            elif f[0:8] == "points00":
                pos_results.append(HoG("samples/s_postive/%s"%f))
            else:
                pass
    
    neg = np.vstack(neg_results)
    pos = np.vstack(pos_results)
    
    np.save("POS_HOG.npy", pos)
    np.save("NEG_HOG.npy", neg)
    
if __name__ == "__main__":
    main()