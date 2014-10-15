# -*- coding: cp936 -*-
"""
Created on Fri Oct 10 10:55:10 2014

特别需要注意：
    scikit image处理的结果往往是float结果，必须得想办法转换成uint8
    主要数据类型的一致性

@author: shuaiyi
"""
import numpy as np
from numpy.random import random
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC # svm
from sklearn.externals import joblib # 保存分类器
from zca import ZCA # 白化处理
from sklearn.pipeline import Pipeline
from skimage.transform import pyramid_reduce
import skimage.io as io

from extractHoG import HoG, imread, HoG_arr
import codebook
import progressbar
from osgeo import gdal

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# big tiff
#from osgeo import gdal
#ds = gdal.Open('test.tif')
#img = ds.ReadAsArray(0,0,100,100)
#img = img.swapaxes(0,2) #交互波段
#img = img.swapaxes(0,1) #交互xy
#imshow(img)
SIZE = [256, 512, 1024]

def random_loc_size(rows, cols):
    index = np.random.randint(len(SIZE))
    x, y = np.random.randint(rows-SIZE[index]), \
           np.random.randint(cols-SIZE[index])
    return x, y, SIZE[index]
    
def getRandom_sample(g_raster):
    x, y, s = random_loc_size(g_raster.RasterYSize, g_raster.RasterXSize)
    img = g_raster.ReadAsArray(y,x,s,s)
    img = img.swapaxes(0,2).swapaxes(0,1)
    if s!= 256:
        return (pyramid_reduce(img, downscale = s/256) * 255).astype(np.uint8)
    else:
        return img

def get_sample(gr, x, y, s):
    img = gr.ReadAsArray(x,y,s,s)
    img = img.swapaxes(0,2).swapaxes(0,1)
    if s!= 256:
        return (pyramid_reduce(img, downscale = s/256) * 255).astype(np.uint8)
    else:
        return img

def random_check_hog(n = 10):
    clf = joblib.load("classifier.pkl")
    g_raster = gdal.Open('test.tif') # test.tif
    plt.axis('off')
    f, axarr = plt.subplots(n, n)
    for i in range(n):
        for j in range(n):
            img = getRandom_sample(g_raster)
            axarr[i,j].imshow(img)
            axarr[i,j].set_title(clf.predict(HoG_arr(img)))
            
def random_check_bow(n = 10):
    km = joblib.load("dict_k_means.pkl")
    clf = joblib.load("classifier_bow.pkl")
    g_raster = gdal.Open('test.tif') # test.tif
    plt.axis('off')
    f, axarr = plt.subplots(n, n)
    for i in range(n):
        for j in range(n):
            img = getRandom_sample(g_raster)
            # print img.dtype, img.shape
            tmp = codebook.BoW(img, km)
            axarr[i,j].imshow(img)
            if tmp != None:
                axarr[i,j].set_title(clf.predict(tmp))
                
def full_check_bow(win_slide=5, win_size=1024):
    km = joblib.load("dict_k_means.pkl")
    clf = joblib.load("classifier_bow.pkl")
    g_raster = gdal.Open('test.tif') # test.tif
    # plt.axis('off')
    # f, axarr = plt.subplots(n, n)
    result = {}
    cols = range(0, g_raster.RasterXSize - win_size, win_slide)
    rows = range(0, g_raster.RasterYSize - win_size, win_slide)
    full = len(rows) * len(cols)
    count = 0
    pbar = progressbar.ProgressBar(maxval=full).start()
    for i in range(0, g_raster.RasterXSize - win_size, win_slide):
        for j in range(0, g_raster.RasterYSize - win_size, win_slide):
            img = get_sample(g_raster, i, j, win_size)
            # print img.dtype, img.shape
            # print "Processing %s %s with size of %s."%(j, i, win_size)
            tmp = codebook.BoW(img, km)
            if tmp != None:
                result[(j,i)] = clf.predict(tmp)
            else:
                result[(j,i)] = 0
            pbar.update(count+1)
            count = count + 1
    pbar.finish()
    arr = np.ones((len(rows), len(cols)))
    for k, v in result.items():
        if v != 0 and v[0] == 2:
            arr[k[0]/win_slide, k[1]/win_slide] = v[0]
    return arr

def full_check_decaf(win_slide=5, win_size=1024, blob_name='fc6_cudanet_out'):
    from decaf.scripts.imagenet import DecafNet
    net = DecafNet()
    clf = joblib.load("420_decaf/classifier_decaf.pkl")
    g_raster = gdal.Open('test.tif') # test.tif
    # plt.axis('off')
    # f, axarr = plt.subplots(n, n)
    result = {}
    cols = range(0, g_raster.RasterXSize - win_size, win_slide)
    rows = range(0, g_raster.RasterYSize - win_size, win_slide)
    full = len(rows) * len(cols)
    count = 0
    pbar = progressbar.ProgressBar(maxval=full).start()
    for i in range(0, g_raster.RasterXSize - win_size, win_slide):
        for j in range(0, g_raster.RasterYSize - win_size, win_slide):
            img = get_sample(g_raster, i, j, win_size)
            net.classify(img, True)
            tmp = net.feature(blob_name) #与训练时候保持一致
            result[(j,i)] = clf.predict(tmp)
            if result[(j,i)] == 2:
                io.imsave("420_decaf/slide_target/%s_%s_%s_%s.png" % (j, i, j+win_size, i+win_size), img)
            pbar.update(count+1)
            count = count + 1
    pbar.finish()
    
    arr = np.ones((len(rows), len(cols)))
    for k, v in result.items():
        if v != 0 and v[0] == 2:
            arr[k[0]/win_slide, k[1]/win_slide] = v[0]
    return arr
    
def main():
    clf = joblib.load("classifier.pkl")
    f, axarr = plt.subplots(2, 10)
    i = j = 0
    for root, dirs, files in os.walk("samples"):
        for f in files:
            if f[0:8] == "points99":
                if random()>0.5 and i < 10 and clf.predict(HoG("samples/s_negtive/%s"%f)) == 1:
                    print f
                    axarr[0,i].imshow(imread("samples/s_negtive/%s"%f))
                    i = i + 1
                    
            elif f[0:8] == "points00":
                if random()>0.5 and j < 10 and clf.predict(HoG("samples/s_postive/%s"%f)) == 2:
                    print f                    
                    axarr[1,j].imshow(imread("samples/s_postive/%s"%f))
                    j = j + 1
            plt.show()
    
if __name__ == "__main__":
    # main()
    #random_check_hog(5)
    #random_check_bow(5)
    result = full_check_decaf(256, 512)
    plt.imshow(result)
    
    
    #with open('result_all.txt', 'w') as f:
    #    for k, v in result.items():
    #        if v != 0 and v[0] == 2:
    #            f.writelines("%s,%s\n"%k)

    #g_raster = gdal.Open('test.tif') # test.tif
    # plt.axis('off')
    # f, axarr = plt.subplots(n, n)
    #win_size = 256
    #win_slide = 64
    #cols = range(0, g_raster.RasterXSize - win_size, win_slide)
    #rows = range(0, g_raster.RasterYSize - win_size, win_slide)
    #arr = np.ones((len(rows), len(cols)))
    #for k, v in result.items():
    #    if v != 0 and v[0] == 2:
    #        arr[k[0]/win_slide, k[1]/win_slide] = v[0]
      