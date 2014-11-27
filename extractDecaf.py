# -*- coding: cp936 -*-
"""
Created on Tue Oct 14 18:58:10 2014

@author: shuaiyi
"""

import logging, os
from skimage import io
import numpy as np
from sklearn.manifold import TSNE
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox 
import random

FEATURES = []
        
def extractFeatures(sample_list, net, blob_name):
    logging.info("Extract features.")
    for record in sample_list:
        print "Extract %s" % record[0]
        img = io.imread(record[0])
        # img = net.oversample(img,True) # center only
        # print net._net.predict(data = img).keys()
        net.classify(img, True)
        feature = net.feature(blob_name)
        FEATURES.append({\
            "name":record[0],\
            "feature":feature,\
            "label":record[1]})
    #return net._net.predict()[blob_name]

def visualization():
    # tsne visualization with picture
    logging.info('TSNE...')
    labels = []
    features = []
    for feature in FEATURES:
        labels.append(int(feature['label']))
        features.append(feature['feature'].ravel())
    labels = np.array(labels)
    features = np.array(features).astype(np.float)
    
    model = TSNE(n_components=2, random_state=0)
    f_tsne = model.fit_transform(features)
    
    for f, r in zip(FEATURES, f_tsne):
        f['tsne'] = (r[0], r[1])
    
    ax = plt.subplot(111)
    logging.info('Generating picture scatter')
    X = []
    for f in FEATURES:
        if random.randint(0,10) >= 5:
            img_path = f['name']
            logging.info('Processing %s'%img_path)
            xy = f['tsne']
            X.append(xy)
            arr_sam = io.imread(img_path) 
        
            imagebox = OffsetImage(arr_sam, zoom=0.1) 
            ab = AnnotationBbox(imagebox, xy, 
                                xycoords='data', 
                                pad=0, 
                                )
            ax.add_artist(ab) 

    X = np.array(X)
    ax.grid(True)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    plt.scatter(X[:,0], X[:,1], 20, labels)
    plt.draw()

def writeFeatures_labels(file_name):
    logging.info('Saving the features and labels.')
    labels = []
    features = []
    for feature in FEATURES:
        labels.append(int(feature['label']))
        features.append(feature['feature'].ravel())
    labels = np.array(labels)
    features = np.array(features)
    np.save(file_name + '_X.npy', features)
    np.save(file_name + '_Y.npy', labels)
   
def load_samples(loc_dir="samples"):
    logging.info("Loading samples from directory (samples)")
    samples = []
    for root, dirs, files in os.walk(loc_dir):
        for f in files:
            print f
            if f[0:8] == "points99":
                samples.append(("samples/s_negtive/%s"%f, 1))
            elif f[0:8] == "points00":
                samples.append(("samples/s_postive/%s"%f, 2))
            else:
                pass
    return samples
       
if __name__ == "__main__":
    """decafnet."""
    from decaf.scripts.imagenet import DecafNet
    logging.getLogger().setLevel(logging.INFO)
    net = DecafNet()
    samples_list = load_samples()
    #extractFeatures(samples_list, net, 'fc6_cudanet_out')
    #writeFeatures_labels('420_decaf/420_decaf')
    visualization()
    joblib.dump({"420_decaf":FEATURES},"420_decaf/420pkl/420.pkl",compress=0) # 压缩导致内存不足
    #import cv2
    #print 'prediction:', net.top_k_prediction(scores, 5)
    #visualize.draw_net_to_file(net._net, 'decafnet.png')
    #print 'Network structure written to decafnet.png'