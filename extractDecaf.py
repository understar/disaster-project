# -*- coding: cp936 -*-
"""
Created on Tue Oct 14 18:58:10 2014

@author: shuaiyi
"""

import logging, os
from skimage import io
import numpy as np
from sklearn.manifold import TSNE

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
    pass

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
    extractFeatures(samples_list, net, 'fc6_cudanet_out')
    writeFeatures_labels('420_decaf/420_decaf')
    
    #import cv2
    #print 'prediction:', net.top_k_prediction(scores, 5)
    #visualize.draw_net_to_file(net._net, 'decafnet.png')
    #print 'Network structure written to decafnet.png'