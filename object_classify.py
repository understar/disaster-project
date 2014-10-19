# -*- coding: cp936 -*-
"""
Created on Fri Oct 17 19:51:20 2014
�����߼���
��ѡ������ֶα�ʶ�Ƿ�Ϊ�����ֺ���
1������ÿһ��region��ȡ��Ӿ���
2����ȡoffset�����㳤��ȡ���ߣ�
3�������Ͻ�Ϊ��㣬ȡ������ͼ�񣻣�һ�ַ����ǲ������Ƿ����֮�����ŵ�256*256��
4�����࣬Ϊ�ֶθ�ֵ

ע������⣺
��֤�����Ĳɼ�����ȷ��; xy�᲻Ҫ�����; ������ȷ
@author: shuaiyi
"""
import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning) 
#�Զ��徯�洦�����������о������ε�
def customwarn(message, category, filename, lineno, file=None, line=None):
    pass

warnings.showwarning = customwarn

import os, sys
import cv2 #ʹ��cv2��resize��ʵ��ͼ�������
import skimage.io as io #��дͼ��
from sklearn.externals import joblib

import progressbar # ������
#import logging
#logging.getLogger()


try:
    from osgeo import ogr, gdal
except:
    import ogr, gdal

def offset(ds, x, y):
    # get georeference info
    transform = ds.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = transform[5]

    # compute pixel offset
    xOffset = int((x - xOrigin) / pixelWidth)
    yOffset = int((y - yOrigin) / pixelHeight)
    return (xOffset, yOffset)

def is_exist(Layer,field_name):
    layerDefinition = Layer.GetLayerDefn()
    for i in range(layerDefinition.GetFieldCount()):
        if layerDefinition.GetFieldDefn(i).GetName() == field_name:
            return True
    return False

def getRegion(r, f): # raster feature
    geo = f.GetGeometryRef()
    lu_x, rd_x, rd_y, lu_y = geo.GetEnvelope()
    lu_offset_x, lu_offset_y = offset(r, lu_x, lu_y)
    rd_offset_x, rd_offset_y = offset(r, rd_x, rd_y)
    w = rd_offset_x-lu_offset_x
    h = rd_offset_y-lu_offset_y

    #print lu_offset_x ,lu_offset_y , w, h
    # һСƬ�����µ����������ԣ���Ҫ�ʵ�������չ
    # ��ȥ��Ե�⣬��������չ50pixel
    expand_size = 50
    if lu_offset_x - expand_size >= 0 and lu_offset_x + w + expand_size <= r.RasterXSize \
    and lu_offset_y - expand_size >= 0 and lu_offset_y + h + expand_size <= r.RasterYSize:
        lu_offset_x = lu_offset_x - expand_size
        lu_offset_y = lu_offset_y - expand_size
        w = w + 2*expand_size
        h = h + 2*expand_size
    img = r.ReadAsArray(lu_offset_x ,lu_offset_y , w, h)
    img = img.swapaxes(0,2).swapaxes(0,1)
    io.imsave("420_decaf/slide_target/%s_%s_%s_%s.png" % \
             (lu_offset_x, lu_offset_y, w, h), img)
    tmp = cv2.imread("420_decaf/slide_target/%s_%s_%s_%s.png" % \
                    (lu_offset_x, lu_offset_y, w, h))
    return cv2.resize(tmp, (256,256), interpolation=cv2.INTER_LINEAR)
    #return resize(img, (256,256))

# ���� decaf �� classifier
from decaf.scripts.imagenet import DecafNet
net = DecafNet()
clf = joblib.load("420_decaf/classifier_decaf.pkl")
blob_name='fc6_cudanet_out'

# ��ȡդ��ͼ��
g_raster = gdal.Open('20-21-22-part2.tif') # ��ָ��ļ���Ӧ��ԭʼդ��
    
# ��ȡ�ָ��� shp �ļ�
driver = ogr.GetDriverByName('ESRI Shapefile')
os.chdir("./420_decaf/segmentation")
fn = "20-21-22-part2.shp"
dataSource = driver.Open(fn, 1) # ��Ҫ��д
os.chdir(os.path.dirname(__file__))
if dataSource is None: 
    print 'Could not open ' + fn
    sys.exit(1) #exit with an error code

layer = dataSource.GetLayer(0)   

# ����ֶ� slide 
# ����Ѿ����ھͲ������
if not is_exist(layer, "slide"):
    fieldDefn = ogr.FieldDefn('slide', ogr.OFTInteger)
    layer.CreateField(fieldDefn)

numFeatures = layer.GetFeatureCount()
print 'Total region count:', numFeatures

#test
img = None
TEST = False
if TEST == True:
    feature = layer.GetNextFeature()
    img = getRegion(g_raster, feature)
    net.classify(img, True)
    tmp = net.feature(blob_name) #��ѵ��ʱ�򱣳�һ��
    is_slide = clf.predict(tmp)
    feature.SetField("slide", is_slide[0])    
else:
    # loop through the regions and predict them
    pbar = progressbar.ProgressBar(maxval=numFeatures).start()
    
    cnt = 0
    feature = layer.GetNextFeature()
    while feature:
        # ��ȡ��Ӧ��ͼ������
        img = getRegion(g_raster, feature)
        
        #imshow(img)
        #raw_input()
        
        net.classify(img, True)
        tmp = net.feature(blob_name) #��ѵ��ʱ�򱣳�һ��
        is_slide = clf.predict(tmp)
        feature.SetField("slide", is_slide[0])
        layer.SetFeature(feature) # ��һ���������ڱ����޸�
        pbar.update(cnt+1)
        cnt = cnt + 1
        feature = layer.GetNextFeature()
            
    pbar.finish()

#f = layer.GetFeature(0)
#geo = f.GetGeometryRef()
#c = geo.Centroid()
#c.GetPoint()
#b = geo.Boundary()
#b.GetPoints()



# close the data source
dataSource.Destroy()
