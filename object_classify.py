# -*- coding: cp936 -*-
"""
Created on Fri Oct 17 19:51:20 2014
程序逻辑：
可选：添加字段标识是否为滑坡灾害；
1、遍历每一个region；取外接矩形
2、获取offset，计算长宽，取长边；
3、以左上角为起点，取正方形图像；（一种方法是不关心是否变形之间缩放到256*256）
4、分类，为字段赋值

注意的问题：
保证样本的采集的正确性; xy轴不要搞错了; 样本正确
@author: shuaiyi
"""
import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning) 
#自定义警告处理函数，将所有警告屏蔽掉
def customwarn(message, category, filename, lineno, file=None, line=None):
    pass

warnings.showwarning = customwarn

import os, sys
import cv2 #使用cv2的resize，实现图像的缩放
import skimage.io as io #读写图像
from skimage.transform import resize
from skimage.util import img_as_ubyte

from sklearn.externals import joblib

import progressbar # 进度条
#import logging
#logging.getLogger()


try:
    from osgeo import ogr, gdal
except:
    import ogr, gdal

"""基于坐标值经纬度，以及栅格的信息，计算影像坐标
"""
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

"""判断shp中是否包含某一字段
"""
def is_exist(Layer,field_name):
    layerDefinition = Layer.GetLayerDefn()
    for i in range(layerDefinition.GetFieldCount()):
        if layerDefinition.GetFieldDefn(i).GetName() == field_name:
            return True
    return False

"""得到单个feature所在的raster范围影像数据
"""
def getRegion(r, f): # raster feature
    geo = f.GetGeometryRef()
    lu_x, rd_x, rd_y, lu_y = geo.GetEnvelope()
    lu_offset_x, lu_offset_y = offset(r, lu_x, lu_y)
    rd_offset_x, rd_offset_y = offset(r, rd_x, rd_y)
    w = rd_offset_x-lu_offset_x
    h = rd_offset_y-lu_offset_y

    # 计算center，比较w\h,获取合适的保持ratio的尺寸
    # 使用skimage的 resize，放弃cv2
    c_x, c_y = (lu_offset_x + rd_offset_x)/2, (lu_offset_y + rd_offset_y)/2
    large = w if w > h else h
    small = w if w <= h else h
    expand_size = 50 # 控制向外扩展的大小
    if c_x - large/2 -expand_size >= 0 and c_x + large/2 +expand_size <= r.RasterXSize \
    and c_y - large/2 -expand_size >= 0 and c_y + large/2 +expand_size <= r.RasterYSize:
        large = large + 2*expand_size
        small = large
  
    #print lu_offset_x ,lu_offset_y , w, h
    """ 一小片，滑坡的特征不明显；需要适当向外扩展
    # 除去边缘外，都向外扩展50pixel
    expand_size = 100
    if lu_offset_x - expand_size >= 0 and lu_offset_x + w + expand_size <= r.RasterXSize \
    and lu_offset_y - expand_size >= 0 and lu_offset_y + h + expand_size <= r.RasterYSize:
        lu_offset_x = lu_offset_x - expand_size
        lu_offset_y = lu_offset_y - expand_size
        w = w + 2*expand_size
        h = h + 2*expand_size
    """
    
    img = r.ReadAsArray(c_x - small/2 ,c_y - small/2, small-1, small-1)
    img = img.swapaxes(0,2).swapaxes(0,1)
    
    #show skimage
    #print img.dtype
    #plt.figure()
    #plt.imshow(img)
    
    return img_as_ubyte(resize(img, (256,256))) # resize 后是float64
    ''' 保存检测过程图像
    io.imsave("420_decaf/slide_target/%s_%s_%s_%s.png" % \
             (lu_offset_x, lu_offset_y, w, h), img)
    tmp = cv2.imread("420_decaf/slide_target/%s_%s_%s_%s.png" % \
                    (lu_offset_x, lu_offset_y, w, h))
    
    return cv2.resize(img, (256,256), interpolation=cv2.INTER_LINEAR)
    #return resize(img, (256,256))
    '''

# 加载 decaf 和 classifier
from decaf.scripts.imagenet import DecafNet
net = DecafNet()
clf = joblib.load("420_decaf/classifier.pkl")
blob_name='fc6_cudanet_out'


# 命令行参数情况
if len(sys.argv) < 3:
    print "usage: object_classify.py path_to_image path_to_segmentation_folder..."
    sys.exit()
else:
    img_path = sys.argv[1]
    segmentation_folder = sys.argv[2:]
# 读取栅格图像
g_raster = gdal.Open(img_path) # 与分割文件对应的原始栅格
    
# 读取分割结果 shp 文件
driver = ogr.GetDriverByName('ESRI Shapefile')
for folder in segmentation_folder:
    print "Process ./420_decaf/segmentation/%s" % folder
    os.chdir("./420_decaf/segmentation/%s" % folder)
    fn = "%s.shp" % img_path[img_path.rfind("/")+1:-4]
    dataSource = driver.Open(fn, 1) # 需要读写
    os.chdir(os.path.dirname(__file__)) # 重新回到启动目录
    if dataSource is None: 
        print 'Could not open ' + fn
        sys.exit(1) #exit with an error code
    
    layer = dataSource.GetLayer(0)   
    
    # 添加字段 slide 
    # 如果已经存在就不再添加
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
        
        # show
        #print img.dtype
        #plt.figure()
        #plt.imshow(img)
        
        net.classify(img, True)
        tmp = net.feature(blob_name) #与训练时候保持一致
        is_slide = clf.predict(tmp)
        feature.SetField("slide", is_slide[0])    
    else:
        # loop through the regions and predict them
        pbar = progressbar.ProgressBar(maxval=numFeatures).start()
        
        cnt = 0
        feature = layer.GetNextFeature()
        while feature:
            # 获取对应的图像样本
            img = getRegion(g_raster, feature)
            
            #imshow(img)
            #raw_input()
            
            net.classify(img, True)
            tmp = net.feature(blob_name) #与训练时候保持一致
            is_slide = clf.predict(tmp)
            feature.SetField("slide", is_slide[0])
            layer.SetFeature(feature) # 这一步可以用于保存修改
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
