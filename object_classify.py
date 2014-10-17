# -*- coding: cp936 -*-
"""
Created on Fri Oct 17 19:51:20 2014

@author: shuaiyi
"""

import sys

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
    
# reading shp file
driver = ogr.GetDriverByName('ESRI Shapefile')

fn = "D:/DEM/disaster-project/trunk/420_decaf/segmentation/bbox.shp"
dataSource = driver.Open(fn, 0)
if dataSource is None:
    print 'Could not open ' + fn
    sys.exit(1) #exit with an error code

layer = dataSource.GetLayer(0)    
numFeatures = layer.GetFeatureCount()
print 'Feature count:', numFeatures

#f = layer.GetFeature(0)
#geo = f.GetGeometryRef()
#c = geo.Centroid()
#c.GetPoint()
#b = geo.Boundary()
#b.GetPoints()

# loop through the features and count them
cnt = 0
feature = layer.GetNextFeature()
while feature:
    cnt = cnt + 1
    feature.Destroy()
    feature = layer.GetNextFeature()
print 'There are ' + str(cnt) + ' features'

# close the data source
# dataSource.Destroy()