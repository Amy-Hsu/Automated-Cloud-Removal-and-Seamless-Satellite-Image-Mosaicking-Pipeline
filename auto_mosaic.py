# -*- coding: utf-8 -*-
"""
Seamless Mosaicking Pipeline for Satellite Imagery

This script automates the generation of a seamless, large-scale mosaic from 
overlapping satellite image tiles. It uses geometric intersection, sparse-graph 
Dijkstra's algorithm for optimal seamline generation, and SSIM-guided Poisson blending.

Key Dependencies: numpy, cv2, rasterio, networkx, geopandas, shapely.

@author: hsu
"""


import numpy as np
import cv2
from osgeo import gdal
import rasterio
import matplotlib.pyplot as plt
import networkx as nx
import scipy.ndimage
from rasterio import Affine as A
import geopandas as gpd
from os import listdir
from os.path import isfile, join
from operator import itemgetter   
import copy
from timeit import default_timer as timer  
import collections
import tqdm
import time
from scipy.spatial import distance
from rasterio.windows import Window
import os    
import pathlib
import json
import gc
from shapely import wkt
import glob
start = timer()

def addAtPos(matrix1, matrix2, xypos, inPlace=True):
    """
    Add matrix2 into matrix1 at position xypos (x,y), in-place or in new matrix.
    Handles matrix2 going off edges of matrix1. 
    """
    x, y = xypos
    h1, w1 = matrix1.shape
    h2, w2 = matrix2.shape

    x1min = max(0, x)
    y1min = max(0, y)
    x1max = max(min(x + w2, w1), 0)
    y1max = max(min(y + h2, h1), 0)
   
    x2min = max(0, -x)
    y2min = max(0, -y)
    x2max = min(-x + w1, w2)
    y2max = min(-y + h1, h2)
    if inPlace:      
        matrix1[y1min:y1max, x1min:x1max] += matrix2[y2min:y2max, x2min:x2max]
    else:       
        matrix1copy = matrix1.copy()        
        matrix1copy[y1min:y1max, x1min:x1max] += matrix2[y2min:y2max, x2min:x2max]
        return matrix1copy

def spiral(X=50, Y=50):
    x = y = 0
    dx = 0
    dy = -1
    spiralList=[]
    for i in range(max(X, Y)**2):
        if (-X/2 < x <= X/2) and (-Y/2 < y <= Y/2):
          
            spiralList.append([x, y])
          
        if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
            dx, dy = -dy, dx
        x, y = x+dx, y+dy        
    return spiralList
def testOverlap(ReadOrder):
    for i in range(1,len(ReadOrder)-1):
        if 'z' not in globals():   
            image1_ds = rasterio.open(ReadOrder[0],nodata=0)
            image2_ds = rasterio.open(ReadOrder[1],nodata=0)
            gt1=image1_ds.transform
            gt2=image2_ds.transform           
            r1 = [gt1[2], gt1[5], gt1[2] + (gt1[0] * image1_ds.width), gt1[5] + (gt1[4] * image1_ds.height)]
            r2 = [gt2[2], gt2[5], gt2[2] + (gt2[0] * image2_ds.width), gt2[5] + (gt2[4] * image2_ds.height)]
        
            # find intersection between bounding boxes
            intersection = [max(r1[0], r2[0]), min(r1[1], r2[1]), min(r1[2], r2[2]), max(r1[3], r2[3])]
            if r1 != r2:        
                if (intersection[2] < intersection[0]) or (intersection[1] < intersection[3]):
                    intersection = None
                    print ('\t***no overlap***')                                
                    ReadOrder[1], ReadOrder[i+1] = ReadOrder[i+1], ReadOrder[1]  
                    
                else:
                    z=0
                    return True,ReadOrder                      
        else:
            z=0
            return True  ,ReadOrder   
                
def findRasterIntersect(raster1,raster2,res,scale_factor):
  
    image1_ds = rasterio.open(raster1,nodata=0)
    image2_ds = rasterio.open(raster2,nodata=0)
    gt1=image1_ds.transform
    gt2=image2_ds.transform
   
    r1 = [gt1[2], gt1[5], gt1[2] + (gt1[0] * image1_ds.width), gt1[5] + (gt1[4] * image1_ds.height)]
    r2 = [gt2[2], gt2[5], gt2[2] + (gt2[0] * image2_ds.width), gt2[5] + (gt2[4] * image2_ds.height)]
    print ('\t1 bounding box: %s' % str(r1))
    print ('\t2 bounding box: %s' % str(r2))

  
    intersection = [max(r1[0], r2[0]), min(r1[1], r2[1]), min(r1[2], r2[2]), max(r1[3], r2[3])]
    if r1 != r2:
        print ('\t** different bounding boxes **')
       
        if (intersection[2] < intersection[0]) or (intersection[1] < intersection[3]):
            intersection = None
            print ('\t***no overlap***')
           
        else:
            print ('\tintersection:',intersection)
            left1 = int(round((intersection[0]-r1[0])/gt1[0])) # difference divided by pixel dimension
            top1 = int(round((intersection[1]-r1[1])/gt1[4]))
            col1 = int(round((intersection[2]-r1[0])/gt1[0])) - left1 # difference minus offset left
            row1 = int(round((intersection[3]-r1[1])/gt1[4])) - top1

            left2 = int(round((intersection[0]-r2[0])/gt2[0])) # difference divided by pixel dimension
            top2 = int(round((intersection[1]-r2[1])/gt2[4]))
            col2 = int(round((intersection[2]-r2[0])/gt2[0])) - left2 # difference minus new left offset
            row2 = int(round((intersection[3]-r2[1])/gt2[4])) - top2
           
            if col1 != col2 or row1 != row2:
                print ("*** MEGA ERROR *** COLS and ROWS DO NOT MATCH ***")
            window1 = rasterio.windows.Window(left1,top1,col1,row1)
            window2 = rasterio.windows.Window(left2,top2,col2,row2)           
            array1b = np.dstack((image1_ds.read(3, window=window1),image1_ds.read(2, window=window1),image1_ds.read(1, window=window1)))  
            array2b = np.dstack((image2_ds.read(3, window=window2),image2_ds.read(2, window=window2),image2_ds.read(1, window=window2)))
                       
    else: 
        col1 = image1_ds.width # = col2
        row1 = image1_ds.height # = row2
        array1b = np.dstack((image1_ds.read(3),image1_ds.read(2),image1_ds.read(1)))
        array2b = np.dstack((image2_ds.read(3),image2_ds.read(2),image2_ds.read(1)))
        
    array1 = scipy.ndimage.zoom(array1b,(scale_factor,scale_factor,1), order=1,) #grid_mode=True
    array2 = scipy.ndimage.zoom(array2b,(scale_factor,scale_factor,1), order=1, )#grid_mode=True
    del array1b,array2b
    gc.collect()
    col1=round(col1*scale_factor)
    row1=round(row1*scale_factor)
    gt = A.translation(max(r1[0], r2[0]), min(r1[1], r2[1])) *  A.scale(res, -res) * image1_ds.transform.scale(
        (image1_ds.width / image1_ds.width),#image1_d.shape[-1]),
        (image1_ds.height / image1_ds.height)# image1_d.shape[-2])
    )   
  
      
    for i in [raster1,raster2]:
        a1=pathlib.Path(i).absolute()
        a2=pathlib.Path('NSPO_GRID_vectorization\GetBoundaryMask.exe').parent.absolute()
        filepath = str(a2)+ os.path.join('\\',i[:-3]+'temp.shp')
        if os.path.isfile(filepath) :
            print('done img2shp:'+i)
        else:

        
            cmdIn='gdal_translate -a_srs EPSG:3826 -of GTiff '+''.join(str(a1)) +' ' +''.join(str(a2))  +  os.path.join('\\', i[:-3]+'tif')
            ret =  os.popen(cmdIn).read()
    
            cmdIn='call '+''.join(str(a2))+ os.path.join('\\','BndPolygonize_v.bat ')+''.join(str(a2))  +  os.path.join('\\', i[:-3]+'tif')
           
            ret = os.popen(cmdIn).read()  
            
            filepath = str(a2)+ os.path.join('\\',i[:-3]+'temp.shp')
            if os.path.isfile(filepath) :
                print('done img2shp:'+i)
            else:
                print('oh! oh! memmory may not enough...')
                time.sleep(1)
            
            
    shapefile1 = gpd.read_file("NSPO_GRID_vectorization\\"+raster1[:-3]+'temp.shp')
    shapefile2 = gpd.read_file("NSPO_GRID_vectorization\\"+raster2[:-3]+'temp.shp')
    
    
    if len(shapefile1)>1 and len(shapefile2)==1:
        shapefile1["area"] = shapefile1.area/ 10**6
        a1=np.argsort(shapefile1["area"] )
        
        geom1 = shapefile1.geometry[a1[(shapefile1.shape[0]-1)]]   
        geom2 =shapefile2.geometry[0] 


    elif len(shapefile2)>1 and len(shapefile1)==1 :
        shapefile2["area"] = shapefile2.area/ 10**6
        a1=np.argsort(shapefile2["area"] )
        
        geom2 = shapefile2.geometry[a1[(shapefile2.shape[0]-1)]]   
        geom1 =shapefile1.geometry[0]
        
    elif len(shapefile2)>1 and len(shapefile1)>1 :
        shapefile2["area"] = shapefile2.area/ 10**6
        a1=np.argsort(shapefile2["area"] )
        
        geom2 = shapefile2.geometry[a1[(shapefile2.shape[0]-1)]]   
        
        
        shapefile1["area"] = shapefile1.area/ 10**6
        a1=np.argsort(shapefile1["area"] )
        
        geom1 = shapefile1.geometry[a1[(shapefile1.shape[0]-1)]]           
    else:
        geom1 = shapefile1.geometry[0]
        geom2 =shapefile2.geometry[0]
        
        
    res_union =gpd.overlay(shapefile1, shapefile2, how='union')
    res_union['new_column'] = 0
    res_union_new = res_union.dissolve(by='new_column')
     
    points = geom1.boundary.intersection(geom2.boundary) 
    try:
        points = list(points.geoms) # 2024/7/31 updated
    except:
        print("There may be no overlap. Check!")
    intersection_points=[]       
    for i in range(len(points)):
        intersection_points.append(points[i].wkt )
    
 
    if len(intersection_points)>2:        
              
        res_diff=gpd.overlay(shapefile2, shapefile1,  how='difference')
       
        res_diff_sep = res_diff['geometry'].values[0]

        res_diff_sep_gdf =gpd.GeoDataFrame({'geometry': gpd.GeoSeries(res_diff_sep)})# 2024/7/31 updated
        res_diff_sep_gdf.columns = ['geometry']
      
       
        res_diff_sep_gdf["area"] = res_diff_sep_gdf.area/ 10**6
        a=np.argsort(res_diff_sep_gdf["area"] )
      
        fin=res_diff_sep_gdf['geometry'][[a[(res_diff_sep_gdf.shape[0]-1)]]]
        fin=gpd.GeoDataFrame(fin)
       
        flist=np.zeros((len(intersection_points),1))
       
        for i in range(len(intersection_points)):
            aq=wkt.loads(points[i].wkt)    
            flist[i]=fin.distance(aq)# 
            aw=np.argsort(flist[:,0])
        intersection_pointsN=[]
        for i in range(2):
            intersection_pointsN.append(intersection_points[int(aw[i])])
        intersection_points=[]
        intersection_points=intersection_pointsN
        
        
        if  len(res_diff_sep_gdf)>1:
            fin=fin.set_crs(res_union_new.crs)      
            deval = res_diff.columns.str.contains('DN_')
            if any(deval):
                res_diff.loc[:,deval ]
                res_diff.drop(res_union_new.loc[:,deval ],inplace=True,axis=1)
            
            res_union2 =gpd.overlay(res_diff, fin, how='difference')
           
            res_union_new=gpd.overlay(res_union_new, res_union2, how='difference') 
            deval = res_union_new.columns.str.contains('DN_')
            if any(deval):
                res_union_new.loc[:,deval ]
                res_union_new.drop(res_union_new.loc[:,deval ],inplace=True,axis=1)
            
            
    
    del shapefile1,shapefile2,geom1,geom2,points,res_union
    union_it = [min(r1[0], r2[0]), max(r1[1], r2[1]), max(r1[2], r2[2]), min(r1[3], r2[3])]#boundbox for union_m
    union_m=np.zeros(( int((max(r1[1], r2[1])-min(r1[3], r2[3]))/res)  ,  int((max(r1[2], r2[2])-min(r1[0], r2[0]))/res)   ))
    union_gt = A.translation(min(r1[0], r2[0]), max(r1[1], r2[1])) *  A.scale(res, -res) 
   
    xypos1=( int( (gt1[2]-union_it[0]) /(res ))  , int( (union_it[1]-gt1[5])/ (res )    )  )
    xypos2=( int( (gt2[2]-union_it[0]) /(res ))  , int( (union_it[1]-gt2[5])/(res )     )  )
   
    u_con=addAtPos(union_m, image1_ds.read(1), xypos1, inPlace=False)
  
    uni_con=addAtPos(u_con, image2_ds.read(1), xypos2, inPlace=False)
    del u_con,union_it#
    gc.collect()
    uni_conn=np.array(uni_con)
    del uni_con
    gc.collect()
    uni_conn[np.where(uni_conn>0)]=1
    union_conn = scipy.ndimage.zoom(uni_conn,(scale_factor,scale_factor), order=1, ) #grid_mode=True# may still have value between 0~1 after interpolation.
    res_union_new.to_file('res_union_new.shp') 
    return array1, array2, col1, row1, intersection,gt,intersection_points,union_gt,union_m,union_conn,res_union_new#,image1_dsf,image2_dsf
   
def BlendPreprocessForOrder(image1_ds,mask_blurred_4chan,res,union_gt):
    image1_ds = rasterio.open(image1_ds)
    b2=image1_ds.read(1)
   
    b1 = np.zeros((mask_blurred_4chan.shape[0],mask_blurred_4chan.shape[1]))    
    
    gt1=image1_ds.transform
      
    xypos1=( int((gt1[2]-union_gt[2])/res) , int((union_gt[5]-gt1[5])/res)  )  
    addAtPos(b1, b2, xypos1, inPlace=True)
    
    im1_0num=round(np.count_nonzero( (mask_blurred_4chan==0) & (b1>0)    ) / np.count_nonzero(mask_blurred_4chan==0),9)
    im1_1num=round(np.count_nonzero( (mask_blurred_4chan==1) & (b1>0)    ) / np.count_nonzero(mask_blurred_4chan==1),9)
    return b1, im1_0num,im1_1num

def MskBlendPreprocessForOrder(image1_ds,mask_blurred_4chan,res,union_gt):
    image1_ds = rasterio.open(image1_ds)
    b2=image1_ds.read(1)#
    b2[np.where(b2<=2)]=0
    b2[np.where(b2>2)]=1

    b1 = np.zeros((mask_blurred_4chan.shape[0],mask_blurred_4chan.shape[1]))    
    
    gt1=image1_ds.transform
    xypos1=( int((gt1[2]-union_gt[2])/res) , int((union_gt[5]-gt1[5])/res)  )  
    addAtPos(b1, b2, xypos1, inPlace=True)    
    
    return b1#
def MskBlendPreprocessForOrderLoop(image1_ds,mask_blurred_4chan,res,union_gt,window):
    image1_ds = rasterio.open(image1_ds,window=window)
    b2=image1_ds.read(1)#
    b1 = np.zeros((mask_blurred_4chan.shape[0],mask_blurred_4chan.shape[1]))   
    
    gt1=image1_ds.transform
    xypos1=( int((gt1[2]-union_gt[2])/res) , int((union_gt[5]-gt1[5])/res)  )  
    addAtPos(b1, b2, xypos1, inPlace=True)    
    return b1

def BlendPreprocess(i,image1_ds,mask_blurred_4chan,res,union_gt):
    image1_ds = rasterio.open(image1_ds)
    b2=image1_ds.read(i)#
  
    b1 = np.zeros((mask_blurred_4chan.shape[0],mask_blurred_4chan.shape[1]))    
    
    gt1=image1_ds.transform
     
    xypos1=( int((gt1[2]-union_gt[2])/res) , int((union_gt[5]-gt1[5])/res)  )  
    addAtPos(b1, b2, xypos1, inPlace=True)
    
  

    return b1


def BlendPreprocessOrderForMosic(image1_ds,image2_ds,mask_blurred_4chan,write_window):
    for id in [image1_ds,image2_ds]:
        if id ==image1_ds:
            ig = rasterio.open(id)
            b2=ig.read(1)
           
            b1= np.zeros((mask_blurred_4chan.shape[0],mask_blurred_4chan.shape[1]))    
            
            gt1=ig.transform
            xypos1=( int((gt1[2]-union_gt[2])/res) , int((union_gt[5]-gt1[5])/res)  )  
            addAtPos(b1, b2, xypos1, inPlace=True)
                 
       
        else:                
            img = rasterio.open(id)
            b2d=img.read(1,window=write_window)
          
            b1d =np.zeros((mask_blurred_4chan.shape[0],mask_blurred_4chan.shape[1]))   #np.zeros((5,5,5), dtype=np.int) # a 5x5x5 matrix of zeroes
           
            addAtPos(b1d, b2d, xypos1, inPlace=True)            
          
    im1_0num=round(np.count_nonzero( (mask_blurred_4chan==0) & (b1>0)    ) / sum(sum(mask_blurred_4chan==0)),9)
    im1_1num=round(np.count_nonzero( (mask_blurred_4chan==1) & (b1>0)    ) / sum(sum(mask_blurred_4chan==1)),9)
    im2_0num=round(np.count_nonzero( (mask_blurred_4chan==0) & (b1d>0)    ) / sum(sum(mask_blurred_4chan==0)),9)
    im2_1num=round(np.count_nonzero( (mask_blurred_4chan==1) & (b1d>0)    ) / sum(sum(mask_blurred_4chan==1)),9)
             
    return b1,b1d,im1_0num,im1_1num,im2_0num,im2_1num

def BlendPreprocessForMosic(ind,image1_ds,image2_ds,mask_blurred_4chan,write_window):
    for id in [image1_ds,image2_ds]:
        if id ==image1_ds:
            ig = rasterio.open(id)
            b2=ig.read(ind)
            b1= np.zeros((mask_blurred_4chan.shape[0],mask_blurred_4chan.shape[1]))    

            gt1=ig.transform
            del ig
            xypos1=( int((gt1[2]-union_gt[2])/res) , int((union_gt[5]-gt1[5])/res)  )  
            addAtPos(b1, b2, xypos1, inPlace=True)
            del b2,gt1
            gc.collect()
        else:                
            img = rasterio.open(id)
            b2d=img.read(ind,window=write_window)
            del img
           
            b1d =np.zeros((mask_blurred_4chan.shape[0],mask_blurred_4chan.shape[1]))#np.zeros((5,5,5), dtype=np.int) # a 5x5x5 matrix of zeroes
            
            addAtPos(b1d, b2d, xypos1, inPlace=True)  
            del b2d,xypos1
            gc.collect()
    return b1,b1d


def bfs(grid, start,wall, clear, goal,width, height):
    queue = collections.deque([[start]])#
    seen = set([start])
    while queue:
        path = queue.popleft()#
        x, y = path[-1]
        if grid[y][x] == goal:
            return path
        for x2, y2 in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):#4 directions
            if 0 <= x2 < width and 0 <= y2 < height and grid[y2][x2] != wall and (x2, y2) not in seen:
                queue.append(path + [(x2, y2)])
                seen.add((x2, y2))
                
def BorderConnection(seamlineff,gt , union_gt,res,scale_factor,fc,fr ):

    
    height=seamlineff.shape[0]
    width=seamlineff.shape[1]
    
    seamlineff3=seamlineff.copy()
    seamlineff3[np.where(seamlineff<2)]=0
    seamlineff3[np.where(seamlineff==2)]=1
    
    x=round((gt[2]-union_gt[2])/(res/scale_factor) ) # col
    y=round((union_gt[5]-gt[5])/(res/scale_factor) ) # row
    colPoint=[fc[0]+x,  fc[1]+x]#start from 0
    rowPoint=[fr[0]+y,  fr[1]+y]#
    for v in range(len(colPoint)):# 
        i=int(rowPoint[v])
        j=int(colPoint[v]) 
        if 'aa' in locals():
            del aa
        if 'ab' in locals():
            del ab
        if i==seamlineff.shape[0]-1 or j==seamlineff.shape[1]-1 or i==seamlineff.shape[0] or j==seamlineff.shape[1]: # check if it's already connected to boder
            pass
        elif i==0 or j==0: # 
            pass         
            
        elif  (  i!=0 and j!=0 and seamlineff[i,j]!=0 and 
             seamlineff[i-1,j-1]!=0 and seamlineff[i-1,j]!=0 and
        seamlineff[i-1,j+1]!=0 and seamlineff[i,j-1]!=0 and seamlineff[i,j+1]!=0 and
        seamlineff[i+1,j-1]!=0 and seamlineff[i+1,j]!=0 and seamlineff[i+1,j+1]!=0 ):
            
           
            spiralList=spiral() 
            seamlineff2=seamlineff.copy()
            seamlineff2[np.where(seamlineff>0)]=-1
            for n in range(len(spiralList)) :#deal with the "1 area".
                if (0<=(i+ spiralList[n][1])<height and  0<= j+ spiralList[n][0]<width  ):
                    try:
                        if ( seamlineff[ i+ spiralList[n][1] ,j+ spiralList[n][0]  ] ==0 and 'ab' not in locals() ) :                    
                            posi=[ i+ spiralList[n][1] ,j+ spiralList[n][0]  ] 
                            ab=0
                            if i>posi[0]:            
                                if j>posi[1]:
                                    seamlineff3[posi[0]:i,j]=1                 
                                    seamlineff3[posi[0],posi[1]:j]=1
                                elif j<posi[1]:
                                    seamlineff3[posi[0]:i,j]=1                 
                                    seamlineff3[posi[0],j:posi[1]+1]=1#
                                    
                                else:
                                    seamlineff3[posi[0]:i,j]=1                 
                                   
                                    
                                    
                            elif i==posi[0]:             
                                if j>posi[1]:
                                    seamlineff3[i,posi[1]:j]=1
                                    
                                elif j<posi[1]:
                                    seamlineff3[i,j:posi[1]+1]=1                
                                else:
                                    print('imposiible case occur!')
                            else:            
                                if j>posi[1]:
                                    seamlineff3[i:posi[0]+1,j]=1                 
                                    seamlineff3[posi[0],posi[1]:j]=1               
                                    
                                elif j<posi[1]:
                                    seamlineff3[i:posi[0]+1,j]=1                 
                                    seamlineff3[posi[0],j:posi[1]+1]=1
                                else:
                                    seamlineff3[i:posi[0]+1,j]=1                 
                                    
                                 
                            i_tolerance=np.linspace(posi[0]-100,posi[0]+100,(posi[0]+100)-(posi[0]-100)+1)
                            j_tolerance=np.linspace(posi[1]-100,posi[1]+100,(posi[1]+100)-(posi[1]-100)+1)
                            i_t=[]
                            j_t=[]
                            i_t[:] = [r for r in i_tolerance if 0 <= r < height]
                            j_t[:] = [c for c in j_tolerance if 0 <= c < width]
                            
                            
                         
                            topList=list(zip(np.zeros(len(j_t)),j_t))
                            bottomList=list(zip((np.ones(len(j_t))*(height-1)),j_t))
                            rightList=list(zip(i_t,np.ones(len(i_t))*(width-1)))
                            leftList=list(zip(i_t,np.zeros(len(i_t))))                         
                            all_list=[]
                            if width-posi[0]+1>posi[0] and posi[1]<height-posi[1]+1:
                                allList=topList+leftList+rightList+bottomList 
                                all_list[:] = [r for r in range(len(allList)) if seamlineff[int(allList[r][0]),int( allList[r][1]) ] ==0]
                                                             
                            elif  width-posi[0]+1<posi[0] and posi[1]<height-posi[1]+1:
                                allList=topList+rightList+bottomList+leftList
                                all_list[:] = [r for r in range(len(allList)) if seamlineff[int(allList[r][0]),int( allList[r][1]) ] ==0]
                                
                            elif  width-posi[0]+1<posi[0] and posi[1]>height-posi[1]+1:
                                allList=bottomList+rightList+topList+leftList
                                all_list[:] = [r for r in range(len(allList)) if seamlineff[int(allList[r][0]),int( allList[r][1]) ] ==0]
                              
                            elif width-posi[0]+1>posi[0] and posi[1]>height-posi[1]+1:
                                allList=bottomList+leftList+topList+rightList
                                all_list[:] = [r for r in range(len(allList)) if seamlineff[int(allList[r][0]),int( allList[r][1]) ] ==0]
                                                      
                            else:
                                allList=topList+bottomList+rightList+leftList
                                all_list[:] = [r for r in range(len(allList)) if seamlineff[int(allList[r][0]),int( allList[r][1]) ] ==0]
                                    
                            
                            seamlineff2=seamlineff.copy()
                            seamlineff2[np.where(seamlineff>0)]=-1
                            wall, clear, goal = -1, 0, 1000
                            width, height = seamlineff.shape[1], seamlineff.shape[0]    
                              
                            for r in range(len(all_list)):
                                seamlineff2[ int(allList[ all_list[r]][0]) , int(allList[ all_list[r]][1])  ]=1000                           
                                                   
                            path2 = bfs(seamlineff2, (posi[1],posi[0] ),wall, clear, goal,width, height)#col first,than row
                            
                                                    
                            for s in range(len(path2)):
                                seamlineff3[path2[s][1],path2[s][0]]=1                            
                        ab    
                    except:#
                        if ( seamlineff[ i+ spiralList[n][1] ,j+ spiralList[n][0]  ] ==1 and 'ab' not in locals() ) :                    
                            posi=[ i+ spiralList[n][1] ,j+ spiralList[n][0]  ] 
                            ab=0
                            if i>posi[0]:            
                                if j>posi[1]:
                                    seamlineff3[posi[0]:i,j]=1                 
                                    seamlineff3[posi[0],posi[1]:j]=1
                                elif j<posi[1]:
                                    seamlineff3[posi[0]:i,j]=1                 
                                    seamlineff3[posi[0],j:posi[1]+1]=1#
                                    
                                else:
                                    seamlineff3[posi[0]:i,j]=1         
                                    
                            elif i==posi[0]:             
                                if j>posi[1]:
                                    seamlineff3[i,posi[1]:j]=1
                                    
                                elif j<posi[1]:
                                    seamlineff3[i,j:posi[1]+1]=1                
                                else:
                                    print('imposiible case occur!')
                            else:            
                                if j>posi[1]:
                                    seamlineff3[i:posi[0]+1,j]=1                 
                                    seamlineff3[posi[0],posi[1]:j]=1               
                                    
                                elif j<posi[1]:
                                    seamlineff3[i:posi[0]+1,j]=1                 
                                    seamlineff3[posi[0],j:posi[1]+1]=1
                                else:
                                    seamlineff3[i:posi[0]+1,j]=1                 
                                    
                                 
                            i_tolerance=np.linspace(posi[0]-100,posi[0]+100,(posi[0]+100)-(posi[0]-100)+1)
                            j_tolerance=np.linspace(posi[1]-100,posi[1]+100,(posi[1]+100)-(posi[1]-100)+1)
                            i_t=[]
                            j_t=[]
                            i_t[:] = [r for r in i_tolerance if 0 <= r < height]
                            j_t[:] = [c for c in j_tolerance if 0 <= c < width]  
                          
                            topList=list(zip(np.zeros(len(j_t)),j_t))
                            bottomList=list(zip((np.ones(len(j_t))*(height-1)),j_t))
                            rightList=list(zip(i_t,np.ones(len(i_t))*(width-1)))
                            leftList=list(zip(i_t,np.zeros(len(i_t))))                         
                            all_list=[]
                            if width-posi[0]+1>posi[0] and posi[1]<height-posi[1]+1:
                                allList=topList+leftList+rightList+bottomList 
                                all_list[:] = [r for r in range(len(allList)) if seamlineff[int(allList[r][0]),int( allList[r][1]) ] ==0]
                                                             
                            elif  width-posi[0]+1<posi[0] and posi[1]<height-posi[1]+1:
                                allList=topList+rightList+bottomList+leftList
                                all_list[:] = [r for r in range(len(allList)) if seamlineff[int(allList[r][0]),int( allList[r][1]) ] ==0]
                                
                            elif  width-posi[0]+1<posi[0] and posi[1]>height-posi[1]+1:
                                allList=bottomList+rightList+topList+leftList
                                all_list[:] = [r for r in range(len(allList)) if seamlineff[int(allList[r][0]),int( allList[r][1]) ] ==0]
                              
                            elif width-posi[0]+1>posi[0] and posi[1]>height-posi[1]+1:
                                allList=bottomList+leftList+topList+rightList
                                all_list[:] = [r for r in range(len(allList)) if seamlineff[int(allList[r][0]),int( allList[r][1]) ] ==0]
                                                      
                            else:
                                allList=topList+bottomList+rightList+leftList
                                all_list[:] = [r for r in range(len(allList)) if seamlineff[int(allList[r][0]),int( allList[r][1]) ] ==0]
                                    
                            seamlineff2=seamlineff.copy()
                            seamlineff2[np.where(seamlineff>1)]=-1
                            wall, clear, goal = -1, 0, 1000
                            width, height = seamlineff.shape[1], seamlineff.shape[0]    
                              
                            for r in range(len(all_list)):
                                seamlineff2[ int(allList[ all_list[r]][0]) , int(allList[ all_list[r]][1])  ]=1000
                                seamlineff2[ 0,:  ]=1000
                                seamlineff2[ :,0  ]=1000
                                seamlineff2[ -1,:  ]=1000
                                seamlineff2[ :,-1  ]=1000
                                                  
                            path2 = bfs(seamlineff2, (posi[1],posi[0] ),wall, clear, goal,width, height)                            
                                                    
                            for s in range(len(path2)):
                                seamlineff3[path2[s][1],path2[s][0]]=1     
        elif seamlineff[i,j]!=0 and i!=0 and j!=0:            
            for k in [1,0,-1]:
                for l in [0,1,-1]:
                    if seamlineff[ i+ k ,j+ l  ] ==0 and 'aa' not in locals():
                        p1=[ i+ k ,j+ l ]#row ,col                        
                        aa=0
                        
                        i_tolerance=np.linspace(p1[0]-100,p1[0]+100,(p1[0]+100)-(p1[0]-100)+1)
                        j_tolerance=np.linspace(p1[1]-100,p1[1]+100,(p1[1]+100)-(p1[1]-100)+1)
                        i_t=[]
                        j_t=[]
                        i_t[:] = [r for r in i_tolerance if 0 <= r < height]
                        j_t[:] = [c for c in j_tolerance if 0 <= c < width]
                       
                        topList=list(zip(np.zeros(len(j_t)),j_t))
                        bottomList=list(zip((np.ones(len(j_t))*(height-1)),j_t))
                        rightList=list(zip(i_t,np.ones(len(i_t))*(width-1)))
                        leftList=list(zip(i_t,np.zeros(len(i_t))))                         
                        all_list=[]
                        if width-p1[0]+1>p1[0] and p1[1]<height-p1[1]+1:
                            allList=topList+leftList+rightList+bottomList 
                            all_list[:] = [r for r in range(len(allList)) if seamlineff[int(allList[r][0]),int( allList[r][1]) ] ==0]
                                                        
                        elif  width-p1[0]+1<p1[0] and p1[1]<height-p1[1]+1:
                            allList=topList+rightList+bottomList+leftList
                            all_list[:] = [r for r in range(len(allList)) if seamlineff[int(allList[r][0]),int( allList[r][1]) ] ==0]
                           
                        elif  width-p1[0]+1<p1[0] and p1[1]>height-p1[1]+1:
                            allList=bottomList+rightList+topList+leftList
                            all_list[:] = [r for r in range(len(allList)) if seamlineff[int(allList[r][0]),int( allList[r][1]) ] ==0]
                            
                        elif width-p1[0]+1>p1[0] and p1[1]>height-p1[1]+1:
                            allList=bottomList+leftList+topList+rightList
                            all_list[:] = [r for r in range(len(allList)) if seamlineff[int(allList[r][0]),int( allList[r][1]) ] ==0]
                                                  
                        else:
                            allList=topList+bottomList+rightList+leftList
                            all_list[:] = [r for r in range(len(allList)) if seamlineff[int(allList[r][0]),int( allList[r][1]) ] ==0]
                                                    
                        
                        seamlineff2=seamlineff.copy()
                        seamlineff2[np.where(seamlineff>0)]=-1
                        wall, clear, goal = -1, 0, 1000
                        width, height = seamlineff.shape[1], seamlineff.shape[0]    
                        
                        
                        for r in range(len(all_list)):
                            seamlineff2[ int(allList[ all_list[r]][0]) , int(allList[ all_list[r]][1])  ]=1000                       
                    
                        print((p1[1],p1[0] ))                      
                        path2 = bfs(seamlineff2, (p1[1],p1[0] ),wall, clear, goal,width, height)#col first,than row
                        
                    
                        for s in range(len(path2)):
                            seamlineff3[path2[s][1],path2[s][0]]=1      
                        
        else:           
            print('start or end_point finding is wrong!')
    
    return seamlineff3

def buildGraph(data_final,data_final2,height,width,path_method,trueID_intersect):
    G = nx.Graph()
    new_d=np.squeeze(np.asarray(data_final)).flatten()
    new_d2=np.squeeze(np.asarray(data_final2)).flatten()    
    algos_r={'a6c':'((1/(abs(new_d[x-1]-new_d[right-1]))) if new_d[x-1]-new_d[right-1]!=0 else 10000000000 ) + (1/(abs(new_d2[x-1]-new_d2[right-1]))if new_d2[x-1]-new_d2[right-1] !=0 else 10000000000 )',
             'a7c':'1/(abs(new_d[x-1]-new_d[right-1])+abs(new_d2[x-1]-new_d2[right-1])) if new_d[x-1]-new_d[right-1]!=0 or new_d2[x-1]-new_d2[right-1]!=0 else 10000000000 '
             }    
    
    algos_d={'a6c':'((1/(abs(new_d[x-1]-new_d[down-1]))) if new_d[x-1]-new_d[down-1]!=0 else 10000000000 ) + (1/(abs(new_d2[x-1]-new_d2[down-1]))if new_d2[x-1]-new_d2[down-1] !=0 else 10000000000 )',
             'a7c':'1/(abs(new_d[x-1]-new_d[down-1])+abs(new_d2[x-1]-new_d2[down-1])) if new_d[x-1]-new_d[down-1]!=0 or new_d2[x-1]-new_d2[down-1]!=0 else 10000000000',
             }    
    j=0            
    rightck=[]
    downck=[]
    xck=[]    
    for x in tqdm.tqdm(range(1,width*height+1)):
    
        conf1=new_d[x-1].tolist()
        conf2=new_d2[x-1].tolist()
        
        if x%width!=0 and x%height!=0 and x<(height-1)*width and conf1 !=0 and conf2 !=0 :
            j+=1
            right=x+1
            G.add_edge(x, right,  weight = eval(algos_r[path_method])   )
            down=x+width        
            G.add_edge(x, down,  weight = eval(algos_d[path_method])   )

            rightck.append(right) 
            xck.append(x) 
                       
        elif x%width==0 and x!=width*height and conf1 !=0 and conf2 !=0 :
            j+=1
            down=x+width         
           
            G.add_edge(x, down,  weight =eval(algos_d[path_method])   )
            
            xck.append(x) 
            downck.append(down)   
        elif x>(height-1)*width and  x!=width*height and conf1 !=0 and conf2 !=0: 
            j+=1
            right=x+1          
            G.add_edge(x, right,  weight =  eval(algos_r[path_method])   )
            
            rightck.append(right)      
            xck.append(x) 
            (np.array(xck[:])==5902489).any()
            
            ck=np.zeros((data_final.shape))
            for f in xck:
                rowck,colck=f//data_final.shape[1],f%data_final.shape[1]-1
        
                ck[rowck,colck]=1            
    source=trueID_intersect[0]
    target=trueID_intersect[1]
    
    path = nx.astar_path(G, source, target, heuristic=None, weight='weight')
    seamline=np.zeros((width*height), dtype=int)
    for i in path:
        seamline[i-1]=1        
    seamlinef=np.resize(seamline,(height,width))
    return seamlinef


def clipImg(usrset,imgg1_ui8_nd_gy,imgg2_ui8_nd_gy,fr,fc):

    pix0=int(usrset/2)
    data = np.array(imgg1_ui8_nd_gy)#.T
   
    msk=np.zeros((data.shape[0],data.shape[1]), np.uint8)
    msk[np.where( data>0   )]=1
    msk[:,-1]=0
    msk[:,0]=0
    msk[0,:]=0
    msk[-1,:]=0
    kernel = np.ones((usrset,usrset), np.uint8)
    erosion = cv2.erode(msk, kernel, iterations = 1)
  
    dataf=copy.deepcopy(data)
    dataf[erosion==0]=0
  
    height, width = imgg2_ui8_nd_gy.shape
    pix=pix0+30
    
    for i in range(0,2):
        if  fr[i]-pix>=0:
            up=fr[i]-pix
        else:
            up=0
        if  fr[i]+pix<=height-1:
            down=fr[i]+pix
        else:
            down=height-1
        if fc[i]-pix>=0:
            left=fc[i]-pix
        else:
            left=0
        if fc[i]+pix<=width-1:
            right=fc[i]+pix
        else:
            right=width-1
        
        dataf[up:down,left:right]=data[up:down,left:right]
    data_final= copy.deepcopy(dataf)
  
    
    data2 = np.array(imgg2_ui8_nd_gy)#.T    
    msk=np.zeros((data2.shape[0],data2.shape[1]), np.uint8)
    msk[np.where(data2>0 )]=1
    msk[:,-1]=0
    msk[:,0]=0
    msk[0,:]=0
    msk[-1,:]=0
    
    erosion = cv2.erode(msk, kernel, iterations = 1)    
  
    
    dataf=copy.deepcopy(data2)
    dataf[erosion==0]=0
    
    for i in range(0,2):
        if  fr[i]-pix>=0:
            up=fr[i]-pix
        else:
            up=0
        if  fr[i]+pix<=height-1:
            down=fr[i]+pix
        else:
            down=height-1
        if fc[i]-pix>=0:
            left=fc[i]-pix
        else:
            left=0
        if fc[i]+pix<=width-1:
            right=fc[i]+pix
        else:
            right=width-1
        dataf[up:down,left:right]=data2[up:down,left:right]#don't need to use addAtPos
   
    data_final2=dataf
    
    
    return data_final,data_final2,height, width

def findRasterIntersectForMosaic(raster1,raster2,MosTai_gt,res,scale_factor,res_union_new):       
    image1_ds = rasterio.open(raster1)     
    gt1=image1_ds.transform    
    image2_ds = rasterio.open(raster2,'r+')
   
    window=Window(  (gt1[2]-MosTai_gt[2])/res  , (MosTai_gt[5]-gt1[5])/res , image1_ds.shape[1], image1_ds.shape[0])
    r1 = [gt1[2], gt1[5], gt1[2] + (gt1[0] * image1_ds.width), gt1[5] + (gt1[4] * image1_ds.height)]
    
    col1 = image1_ds.width 
    row1 = image1_ds.height 
    array1 = scipy.ndimage.zoom(np.dstack((image1_ds.read(3),image1_ds.read(2),image1_ds.read(1))),(scale_factor,scale_factor,1), order=1,) 
    array2 = scipy.ndimage.zoom(np.dstack((image2_ds.read(3,window=window),image2_ds.read(2,window=window),image2_ds.read(1,window=window))),(scale_factor,scale_factor,1), order=1, )

    gt = A.translation(r1[0], r1[1]) *  A.scale(res, -res) * image1_ds.transform.scale(
        (image1_ds.width / image1_ds.width ), 
        (image1_ds.height / image1_ds.height ) 
    )  
    a1=pathlib.Path(raster1).absolute()
    a2=pathlib.Path('NSPO_GRID_vectorization\GetBoundaryMask.exe').parent.absolute() 
    filepath = str(a2)+ os.path.join('\\',raster1[:-3]+'temp.shp')
    if os.path.isfile(filepath) :
        print('done img2shp:'+raster1)
    else:       
        
        cmdIn='gdal_translate -a_srs EPSG:3826 -of GTiff '+''.join(str(a1)) +' ' +''.join(str(a2))  +  os.path.join('\\', raster1[:-3]+'tif')
        ret =  os.popen(cmdIn).read()
    
        cmdIn='call '+''.join(str(a2))+ os.path.join('\\','BndPolygonize_v.bat ')+''.join(str(a2))  +  os.path.join('\\', raster1[:-3]+'tif')
      
        ret = os.popen(cmdIn).read()  
        filepath = str(a2)+ os.path.join('\\',raster1[:-3]+'temp.shp')
        if os.path.isfile(filepath) :
            print('done img2shp:'+raster1)
        else:
            print('oh! oh! memmory may not enough...')
            time.sleep(1)
    
    shapefile1 = gpd.read_file("NSPO_GRID_vectorization\\"+raster1[:-3]+'temp.shp')

    
    if len(shapefile1)>1:
        shapefile1["area"] = shapefile1.area/ 10**6
        a1=np.argsort(shapefile1["area"] )

        geom1 = shapefile1.geometry[a1[(shapefile1.shape[0]-1)]]
        geom2 =res_union_new.geometry[0]
    else:
        geom1 = shapefile1.geometry[0]
        geom2 =res_union_new.geometry[0] 
    
   
    points = geom1.boundary.intersection(geom2.boundary)
    Lpoints = list(points) 
    del geom1,geom2
    gc.collect()
    intersection_points=[]       
    for i in range(len(Lpoints)):
        intersection_points.append(Lpoints[i].wkt )        
    
        
    if len(intersection_points)>2:        
       
        res_diff=gpd.overlay(shapefile1, res_union_new, how='difference')
       
        res_diff_sep = res_diff['geometry'].values[0]        
        res_diff_sep_gdf = gpd.GeoDataFrame(res_diff_sep)
        res_diff_sep_gdf.columns = ['geometry']      
      
        res_diff_sep_gdf["area"] = res_diff_sep_gdf.area/ 10**6
        a=np.argsort(res_diff_sep_gdf["area"] )
      
        fin=res_diff_sep_gdf['geometry'][[a[(res_diff_sep_gdf.shape[0]-1)]]]
        fin=gpd.GeoDataFrame(fin)
        flist=np.zeros((len(intersection_points),1))
       
        for i in range(len(intersection_points)):
            aq=wkt.loads(points[i].wkt)    
            flist[i]=fin.distance(aq)# 
            aw=np.argsort(flist[:,0])
        intersection_pointsN=[]
        for i in range(2):
            intersection_pointsN.append(intersection_points[int(aw[i])])
        intersection_points=[]
        intersection_points=intersection_pointsN                
        
        if len(res_diff_sep_gdf)>1:
            res_union =gpd.overlay(shapefile1, res_union_new, how='union')
            res_union['new_column'] = 0
            res_union_new = res_union.dissolve(by='new_column')
           
            deval = res_union_new.columns.str.contains('DN_')
            if any(deval):
                res_union_new.loc[:,deval ]
                res_union_new.drop(res_union_new.loc[:,deval ],inplace=True,axis=1)
           
            del deval 
            
            fin=fin.set_crs(res_union_new.crs)      
            deval = res_diff.columns.str.contains('DN_')
            if any(deval):
                res_diff.loc[:,deval ]
                res_diff.drop(res_union_new.loc[:,deval ],inplace=True,axis=1)
            
            res_union2 =gpd.overlay(res_diff, fin, how='difference')
         
            res_union_new=gpd.overlay(res_union_new, res_union2, how='difference') 
            deval = res_union_new.columns.str.contains('DN_')
            if any(deval):
                res_union_new.loc[:,deval ]
                res_union_new.drop(res_union_new.loc[:,deval ],inplace=True,axis=1)
                
        else:
            res_union =gpd.overlay(shapefile1, res_union_new, how='union')
            res_union['new_column'] = 0
            res_union_new = res_union.dissolve(by='new_column')
      
            deval = res_union_new.columns.str.contains('DN_')
            if any(deval):
                res_union_new.loc[:,deval ]
                res_union_new.drop(res_union_new.loc[:,deval ],inplace=True,axis=1)
           
            del deval   
            
        del res_diff,res_diff_sep,a,fin,flist,aq,aw,intersection_pointsN,res_diff_sep_gdf
        gc.collect()
    #-----------------------     
    else:
        res_union =gpd.overlay(shapefile1, res_union_new, how='union')
        res_union['new_column'] = 0
        res_union_new = res_union.dissolve(by='new_column')
      
        deval = res_union_new.columns.str.contains('DN_')
        if any(deval):
            res_union_new.loc[:,deval ]
            res_union_new.drop(res_union_new.loc[:,deval ],inplace=True,axis=1)
       
        del deval 
  
        
    union_it = [r1[0], r1[1], r1[2], r1[3]]#boundbox for union_m
    xypos1=( int( (gt1[2]-union_it[0]) /(res ))  , int( (union_it[1]-gt1[5])/ (res )    )  )
    del union_it,gt1
    gc.collect()
    union_m=np.zeros( (int((r1[1]-r1[3])/res)  ,  int((r1[2]-r1[0])/res))  , dtype=np.int32)
 
    u_con=addAtPos(union_m, image1_ds.read(1), xypos1, inPlace=False)
    uni_con=addAtPos(u_con, image2_ds.read(1,window=window), xypos1, inPlace=False) 

    del u_con,xypos1
    gc.collect()
    union_gt = A.translation(r1[0], r1[1]) *  A.scale(res, -res)      
    uni_conn=np.array(uni_con)
    del uni_con
    gc.collect()
    uni_conn[np.where(uni_conn>0)]=1
    union_conn = scipy.ndimage.zoom(uni_conn,(scale_factor,scale_factor), order=1, ) 
    res_union_new.to_file('res_union_new.shp') 
    return array1, array2, col1, row1, gt,intersection_points,union_gt,union_m,union_conn,window,res_union_new

def MakeTrueID(intersection_points,imgg1_ui8_nd_gy,imgg2_ui8_nd_gy,gt,res,scale_factor):
    height, width = imgg2_ui8_nd_gy.shape
    p_left=[]
    p_top=[]
    for i in range(len(intersection_points)):
        if intersection_points[i][0:5]=='POINT':
          
            p_l=intersection_points[i][7:13]
            p_t=intersection_points[i][14:-1]
            
            p_lf=''.join([t for t in p_l ])
            p_tp=''.join([t for t in p_t ])
            p_left.append(float(p_lf))
            p_top.append(float(p_tp))
        elif  intersection_points[i][0:10]=='LINESTRING':
            p_l=intersection_points[i][12:18]
            p_t=intersection_points[i][19:26]
            
            p_lf=''.join([t for t in p_l ])
            p_tp=''.join([t for t in p_t ])
            p_left.append(float(p_lf))
            p_top.append(float(p_tp))        
    
    
    LeftTop_overlapped=[gt[2],gt[5]]
    
    pl_2rc=[]
    pt_2rc=[]
    for i in range(len(p_left)):
        pl_2rc.append(((p_left[i]-LeftTop_overlapped[0])/(res)))
        pt_2rc.append(((LeftTop_overlapped[1]-p_top[i])/(res)))
    fc=[]
    fr=[]
    if (1/scale_factor)==1:
        trueID_intersect=[]
        for i in range(len(pl_2rc)):   
            c=round(pl_2rc[i])
            r=round(pt_2rc[i])
            if imgg1_ui8_nd_gy[r,c]!=0 and imgg2_ui8_nd_gy[r,c]!=0:
                trueID_intersect.append((r+1-1)*width+c+1)
            else:
             
                if 'aa' in locals():
                    del aa
                if 'ab' in locals():
                    del ab
                spiralList=spiral(35,35)            
                for n in range(len(spiralList)) :
                    if r+spiralList[n][1]<imgg1_ui8_nd_gy.shape[0] and c+spiralList[n][0]<imgg1_ui8_nd_gy.shape[1]:
                        if imgg1_ui8_nd_gy[r+spiralList[n][1],c+spiralList[n][0]]!=0 and imgg2_ui8_nd_gy[r+spiralList[n][1],c+spiralList[n][0]]!=0 and 'aa' not in locals()  :
                           trueID_intersect.append((r+spiralList[n][1]+1-1)*width+c+spiralList[n][0]+1)
                           r=r+spiralList[n][1]
                           c=c+spiralList[n][0]
                           aa=0
                          
            fc.append(c)
            fr.append(r)
    else:
       
        trueID_intersect=[]
        for i in range(len(pl_2rc)):   
            c=int(pl_2rc[i]/(1/scale_factor))
            r=int(pt_2rc[i]/(1/scale_factor)) 
            if c==imgg1_ui8_nd_gy.shape[1]:
                c=c-1                
            if r==imgg1_ui8_nd_gy.shape[0]:
                r=r-1                
            if imgg1_ui8_nd_gy[r,c]!=0 and imgg2_ui8_nd_gy[r,c]!=0:
                trueID_intersect.append((r+1-1)*width+c+1) 
                
            else:
              
                spiralList=spiral(35,35)     
                if 'aa' in locals():
                    del aa
                if 'ab' in locals():
                    del ab
                for n in range(len(spiralList)) :
                    if r+spiralList[n][1]<imgg1_ui8_nd_gy.shape[0] and c+spiralList[n][0]<imgg1_ui8_nd_gy.shape[1]:
                        if imgg1_ui8_nd_gy[r+spiralList[n][1],c+spiralList[n][0]]!=0 and imgg2_ui8_nd_gy[r+spiralList[n][1],c+spiralList[n][0]]!=0 and 'ab' not in locals()  :
                           trueID_intersect.append((r+spiralList[n][1]+1-1)*width+c+spiralList[n][0]+1)
                           r=r+spiralList[n][1]
                           c=c+spiralList[n][0]            
                           ab=0
                                   
            fc.append(c)# 
            fr.append(r)# 
               
    return fc ,fr, trueID_intersect

def blendMask(blendPix,im_floodfill_f):
    im_floodfill_ftest=im_floodfill_f.copy()
    im_floodfill_ftest[np.where(im_floodfill_f==1)]=(blendPix+1)
    mask_blurred  = cv2.GaussianBlur(im_floodfill_ftest,(blendPix,blendPix),0)
    mask_blurred_1chan = mask_blurred.astype('float32') / (blendPix+1)
    return mask_blurred_1chan

with open('para.txt', 'r') as f:
    datastore = json.load(f)

path_method=datastore["path_method"]
resampleTo=float(datastore["resampleTo"])
res=float(datastore["res"])
trueBits=int(datastore["trueBits"])
mypath=datastore["imgPath"]

kernel=int(datastore["kernel"])
pan_res=float(datastore["pan_res"])


scale_factor=1/resampleTo
del resampleTo


NOscale_factor=1/scale_factor
imgBits=(2**trueBits)/(2**8)
if trueBits==12:
    imgtype='uint16'
else:
    imgtype='uint'+str(trueBits)
    
files = listdir(mypath)
gt_two=[]
file=[]
gt_five=[]

for f in files:

  fullpath = join(mypath, f)

  if isfile(fullpath) and  f.endswith(".img"):

    image=rasterio.open(f)
    file.append(f)
    gt_two.append(image.transform[2] )
    gt_five.append(image.transform[5] )


coords=np.array([[gt_two[i],gt_five[i]] for i in range(len(gt_two))],dtype='float32')    

dist=np.squeeze(distance.cdist( coords,[coords[0]] , 'euclidean'))
distIndex=np.argsort(dist) 
ReadOrder=[file[i] for i in distIndex.tolist()]
print("Processing Order：", ReadOrder)

allWindow=[]
ImgInd=[]
Union_gt=[]
for i in range(len(ReadOrder)-1):   
    if i!=0:
        loop=i
        image1_ds = ReadOrder[i+1]
        image2_ds = 'MosTai.tif'
        print('Image Process Progress:',i+2,'/',len(ReadOrder))
        print('Image Preprocessing...')
        image1_isect_array, image2_isect_array, col, row, gt, intersection_points, union_gt, union_m, union_conn,ReadWindow,res_union_new=findRasterIntersectForMosaic(image1_ds, image2_ds,MosTai_gt,res,scale_factor,res_union_new)
        Union_gt.append(union_gt)
       
        imgg1=cv2.cvtColor(image1_isect_array,cv2.COLOR_RGB2GRAY)
        imgg2=cv2.cvtColor(image2_isect_array,cv2.COLOR_RGB2GRAY)
        
        imgg1_ui8=image1_isect_array/imgBits
        imgg2_ui8=image2_isect_array/imgBits
    
        imgg1_ui8.astype(np.uint8)
        imgg2_ui8.astype(np.uint8)
      
        imgg1_ui8_nd = np.array(imgg1_ui8, dtype=np.uint8)
        imgg2_ui8_nd = np.array(imgg2_ui8, dtype=np.uint8)
        
        imgg1_ui8_nd_gy=cv2.cvtColor(imgg1_ui8_nd,cv2.COLOR_RGB2GRAY)
        imgg2_ui8_nd_gy=cv2.cvtColor(imgg2_ui8_nd,cv2.COLOR_RGB2GRAY)
        del  imgg1,imgg2,imgg1_ui8,imgg2_ui8,imgg1_ui8_nd,imgg2_ui8_nd,image2_isect_array
        gc.collect()
        
        fc,fr,trueID_intersect=MakeTrueID(intersection_points,imgg1_ui8_nd_gy,imgg2_ui8_nd_gy,gt,res,scale_factor)
        if len(trueID_intersect)==2:
            try:
                data_final,data_final2,height, width=clipImg(kernel,imgg1_ui8_nd_gy,imgg2_ui8_nd_gy,fr,fc)
                seamlinef=buildGraph(data_final,data_final2,height,width,path_method,trueID_intersect)
            except:
                data_final,data_final2,height, width=clipImg(11,imgg1_ui8_nd_gy,imgg2_ui8_nd_gy,fr,fc)
                seamlinef=buildGraph(data_final,data_final2,height,width,path_method,trueID_intersect)
        else:
            print("please check the intersection of image: " +image1_ds+' and image: '+image2_ds)
        del  data_final,data_final2,imgg1_ui8_nd_gy,imgg2_ui8_nd_gy
        print('Generating seamline ...')
        x=round((gt[2]-union_gt[2])/(res/scale_factor) ) # col
        y=round((union_gt[5]-gt[5])/(res/scale_factor) ) # row
        xypos=(x,y)
        union_conn[union_conn>0]=1
        seamlineff=addAtPos(union_conn, seamlinef, xypos, inPlace=False)  
        seamlineff3=BorderConnection(seamlineff, gt, union_gt,res,scale_factor,fc,fr )
        
        seamlineff4 = cv2.resize(seamlineff3,(union_m.shape[1],union_m.shape[0]) ,interpolation=cv2.INTER_NEAREST)
               
        sf = cv2.resize(seamlineff4,(union_m.shape[1],union_m.shape[0]) ,interpolation=cv2.INTER_NEAREST)#np.resize =! cv2.resize
        del seamlineff,seamlineff3 #,seamlinef
        gc.collect()   
    
        if sf.shape[0]>65534 and (sf[60000:,:]==1).any():
            sff=sf[0:60000,:]
            sff2=sf[60000:,:]
            th, im_th = cv2.threshold(sff.astype(np.float64), 0, 1, cv2.THRESH_BINARY_INV)
            im_floodfill1 =im_th.copy().astype(np.uint8)   
            h, w = sff.shape[:2]
            mask = np.zeros((h+2, w+2), np.uint8)
            cv2.floodFill(im_floodfill1, mask, (1,1), 0)  
          
            th, im_th = cv2.threshold(sff2.astype(np.float64), 0, 1, cv2.THRESH_BINARY_INV)
            im_floodfill2 =im_th.copy().astype(np.uint8)   
            h, w = sff2.shape[:2]
            mask = np.zeros((h+2, w+2), np.uint8)
           
            if im_floodfill1[59999,0]==0:
                cv2.floodFill(im_floodfill2, mask, (1,1),  np.uint32(im_floodfill1[59999,0]).item() )  
            elif im_floodfill1[59999,im_floodfill1.shape[1]-1]==0 :
                cv2.floodFill(im_floodfill2, mask, (im_floodfill2.shape[1]-1,1),  np.uint32(im_floodfill1[59999,im_floodfill1.shape[1]-1]).item() )  
            else:
                print('floodfill process has something wrong!')
            
            h, w = sf.shape[:2]
            im_floodfill=np.zeros(sf.shape[:2], np.uint8)
            addAtPos(im_floodfill, im_floodfill1, (0,0), inPlace=True)
            addAtPos(im_floodfill, im_floodfill2, (0,60000), inPlace=True)
           
            del sff,sff2,im_floodfill1,im_floodfill2
            gc.collect()
        else:
            th, im_th = cv2.threshold(sf.astype(np.float64), 0, 1, cv2.THRESH_BINARY_INV)
            im_floodfill = im_th.copy().astype(np.uint8)    
         
            h, w = im_th.shape[:2]
            mask = np.zeros((h+2, w+2), np.uint8)
            cv2.floodFill(im_floodfill, mask, (1,1), 0)   
          
        num_objects, labels = cv2.connectedComponents(im_floodfill, connectivity=8)
        classN=[np.sum(labels==a) for a in range(num_objects)]
        classN_ind=np.argsort(classN) 
        classN_index=classN_ind.tolist()
        classN_index.reverse()
        
        cind=[]
        for i in classN_index:
            if classN[i]>1000:
                cind.append(i)
        for h in range(len(cind)): 
            if len(cind)==3:
                m=h+1
                if m>len(cind)-1:
                    m=h-2
                    if all(im_floodfill[np.where(labels==cind[h] )]==1) and all(im_floodfill[np.where(labels==cind[m] )]==1): 
                        im_floodfill_f=im_floodfill.copy()                                 
                        im_floodfill_f[np.where((labels!=cind[h]) &( labels!=cind[m]) )]=0
                else: 
                    if all(im_floodfill[np.where(labels==cind[h] )]==1) and all(im_floodfill[np.where(labels==cind[m] )]==1): 
                        im_floodfill_f=im_floodfill.copy()                                 
                        im_floodfill_f[np.where((labels!=cind[h]) &( labels!=cind[m]) )]=0        
            else:
                if all(im_floodfill[np.where(labels==cind[h])]==1):  
                    im_floodfill_f=im_floodfill.copy()                                 
                    im_floodfill_f[np.where(labels!=cind[h])]=0 
                    
        
        del th, im_th,mask,num_objects, labels,classN,classN_ind,classN_index,cind
        gc.collect()
        print('Saving seamline for:"',image1_ds[:-4],'"...')
        with rasterio.open(
            str(image1_ds[:-4])+'MosTai_sl.img',
            'w+',           
            height=sf.shape[0],
            width=sf.shape[1],
            count=1,
            dtype=rasterio.uint8,          
            crs= {'init': 'EPSG:3826'},#Initialize from a named CRS
            transform=union_gt,
        ) as dst:
            dst.write(sf.astype(rasterio.uint8), 1)
        del sf
        
        print('Mosaic "',image1_ds[:-4],'" images...')
        mask_blurred_4chan=blendMask(57,im_floodfill_f) 
        
        with rasterio.open(
            str(image1_ds[:-4])+'MosTai_sl_polygon.img',
            'w+',           
            height=mask_blurred_4chan.shape[0]*int(res/pan_res),
            width=mask_blurred_4chan.shape[1]*int(res/pan_res),
            count=1,
            dtype=rasterio.float32,          
            crs= {'init': 'EPSG:3826'},#   
            transform= A.translation(union_gt[2], union_gt[5]) * A.scale(pan_res, -pan_res) #  
        ) as dst:
            dst.write(cv2.resize(mask_blurred_4chan,(mask_blurred_4chan.shape[1]*int(res/pan_res) ,mask_blurred_4chan.shape[0]*int(res/pan_res)),interpolation=cv2.INTER_NEAREST).astype(rasterio.float32), 1)        
        
        del im_floodfill,im_floodfill_f        
        b1,b1d,im1_0num,im1_1num,im2_0num,im2_1num=BlendPreprocessOrderForMosic(image1_ds,image2_ds,mask_blurred_4chan,ReadWindow)        
        
        img=[(0,image1_ds,im1_0num, 'b1d','b1' ),
              (1,image1_ds,im1_1num, 'b1d','b1' ),
              (0,image2_ds,im2_0num, 'b1','b1d' ),
              (1,image2_ds,im2_1num, 'b1','b1d')]
        imgind=sorted(img,reverse=True,  key = itemgetter(0,2 ))   
        ImgInd.append(imgind)
       
        MosTai= rasterio.open(
                    'MosTai.tif',
                    'r+',
                  
                    height=int( ( float(max( gt_five))-  2400000   )/res ),
                    width= int( (   406000-  float(min( gt_five)   ))/res ),
                    count=4,
                    dtype=image1_isect_array.dtype ,
                  
                    crs= {'init': 'EPSG:3826'},#Initialize from a named CRS
                    transform=MosTai_gt)
        kwargs = MosTai.meta.copy()
        kwargs.update(BIGTIFF="IF_SAFER")
      
        
        # plt.imshow(b1d)        
        for k in range(4): 
            if k==0:
                try:
                    
                    write_window = Window(   (union_gt[2]-MosTai_gt[2])/res,  (MosTai_gt[5]-union_gt[5])/res  ,    union_m.shape[1], union_m.shape[0]     )# 
                    outcome= eval(imgind[0][3]).astype(imgtype) *(1- mask_blurred_4chan)+ eval(imgind[0][4]).astype(imgtype) *( mask_blurred_4chan)      
                    
                    outcome=outcome.astype(imgtype)
                    
                    
                 
                    print('Writing result:',image1_ds[:-4],'band:',k+1,'to disk...')
                    MosTai.write( outcome, indexes=k+1, window=write_window)
              
                    del outcome ,b1,b1d 
                    gc.collect()
                
                except:
                   
                    h=int(mask_blurred_4chan.shape[0]/2)
                    w=int(mask_blurred_4chan.shape[1]/2)
                    outcome= eval(imgind[0][3])[0:h,:].astype(imgtype) *(1- mask_blurred_4chan)[0:h,:]+ eval(imgind[0][4])[0:h,:].astype(imgtype) *( mask_blurred_4chan)[0:h,:]     
                    outcome=outcome.astype(imgtype)
                    write_window2 = Window(   (union_gt[2]-MosTai_gt[2])/res,  (MosTai_gt[5]-union_gt[5])/res  ,    outcome.shape[1], outcome.shape[0]     )# write_window.height = first, write_window.width = second
                   
                    print('Writing 1/2 result:',image1_ds[:-4],'band:',k+1,'to disk...')
                    MosTai.write( outcome, indexes=k+1, window=write_window2)
                    del outcome ,write_window2
                    gc.collect()
                    
                    outcome= eval(imgind[0][3])[h:,:].astype(imgtype) *(1- mask_blurred_4chan)[h:,:]+ eval(imgind[0][4])[h:,:].astype(imgtype) *( mask_blurred_4chan)[h:,:]     
                    outcome=outcome.astype(imgtype)
                    write_window3 = Window(   (union_gt[2]-MosTai_gt[2])/res,  (MosTai_gt[5]-union_gt[5])/res+h  ,    outcome.shape[1], outcome.shape[0]     )# write_window.height = first, write_window.width = second
                   
                    print('Writing 2/2 result:',image1_ds[:-4],'band:',k+1,'to disk...')
                    MosTai.write( outcome, indexes=k+1, window=write_window3)
             
                    del outcome ,b1,b1d ,write_window3
                    gc.collect()
                 
                
            else:
                
                ind=k+1
                b1,b1d=BlendPreprocessForMosic(ind,image1_ds,image2_ds,mask_blurred_4chan,ReadWindow)  
                    
                try:
                
                    outcome= eval(imgind[0][3]).astype(imgtype) *(1- mask_blurred_4chan)+ eval(imgind[0][4]).astype(imgtype) *( mask_blurred_4chan)      
                    outcome=outcome.astype(imgtype)
                    print('Writing result:',image1_ds[:-4],'band:',k+1,'to disk...')
                    MosTai.write( outcome, indexes=k+1, window=write_window)
            
                    del outcome, b1,b1d 
                    gc.collect()
                
                except:
                    
                   
                    h=int(mask_blurred_4chan.shape[0]/2)
                    w=int(mask_blurred_4chan.shape[1]/2)
                    outcome= eval(imgind[0][3])[0:h,:].astype(imgtype) *(1- mask_blurred_4chan)[0:h,:]+ eval(imgind[0][4])[0:h,:].astype(imgtype) *( mask_blurred_4chan)[0:h,:]      
                    outcome=outcome.astype(imgtype)
                    write_window2 = Window(   (union_gt[2]-MosTai_gt[2])/res,  (MosTai_gt[5]-union_gt[5])/res  ,    outcome.shape[1], outcome.shape[0]     )# write_window.height = first, write_window.width = second
                    print('Writing 1/2 result:',image1_ds[:-4],'band:',k+1,'to disk...')
                    MosTai.write( outcome, indexes=k+1, window=write_window2)
                  
                    del outcome,write_window2
                    
                   
                    outcome= eval(imgind[0][3])[h:,:].astype(imgtype) *(1- mask_blurred_4chan)[h:,:]+ eval(imgind[0][4])[h:,:].astype(imgtype) *( mask_blurred_4chan)[h:,:]    
                    outcome=outcome.astype(imgtype)
                    write_window3 = Window(   (union_gt[2]-MosTai_gt[2])/res,  (MosTai_gt[5]-union_gt[5])/res+h  ,    outcome.shape[1], outcome.shape[0]     )# write_window.height = first, write_window.width = second
                   
                    print('Writing 2/2 result:',image1_ds[:-4],'band:',k+1,'to disk...')
                    MosTai.write( outcome, indexes=k+1, window=write_window3)
            
                    del outcome, b1,b1d ,write_window3
                    gc.collect()       
                
        MosTai.close()
        
        TaiMas= rasterio.open(
            'TaiMas.tif',
            'r+',
           
            height= int( ( float(max( gt_five))-  2400000   )/res ),
            width=  int( (   406000-  float(min( gt_two)   ))/res ),
            count=1,
            dtype=image1_isect_array.dtype,
            
            crs= {'init': 'EPSG:3826'},
            transform=TaiMas_gt)
        kwargs = TaiMas.meta.copy()
        kwargs.update(BIGTIFF="IF_SAFER")
       
        cldList=glob.glob(''.join(str(pathlib.Path().absolute()))+'\\'+'SPOT67Cloud_Hsiao_Rou'+'\\'+'*'+image1_ds[:-4]+'*.tif')        
        b1=MskBlendPreprocessForOrder(cldList[0],mask_blurred_4chan,res,union_gt)     
       
        b1d=MskBlendPreprocessForOrderLoop('TaiMas.tif',mask_blurred_4chan,res,union_gt,ReadWindow)     
        
        cldMas=eval(imgind[0][3]) *(1- mask_blurred_4chan)+ eval(imgind[0][4]) *( mask_blurred_4chan)
        cldMas[np.where(cldMas>0)]=1

        cldMas=cldMas.astype(imgtype)
        print('Writing cloud mask to disk...')
        TaiMas.write( cldMas, indexes=1, window=write_window)          
        TaiMas.close()
        allWindow.append(write_window)
        del mask_blurred_4chan ,cldMas,cldList,b1,b1d    
        gc.collect()
    else:

        image1_ds =ReadOrder[i]
        image2_ds =ReadOrder[i+1]
        if (testOverlap(ReadOrder)[0]==True):
            image1_ds =ReadOrder[i]#
            image2_ds =ReadOrder[i+1]#'
            image1_isect_array, image2_isect_array, col, row, isect_bb, gt, intersection_points, union_gt, union_m, union_conn,res_union_new=findRasterIntersect(image1_ds, image2_ds,res,scale_factor)
            fp = open("ProcessOrder.txt", "w")         
            fp.write(str(ReadOrder) )     
            fp.close()
        else:
            image1_isect_array, image2_isect_array, col, row, isect_bb, gt, intersection_points, union_gt, union_m, union_conn,res_union_new=findRasterIntersect(image1_ds, image2_ds,res,scale_factor)
        Union_gt.append(union_gt)    
        MosTai_gt = A.translation(float(min( gt_two)), float(max( gt_five))) *  A.scale(res, -res) 
        TaiMas_gt = A.translation(float(min( gt_two)), float(max( gt_five))) *  A.scale(res, -res) 
       
        MosTai= rasterio.open(
            'MosTai.tif',
            'w+',
           
            height= int( ( float(max( gt_five))-  2400000   )/res ),
            width=  int( (   406000-  float(min( gt_two)   ))/res ),
            count=4,
            dtype=image1_isect_array.dtype,
          
            crs= {'init': 'EPSG:3826'},#
            transform=MosTai_gt)
        kwargs = MosTai.meta.copy()
        kwargs.update(BIGTIFF="IF_SAFER")
        
      
        TaiMas= rasterio.open(
            'TaiMas.tif',
            'w+',
           
            height= int( ( float(max( gt_five))-  2400000   )/res ),
            width=  int( (   406000-  float(min( gt_two)   ))/res ),
            count=1,
            dtype=image1_isect_array.dtype,
           
            crs= {'init': 'EPSG:3826'},#
            transform=TaiMas_gt)
        kwargs = TaiMas.meta.copy()
        kwargs.update(BIGTIFF="IF_SAFER")
        
                
     
        imgg1=cv2.cvtColor(image1_isect_array,cv2.COLOR_RGB2GRAY)
        imgg2=cv2.cvtColor(image2_isect_array,cv2.COLOR_RGB2GRAY)
        
       
        imgg1_ui8=image1_isect_array/imgBits
        imgg2_ui8=image2_isect_array/imgBits
        
       
        imgg1_ui8.astype(np.uint8)
        imgg2_ui8.astype(np.uint8)
        
        
        imgg1_ui8_nd = np.array(imgg1_ui8, dtype=np.uint8)
        imgg2_ui8_nd = np.array(imgg2_ui8, dtype=np.uint8)
        
        
       
        imgg1_ui8_nd_gy=cv2.cvtColor(imgg1_ui8_nd,cv2.COLOR_RGB2GRAY)
        imgg2_ui8_nd_gy=cv2.cvtColor(imgg2_ui8_nd,cv2.COLOR_RGB2GRAY)
        del  imgg1,imgg2,imgg1_ui8,imgg2_ui8,imgg1_ui8_nd,imgg2_ui8_nd,image1_isect_array,image2_isect_array
        gc.collect()
        
        fc,fr,trueID_intersect=MakeTrueID(intersection_points,imgg1_ui8_nd_gy,imgg2_ui8_nd_gy,gt,res,scale_factor)
        if len(trueID_intersect)==2:
            try:
                data_final,data_final2,height, width=clipImg(kernel,imgg1_ui8_nd_gy,imgg2_ui8_nd_gy,fr,fc)
                seamlinef=buildGraph(data_final,data_final2,height,width,path_method,trueID_intersect)
            except:
                data_final,data_final2,height, width=clipImg(11,imgg1_ui8_nd_gy,imgg2_ui8_nd_gy,fr,fc)
                seamlinef=buildGraph(data_final,data_final2,height,width,path_method,trueID_intersect)
        else:
            print("please check the intersection of image: " +image1_ds+' and image: '+image2_ds)
        del  data_final,data_final2,imgg1_ui8_nd_gy,imgg2_ui8_nd_gy
        print('Generating seamline ...')
        x=round((gt[2]-union_gt[2])/(res/scale_factor) ) # col
        y=round((union_gt[5]-gt[5])/(res/scale_factor) ) # row
        xypos=(x,y)
        union_conn[union_conn>0]=1
        seamlineff=addAtPos(union_conn, seamlinef, xypos, inPlace=False)
       
        ##################################################################################################
        
        seamlineff3=BorderConnection(seamlineff, gt, union_gt, res, scale_factor, fc,fr )
        ###########################################################################################################
        
        seamlineff4 = cv2.resize(seamlineff3,(union_m.shape[1],union_m.shape[0]) ,interpolation=cv2.INTER_NEAREST)
        sf = cv2.resize(seamlineff4,(union_m.shape[1],union_m.shape[0]) ,interpolation=cv2.INTER_NEAREST)
        
        del seamlineff4,union_conn,seamlineff3,seamlineff,seamlinef
        gc.collect()
      
        if sf.shape[0]>65534 and (sf[60000:,:]==1).any():
            sff=sf[0:60000,:]
            sff2=sf[60000:,:]
            th, im_th = cv2.threshold(sff.astype(np.float64), 0, 1, cv2.THRESH_BINARY_INV)
            im_floodfill1 =im_th.copy().astype(np.uint8)   
            h, w = sff.shape[:2]
            mask = np.zeros((h+2, w+2), np.uint8)
            cv2.floodFill(im_floodfill1, mask, (1,1), 0)  
           
            th, im_th = cv2.threshold(sff2.astype(np.float64), 0, 1, cv2.THRESH_BINARY_INV)
            im_floodfill2 =im_th.copy().astype(np.uint8)   
            h, w = sff2.shape[:2]
            mask = np.zeros((h+2, w+2), np.uint8)
            
            if im_floodfill1[59999,0]==0:
                cv2.floodFill(im_floodfill2, mask, (1,1),  np.uint32(im_floodfill1[59999,0]).item() )  
            elif im_floodfill1[59999,im_floodfill1.shape[1]-1]==0 :
                cv2.floodFill(im_floodfill2, mask, (im_floodfill2.shape[1]-1,1),  np.uint32(im_floodfill1[59999,im_floodfill1.shape[1]-1]).item() )  
            else:
                print('floodfill process has something wrong!')
           
            h, w = sf.shape[:2]
            im_floodfill=np.zeros(sf.shape[:2], np.uint8)
            addAtPos(im_floodfill, im_floodfill1, (0,0), inPlace=True)
            addAtPos(im_floodfill, im_floodfill2, (0,60000), inPlace=True)
            
            del sff,sff2,im_floodfill1,im_floodfill2
            gc.collect()
        else:
            th, im_th = cv2.threshold(sf.astype(np.float64), 0, 1, cv2.THRESH_BINARY_INV)
            im_floodfill = im_th.copy().astype(np.uint8)
            
            h, w = im_th.shape[:2]
            mask = np.zeros((h+2, w+2), np.uint8)
            cv2.floodFill(im_floodfill, mask, (1,1), 0)
                        
        
        num_objects, labels = cv2.connectedComponents(im_floodfill, connectivity=8)
        classN=[np.sum(labels==i) for i in range(num_objects)]
        classN_ind=np.argsort(classN) 
        classN_index=classN_ind.tolist()
        classN_index.reverse()
       
        cind=[]
        for i in classN_index:
            if classN[i]>1000:
                cind.append(i)
        for h in range(len(cind)): 
            if len(cind)==3:
                m=h+1
                if m>len(cind)-1:
                    m=h-2
                    if all(im_floodfill[np.where(labels==cind[h] )]==1) and all(im_floodfill[np.where(labels==cind[m] )]==1): 
                        im_floodfill_f=im_floodfill.copy()                                 
                        im_floodfill_f[np.where((labels!=cind[h]) &( labels!=cind[m]) )]=0
                else: 
                    if all(im_floodfill[np.where(labels==cind[h] )]==1) and all(im_floodfill[np.where(labels==cind[m] )]==1): 
                        im_floodfill_f=im_floodfill.copy()                                 
                        im_floodfill_f[np.where((labels!=cind[h]) &( labels!=cind[m]) )]=0                        
            else:
                if all(im_floodfill[np.where(labels==cind[h])]==1):  
                    im_floodfill_f=im_floodfill.copy()                                 
                    im_floodfill_f[np.where(labels!=cind[h])]=0 
                    
        
     
        del th, im_th,mask,num_objects, labels,classN,classN_ind,classN_index,cind
        gc.collect()
        print('Saving "',image1_ds[:-4],'+',image2_ds[:-4],'"seamline ...')
        with rasterio.open(
            str(image1_ds[:-4])+str(image2_ds[:-4])+'_sl.img',
            'w+',           
            height=sf.shape[0],
            width=sf.shape[1],
            count=1,
            dtype=rasterio.uint8,           
            crs= {'init': 'EPSG:3826'},#Initialize from a named CRS
            transform=union_gt,
        ) as dst:
            dst.write(sf.astype(rasterio.uint8), 1)   
        del sf
        print('Mosaic "',image1_ds[:-4],'+',image2_ds[:-4],'" images...')
        mask_blurred_4chan=blendMask(57,im_floodfill_f)  
     
        with rasterio.open(
            str(image1_ds[:-4])+str(image2_ds[:-4])+'MosTai_sl_polygon.img',
            'w+',           
            height=mask_blurred_4chan.shape[0]*int(res/pan_res),
            width=mask_blurred_4chan.shape[1]*int(res/pan_res),
            count=1,
            dtype=rasterio.float32,          
            crs= {'init': 'EPSG:3826'},#Initialize from a named CRS            
            transform= A.translation(union_gt[2], union_gt[5]) * A.scale(pan_res, -pan_res) #  union_gt,
        ) as dst:
            dst.write(cv2.resize(mask_blurred_4chan,(mask_blurred_4chan.shape[1]*int(res/pan_res) ,mask_blurred_4chan.shape[0]*int(res/pan_res)),interpolation=cv2.INTER_NEAREST).astype(rasterio.float32), 1)        
        
            
        del im_floodfill,im_floodfill_f
        
        b1,im1_0num,im1_1num=BlendPreprocessForOrder(image1_ds,mask_blurred_4chan,res,union_gt)
        b1d,im2_0num,im2_1num=BlendPreprocessForOrder(image2_ds,mask_blurred_4chan,res,union_gt)
        
    
        
        img=[(0,image1_ds,im1_0num, 'b1d','b1' ),
              (1,image1_ds,im1_1num, 'b1d','b1' ),
              (0,image2_ds,im2_0num, 'b1','b1d' ),
              (1,image2_ds,im2_1num, 'b1','b1d')]
        imgind=sorted(img,reverse=True,  key = itemgetter(0,2 )) 
        ImgInd.append(imgind)
        kwargs = MosTai.meta.copy()
        kwargs.update(BIGTIFF="IF_SAFER")
        for k in range(4):
            if k==0:
                try:
                    write_window = Window(   (union_gt[2]-MosTai_gt[2])/res,  (MosTai_gt[5]-union_gt[5])/res  ,    union_m.shape[1], union_m.shape[0]     )
                    outcome= eval(imgind[0][3]).astype(imgtype) *(1- mask_blurred_4chan)+ eval(imgind[0][4]).astype(imgtype) *( mask_blurred_4chan)      
                    outcome=outcome.astype(imgtype)                    
                    print('Writing result',image1_ds[:-4],image2_ds[:-4],'band:',k+1,'to disk...')
                    MosTai.write( outcome, indexes=k+1, window=write_window)
                   
                    del outcome,b1,b1d,union_m
                    gc.collect()
                    
                except:
                    
                    h=int(mask_blurred_4chan.shape[0]/2)
                    w=int(mask_blurred_4chan.shape[1]/2)
                    outcome= eval(imgind[0][3])[0:h,:].astype(imgtype) *(1- mask_blurred_4chan)[0:h,:]+ eval(imgind[0][4])[0:h,:].astype(imgtype) *( mask_blurred_4chan)[0:h,:]     
                    outcome=outcome.astype(imgtype)
                    write_window2 = Window(   (union_gt[2]-MosTai_gt[2])/res,  (MosTai_gt[5]-union_gt[5])/res  ,    outcome.shape[1], outcome.shape[0]     )# write_window.height = first, write_window.width = second
                    print('Writing 1/2 result:',image1_ds[:-4],'band:',k+1,'to disk...')
                    MosTai.write( outcome, indexes=k+1, window=write_window2)
                    del outcome ,write_window2
                    gc.collect()
                    
                    outcome= eval(imgind[0][3])[h:,:].astype(imgtype) *(1- mask_blurred_4chan)[h:,:]+ eval(imgind[0][4])[h:,:].astype(imgtype) *( mask_blurred_4chan)[h:,:]     
                    outcome=outcome.astype(imgtype)
                    write_window3 = Window(   (union_gt[2]-MosTai_gt[2])/res,  (MosTai_gt[5]-union_gt[5])/res+h  ,    outcome.shape[1], outcome.shape[0]     )# write_window.height = first, write_window.width = second
                    print('Writing 2/2 result:',image1_ds[:-4],'band:',k+1,'to disk...')
                    MosTai.write( outcome, indexes=k+1, window=write_window3)
                    # 
                    del outcome ,b1,b1d ,write_window3
                    gc.collect()                    
                    
                    
            else:  
                
                ind=k+1
                b1=BlendPreprocess(ind,image1_ds,mask_blurred_4chan,res,union_gt)
                b1d=BlendPreprocess(ind,image2_ds,mask_blurred_4chan,res,union_gt)
                   
                
                try:
                    
                     
                    outcome= eval(imgind[0][3]).astype(imgtype) *(1- mask_blurred_4chan)+ eval(imgind[0][4]).astype(imgtype) *( mask_blurred_4chan)      
                    outcome=outcome.astype(imgtype)
                    # 
                    print('Writing result',image1_ds[:-4],image2_ds[:-4],'band:',k+1,'to disk...')
                    MosTai.write( outcome, indexes=k+1, window=write_window)     
                    # 
                    del outcome,b1,b1d
                    gc.collect()
                
                
                except:
                   
                    h=int(mask_blurred_4chan.shape[0]/2)
                    w=int(mask_blurred_4chan.shape[1]/2)
                    outcome= eval(imgind[0][3])[0:h,:].astype(imgtype) *(1- mask_blurred_4chan)[0:h,:]+ eval(imgind[0][4])[0:h,:].astype(imgtype) *( mask_blurred_4chan)[0:h,:]      
                    outcome=outcome.astype(imgtype)
                    write_window2 = Window(   (union_gt[2]-MosTai_gt[2])/res,  (MosTai_gt[5]-union_gt[5])/res  ,    outcome.shape[1], outcome.shape[0]     )# write_window.height = first, write_window.width = second
                   
                    print('Writing 1/2 result:',image1_ds[:-4],'band:',k+1,'to disk...')
                    MosTai.write( outcome, indexes=k+1, window=write_window2)
                   
                    del outcome,write_window2
                    
                   
                    outcome= eval(imgind[0][3])[h:,:].astype(imgtype) *(1- mask_blurred_4chan)[h:,:]+ eval(imgind[0][4])[h:,:].astype(imgtype) *( mask_blurred_4chan)[h:,:]    
                    outcome=outcome.astype(imgtype)
                    write_window3 = Window(   (union_gt[2]-MosTai_gt[2])/res,  (MosTai_gt[5]-union_gt[5])/res+h  ,    outcome.shape[1], outcome.shape[0]     )# write_window.height = first, write_window.width = second
                   
                    print('Writing 2/2 result:',image1_ds[:-4],'band:',k+1,'to disk...')
                    MosTai.write( outcome, indexes=k+1, window=write_window3)
                    
                    del outcome, b1,b1d ,write_window3
                    gc.collect()                    
                    
                    
        MosTai.close()
        
       
        cldList=glob.glob(''.join(str(pathlib.Path().absolute()))+'\\'+'SPOT67Cloud_Hsiao_Rou'+'\\'+'*'+image1_ds[:-4]+'*.tif')        
        b1=MskBlendPreprocessForOrder(cldList[0],mask_blurred_4chan,res,union_gt)
        cldList=glob.glob(''.join(str(pathlib.Path().absolute()))+'\\'+'SPOT67Cloud_Hsiao_Rou'+'\\'+'*'+image2_ds[:-4]+'*.tif')
        b1d=MskBlendPreprocessForOrder(cldList[0],mask_blurred_4chan,res,union_gt)
       
       
        cldMas=eval(imgind[0][3]).astype(imgtype) *(1- mask_blurred_4chan)+ eval(imgind[0][4]).astype(imgtype) *( mask_blurred_4chan)
        cldMas[np.where(cldMas>0)]=1
        cldMas=cldMas.astype(imgtype)
        print('Writing cloud mask to disk...')
        TaiMas.write( cldMas, indexes=1, window=write_window)     
        TaiMas.close()
        allWindow.append(write_window)
        del mask_blurred_4chan,union_gt,cldMas,cldList,b1,b1d
        gc.collect()

fp.close()  
with open('allWindow.npy', 'wb') as f:
    np.save(f, allWindow)      
with open('ImgInd.npy', 'wb') as f:
    np.save(f, ImgInd)      
with open('Union_gt.npy', 'wb') as f:
    np.save(f, Union_gt) 
    
gt_two=[]
file=[]
gt_five=[]

for f in ReadOrder:

  fullpath = join(mypath, f)

  if isfile(fullpath) and  f.endswith(".img"):

    image=rasterio.open(f)
    file.append(f)
    gt_two.append(image.transform[2] )
    gt_five.append(image.transform[5] )
    
coords=np.array([[gt_two[i],gt_five[i]] for i in range(len(gt_two))],dtype='float32')    
with open('transform.npy', 'wb') as f:
    np.save(f, coords)      
print('All done!')
end = timer()
print('Total time spent(hr):',(end - start)/60/60) # Time in seconds # Time in seconds



