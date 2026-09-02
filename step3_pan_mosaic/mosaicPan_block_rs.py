# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 14:24:57 2021

@author: amy hsu
"""
import numpy as np
import cv2
import rasterio
import matplotlib.pyplot as plt
from rasterio import Affine as A
from timeit import default_timer as timer  
import tqdm
from rasterio.windows import Window  
import pathlib
import os
import gc
import json
from rasterio.enums import Resampling


start = timer()
with open('para_pan.txt', 'r') as f:
    datastore = json.load(f)

res=float(datastore["res"])
trueBits=int(datastore["trueBits"])
StartLetter=datastore["StartLetter"]#
upscale_factor=int(datastore["upscale_factor"])

imgBits=(2**trueBits)/(2**8)
if trueBits==12:
    imgtype='uint16'
else:
    imgtype='uint'+str(trueBits)


my_file = open('ProcessOrder.txt', "r")
ReadOrder =  my_file.read().split(",")
my_file.close()

allWindow = np.load('allWindow.npy',allow_pickle=True)
allTransform= np.load('transform.npy',allow_pickle=True) 
gt_two,gt_five=  allTransform[:,0],allTransform[:,1]
ImgInd = np.load('ImgInd.npy',allow_pickle=True)
Union_gt = np.load('Union_gt.npy',allow_pickle=True)

for i in range(len(ReadOrder)-1):   
    if i!=0:
        print('Image Process Progress:',i+2,'/',len(ReadOrder))   
        img1_ori = ReadOrder[i+1][ReadOrder[i+1].find(StartLetter):ReadOrder[i+1].find(StartLetter)+8]
        image1_ds = StartLetter+str(int(ReadOrder[i+1][ReadOrder[i+1].find('0'):ReadOrder[i+1].find(StartLetter)+8])+1).zfill(7)+('.img')#
        image2_ds = 'MosTai_Pan.tif'#
        
        raw1Path = pathlib.Path('raw_pan\\'+image1_ds[:-4].upper()+'\\'+image1_ds[:-4].upper()+'_B1.DAT.raw').absolute()
        new_rawPath = str(pathlib.Path('new_pan').absolute() )
        slPath = pathlib.Path(img1_ori+'MosTai_sl_polygon.img').absolute()
        MosTai_Pan = str(pathlib.Path(image2_ds).absolute() )
        
        a=rasterio.open(slPath)
        x_min = a.transform[2]
        y_min = a.transform[5] - a.transform[0]*a.shape[0]
        x_max = a.transform[2] + a.transform[0]*a.shape[1]
        y_max = a.transform[5]
        
        print('Clipping...')
        if os.path.isfile(new_rawPath+'\\'+image1_ds+'.tif') :
            print('Done with image clipping:',image1_ds+'.tif')
        else:
            cmdIn =' gdalwarp  -t_srs EPSG:3826 -te '+ str(x_min) +' ' + str(y_min) +' '+ str(x_max) +' '+ str(y_max) +' '+ ''.join(str(raw1Path)) +' '+''.join(str(new_rawPath ))+'\\'+image1_ds+'.tif'
            ret =  os.popen(cmdIn).read()
      
        seamline_polygon = rasterio.open(slPath)
       
        
        print('Mosaicking and writing to disk...')
        with rasterio.open(new_rawPath+'\\'+image1_ds+'.tif') as src:
            for ji, window in src.block_windows(1): 
               
                 MosTai= rasterio.open( 'MosTai_Pan.tif','r+')# 
                 
                 seamlinef=seamline_polygon.read(1,window=window)
                 
                 b1 = src.read(1, window=window)#
                 write_window = Window(   (  src.transform[2]  -MosTai_pan_gt[2])/res,  (MosTai_pan_gt[5]-  (src.transform[5]-(ji[0]*res))  )/res  ,    seamlinef.shape[1], seamlinef.shape[0]     )
                 b1d = MosTai.read(1 ,window=write_window)
                 outcome=( eval(ImgInd[i][0][3]).astype(imgtype) *(1- seamlinef)+ eval(ImgInd[i][0][4]).astype(imgtype) *( seamlinef)  ).astype(imgtype)
                
                 MosTai.write( outcome, 1, window=write_window)

        
    else:
        print('Image Process Progress: 1,2','/',len(ReadOrder))
        img1_ori=ReadOrder[i][ReadOrder[i].find(StartLetter):ReadOrder[i].find(StartLetter)+8]
        img2_ori=ReadOrder[i+1][ReadOrder[i+1].find(StartLetter):ReadOrder[i+1].find(StartLetter)+8]
        image1_ds= StartLetter+str(int(ReadOrder[i][ReadOrder[i].find('0'):ReadOrder[i].find(StartLetter)+8])+1).zfill(7)+('.img')#spot 
        image2_ds =StartLetter+str(int(ReadOrder[i+1][ReadOrder[i+1].find('0'):ReadOrder[i+1].find(StartLetter)+8])+1).zfill(7)+('.img')#spot 
        
       
        raw1Path = pathlib.Path('raw_pan\\'+image1_ds[:-4].upper()+'\\'+image1_ds[:-4].upper()+'_B1.DAT.raw').absolute()
        raw2Path = pathlib.Path('raw_pan\\'+image2_ds[:-4].upper()+'\\'+image2_ds[:-4].upper()+'_B1.DAT.raw').absolute()
        
        
        new_rawPath = str(pathlib.Path('new_pan').absolute() )
        slPath = pathlib.Path(img1_ori+img2_ori+'MosTai_sl_polygon.img').absolute()
        
        
        a=rasterio.open(slPath)
        x_min = a.transform[2]
        y_min = a.transform[5] - a.transform[0]*a.shape[0]
        x_max = a.transform[2] + a.transform[0]*a.shape[1]
        y_max = a.transform[5]
        
        print('Clipping...')
        if os.path.isfile(new_rawPath+'\\'+image1_ds+'.tif') :
            print('Done with image clipping:',image1_ds+'.tif')
        else:
            cmdIn =' gdalwarp  -t_srs EPSG:3826 -te '+ str(x_min) +' ' + str(y_min) +' '+ str(x_max) +' '+ str(y_max) +' '+ ''.join(str(raw1Path)) +' '+''.join(str(new_rawPath ))+'\\'+image1_ds+'.tif'
            ret =  os.popen(cmdIn).read()
            
        if os.path.isfile(new_rawPath+'\\'+image2_ds+'.tif') :
            print('Done with image clipping:',image2_ds+'.tif')
        else:        
            cmdIn =' gdalwarp  -t_srs EPSG:3826 -te '+ str(x_min) +' ' + str(y_min) +' '+ str(x_max) +' '+ str(y_max) +' '+ ''.join(str(raw2Path)) +' '+''.join(str(new_rawPath ))+'\\'+image2_ds+'.tif'
            ret =  os.popen(cmdIn).read()
       
        
        
        img2 = rasterio.open(new_rawPath+'\\'+image2_ds+'.tif')
        seamline_polygon = rasterio.open(slPath)
        
        
        MosTai_pan_gt = A.translation(float(min( gt_two)), float(max( gt_five))) *  A.scale(res, -res) 
        
        MosTai= rasterio.open(
            'MosTai_Pan.tif',
            'w+',
           
            height= int( ( float(max( gt_five))-  2400000   )/res ),
            width=  int( (   406000-  float(min( gt_two)   ))/res ),
            count=1,
            dtype=img2.meta['dtype'],
           
            crs= {'init': 'EPSG:3826'},
            transform=MosTai_pan_gt)
        kwargs = MosTai.meta.copy()
        kwargs.update(BIGTIFF="IF_SAFER")
        
        print('Mosaicking and writing to disk...')
        with rasterio.open(new_rawPath+'\\'+image1_ds+'.tif') as src:
            for ji, window in src.block_windows(1): 
              
                 seamlinef=seamline_polygon.read(1,window=window)
              
                 b1 = src.read(1, window=window)
                 b1d = img2.read(1 ,window=window)
                 outcome=( eval(ImgInd[i][0][3]).astype(imgtype) *(1- seamlinef)+ eval(ImgInd[i][0][4]).astype(imgtype) *( seamlinef)  ).astype(imgtype)
                 write_window = Window(   (  img2.transform[2]  -MosTai_pan_gt[2])/res,  (MosTai_pan_gt[5]-  (img2.transform[5]-(ji[0]*res))  )/res  ,    width=seamlinef.shape[1], height=seamlinef.shape[0]     )# 
                 
                 MosTai.write(outcome , 1, window=write_window)
        
MosTai.close()         
print('All done!')        
end = timer()
print('Total time spent(hr):',(end - start)/60/60)
