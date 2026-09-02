# -*- coding: utf-8 -*-
"""
Panchromatic mosaic generation (Step 3).

Mosaics Pan scenes reusing the seamlines found in Step 2, producing a
single panchromatic mosaic that is geometrically consistent with the
multispectral mosaic.

Author: Hsiao-Jou Hsu
License: MIT
"""
import numpy as np
import cv2
import rasterio
from rasterio import Affine as A
from timeit import default_timer as timer
from rasterio.windows import Window
import pathlib
import os
import gc
import json
from rasterio.enums import Resampling


start = timer()
with open('para_pan.txt', 'r') as f:
    datastore = json.load(f)

canvas_crs = datastore.get("canvas_crs", "EPSG:3826")
canvas_x_max = float(datastore.get("canvas_x_max", "406000"))
canvas_y_min = float(datastore.get("canvas_y_min", "2400000"))

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
        image2_ds = 'mosaic_pan.tif'#

        raw1Path = pathlib.Path('raw_pan\\'+image1_ds[:-4].upper()+'\\'+image1_ds[:-4].upper()+'_B1.DAT.raw').absolute()
        new_rawPath = str(pathlib.Path('new_pan').absolute())
        slPath = pathlib.Path(img1_ori+'_seamline_polygon.img').absolute()

        a=rasterio.open(slPath)
        x_min = a.transform[2]
        y_min = a.transform[5] - a.transform[0]*a.shape[0]
        x_max = a.transform[2] + a.transform[0]*a.shape[1]
        y_max = a.transform[5]

        print('Clipping...')
        if os.path.isfile(new_rawPath+'\\'+image1_ds+'.tif'):
            print('Done with image clipping:',image1_ds+'.tif')
        else:
            cmdIn =' gdalwarp  -t_srs '+canvas_crs+' -te '+ str(x_min) +' ' + str(y_min) +' '+ str(x_max) +' '+ str(y_max) +' '+ ''.join(str(raw1Path)) +' '+''.join(str(new_rawPath ))+'\\'+image1_ds+'.tif'
            ret =  os.popen(cmdIn).read()

        seamline_polygon = rasterio.open(slPath)

        print('Mosaicking and writing to disk...')
        with rasterio.open(new_rawPath+'\\'+image1_ds+'.tif') as src:
            for ji, window in src.block_windows(1):

                 mosaic_out= rasterio.open( 'mosaic_pan.tif','r+')#

                 seamlinef=seamline_polygon.read(1,window=window)

                 b1 = src.read(1, window=window)#
                 write_window = Window(   (  src.transform[2]  -mosaic_pan_gt[2])/res,  (mosaic_pan_gt[5]-  (src.transform[5]-(ji[0]*res))  )/res  ,    seamlinef.shape[1], seamlinef.shape[0]     )
                 b1d = mosaic_out.read(1 ,window=write_window)
                 outcome=( eval(ImgInd[i][0][3]).astype(imgtype) *(1- seamlinef)+ eval(ImgInd[i][0][4]).astype(imgtype) *( seamlinef)  ).astype(imgtype)

                 mosaic_out.write( outcome, 1, window=write_window)

    else:
        print('Image Process Progress: 1,2','/',len(ReadOrder))
        img1_ori=ReadOrder[i][ReadOrder[i].find(StartLetter):ReadOrder[i].find(StartLetter)+8]
        img2_ori=ReadOrder[i+1][ReadOrder[i+1].find(StartLetter):ReadOrder[i+1].find(StartLetter)+8]
        image1_ds= StartLetter+str(int(ReadOrder[i][ReadOrder[i].find('0'):ReadOrder[i].find(StartLetter)+8])+1).zfill(7)+('.img')#spot
        image2_ds =StartLetter+str(int(ReadOrder[i+1][ReadOrder[i+1].find('0'):ReadOrder[i+1].find(StartLetter)+8])+1).zfill(7)+('.img')#spot

        raw1Path = pathlib.Path('raw_pan\\'+image1_ds[:-4].upper()+'\\'+image1_ds[:-4].upper()+'_B1.DAT.raw').absolute()
        raw2Path = pathlib.Path('raw_pan\\'+image2_ds[:-4].upper()+'\\'+image2_ds[:-4].upper()+'_B1.DAT.raw').absolute()

        new_rawPath = str(pathlib.Path('new_pan').absolute())
        slPath = pathlib.Path(img1_ori+img2_ori+'_seamline_polygon.img').absolute()

        a=rasterio.open(slPath)
        x_min = a.transform[2]
        y_min = a.transform[5] - a.transform[0]*a.shape[0]
        x_max = a.transform[2] + a.transform[0]*a.shape[1]
        y_max = a.transform[5]

        print('Clipping...')
        if os.path.isfile(new_rawPath+'\\'+image1_ds+'.tif'):
            print('Done with image clipping:',image1_ds+'.tif')
        else:
            cmdIn =' gdalwarp  -t_srs '+canvas_crs+' -te '+ str(x_min) +' ' + str(y_min) +' '+ str(x_max) +' '+ str(y_max) +' '+ ''.join(str(raw1Path)) +' '+''.join(str(new_rawPath ))+'\\'+image1_ds+'.tif'
            ret =  os.popen(cmdIn).read()

        if os.path.isfile(new_rawPath+'\\'+image2_ds+'.tif'):
            print('Done with image clipping:',image2_ds+'.tif')
        else:
            cmdIn =' gdalwarp  -t_srs '+canvas_crs+' -te '+ str(x_min) +' ' + str(y_min) +' '+ str(x_max) +' '+ str(y_max) +' '+ ''.join(str(raw2Path)) +' '+''.join(str(new_rawPath ))+'\\'+image2_ds+'.tif'
            ret =  os.popen(cmdIn).read()

        img2 = rasterio.open(new_rawPath+'\\'+image2_ds+'.tif')
        seamline_polygon = rasterio.open(slPath)

        mosaic_pan_gt = A.translation(float(min( gt_two)), float(max( gt_five))) *  A.scale(res, -res)

        mosaic_out= rasterio.open(
            'mosaic_pan.tif',
            'w+',
            height= int( ( float(max( gt_five))-  canvas_y_min   )/res ),
            width=  int( (   canvas_x_max-  float(min( gt_two)   ))/res ),
            count=1,
            dtype=img2.meta['dtype'],
            crs= {'init': canvas_crs},
            transform=mosaic_pan_gt)
        kwargs = mosaic_out.meta.copy()
        kwargs.update(BIGTIFF="IF_SAFER")

        print('Mosaicking and writing to disk...')
        with rasterio.open(new_rawPath+'\\'+image1_ds+'.tif') as src:
            for ji, window in src.block_windows(1):

                 seamlinef=seamline_polygon.read(1,window=window)

                 b1 = src.read(1, window=window)
                 b1d = img2.read(1 ,window=window)
                 outcome=( eval(ImgInd[i][0][3]).astype(imgtype) *(1- seamlinef)+ eval(ImgInd[i][0][4]).astype(imgtype) *( seamlinef)  ).astype(imgtype)
                 write_window = Window(   (  img2.transform[2]  -mosaic_pan_gt[2])/res,  (mosaic_pan_gt[5]-  (img2.transform[5]-(ji[0]*res))  )/res  ,    width=seamlinef.shape[1], height=seamlinef.shape[0]     )#

                 mosaic_out.write(outcome , 1, window=write_window)

mosaic_out.close()
print('All done!')
end = timer()
print('Total time spent(hr):',(end - start)/60/60)
