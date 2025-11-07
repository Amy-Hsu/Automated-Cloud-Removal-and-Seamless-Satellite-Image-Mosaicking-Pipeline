# -*- coding: utf-8 -*-
"""
A Computational Pipeline for Automated Cloud Removal and Seamless Satellite Image Mosaicking

STAGE: SSIM-GUIDED SELECTION AND POISSON BLENDING

This script handles the final, critical stage of the pipeline:
1. Reads the current cloud mask and segments it into individual patches.
2. Calculates the Structural Similarity (SSIM) between the surrounding cloud-free area of the large mosaic
   and corresponding areas in a pool of candidate images.
3. Selects the optimal (highest SSIM) candidate patch for each cloud area.
4. Performs seamless replacement using OpenCV's cv2.seamlessClone (Poisson Blending).

Author: Hsu
"""

import numpy as np
import cv2
import rasterio
import matplotlib.pyplot as plt
from rasterio import Affine as A
from os import listdir
from os.path import isfile, join
from timeit import default_timer as timer  
from rasterio.windows import Window
import pathlib
import glob
from scipy import ndimage
from rasterio import windows
import os
import shutil
from skimage.metrics import structural_similarity
import json

from rasterio.windows import from_bounds
def border_elems_generic(a, W): # Input array : a, Edgewidth : W
    n1 = a.shape[0]
    r1 = np.minimum(np.arange(n1)[::-1], np.arange(n1))
    n2 = a.shape[1]
    r2 = np.minimum(np.arange(n2)[::-1], np.arange(n2))
    return a[np.minimum(r1[:,None],r2)<W]
start = timer() 

with open('para_cld.txt', 'r') as f:
    datastore = json.load(f)
#Use the new datastore datastructure
res=float(datastore["res"])
trueBits=int(datastore["trueBits"])
mypath=datastore["CandiImgPath"]
cldpath=datastore["CldMaskPath"]
bigImgAddr=datastore["MosaicImgAddress"]
candi=datastore["CandiMaskFolderName"]
candiCld=datastore["MosaicCldAddress"]
pan_res=float(datastore["pan_res"])
bigPan_Addr=datastore["PanMosaicImgAddress"]
Pan_Addr=datastore["PanImgAddress"]


imgBits=(2**trueBits)/(2**8)
if trueBits==12:
    imgtype='uint16'
else:
    imgtype='uint'+str(trueBits)


my_file = open('ProcessOrder.txt', "r")
ReadOrder =  my_file.read().split(",")
my_file.close()
TaiMas=rasterio.open(candiCld,'r+')
TaiMas_t=TaiMas.transform

allWindow = np.load('allWindow.npy',allow_pickle=True)
cd1 = np.load('transform.npy',allow_pickle=True)   
Count=0
total=0
stage1=0
stage2=0
stage3=0
delnum=0
cldzeroNum=0
zeroNum=0
right=0
up=0
low=0
left=0
tw6m=rasterio.open('tw_city6m.img')
bigtw6mShp=tw6m.transform[2],tw6m.transform[5]
bigPan=rasterio.open(bigPan_Addr)
bigPanShp=bigPan.transform[2],bigPan.transform[5]
bigPan.close()

for i in range(len(ReadOrder)-1):#    

    TaiMas=rasterio.open(candiCld,'r+')
    cldRdbineary0=TaiMas.read(1,window=allWindow[i])
    cldRdbineary=cldRdbineary0.copy()
    cldRdbineary[cldRdbineary0!=1]=0   
  
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cldRdbineary.astype('uint8'), connectivity=8, ltype=None)
   
    for a in range(1,num_labels):
        if stats[a,4]<500:
            delnum+=1
            cldRdbineary[labels==a]=0   

    print('scraps num:',delnum)

    kernel = np.ones((50,50), np.uint8)
    dilation = cv2.dilate(cldRdbineary, kernel, iterations = 1).astype('uint8')
      
    contours, hierarchy = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img=cv2.drawContours(dilation.copy(), contours, -1, (128, 0, 255), 2)
   
    num_objects, labels = cv2.connectedComponents(dilation, connectivity=8)      
  
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours] 
    for bbox in bounding_boxes:
         [x , y, w, h] = bbox
         cv2.rectangle(dilation, (x-10, y-10), (x + w + 10, y + h + 10), (255,0, 0), 3)
   
    
    out = img*125 + dilation 
  
    if i==0:
        cdc=min(cd1[i][0],cd1[i+1][0])
        cdr=max(cd1[i][1],cd1[i+1][1])     
    else:
        
        cdc=cd1[i+1][0]
        cdr=cd1[i+1][1]
        # i=i-1    

    print('cloud spiliting...',i+1,'/',len(allWindow)) 
    c=0
    for bbox in bounding_boxes:
        c+=1
        [x , y, w, h] = bbox
        
        expdx=10
        expdy=10
        expdw=10
        expdh=10
        
        if x-10<0:
            expdx=x
        if y-10<0:
            expdy=y
        if y + h + 10 >dilation.shape[0]:
            expdh=dilation.shape[0]-y-h
        if x + w + 10 >dilation.shape[1]:
            expdw=dilation.shape[1]-x-w  
        cv2.rectangle(dilation, (x-expdx, y-expdy), (x + w + expdw, y + h + expdh), (255,0, 0), 3)
        cld_gt=A.translation( cdc+(x-expdx)*res , cdr-(y-expdy)*res ) * A.scale(res, -res)
        write_window = Window( cld_gt[2] , cld_gt[5]   ,  w+(expdw+expdh), h+(expdw+expdh) )        
        clas=[np.sum(labels[y-expdy:y+h+expdh,x-expdx:x+w+expdw]==a) for a in range(num_objects)]
        classN=clas[1:]
        classNIndex=np.argsort(classN)[::-1]      
        labelscy=np.ones((labels[y-expdy:y+h+expdh,x-expdx:x+w+expdw].shape[0],labels[y-expdy:y+h+expdh,x-expdx:x+w+expdw].shape[1]))
        labelscy [np.where(labels[y-expdy:y+h+expdh,x-expdx:x+w+expdw]!=(classNIndex[0]+1))]=0        
        
        labelscy1=ndimage.binary_fill_holes(labelscy.astype(int))       
        labelscy2=np.zeros((labelscy.shape[0],labelscy.shape[1]))
        labelscy2[np.where(labelscy1==True)]=1
       
        labelscy2=labelscy2.astype(rasterio.uint8)        
    
        total+=1
        with rasterio.open(
                    'cldDatabase\\'+str(c)+'_cld.img',
                    'w+',           
                    height=h+(expdw+expdh),
                    width=w+(expdw+expdh),
                    count=1,
                    dtype=rasterio.uint8,           
                    crs= {'init': 'EPSG:3826'},
                    transform=cld_gt
                ) as dst:
                    dst.write(labelscy2.astype(rasterio.uint8), 1)   

    if i ==0:       
        files = listdir(mypath)
        gt_two=[]
        file=[]
        gt_five=[]       
        for f in files:          
          fullpath1 = join(mypath, f)         
          if isfile(fullpath1) and  f.endswith(".img"):          
            image=rasterio.open(mypath+'\\'+f)
            file.append(f)
            gt_two.append(image.transform[2] )
            gt_five.append(image.transform[5] )
       
  
    files = listdir(cldpath)
   
    cldfile=[]
  
    for f in files:
     
      fullpath = join(cldpath, f)
     
      if isfile(fullpath) and  f.endswith(".img"):
      
        cldfile.append(f)
       
    bigImg= rasterio.open(bigImgAddr)
    ssim= np.empty((50,len(cldfile),))* np.nan
    ssim=np.zeros((50,len(cldfile),))
  
    bigShp=bigImg.transform[2],bigImg.transform[5]

    print('calculate ssim...',i+1,'/',len(allWindow))  
    a=0                        
    for j in cldfile:
        cld_ds = rasterio.open(cldpath+'\\'+j)
        cld_gt=cld_ds.transform                  
        cldWin=Window(((cld_gt[2]-bigShp[0])/res if (cld_gt[2]-bigShp[0])/res>0 else (cld_gt[2]-bigShp[0])/res ),((bigShp[1]-cld_gt[5])/res if (bigShp[1]-cld_gt[5])/res>0 else (bigShp[1]-cld_gt[5])/res ), width=cld_ds.width ,height=cld_ds.height  )   #full cld mask                    
        cldWin6m=Window(((cld_gt[2]-bigtw6mShp[0])/res if (cld_gt[2]-bigtw6mShp[0])/res>0 else (cld_gt[2]-bigtw6mShp[0])/res ),((bigtw6mShp[1]-cld_gt[5])/res if (bigtw6mShp[1]-cld_gt[5])/res>0 else (bigtw6mShp[1]-cld_gt[5])/res ), width=cld_ds.width ,height=cld_ds.height  )   #full cld mask                    
        
       
        b=0
        for k in file:
            imagePool= rasterio.open(mypath+'\\'+k)        
            gt1=imagePool.transform             
            imgWin=Window(((gt1[2]-bigShp[0])/res if (gt1[2]-bigShp[0])/res>0 else (gt1[2]-bigShp[0])/res ),((bigShp[1]-gt1[5])/res if (bigShp[1]-gt1[5])/res>0 else (bigShp[1]-gt1[5])/res ),  width=imagePool.width ,height=imagePool.height  )
            tw6mWin=Window(((gt1[2]-bigtw6mShp[0])/res if (gt1[2]-bigtw6mShp[0])/res>0 else (gt1[2]-bigtw6mShp[0])/res ),((bigtw6mShp[1]-gt1[5])/res if (bigtw6mShp[1]-gt1[5])/res>0 else (bigtw6mShp[1]-gt1[5])/res ),  width=imagePool.width ,height=imagePool.height  )

            try:
          
                ints_window=windows.intersection(imgWin, cldWin)              
                ints_window_tw6m=windows.intersection(tw6mWin, cldWin6m) 
                write_window= Window(abs(imgWin.col_off-ints_window.col_off), abs(ints_window.row_off-imgWin.row_off), height= ints_window.height , width=ints_window.width)
                cld_window=Window(abs(cldWin.col_off-ints_window.col_off), abs(ints_window.row_off-cldWin.row_off), height= ints_window.height , width=ints_window.width)# intersection cloud mask
                
                cld= cld_ds.read(1,window=cld_window)              
                bigImgrgb=np.dstack((bigImg.read(3,window=ints_window)/imgBits, bigImg.read(2,window=ints_window)/imgBits,bigImg.read(1,window=ints_window)/imgBits)).astype('uint8')
                bigimgRGB=bigImgrgb.copy()
                for d in range(3):
                    bigimgRGB[:,:,d][np.where(cld==1)]=0 
                    
                bigImg2gray=cv2.cvtColor(bigImgrgb,cv2.COLOR_RGB2GRAY) 
                bigImg2gray[np.where(cld==1)]=0# 
               
                imgrgb= np.dstack((imagePool.read(3,window=write_window)/imgBits, imagePool.read(2,window=write_window)/imgBits,imagePool.read(1,window=write_window)/imgBits)).astype('uint8')
                imgRGB=imgrgb.copy()
                for d in range(3):
                    imgRGB[:,:,d][np.where(cld==1)]=0 
                
                tw6mMask=tw6m.read(1,window=ints_window_tw6m)
                
                imgRGB[tw6mMask==0]=0
                
    
                ssim[b,a]=  structural_similarity(bigimgRGB, imgRGB, multichannel=True,dynamic_range=255,K1=0.01,K2=0.03) if cld_window==cldWin else structural_similarity(bigimgRGB, imgRGB, multichannel=True,dynamic_range=255,K1=0.01,K2=0.03)-0.05
                
               
                
                if np.all(bigimgRGB[:]==0) and np.all(imgRGB[:]==0):
                    ssim[b,a]=-3
                if np.all(imgRGB[cld==0]==0):#
                    ssim[b,a]=-3
            except:
              
                ssim[b,a]=-1            
             
            finally:
                if i==0:
                    if (k == ReadOrder[i][ReadOrder[i].find(str(k[0:2])):ReadOrder[i].find(k[0:2])+len(k)] 
                    or k == ReadOrder[i+1][ReadOrder[i+1].find(str(k[0:2])):ReadOrder[i+1].find(k[0:2])+len(k)]):
                        ssim[b,a]=-2
                else:
                    if k == ReadOrder[i+1][ReadOrder[i+1].find(str(k[0:2])):ReadOrder[i+1].find(k[0:2])+len(k)]:
                        ssim[b,a]=-2
                b+=1 
        a+=1
    
    bigImg.close()
   
    a=0
  
    print('cloud removal...',i+1,'/',len(allWindow)) 
    for j in cldfile:      
        cld_ds = rasterio.open(cldpath+'\\'+j)
        cld_gt=cld_ds.transform                  
        cldWin=Window(((cld_gt[2]-bigShp[0])/res if (cld_gt[2]-bigShp[0])/res>0 else (cld_gt[2]-bigShp[0])/res ),((bigShp[1]-cld_gt[5])/res if (bigShp[1]-cld_gt[5])/res>0 else (bigShp[1]-cld_gt[5])/res ), width=cld_ds.width ,height=cld_ds.height  )                       
        cldWin6m=Window(((cld_gt[2]-bigtw6mShp[0])/res if (cld_gt[2]-bigtw6mShp[0])/res>0 else (cld_gt[2]-bigtw6mShp[0])/res ),((bigtw6mShp[1]-cld_gt[5])/res if (bigtw6mShp[1]-cld_gt[5])/res>0 else (bigtw6mShp[1]-cld_gt[5])/res ), width=cld_ds.width ,height=cld_ds.height  )   #full cld mask                    

        PancldWin=Window((cld_gt[2]-bigPanShp[0])/pan_res ,(bigPanShp[1]-cld_gt[5])/pan_res , width=cld_ds.width*(res/pan_res) ,height=cld_ds.height*(res/pan_res)  )                       
       
        TaicldWin=Window(((cld_gt[2]-TaiMas_t[2])/res if (cld_gt[2]-TaiMas_t[2])/res>0 else (cld_gt[2]-TaiMas_t[2])/res ),((TaiMas_t[5]-cld_gt[5])/res if (TaiMas_t[5]-cld_gt[5])/res>0 else (TaiMas_t[5]-cld_gt[5])/res ), width=cld_ds.width ,height=cld_ds.height  )  #big cld mask                     

        
        ssimIndex=np.argsort(ssim[:,a])  
        a+=1
        ss=0
        for s in list(range(1,np.count_nonzero(ssim[:,a-1]>0.10)+1)):
            if ss==0:
                if ssim[ssimIndex[-s],a-1]>= 0.10:                    
                    fName=file[ssimIndex[-s]]                    
                    imagePool= rasterio.open(mypath+'\\'+fName)                    
                    gt1=imagePool.transform                    
                  
                    imgWin=Window(((gt1[2]-bigShp[0])/res if (gt1[2]-bigShp[0])/res>0 else (gt1[2]-bigShp[0])/res ),((bigShp[1]-gt1[5])/res if (bigShp[1]-gt1[5])/res>0 else (bigShp[1]-gt1[5])/res ),  width=imagePool.width ,height=imagePool.height  )
                    tw6mWin=Window(((gt1[2]-bigtw6mShp[0])/res if (gt1[2]-bigtw6mShp[0])/res>0 else (gt1[2]-bigtw6mShp[0])/res ),((bigtw6mShp[1]-gt1[5])/res if (bigtw6mShp[1]-gt1[5])/res>0 else (bigtw6mShp[1]-gt1[5])/res ),  width=imagePool.width ,height=imagePool.height  )
                    
                    imgWin_tai=Window(((gt1[2]-TaiMas_t[2])/res if (gt1[2]-TaiMas_t[2])/res>0 else (gt1[2]-TaiMas_t[2])/res ),((TaiMas_t[5]-gt1[5])/res if (TaiMas_t[5]-gt1[5])/res>0 else (TaiMas_t[5]-gt1[5])/res ),  width=imagePool.width ,height=imagePool.height  )#big cld mask 
                  
                    try:
                        
                        ints_window=windows.intersection(imgWin, cldWin)  #big taiwan
                        ints_window_tw6m=windows.intersection(tw6mWin, cldWin6m)  #tw6m
                        ints_window_TaicldWin=windows.intersection(imgWin_tai, TaicldWin) #big cld mask
                        write_window= Window(abs(imgWin.col_off-ints_window.col_off), abs(ints_window.row_off-imgWin.row_off), height= ints_window.height , width=ints_window.width)
                        cld_window=Window(abs(cldWin.col_off-ints_window.col_off), abs(ints_window.row_off-cldWin.row_off), height= ints_window.height , width=ints_window.width)
                                   
                        cld= cld_ds.read(1,window=cld_window)
                      
                        if np.any(border_elems_generic(cld,1)==1)==False:
                
                            cldbList=glob.glob(''.join(str(pathlib.Path().absolute()))+'\\'+candi+'\\'+'*'+fName[:-4]+'*.tif')        
                            cldb=rasterio.open(cldbList[0])
                            cldb=cldb.read(1,window=write_window)
                          
                             
                            imgrgb=imagePool.read(window=write_window)
                           
                            thresh=0.75
                            if ssim[ssimIndex[-1],a-1]==-0.91 and  np.any(imgrgb[0,:,:][cld==1]==0)==False and np.count_nonzero(cldb[cld==1]<10)/np.count_nonzero(cldb[cld==1])>0.85 :
                                cldcy=cld.copy()
                                blendPix=65
                                cldcy[np.where(cld==1)]=(blendPix+1)
                                try:                            
                                    mask_blurred  = cv2.GaussianBlur(cldcy,(blendPix,blendPix),0)
                                except:
                                    blendPix=35
                                    cldcy=cld.copy()
                                    cldcy[np.where(cld==1)]=(blendPix+1)
                                    mask_blurred  = cv2.GaussianBlur(cldcy,(blendPix,blendPix),0)  
                                if np.any(mask_blurred[0,:]!=0) or np.any(mask_blurred[-1,:]!=0) or np.any(mask_blurred[:,0]!=0) or np.any(mask_blurred[:,-1]!=0):
                                    blendPix=25
                                    cldcy=cld.copy()
                                    cldcy[np.where(cld==1)]=(blendPix+1)
                                    mask_blurred  = cv2.GaussianBlur(cldcy,(blendPix,blendPix),0) 
                               
                                mask_blurred_3chan = cv2.cvtColor(mask_blurred, cv2.COLOR_GRAY2BGR).astype('float32') / (blendPix+1)
                                mask_blurred_4chan = np.concatenate( [mask_blurred_3chan,mask_blurred[...,None]/(blendPix+1)], axis=2)
                                mask_blurred_4chan.astype('float32')
                              
                                
                                mask_blurred_4chan=np.moveaxis(mask_blurred_4chan, [0, 1, 2], [1, 2, 0]) 
                                
                                tw6mMask=tw6m.read(1,window=ints_window_tw6m)                            
                                out = 255 * np.zeros(bigImgrgb.shape)                        
                                temp=cldb.copy()
                                temp[cld==0]=1
                                temp[temp<=2]=1
                                temp[temp>10]=10                            
                              
                                kernel = np.ones((10,10), np.uint8)
                                temp = cv2.dilate(temp, kernel, iterations = 1)
                              
                                h, w = temp.shape[:2]
                                mask = np.zeros((h+2, w+2), np.uint8)
                                cv2.floodFill(temp, mask, (1,1), 0)  
                                temp[temp!=0]=1
                                
                                bigImg= rasterio.open(bigImgAddr)
                                
                                bigImgrgb=bigImg.read(window=ints_window)
                                outcome=bigImgrgb.astype(imgtype) *(1- mask_blurred_4chan)+ imgrgb.astype(imgtype) *( mask_blurred_4chan)      
                              
                                stage1+=1                                
                                Count+=1
                                outcome=outcome.astype(imgtype)                               
                                
                                with rasterio.open('MosTai.tif', 'r+') as dst:
                                    dst.write(outcome,  window=ints_window) 
                                bigImg.close() 
                                
                                TaiMas=rasterio.open(candiCld,'r+')
                                taicld=TaiMas.read(1,window=ints_window_TaicldWin)
                              
                                taicld[np.where((cld==1) & (temp==0) & (tw6mMask==1))]=0 
                                taicld[np.where( (temp==1) & (tw6mMask==1) ) ]=(i+1)*10
                                TaiMas.close()
                                with rasterio.open('TaiMas.tif', 'r+') as dst:
                                    dst.write(taicld.astype(imgtype),1,  window=ints_window_TaicldWin) 
                                                                            
                                mask_blurred_pan_chan=cv2.resize( mask_blurred_4chan[1,:,:],(int(mask_blurred_4chan[1,:,:].shape[1]*(res/pan_res)),int(mask_blurred_4chan[1,:,:].shape[0]*(res/pan_res))) ,interpolation=cv2.INTER_NEAREST)                                
                                                              
                                with rasterio.open( bigImgAddr) as src:                                   
                                    
                                    win_transform = src.window_transform(ints_window)
                                   
                                    panBounds=from_bounds(win_transform[2], win_transform[5]-ints_window.height*res, win_transform[2]+ints_window.width*res, win_transform[5], src.transform)#left, bottom, right, top, src.transform
                                MosTai_pan= rasterio.open( bigPan_Addr,'r+')# 
                              
                                bigpan = MosTai_pan.read(1 ,window=from_bounds(win_transform[2], win_transform[5]-ints_window.height*res, win_transform[2]+ints_window.width*res, win_transform[5], MosTai_pan.transform))
                                MosTai_pan.close()
                                
                                PAN_fName=fName[0]+str(int(fName[fName.find('0'):fName.find('0')+7])+1).zfill(7)
                                pan_imagePool = rasterio.open(Pan_Addr+'\\'+PAN_fName+'\\'+PAN_fName+'_B1.DAT.raw')       
                               
                                smallpan=pan_imagePool.read(1,window= from_bounds(win_transform[2], win_transform[5]-ints_window.height*res, win_transform[2]+ints_window.width*res, win_transform[5], pan_imagePool.transform))                                
                                pan_imagePool.close()
                                outcome=(smallpan.astype(imgtype) * mask_blurred_pan_chan + bigpan.astype(imgtype) *( 1-mask_blurred_pan_chan) ).astype(imgtype)    
                              
                                with rasterio.open( bigPan_Addr,'r+') as dst:                                
                                    dst.write( outcome, 1, window= from_bounds(win_transform[2], win_transform[5]-ints_window.height*res, win_transform[2]+ints_window.width*res, win_transform[5], dst.transform ) )
                                del outcome,smallpan,bigpan
                                ss=1    
                                    
                            elif np.any(imgrgb[0,:,:][cld==1]==0)==False and np.count_nonzero(cldb[cld==1]<10)/np.count_nonzero(cldb[cld==1])>thresh : #(cldb[cld==1]<10).all()==True   #cldb.all()<10
                                bigImg= rasterio.open(bigImgAddr)     
                                if np.count_nonzero(cld)==2000000:  
                                    cldcy=cld.copy()
                                    blendPix=65
                                    cldcy[np.where(cld==1)]=(blendPix+1)
                                    try:
                                      
                                        mask_blurred  = cv2.GaussianBlur(cldcy,(blendPix,blendPix),0)
                                    except:
                                        blendPix=35
                                        cldcy=cld.copy()
                                        cldcy[np.where(cld==1)]=(blendPix+1)
                                        mask_blurred  = cv2.GaussianBlur(cldcy,(blendPix,blendPix),0)  
                                    if np.any(mask_blurred[0,:]!=0) or np.any(mask_blurred[-1,:]!=0) or np.any(mask_blurred[:,0]!=0) or np.any(mask_blurred[:,-1]!=0):
                                        blendPix=25
                                        cldcy=cld.copy()
                                        cldcy[np.where(cld==1)]=(blendPix+1)
                                        mask_blurred  = cv2.GaussianBlur(cldcy,(blendPix,blendPix),0) 
                                   
                                    mask_blurred_3chan = cv2.cvtColor(mask_blurred, cv2.COLOR_GRAY2BGR).astype('float32') / (blendPix+1)
                                    mask_blurred_4chan = np.concatenate( [mask_blurred_3chan,mask_blurred[...,None]/(blendPix+1)], axis=2)
                                    mask_blurred_4chan.astype('float32')
                                  
                                    
                                    mask_blurred_4chan=np.moveaxis(mask_blurred_4chan, [0, 1, 2], [1, 2, 0]) 
                                    
                                    
                                    
                                    bigImgrgb=bigImg.read(window=ints_window)#/imgBits
                                    outcome=bigImgrgb.astype(imgtype) *(1- mask_blurred_4chan)+ imgrgb.astype(imgtype) *( mask_blurred_4chan)      
                               
                                
                                else:
                                    code=0
                                    imgrgb=imagePool.read(window=write_window)/imgBits
                                    bigImgrgb=bigImg.read(window=ints_window)/imgBits
                                   
                                     
                                    tw6mMask=tw6m.read(1,window=ints_window_tw6m)
                                    
                                    out = 255 * np.zeros(bigImgrgb.shape)
                                    
                                    temp=cldb.copy()
                                    temp[cld==0]=1
                                    temp[temp<=2]=1
                                    temp[temp>10]=10                            
                                  
                                    kernel = np.ones((10,10), np.uint8)
                                    temp = cv2.dilate(temp, kernel, iterations = 1)                              
                                    
                                    h, w = temp.shape[:2]
                                    mask = np.zeros((h+2, w+2), np.uint8)
                                    cv2.floodFill(temp, mask, (1,1), 0)  
                                    temp[temp!=0]=1
                                    
                                    # find center-----------
                                    br = cv2.boundingRect(cld) 
                                    centerOfBR = (br[0] + br[2] // 2, br[1] + br[3] // 2)
                                    
                                    centerOfBRn=list(centerOfBR)
                                    if  br[0] + br[2] == cld.shape[1]:
                                        centerOfBRn[0]=centerOfBRn[0]-1
                                        print('right:', 'loop:',i,'; cld_temp:',j,'; fName:', fName )
                                        right+=1
                                    if br[1] == 0:
                                        centerOfBRn[1]=centerOfBRn[1]+1
                                        print('up:', 'loop:',i,'; cld_temp:',j ,'; fName:', fName)
                                        up+=1
                                    if br[0] == 0:
                                      
                                        print('left:', 'loop:',i,'; cld_temp:',j,'; fName:', fName )
                                        left+=1
                                    if br[1] + br[3] == cld.shape[0]:
                                        centerOfBRn[1]=centerOfBRn[1]-1
                                        print('low:', 'loop:',i,'; cld_temp:',j ,'; fName:', fName)
                                        low+=1
                                    centerOfBR=tuple(centerOfBRn)                       
                                             
                                    for h  in [1,0]:
                                       
                                        bigImgrgb1=bigImgrgb[h:h+3,:,:].astype('uint8')
                                        imgrgb1=imgrgb[h:h+3,:,:].astype('uint8')
                                       
                                        imgrgb2=np.moveaxis(imgrgb1, [0, 1, 2], [2, 0, 1]).astype('uint8')
                                        bigImgrgb2=np.moveaxis(bigImgrgb1, [0, 1, 2], [2, 0, 1]).astype('uint8')
                                        imgrgb2[tw6mMask==0,:]=0
                                      
                                        imgrgb2[temp==1,:]=0
                                        bigImgrgb2[temp==1,:]=0
                                        cld[cld==1]=255
                                        mixed_clone = cv2.seamlessClone( imgrgb2, bigImgrgb2, cld, centerOfBR,cv2.NORMAL_CLONE)#cv2.MONOCHROME_TRANSFER)#cv2.NORMAL_CLONE
                                        
                                        
                                        mixed_clone=np.moveaxis(mixed_clone, [0, 1, 2], [1, 2, 0]).astype('uint16')
                                        out[h:h+3,:,:]=mixed_clone                             
                                    
                                    outcome =out *imgBits
                                    out2=out *imgBits
                                    for p in [0,1,2,3]:
                                        imgrgb=imagePool.read(p+1,window=write_window)
                                        out2[p,:,:][temp==1]=imgrgb[temp==1]#outcome
                                    if np.any(temp==1)==True:
                                        cldzeroNum+=1
                                        print('loop:',i,'; cld_temp:',j)
                                    if np.any(temp==1)==True and np.any(out2[0,:,:][tw6mMask==1]==0)==True:
                                        zeroNum+=1
                                     
                                        print('loop:',i,'; hole_Name:',j,'code:',code)
                                   
                                    
                                    if code==0:
                                        cldblendPix=25
                                        temp[np.where(temp==0)]=(cldblendPix+1)
                                        
                                        mask_blurred  = cv2.GaussianBlur(temp,(cldblendPix,cldblendPix),0)    
                                      
                                        mask_blurred_3chan = cv2.cvtColor(mask_blurred, cv2.COLOR_GRAY2BGR).astype('float32') / (cldblendPix+1)
                                        mask_blurred_4chan = np.concatenate( [mask_blurred_3chan,mask_blurred[...,None]/(cldblendPix+1)], axis=2)
                                        mask_blurred_4chan.astype('float32')
                                                              
                                        mask_blurred_4chan=np.moveaxis(mask_blurred_4chan, [0, 1, 2], [1, 2, 0]) 
                                        mask_blurred_4chan[:,temp==1]=0
                                        
                                      
                                        
                                        imgrgb=imagePool.read(window=write_window)
                                        outcome=outcome.astype(imgtype) * mask_blurred_4chan + imgrgb.astype(imgtype) *( 1-mask_blurred_4chan)      
                                       
                                        
                                        cldcy=cld.copy()
                                        blendPix=65
                                        cldcy[np.where(cld==255)]=(blendPix+1)
                                        
                                        bigImgrgb=bigImg.read(window=ints_window)
                                        
                                        try:
                                         
                                            mask_blurred  = cv2.GaussianBlur(cldcy,(blendPix,blendPix),0)
                                        except:
                                            blendPix=35
                                            cldcy=cld.copy()
                                            cldcy[np.where(cld==255)]=(blendPix+1)
                                            mask_blurred  = cv2.GaussianBlur(cldcy,(blendPix,blendPix),0)  
                                        if np.any(mask_blurred[0,:]!=0) or np.any(mask_blurred[-1,:]!=0) or np.any(mask_blurred[:,0]!=0) or np.any(mask_blurred[:,-1]!=0):
                                            blendPix=25
                                            cldcy=cld.copy()
                                            cldcy[np.where(cld==255)]=(blendPix+1)
                                            mask_blurred  = cv2.GaussianBlur(cldcy,(blendPix,blendPix),0) 
                               
                                       
                                        mask_blurred_3chan = cv2.cvtColor(mask_blurred, cv2.COLOR_GRAY2BGR).astype('float32') / (blendPix+1)
                                        mask_blurred_4chan = np.concatenate( [mask_blurred_3chan,mask_blurred[...,None]/(blendPix+1)], axis=2)
                                        mask_blurred_4chan.astype('float32')
                                        mask_blurred_4chan=np.moveaxis(mask_blurred_4chan, [0, 1, 2], [1, 2, 0]) 
                                        outcome=outcome.astype(imgtype) * mask_blurred_4chan + bigImgrgb.astype(imgtype) *( 1-mask_blurred_4chan)      
                                    
                #------------------------------------------------------------------------------------------------
                                if code==0:
                                    
                                    stage2+=1
                                    
                                    Count+=1
                                    outcome=outcome.astype(imgtype)
                                  
                                    
                                    with rasterio.open('MosTai.tif', 'r+') as dst:                                
                                        dst.write(outcome,  window=ints_window) 
                                    bigImg.close()    
                                    
                                    TaiMas=rasterio.open(candiCld,'r+')
                                    taicld=TaiMas.read(1,window=ints_window_TaicldWin)
                                    
                                    taicld[np.where((cld==255) & (temp!=1) & (tw6mMask==1))]=0 
                                    taicld[np.where( (temp==1) & (tw6mMask==1) ) ]=(i+1)*10
                                    TaiMas.close()
                                    with rasterio.open('TaiMas.tif', 'r+') as dst:                                
                                        dst.write(taicld.astype(imgtype),1,  window=ints_window_TaicldWin) 
                                 
                                    #----------for pan-------------------------------------------                                 
                                                       
                                    mask_blurred_pan_chan=cv2.resize( mask_blurred_4chan[1,:,:],(int(mask_blurred_4chan[1,:,:].shape[1]*(res/pan_res)),int(mask_blurred_4chan[1,:,:].shape[0]*(res/pan_res))) ,interpolation=cv2.INTER_NEAREST)                                
                                    
                                    with rasterio.open( bigImgAddr) as src:  
                                        win_transform = src.window_transform(ints_window)                                       
                                        panBounds=from_bounds(win_transform[2], win_transform[5]-ints_window.height*res, win_transform[2]+ints_window.width*res, win_transform[5], src.transform)#left, bottom, right, top, src.transform
                                    MosTai_pan= rasterio.open( bigPan_Addr,'r+')                                   
                                    bigpan = MosTai_pan.read(1 ,window=from_bounds(win_transform[2], win_transform[5]-ints_window.height*res, win_transform[2]+ints_window.width*res, win_transform[5], MosTai_pan.transform))
                                    MosTai_pan.close()   
                                    PAN_fName=fName[0]+str(int(fName[fName.find('0'):fName.find('0')+7])+1).zfill(7)
                                    pan_imagePool = rasterio.open(Pan_Addr+'\\'+PAN_fName+'\\'+PAN_fName+'_B1.DAT.raw')       
                                    
                                    smallpan=pan_imagePool.read(1,window= from_bounds(win_transform[2], win_transform[5]-ints_window.height*res, win_transform[2]+ints_window.width*res, win_transform[5], pan_imagePool.transform))                                
                                    pan_imagePool.close()
                                    outcome=(smallpan.astype(imgtype) * mask_blurred_pan_chan + bigpan.astype(imgtype) *( 1-mask_blurred_pan_chan) ).astype(imgtype)    
                                    with rasterio.open( bigPan_Addr,'r+') as dst:                                
                                        dst.write( outcome, 1, window= from_bounds(win_transform[2], win_transform[5]-ints_window.height*res, win_transform[2]+ints_window.width*res, win_transform[5], dst.transform ) )
                                    del outcome,smallpan,bigpan
                                    ss=1                     
                                
                            elif ssim[ssimIndex[-1],a-1]==-0.9 and np.any(imgrgb[0,:,:][cld==1]==0)==False and np.count_nonzero(cldb[cld==1]<10)/np.count_nonzero(cldb[cld==1])>0.55:
                                cldcy=cld.copy()
                                blendPix=65
                                cldcy[np.where(cld==1)]=(blendPix+1)
                                try:
                                    mask_blurred  = cv2.GaussianBlur(cldcy,(blendPix,blendPix),0)
                                except:
                                    blendPix=35
                                    cldcy=cld.copy()
                                    cldcy[np.where(cld==1)]=(blendPix+1)
                                    mask_blurred  = cv2.GaussianBlur(cldcy,(blendPix,blendPix),0)  
                                if np.any(mask_blurred[0,:]!=0) or np.any(mask_blurred[-1,:]!=0) or np.any(mask_blurred[:,0]!=0) or np.any(mask_blurred[:,-1]!=0):
                                    blendPix=25
                                    cldcy=cld.copy()
                                    cldcy[np.where(cld==1)]=(blendPix+1)
                                    mask_blurred  = cv2.GaussianBlur(cldcy,(blendPix,blendPix),0)                              
                                mask_blurred_3chan = cv2.cvtColor(mask_blurred, cv2.COLOR_GRAY2BGR).astype('float32') / (blendPix+1)
                                mask_blurred_4chan = np.concatenate( [mask_blurred_3chan,mask_blurred[...,None]/(blendPix+1)], axis=2)
                                mask_blurred_4chan.astype('float32')                                
                                
                                mask_blurred_4chan=np.moveaxis(mask_blurred_4chan, [0, 1, 2], [1, 2, 0])                                 
                                tw6mMask=tw6m.read(1,window=ints_window_tw6m)                            
                                out = 255 * np.zeros(bigImgrgb.shape)                        
                                temp=cldb.copy()
                                temp[cld==0]=1
                                temp[temp<=2]=1
                                temp[temp>10]=10   
                                kernel = np.ones((10,10), np.uint8)
                                temp = cv2.dilate(temp, kernel, iterations = 1)                              
                                h, w = temp.shape[:2]
                                mask = np.zeros((h+2, w+2), np.uint8)
                                cv2.floodFill(temp, mask, (1,1), 0)  
                                temp[temp!=0]=1                                
                                bigImg= rasterio.open(bigImgAddr)                                
                                bigImgrgb=bigImg.read(window=ints_window)
                                outcome=bigImgrgb.astype(imgtype) *(1- mask_blurred_4chan)+ imgrgb.astype(imgtype) *( mask_blurred_4chan)      
                               
                                stage3+=1                                
                                Count+=1
                                outcome=outcome.astype(imgtype)                                
                                
                                with rasterio.open('MosTai.tif', 'r+') as dst:
                                    dst.write(outcome,  window=ints_window) 
                                bigImg.close() 
                                
                                TaiMas=rasterio.open(candiCld,'r+')
                                taicld=TaiMas.read(1,window=ints_window_TaicldWin)                                
                                taicld[np.where((cld==1) & (temp==0) & (tw6mMask==1))]=0 
                                taicld[np.where( (temp==1) & (tw6mMask==1) ) ]=(i+1)*10
                                TaiMas.close()
                                with rasterio.open('TaiMas.tif', 'r+') as dst:
                                    dst.write(taicld.astype(imgtype), 1, window=ints_window_TaicldWin)
                                mask_blurred_pan_chan=cv2.resize( mask_blurred_4chan[1,:,:],(int(mask_blurred_4chan[1,:,:].shape[1]*(res/pan_res)),int(mask_blurred_4chan[1,:,:].shape[0]*(res/pan_res))) ,interpolation=cv2.INTER_NEAREST)                                
                                                                
                                with rasterio.open( bigImgAddr) as src:    
                                    win_transform = src.window_transform(ints_window)
                                    panBounds=from_bounds(win_transform[2], win_transform[5]-ints_window.height*res, win_transform[2]+ints_window.width*res, win_transform[5], src.transform)#left, bottom, right, top, src.transform
                                MosTai_pan= rasterio.open( bigPan_Addr,'r+')                                
                                bigpan = MosTai_pan.read(1 ,window=from_bounds(win_transform[2], win_transform[5]-ints_window.height*res, win_transform[2]+ints_window.width*res, win_transform[5], MosTai_pan.transform))
                                MosTai_pan.close()                             
                                PAN_fName=fName[0]+str(int(fName[fName.find('0'):fName.find('0')+7])+1).zfill(7)
                                pan_imagePool = rasterio.open(Pan_Addr+'\\'+PAN_fName+'\\'+PAN_fName+'_B1.DAT.raw')       
                                
                                smallpan=pan_imagePool.read(1,window= from_bounds(win_transform[2], win_transform[5]-ints_window.height*res, win_transform[2]+ints_window.width*res, win_transform[5], pan_imagePool.transform))                                
                                pan_imagePool.close()
                                outcome=(smallpan.astype(imgtype) * mask_blurred_pan_chan + bigpan.astype(imgtype) *( 1-mask_blurred_pan_chan) ).astype(imgtype)    
                                with rasterio.open( bigPan_Addr,'r+') as dst:                                
                                    dst.write( outcome, 1, window= from_bounds(win_transform[2], win_transform[5]-ints_window.height*res, win_transform[2]+ints_window.width*res, win_transform[5], dst.transform ) )
                                del outcome,smallpan,bigpan                                
                                ss=1  
                                
                    except:
                      
                        pass
                    
    print('stage1:',stage1)
    print('stage2:',stage2) 
    print('cldzeroNum:',cldzeroNum)
    print('zeroExist:',zeroNum)
    print('stage3:',stage3)
    print('right,up,left,low:',right,up,left,low)
    cld_ds.close()
    TaiMas.close()
    dirpath=cldpath+'\\'
    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)
print('All done!')
print(Count,'/',total,' reconstructions of cloud-contaminated area')
end = timer()
print('time(hr):',(end - start)/60/60) # Time in seconds


