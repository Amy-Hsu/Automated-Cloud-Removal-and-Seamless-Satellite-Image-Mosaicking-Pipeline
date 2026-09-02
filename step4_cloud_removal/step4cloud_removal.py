# -*- coding: utf-8 -*-
"""
Created on Thu May 13 09:52:10 2021

@author: amy hsu
"""
#process pan at the same  time2021/11/25

# cloud Process
import numpy as np
import cv2
import rasterio
import matplotlib.pyplot as plt
# import networkx as nx
# import scipy.ndimage
from rasterio import Affine as A
# import geopandas as gpd
from os import listdir
from os.path import isfile, join
# from operator import itemgetter   
# import copy
from timeit import default_timer as timer  
# import collections
# import tqdm
# import time
from rasterio.windows import Window
# import os    
import pathlib
# import json
import glob
from scipy import ndimage
from rasterio import windows
import os
import shutil
from skimage.metrics import structural_similarity
# from skimage import exposure
import json
# import shutil
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

# blendPix=65#odd number only

imgBits=(2**trueBits)/(2**8)
if trueBits==12:
    imgtype='uint16'
else:
    imgtype='uint'+str(trueBits)
#---------read cld mask--------------
# with open('ProcessOrder.txt') as f:
#     for line in f.readlines():
#         ReadOrder = line.split(',')  

my_file = open('ProcessOrder.txt', "r")
ReadOrder =  my_file.read().split(",")
my_file.close()
TaiMas=rasterio.open(candiCld,'r+')
TaiMas_t=TaiMas.transform
# shutil.copy(candiCld, 'TaiMasRaw.tif')
# shutil.copy(Pan_Addr, 'MosTai_Pan_Raw.tif')
#to load allWindow.txt:
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
#%%
for i in range(len(ReadOrder)-1):#[0]:#

    #read TaiMas correspondening to allwindow.
    TaiMas=rasterio.open(candiCld,'r+')
    cldRdbineary0=TaiMas.read(1,window=allWindow[i])
    cldRdbineary=cldRdbineary0.copy()
    cldRdbineary[cldRdbineary0!=1]=0
    #read candidates from cloud database:
    # cloud_database
    # cldRdbineary[cldRdbineary!=i+1]=0
    
    # num_objects, labels = cv2.connectedComponents(cldRdbineary.astype('uint8'))
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cldRdbineary.astype('uint8'), connectivity=8, ltype=None)
    # num_labels, labels= cv2.connectedComponentsWithStats(cldRdbineary.astype('uint8'), connectivity=8)
    for a in range(1,num_labels):
        if stats[a,4]<500:#you can adjust manually. default is 500
            delnum+=1
            cldRdbineary[labels==a]=0
   

    print('scraps num:',delnum)
#--------------------------

    kernel = np.ones((50,50), np.uint8)
    dilation = cv2.dilate(cldRdbineary, kernel, iterations = 1).astype('uint8')
    
    contours, hierarchy = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img=cv2.drawContours(dilation.copy(), contours, -1, (128, 0, 255), 2)
    
    
    num_objects, labels = cv2.connectedComponents(dilation, connectivity=8)
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours] 
    for bbox in bounding_boxes:
         [x , y, w, h] = bbox
         cv2.rectangle(dilation, (x-10, y-10), (x + w + 10, y + h + 10), (255,0, 0), 3)

    out = img*125 + dilation # can show picture by combining dilation and object detecton retangle
    
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
        
        # Mt= rasterio.open(
        #         str(c)+'_cld.img',
        #         'w+',
               
        #         height= h+20,
        #         width=  w+20,
        #         count=1,
        #         dtype=rasterio.uint8,
        #         # nodata=0,
        #         crs= {'init': 'EPSG:3826'},#Initialize from a named CRS
        #         transform=cld_gt)
        # Mt.write(labelscy2.astype(rasterio.uint8), 1) 
        # Mt.close()
        total+=1
        with rasterio.open(
                    'cldDatabase\\'+str(c)+'_cld.img',
                    'w+',           
                    height=h+(expdw+expdh),
                    width=w+(expdw+expdh),
                    count=1,
                    dtype=rasterio.uint8,           
                    crs= {'init': 'EPSG:3826'},#Initialize from a named CRS
                    transform=cld_gt
                ) as dst:
                    dst.write(labelscy2.astype(rasterio.uint8), 1)

    if i ==0:
        files = listdir(mypath)
        gt_two=[]
        file=[]
        gt_five=[]
        # Loop through files
        for f in files:
          # Build absolute path
          fullpath1 = join(mypath, f)
          # Check if it's a file or directory
          if isfile(fullpath1) and  f.endswith(".img"):
            image=rasterio.open(mypath+'\\'+f)
            file.append(f)
            gt_two.append(image.transform[2] )
            gt_five.append(image.transform[5] )
            
            
        # bigExt= int( ( float(max( gt_five))-  2416000   )/res ),int( (   406000-  float(min( gt_two)   ))/res )
        # bigShp= min( gt_two),max( gt_five)
        
        
    files = listdir(cldpath)
    # gt_two=[]
    cldfile=[]
    # gt_five=[]
    # Loop through files
    for f in files:
      # Build absolute path
      fullpath = join(cldpath, f)
      # Check if it's a file or directory
      if isfile(fullpath) and  f.endswith(".img"):
        # image=rasterio.open(f)
        cldfile.append(f)
        # gt_two.append(image.transform[2] )
        # gt_five.append(image.transform[5] )
    bigImg= rasterio.open(bigImgAddr)
    ssim= np.empty((50,len(cldfile),))* np.nan
    ssim=np.zeros((50,len(cldfile),))
    # intsWins=np.empty((20,len(cldfile),))* np.nan
    # cldWins=np.empty((20,len(cldfile),))* np.nan
    # imgWins=np.empty((20,len(cldfile),))* np.nan
    
    bigShp=bigImg.transform[2],bigImg.transform[5]

    print('calculate ssim...',i+1,'/',len(allWindow))  
    a=0                        
    for j in cldfile:
        cld_ds = rasterio.open(cldpath+'\\'+j)
        cld_gt=cld_ds.transform                  
        cldWin=Window(((cld_gt[2]-bigShp[0])/res if (cld_gt[2]-bigShp[0])/res>0 else (cld_gt[2]-bigShp[0])/res ),((bigShp[1]-cld_gt[5])/res if (bigShp[1]-cld_gt[5])/res>0 else (bigShp[1]-cld_gt[5])/res ), width=cld_ds.width ,height=cld_ds.height  )   #full cld mask                    
        cldWin6m=Window(((cld_gt[2]-bigtw6mShp[0])/res if (cld_gt[2]-bigtw6mShp[0])/res>0 else (cld_gt[2]-bigtw6mShp[0])/res ),((bigtw6mShp[1]-cld_gt[5])/res if (bigtw6mShp[1]-cld_gt[5])/res>0 else (bigtw6mShp[1]-cld_gt[5])/res ), width=cld_ds.width ,height=cld_ds.height  )   #full cld mask                    
        
        b=0
        for k in file:# candidates images pool
            imagePool= rasterio.open(mypath+'\\'+k)
        
            gt1=imagePool.transform
            
            # this = sys.modules[__name__] 
            # setattr(this, 'ssim_cld%s' % a, [])
            
            # imgWin=(gt1[2]-bigShp[0])/res-1,(bigShp[1]-gt1[5])/res-1
            imgWin=Window(((gt1[2]-bigShp[0])/res if (gt1[2]-bigShp[0])/res>0 else (gt1[2]-bigShp[0])/res ),((bigShp[1]-gt1[5])/res if (bigShp[1]-gt1[5])/res>0 else (bigShp[1]-gt1[5])/res ),  width=imagePool.width ,height=imagePool.height  )
            tw6mWin=Window(((gt1[2]-bigtw6mShp[0])/res if (gt1[2]-bigtw6mShp[0])/res>0 else (gt1[2]-bigtw6mShp[0])/res ),((bigtw6mShp[1]-gt1[5])/res if (bigtw6mShp[1]-gt1[5])/res>0 else (bigtw6mShp[1]-gt1[5])/res ),  width=imagePool.width ,height=imagePool.height  )

            try:
          
                ints_window=windows.intersection(imgWin, cldWin)  #big taiwan                
                ints_window_tw6m=windows.intersection(tw6mWin, cldWin6m)  #tw6m
                write_window= Window(abs(imgWin.col_off-ints_window.col_off), abs(ints_window.row_off-imgWin.row_off), height= ints_window.height , width=ints_window.width)
                cld_window=Window(abs(cldWin.col_off-ints_window.col_off), abs(ints_window.row_off-cldWin.row_off), height= ints_window.height , width=ints_window.width)# intersection cloud mask
                # intsWins[b,a]=ints_window
                # cldWins[b,a]=cld_window
                # imgWins[b,a]=write_window
    
                
                cld= cld_ds.read(1,window=cld_window)
                
                
                bigImgrgb=np.dstack((bigImg.read(3,window=ints_window)/imgBits, bigImg.read(2,window=ints_window)/imgBits,bigImg.read(1,window=ints_window)/imgBits)).astype('uint8')
                bigimgRGB=bigImgrgb.copy()
                for d in range(3):
                    bigimgRGB[:,:,d][np.where(cld==1)]=0 # hallow the cld rgb
                    
                bigImg2gray=cv2.cvtColor(bigImgrgb,cv2.COLOR_RGB2GRAY) 
                bigImg2gray[np.where(cld==1)]=0# hallow the cld gray
                
                imgrgb= np.dstack((imagePool.read(3,window=write_window)/imgBits, imagePool.read(2,window=write_window)/imgBits,imagePool.read(1,window=write_window)/imgBits)).astype('uint8')
                imgRGB=imgrgb.copy()
                for d in range(3):
                    imgRGB[:,:,d][np.where(cld==1)]=0 # hallow the cld rgb
                    
                
                tw6mMask=tw6m.read(1,window=ints_window_tw6m)
                
                imgRGB[tw6mMask==0]=0
                
    
                ssim[b,a]=  structural_similarity(bigimgRGB, imgRGB, multichannel=True,dynamic_range=255,K1=0.01,K2=0.03) if cld_window==cldWin else structural_similarity(bigimgRGB, imgRGB, multichannel=True,dynamic_range=255,K1=0.01,K2=0.03)-0.05
                
                # b+=1
                # # from SSIM_PIL import compare_ssim 
                # # value = compare_ssim(imgGray, bigImg2gray) # Compare images using OpenCL by default
                
                if np.all(bigimgRGB[:]==0) and np.all(imgRGB[:]==0):#to exclude the image which only contains 0.
                    ssim[b,a]=-3
                if np.all(imgRGB[cld==0]==0):# to exclude the image whose intersecting area only contains 0.
                    ssim[b,a]=-3
            except:
                ssim[b,a]=-1
                # b+=1
             
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
    # to use possion or feather to mend cloud hole from the full extent Taiwan image.
    a=0
#%%    
    print('cloud removal...',i+1,'/',len(allWindow)) 
    for j in cldfile:
      
        cld_ds = rasterio.open(cldpath+'\\'+j)
        cld_gt=cld_ds.transform 
                 
        cldWin=Window(((cld_gt[2]-bigShp[0])/res if (cld_gt[2]-bigShp[0])/res>0 else (cld_gt[2]-bigShp[0])/res ),((bigShp[1]-cld_gt[5])/res if (bigShp[1]-cld_gt[5])/res>0 else (bigShp[1]-cld_gt[5])/res ), width=cld_ds.width ,height=cld_ds.height  )                       
        cldWin6m=Window(((cld_gt[2]-bigtw6mShp[0])/res if (cld_gt[2]-bigtw6mShp[0])/res>0 else (cld_gt[2]-bigtw6mShp[0])/res ),((bigtw6mShp[1]-cld_gt[5])/res if (bigtw6mShp[1]-cld_gt[5])/res>0 else (bigtw6mShp[1]-cld_gt[5])/res ), width=cld_ds.width ,height=cld_ds.height  )   #full cld mask                    

        PancldWin=Window((cld_gt[2]-bigPanShp[0])/pan_res ,(bigPanShp[1]-cld_gt[5])/pan_res , width=cld_ds.width*(res/pan_res) ,height=cld_ds.height*(res/pan_res)  )                       
        # cldWin=Window(((cld_gt[2]-bigShp[0])/res if (cld_gt[2]-bigShp[0])/res>=0 else 0),((bigShp[1]-cld_gt[5])/res if (bigShp[1]-cld_gt[5])/res>=0 else 0 ), width=cld_ds.width ,height=cld_ds.height  )                       
        
        TaicldWin=Window(((cld_gt[2]-TaiMas_t[2])/res if (cld_gt[2]-TaiMas_t[2])/res>0 else (cld_gt[2]-TaiMas_t[2])/res ),((TaiMas_t[5]-cld_gt[5])/res if (TaiMas_t[5]-cld_gt[5])/res>0 else (TaiMas_t[5]-cld_gt[5])/res ), width=cld_ds.width ,height=cld_ds.height  )  #big cld mask                     

        
        ssimIndex=np.argsort(ssim[:,a])  
        a+=1
        ss=0
        for s in list(range(1,np.count_nonzero(ssim[:,a-1]>0.10)+1)):
            if ss==0:
                if ssim[ssimIndex[-s],a-1]>= 0.10:#set threshold to filter not good enough img.
                
                # # to exclude the replicated used image. may encounter problems when eligible image is only one!?
                # for p in range(-1, -(len(ssimIndex)+1) ,-1) :
                #     if ssim[ssimIndex[p],a-1]>= 0.10:
                #         if i==0 :
                #             if ( (file[ssimIndex[p]] ==  ReadOrder[i][ReadOrder[i].find(str(file[ssimIndex[-1]][0:2])):ReadOrder[i].find(file[ssimIndex[-1]][0:2])+len(file[ssimIndex[-1]])] )
                #             or (file[ssimIndex[p]] ==  ReadOrder[i+1][ReadOrder[i+1].find(str(file[ssimIndex[-1]][0:2])):ReadOrder[i+1].find(file[ssimIndex[-1]][0:2])+len(file[ssimIndex[-1]])]  )):
                #                 if ( (file[ssimIndex[p-1]] ==  ReadOrder[i][ReadOrder[i].find(str(file[ssimIndex[-1]][0:2])):ReadOrder[i].find(file[ssimIndex[-1]][0:2])+len(file[ssimIndex[-1]])] )
                #                 or (file[ssimIndex[p-1]] ==  ReadOrder[i+1][ReadOrder[i+1].find(str(file[ssimIndex[-1]][0:2])):ReadOrder[i+1].find(file[ssimIndex[-1][0:2]])+len(file[ssimIndex[-1]])]  )):
                #                     fName=file[ssimIndex[p-2]]
                #                 else:
                #                     fName=file[ssimIndex[p-1]]
                #             else:
                #                 fName=file[ssimIndex[p]]
                #         else:
                #             if file[ssimIndex[p]] ==  ReadOrder[i+1][ReadOrder[i+1].find(str(file[ssimIndex[-1]][0:2])):ReadOrder[i+1].find(file[ssimIndex[-1]][0:2])+len(file[ssimIndex[-1]])]:
                #                 fName=file[ssimIndex[p-1]]
                #             else:
                #                 fName=file[ssimIndex[p]]
                # #-------------------------------------
                    
                    fName=file[ssimIndex[-s]]
                    
                    imagePool= rasterio.open(mypath+'\\'+fName)
                    
                    gt1=imagePool.transform
                    
                    # this = sys.modules[__name__] 
                    # setattr(this, 'ssim_cld%s' % a, [])
                            
                    # imgWin=(gt1[2]-bigShp[0])/res-1,(bigShp[1]-gt1[5])/res-1
                    
                    imgWin=Window(((gt1[2]-bigShp[0])/res if (gt1[2]-bigShp[0])/res>0 else (gt1[2]-bigShp[0])/res ),((bigShp[1]-gt1[5])/res if (bigShp[1]-gt1[5])/res>0 else (bigShp[1]-gt1[5])/res ),  width=imagePool.width ,height=imagePool.height  )
                    tw6mWin=Window(((gt1[2]-bigtw6mShp[0])/res if (gt1[2]-bigtw6mShp[0])/res>0 else (gt1[2]-bigtw6mShp[0])/res ),((bigtw6mShp[1]-gt1[5])/res if (bigtw6mShp[1]-gt1[5])/res>0 else (bigtw6mShp[1]-gt1[5])/res ),  width=imagePool.width ,height=imagePool.height  )
                    
                    imgWin_tai=Window(((gt1[2]-TaiMas_t[2])/res if (gt1[2]-TaiMas_t[2])/res>0 else (gt1[2]-TaiMas_t[2])/res ),((TaiMas_t[5]-gt1[5])/res if (TaiMas_t[5]-gt1[5])/res>0 else (TaiMas_t[5]-gt1[5])/res ),  width=imagePool.width ,height=imagePool.height  )#big cld mask 
                    
                    
                    # imgWin=Window(((gt1[2]-bigShp[0])/res if (gt1[2]-bigShp[0])/res>=0 else 0 ),((bigShp[1]-gt1[5])/res if (bigShp[1]-gt1[5])/res>=0 else 0 ),  width=imagePool.width ,height=imagePool.height  )
                    # tw6mWin=Window(((gt1[2]-bigtw6mShp[0])/res if (gt1[2]-bigtw6mShp[0])/res>=0 else 0 ),((bigtw6mShp[1]-gt1[5])/res if (bigtw6mShp[1]-gt1[5])/res>=0 else 0 ),  width=imagePool.width ,height=imagePool.height  )
                    
                    
                    try:
                        
                        ints_window=windows.intersection(imgWin, cldWin)  #big taiwan
                        ints_window_tw6m=windows.intersection(tw6mWin, cldWin6m)  #tw6m
                        ints_window_TaicldWin=windows.intersection(imgWin_tai, TaicldWin) #big cld mask
                        write_window= Window(abs(imgWin.col_off-ints_window.col_off), abs(ints_window.row_off-imgWin.row_off), height= ints_window.height , width=ints_window.width)
                        cld_window=Window(abs(cldWin.col_off-ints_window.col_off), abs(ints_window.row_off-cldWin.row_off), height= ints_window.height , width=ints_window.width)
                                   
                        cld= cld_ds.read(1,window=cld_window)
                        
                        # If 1 exists in border; then skip the cld file.
                        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        if np.any(border_elems_generic(cld,1)==1)==False:# or cld.shape[-2]*cld.shape[-1]>36000000:# to prevent replicated cld mask from reprocessing several times.
                
                            cldbList=glob.glob(''.join(str(pathlib.Path().absolute()))+'\\'+candi+'\\'+'*'+fName[:-4]+'*.tif')        
                            cldb=rasterio.open(cldbList[0])
                            cldb=cldb.read(1,window=write_window)
                             
                            imgrgb=imagePool.read(window=write_window)#/imgBits
                            
                            # if imgrgb.shape[-2]*imgrgb.shape[-1]>36000000: #200000:
                            #     thresh=0.90
                            # else:
                            #     thresh=0.99                 
                            thresh=0.75#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
                                # mask_blurred_1chan = mask_blurred.astype('float32') / (blendPix+1)
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
                                temp = cv2.dilate(temp, kernel, iterations = 1)#.astype('uint8')
                                h, w = temp.shape[:2]
                                mask = np.zeros((h+2, w+2), np.uint8)
                                cv2.floodFill(temp, mask, (1,1), 0)  
                                temp[temp!=0]=1
                                
                                bigImg= rasterio.open(bigImgAddr)
                                
                                bigImgrgb=bigImg.read(window=ints_window)#/imgBits
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
                                # TaiMas=rasterio.open(candiCld,'r+')    
                                
                                #----------for pan-------------------------------------------                                 
                                #resize cldb mask to pan's res                            
                                mask_blurred_pan_chan=cv2.resize( mask_blurred_4chan[1,:,:],(int(mask_blurred_4chan[1,:,:].shape[1]*(res/pan_res)),int(mask_blurred_4chan[1,:,:].shape[0]*(res/pan_res))) ,interpolation=cv2.INTER_NEAREST)                                
                                
                                
                                #read big pan which is cldb mask's counterpart 
                                with rasterio.open( bigImgAddr) as src:                                   
                                    
                                    win_transform = src.window_transform(ints_window)
                                    # bigImg_win_transform=win_transform[2],win_transform[5]
                                    # del win_transform
                                    panBounds=from_bounds(win_transform[2], win_transform[5]-ints_window.height*res, win_transform[2]+ints_window.width*res, win_transform[5], src.transform)#left, bottom, right, top, src.transform
                                MosTai_pan= rasterio.open( bigPan_Addr,'r+')# 
                                # MosTai_panWin=MosTai_pan.transform[2],MosTai_pan.transform[5]
                                # profile = MosTai_pan.profile
                                # profile.update({
                                #     'height': ints_window.height*(res/pan_res) ,
                                #     'width':ints_window.width*(res/pan_res),
                                #     'transform': win_transform})
                                bigpan = MosTai_pan.read(1 ,window=from_bounds(win_transform[2], win_transform[5]-ints_window.height*res, win_transform[2]+ints_window.width*res, win_transform[5], MosTai_pan.transform))
                                MosTai_pan.close()
                                
                                
                                #read pan_imagePool which is cldb mask's counterpart 
                                PAN_fName=fName[0]+str(int(fName[fName.find('0'):fName.find('0')+7])+1).zfill(7)
                                pan_imagePool = rasterio.open(Pan_Addr+'\\'+PAN_fName+'\\'+PAN_fName+'_B1.DAT.raw')       
                                # write_window= Window(abs(imgWin.col_off-ints_window.col_off), abs(ints_window.row_off-imgWin.row_off), height= ints_window.height , width=ints_window.width)
                                
                                smallpan=pan_imagePool.read(1,window= from_bounds(win_transform[2], win_transform[5]-ints_window.height*res, win_transform[2]+ints_window.width*res, win_transform[5], pan_imagePool.transform))                                
                                # panWin=Window( (pan_imagePool.transform[2]-MosTai_panWin[0])/pan_res ,(MosTai_panWin[1]-pan_imagePool.transform[5])/pan_res ,  width=pan_imagePool.width ,height=pan_imagePool.height  )
                                pan_imagePool.close()
                                outcome=(smallpan.astype(imgtype) * mask_blurred_pan_chan + bigpan.astype(imgtype) *( 1-mask_blurred_pan_chan) ).astype(imgtype)
                                with rasterio.open( bigPan_Addr,'r+') as dst:                                
                                    dst.write( outcome, 1, window= from_bounds(win_transform[2], win_transform[5]-ints_window.height*res, win_transform[2]+ints_window.width*res, win_transform[5], dst.transform ) )
                                del outcome,smallpan,bigpan
                                ss=1    
                                    
                            elif np.any(imgrgb[0,:,:][cld==1]==0)==False and np.count_nonzero(cldb[cld==1]<10)/np.count_nonzero(cldb[cld==1])>thresh : #(cldb[cld==1]<10).all()==True   #cldb.all()<10
                                # aa=1
                                #-way1---blend---------------------------------------------------------------------------------
                                bigImg= rasterio.open(bigImgAddr)     
                                if np.count_nonzero(cld)==2000000: #ssim[ssimIndex[-1],a-1]>=0.95  # > selectable !!!!!!!!  >1: 
                                    cldcy=cld.copy()
                                    blendPix=65
                                    cldcy[np.where(cld==1)]=(blendPix+1)
                                    try:
                                        # blendPix=65
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
                                    # mask_blurred_1chan = mask_blurred.astype('float32') / (blendPix+1)
                                    mask_blurred_3chan = cv2.cvtColor(mask_blurred, cv2.COLOR_GRAY2BGR).astype('float32') / (blendPix+1)
                                    mask_blurred_4chan = np.concatenate( [mask_blurred_3chan,mask_blurred[...,None]/(blendPix+1)], axis=2)
                                    mask_blurred_4chan.astype('float32')

                                    mask_blurred_4chan=np.moveaxis(mask_blurred_4chan, [0, 1, 2], [1, 2, 0])

                                    bigImgrgb=bigImg.read(window=ints_window)#/imgBits
                                    outcome=bigImgrgb.astype(imgtype) *(1- mask_blurred_4chan)+ imgrgb.astype(imgtype) *( mask_blurred_4chan)
                                #---------------------------------------------------------------------------------------------------
                                
                                #--histogram matching-20210716 closed testing,dead end.--
                                    
                                #-----
                                
                                else:
                                #-way2---poisson---------------------------------------------------------------------------------
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
                                    temp = cv2.dilate(temp, kernel, iterations = 1)#.astype('uint8')

                                    h, w = temp.shape[:2]
                                    mask = np.zeros((h+2, w+2), np.uint8)
                                    cv2.floodFill(temp, mask, (1,1), 0)  
                                    temp[temp!=0]=1
                                    
                                    # find center-----------
                                    br = cv2.boundingRect(cld) # bounding rect (x,y,width,height)
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
                                    if br[0] == 0:#!!!!!needs more sample
                                        # centerOfBRn[0]=centerOfBRn[0]+1
                                        print('left:', 'loop:',i,'; cld_temp:',j,'; fName:', fName )
                                        left+=1
                                    if br[1] + br[3] == cld.shape[0]:
                                        centerOfBRn[1]=centerOfBRn[1]-1
                                        print('low:', 'loop:',i,'; cld_temp:',j ,'; fName:', fName)
                                        low+=1
                                    centerOfBR=tuple(centerOfBRn)                       
                                    # -------------------------------
            
                                    for h  in [1,0]:
                                        # imgrgb=imagePool.read(window=write_window)/imgBits
                                        # bigImgrgb=bigImg.read(window=ints_window)/imgBits
                                        bigImgrgb1=bigImgrgb[h:h+3,:,:].astype('uint8')
                                        imgrgb1=imgrgb[h:h+3,:,:].astype('uint8')
                                        
                                        # way1:to prevent the shift position after seamlessClone the process.
                                        # https://stackoverflow.com/questions/47827198/opencv-seamless-cloning-shift-position-after-finish-the-process
        
                                        
                                        # poissonImage = cv2.seamlessClone(srcImage, dstImage, maskImage, centerOfBR )
                                        #way2
                                        # center = (bigImgrgb1.shape[2]//2, bigImgrgb1.shape[1]//2)
                                        
                                        
                                        # BI=bigImg.transform
                                        # cooryx=BI[5]-(center[0]*res),BI[2]-(center[1]*res)
                                        # cooryx
                                        # center=( 120 ,80 )
                                        # try:
                                        # center=(3794,5625)
                                        imgrgb2=np.moveaxis(imgrgb1, [0, 1, 2], [2, 0, 1]).astype('uint8')
                                        bigImgrgb2=np.moveaxis(bigImgrgb1, [0, 1, 2], [2, 0, 1]).astype('uint8')
                                        imgrgb2[tw6mMask==0,:]=0
                                        
                                        imgrgb2[temp==1,:]=0
                                        bigImgrgb2[temp==1,:]=0
                                        # mask = 255 * np.ones(imgrgb2.shape)
                                        # mask = 1 * np.ones((bigImgrgb2.shape[0],bigImgrgb2.shape[1],bigImgrgb2.shape[2]))
                                        cld[cld==1]=255
                                        # mask = 255 * np.ones(cld.shape)
                                        mixed_clone = cv2.seamlessClone( imgrgb2, bigImgrgb2, cld, centerOfBR,cv2.NORMAL_CLONE)#cv2.MONOCHROME_TRANSFER)#cv2.NORMAL_CLONE

                                        mixed_clone=np.moveaxis(mixed_clone, [0, 1, 2], [1, 2, 0]).astype('uint16')
                                        out[h:h+3,:,:]=mixed_clone#*imgBits

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
                                        # code=1
                                        print('loop:',i,'; hole_Name:',j,'code:',code)
                                    
                                    if code==0:
                                        cldblendPix=25
                                        temp[np.where(temp==0)]=(cldblendPix+1)
                                        
                                        mask_blurred  = cv2.GaussianBlur(temp,(cldblendPix,cldblendPix),0)    
                                        # mask_blurred_1chan = mask_blurred.astype('float32') / (cldblendPix+1)
                                        mask_blurred_3chan = cv2.cvtColor(mask_blurred, cv2.COLOR_GRAY2BGR).astype('float32') / (cldblendPix+1)
                                        mask_blurred_4chan = np.concatenate( [mask_blurred_3chan,mask_blurred[...,None]/(cldblendPix+1)], axis=2)
                                        mask_blurred_4chan.astype('float32')
                                        mask_blurred_4chan=np.moveaxis(mask_blurred_4chan, [0, 1, 2], [1, 2, 0]) 
                                        mask_blurred_4chan[:,temp==1]=0
                                        
                                        imgrgb=imagePool.read(window=write_window)
                                        outcome=outcome.astype(imgtype) * mask_blurred_4chan + imgrgb.astype(imgtype) *( 1-mask_blurred_4chan)      
                                        # del 
                                        
                                        cldcy=cld.copy()
                                        blendPix=65
                                        cldcy[np.where(cld==255)]=(blendPix+1)
                                        
                                        bigImgrgb=bigImg.read(window=ints_window)
                                        
                                        try:
                                            # blendPix=65
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
                               
                                        # mask_blurred_1chan = mask_blurred.astype('float32') / (blendPix+1)
                                        mask_blurred_3chan = cv2.cvtColor(mask_blurred, cv2.COLOR_GRAY2BGR).astype('float32') / (blendPix+1)
                                        mask_blurred_4chan = np.concatenate( [mask_blurred_3chan,mask_blurred[...,None]/(blendPix+1)], axis=2)
                                        mask_blurred_4chan.astype('float32')
                                        mask_blurred_4chan=np.moveaxis(mask_blurred_4chan, [0, 1, 2], [1, 2, 0]) 
                                        outcome=outcome.astype(imgtype) * mask_blurred_4chan + bigImgrgb.astype(imgtype) *( 1-mask_blurred_4chan)      
                                        
                                       
                                    #------------------------------------------------------------------------------------------------
                                    #----way3 histogram matching again------------------------------------------------------------
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
                                    # taicld[cld==255]=0      
                                    taicld[np.where((cld==255) & (temp!=1) & (tw6mMask==1))]=0 
                                    taicld[np.where( (temp==1) & (tw6mMask==1) ) ]=(i+1)*10
                                    TaiMas.close()
                                    with rasterio.open('TaiMas.tif', 'r+') as dst:                                
                                        dst.write(taicld.astype(imgtype),1,  window=ints_window_TaicldWin) 
                                    # TaiMas=rasterio.open(candiCld,'r+')
                                
                                    #----------for pan-------------------------------------------                                 
                                    #resize cldb mask to pan's res                            
                                    mask_blurred_pan_chan=cv2.resize( mask_blurred_4chan[1,:,:],(int(mask_blurred_4chan[1,:,:].shape[1]*(res/pan_res)),int(mask_blurred_4chan[1,:,:].shape[0]*(res/pan_res))) ,interpolation=cv2.INTER_NEAREST)                                
                                    
                                    
                                    #read big pan which is cldb mask's counterpart 
                                    with rasterio.open( bigImgAddr) as src:                                   
                                        
                                        win_transform = src.window_transform(ints_window)
                                        # bigImg_win_transform=win_transform[2],win_transform[5]
                                        # del win_transform
                                        panBounds=from_bounds(win_transform[2], win_transform[5]-ints_window.height*res, win_transform[2]+ints_window.width*res, win_transform[5], src.transform)#left, bottom, right, top, src.transform
                                    MosTai_pan= rasterio.open( bigPan_Addr,'r+')# 
                                    # MosTai_panWin=MosTai_pan.transform[2],MosTai_pan.transform[5]
                                    # profile = MosTai_pan.profile
                                    # profile.update({
                                    #     'height': ints_window.height*(res/pan_res) ,
                                    #     'width':ints_window.width*(res/pan_res),
                                    #     'transform': win_transform})
                                    bigpan = MosTai_pan.read(1 ,window=from_bounds(win_transform[2], win_transform[5]-ints_window.height*res, win_transform[2]+ints_window.width*res, win_transform[5], MosTai_pan.transform))
                                    MosTai_pan.close()
                                    
                                    
                                    #read pan_imagePool which is cldb mask's counterpart 
                                    PAN_fName=fName[0]+str(int(fName[fName.find('0'):fName.find('0')+7])+1).zfill(7)
                                    pan_imagePool = rasterio.open(Pan_Addr+'\\'+PAN_fName+'\\'+PAN_fName+'_B1.DAT.raw')       
                                    # write_window= Window(abs(imgWin.col_off-ints_window.col_off), abs(ints_window.row_off-imgWin.row_off), height= ints_window.height , width=ints_window.width)
                                    
                                    smallpan=pan_imagePool.read(1,window= from_bounds(win_transform[2], win_transform[5]-ints_window.height*res, win_transform[2]+ints_window.width*res, win_transform[5], pan_imagePool.transform))                                
                                    # panWin=Window( (pan_imagePool.transform[2]-MosTai_panWin[0])/pan_res ,(MosTai_panWin[1]-pan_imagePool.transform[5])/pan_res ,  width=pan_imagePool.width ,height=pan_imagePool.height  )
                                    pan_imagePool.close()
                                    outcome=(smallpan.astype(imgtype) * mask_blurred_pan_chan + bigpan.astype(imgtype) *( 1-mask_blurred_pan_chan) ).astype(imgtype)
                                    with rasterio.open( bigPan_Addr,'r+') as dst:
                                        dst.write( outcome, 1, window= from_bounds(win_transform[2], win_transform[5]-ints_window.height*res, win_transform[2]+ints_window.width*res, win_transform[5], dst.transform ) )
                                    del outcome,smallpan,bigpan
                                    ##----------------------------------------------------------------------
                                    ss=1          
                                
                                # a+=1
                                
                                
                            elif ssim[ssimIndex[-1],a-1]==-0.9 and np.any(imgrgb[0,:,:][cld==1]==0)==False and np.count_nonzero(cldb[cld==1]<10)/np.count_nonzero(cldb[cld==1])>0.55:
                                cldcy=cld.copy()
                                blendPix=65
                                cldcy[np.where(cld==1)]=(blendPix+1)
                                try:
                                    # blendPix=65
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
                             
                                # mask_blurred_1chan = mask_blurred.astype('float32') / (blendPix+1)
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
                                temp = cv2.dilate(temp, kernel, iterations = 1)#.astype('uint8')
                                h, w = temp.shape[:2]
                                mask = np.zeros((h+2, w+2), np.uint8)
                                cv2.floodFill(temp, mask, (1,1), 0)  
                                temp[temp!=0]=1
                                
                                
                                bigImg= rasterio.open(bigImgAddr)
                                
                                bigImgrgb=bigImg.read(window=ints_window)#/imgBits
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
                                # TaiMas=rasterio.open(candiCld,'r+')
                                #----------for pan-------------------------------------------                                 
                                #resize cldb mask to pan's res                            
                                mask_blurred_pan_chan=cv2.resize( mask_blurred_4chan[1,:,:],(int(mask_blurred_4chan[1,:,:].shape[1]*(res/pan_res)),int(mask_blurred_4chan[1,:,:].shape[0]*(res/pan_res))) ,interpolation=cv2.INTER_NEAREST)                                
                                
                                
                                #read big pan which is cldb mask's counterpart 
                                with rasterio.open( bigImgAddr) as src:                                   
                                    
                                    win_transform = src.window_transform(ints_window)
                                    # bigImg_win_transform=win_transform[2],win_transform[5]
                                    # del win_transform
                                    panBounds=from_bounds(win_transform[2], win_transform[5]-ints_window.height*res, win_transform[2]+ints_window.width*res, win_transform[5], src.transform)#left, bottom, right, top, src.transform
                                MosTai_pan= rasterio.open( bigPan_Addr,'r+')# 
                                # MosTai_panWin=MosTai_pan.transform[2],MosTai_pan.transform[5]
                                # profile = MosTai_pan.profile
                                # profile.update({
                                #     'height': ints_window.height*(res/pan_res) ,
                                #     'width':ints_window.width*(res/pan_res),
                                #     'transform': win_transform})
                                bigpan = MosTai_pan.read(1 ,window=from_bounds(win_transform[2], win_transform[5]-ints_window.height*res, win_transform[2]+ints_window.width*res, win_transform[5], MosTai_pan.transform))
                                MosTai_pan.close()
                                
                                
                                #read pan_imagePool which is cldb mask's counterpart 
                                PAN_fName=fName[0]+str(int(fName[fName.find('0'):fName.find('0')+7])+1).zfill(7)
                                pan_imagePool = rasterio.open(Pan_Addr+'\\'+PAN_fName+'\\'+PAN_fName+'_B1.DAT.raw')       
                                # write_window= Window(abs(imgWin.col_off-ints_window.col_off), abs(ints_window.row_off-imgWin.row_off), height= ints_window.height , width=ints_window.width)
                                
                                smallpan=pan_imagePool.read(1,window= from_bounds(win_transform[2], win_transform[5]-ints_window.height*res, win_transform[2]+ints_window.width*res, win_transform[5], pan_imagePool.transform))                                
                                # panWin=Window( (pan_imagePool.transform[2]-MosTai_panWin[0])/pan_res ,(MosTai_panWin[1]-pan_imagePool.transform[5])/pan_res ,  width=pan_imagePool.width ,height=pan_imagePool.height  )
                                pan_imagePool.close()
                                outcome=(smallpan.astype(imgtype) * mask_blurred_pan_chan + bigpan.astype(imgtype) *( 1-mask_blurred_pan_chan) ).astype(imgtype)
                                with rasterio.open( bigPan_Addr,'r+') as dst:                                
                                    dst.write( outcome, 1, window= from_bounds(win_transform[2], win_transform[5]-ints_window.height*res, win_transform[2]+ints_window.width*res, win_transform[5], dst.transform ) )
                                del outcome,smallpan,bigpan
                                
                                ss=1  
                                
                    except:
                        # a+=1
                        pass
                    
    print('stage1:',stage1)
    print('stage2:',stage2) 
    print('cldzeroNum:',cldzeroNum)
    print('zeroExist:',zeroNum)
    print('stage3:',stage3)
    print('right,up,left,low:',right,up,left,low)
    cld_ds.close()# 
    TaiMas.close()# 
    dirpath=cldpath+'\\'
    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)


print('All done!')
print(Count,'/',total,' reconstructions of cloud-contaminated area')#warning!! total number includes replicated cld masks.
end = timer()
print('time(hr):',(end - start)/60/60) # Time in seconds


