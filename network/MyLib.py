# -*- coding: utf-8 -*-
"""
Created on Tue May  1 12:50:06 2018

@author: XieQi
"""

import numpy as np
import matplotlib.pyplot as plt
import MyLib as ML
import os

def normalized(X):
    maxX = np.max(X)
    minX = np.min(X)
    X = (X-minX)/(maxX - minX)
    return X

def setRange(X, maxX = 1, minX = 0):
    X = (X-minX)/(maxX - minX)
    return X


def get3band_of_tensor(outX,nbanch=0,nframe=[0,1,2]):
    X = outX[:,:,:,nframe]
    X = X[nbanch,:,:,:]
    return X

def imshow(X):
#    X = ML.normalized(X)
    X = np.maximum(X,0)
    X = np.minimum(X,1)
    plt.imshow(X)
    plt.axis('off') 
    plt.show()  
    
def imwrite(X,saveload='tempIm'):
    # print(np.shape(X))
    # print(np.shape(np.tile(X, [1, 1, 3])))
    X = np.tile(np.expand_dims(X,2), [1, 1, 3])*2
    #print(np.shape(X))
    X = np.maximum(X,0)
    X = np.minimum(X,1)
    #X1 = np.expand_dims(X,-1).repeat(3,axis=-1)
    plt.imsave(saveload, X)
    #plt.imsave(saveload, ML.normalized(X))

def savefig(X,saveload='tempIm'):
    plt.imshow(X,cmap='YlGnBu_r')
    plt.savefig(saveload)



def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
		print("---  new folder...  ---")
		print("---  "+path+"  ---")
	else:
		print("---  There is "+ path + " !  ---")
		
#file = "test/"
#mkdir(file)  