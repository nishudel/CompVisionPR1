# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 22:16:27 2020

@author: Aamanku
"""

import cv2
import numpy as np

img=cv2.imread('totoro.jpg')
# cv2.imshow('out', img)
# cv2.waitKey(0)
h=np.array([[1,2,1],[2,4,2],[1,2,1]])/16
# h=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
H=np.repeat(h[:, :, np.newaxis], 3, axis=2)
# 
# H=np.ones((10,10))/100
pad_size=np.array(h.shape)//2
img_padded=np.pad(img,(pad_size,pad_size,(0,0)),'constant')
# cv2.imshow('padded out',img_padded)
# cv2.waitKey(0)
#img_convolved=cv2.filter2D(img,-1,h)
#cv2.imshow('filtered',img_convolved)
#cv2.waitKey(0)
img_out=np.uint8(np.zeros(img.shape))
H_size=H.size
H_reshaped=H.reshape([1,H_size])
h_x=int(H.shape[0]/2)
h_y=int(H.shape[1]/2)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        # img_out[i,j,:]=H*img_padded[i:(i+2*h_x+1),j:(j+2*h_y+1),:]
        img_out[i,j,:]=np.sum(np.sum(H*img_padded[i:(i+2*h_x+1),j:(j+2*h_y+1),:],0),0)
    print(i)

cv2.imshow('out',img_out)
cv2.waitKey(0)
    
    
