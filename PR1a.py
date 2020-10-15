import numpy as np
import functionspr1a as fnc
import pdb
import cv2


if __name__ == '__main__':
	img=cv2.imread('totoro.jpg') 
	img1=img
	

	H=(1/16)*np.array([[1,2,1],[2,4,2],[1,2,1]])

	r=fnc.convolfn(img[:,:,0],H)
	b=fnc.convolfn(img[:,:,1],H)
	g=fnc.convolfn(img[:,:,2],H)

	img_cnv=cv2.merge((r,b,g))
	inblt_img=cv2.filter2D(img,-1,H)

	resid_img=img1-img_cnv
#	cv2.imshow('img_cnv',img_cnv)
#	cv2.imshow('img_inb',inblt_img)
	cv2.imshow('img',img1)
	cv2.imshow('residual_img',resid_img)
	cv2.waitKey(0)