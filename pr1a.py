import numpy as np
import functionspr1a as fnc
import pdb
import cv2


if __name__ == '__main__':
	#img=cv2.imread('ball.png') 
	
	#H=(1/16)*np.array([[1,2,1],[2,4,2],[1,2,1]])
	
	'''
	#pdb.set_trace()
	conv_img=fnc.Convolve(img,H)
	cv2.imshow('img',img)
	cv2.imshow('convoluted img',(conv_img))
	cv2.waitKey(0)
	'''
	############################ Reduce ############################
	'''
	cv2.imshow('img',img)
	cv2.imshow('imgnew',fnc.Reduce(img))
	cv2.waitKey(0)
	'''
	############################ Generate Gaussian Pyramid ############################
	#fnc.GaussianPyramid(img,4)
	############################ Generate Laplacian Pyramid ############################
	'''
	n=4
	Laplacian_pyramid=fnc.LaplacianPyramids(img,n)
	Reconstruct_img=fnc.Reconstruct(Laplacian_pyramid,n)
	cv2.imshow('error',img-Reconstruct_img)
	cv2.waitKey(0)
	'''
	############################ Merging ############################
	img1=cv2.imread('im1_1.JPG') 
	img2=cv2.imread('im1_2.JPG')
	image1=fnc.image_crop(img1)
	image2=fnc.image_crop(img2)
	#print(np.dtype(img1[0,1,0]),np.dtype(image1[0,1,0]))
	#cv2.imshow('img1',image1)
	#cv2.imshow('img2',image2)
	#cv2.waitKey(0)
	fnc.normalblending(image1,image2)