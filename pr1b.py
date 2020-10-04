import numpy as np
import cv2
import matplotlib.pyplot as plt
import math as m
import random
import matgen 
from matgen import *


def blockfn(x,y,x_t,y_t):
	block=[[-x,-y,-1,0,0,0,x_t*x,x_t*y,x_t],[0,0,0,-x,-y,-1,y_t*x,y_t*y,y_t]]
	return block



if __name__ == '__main__':
	#reading the image
	img1=cv2.imread('totoro.jpg')
	img = img1[...,::-1].copy()
	rows,cols,ch = img.shape
	#cv2.imshow('image',img)
	#k= cv2.waitKey(0) 



	#setting random values 
	theta=random.random()
	tx=random.random()
	ty=random.random()	
	sx=1.5
	sy=0.9
	#shifting centre and bringing back
	px=0.5*rows
	py=0.5*cols
	tM=matgen.translmat(-px,-py)
	tiM=matgen.translmat(px,py)
	
	#affine transformation	
	tfmat_af=matgen.translmat(tx,ty)@matgen.scalemat(sx,sy)@matgen.rotatmat(theta)
	tfmat_af=tiM@tfmat_af@tM
	#print(tfmat_af)
	#applying the transformation to the image	
	mat_af=cv2.warpPerspective(img,tfmat_af,(rows,cols))
	#cv2.imshow('img',mat_af)
	#k= cv2.waitKey(0) 
	
	#perspective transformation
	tfmat_pr=matgen.translmat(300,150)@matgen.scalemat(1,1)@matgen.rotatmat(m.pi/3)
	tfmat_pr[2,0]=0.0005
	tfmat_pr[2,1]=0.0005
	tfmat_pr=tiM@tfmat_pr@tM
	#print('tfmat_pr:   ')
	print(tfmat_pr)
	#applying the transformation to the image	
	mat_pr=cv2.warpPerspective(img,tfmat_pr,(rows,cols))
	#cv2.imshow('img',mat_pr)
	#k= cv2.waitKey(0) 

	#######The following is to test the validity of the algo - using points directly #######
	X=np.array([[200,800,800,200],[100,100,700,700],[1,1,1,1]])
	X_tf=tfmat_pr@X
	#X_tf[2,:]=[1,1,1,1]
	X_tf=X_tf.T
	X=X.T
	
	i=0
	while (i<4):
		i=i+1
		blk=np.array(blockfn(X[i-1][0],X[i-1][1],X_tf[i-1][0],X_tf[i-1][1]))
		#a=np.shape(blk)
		#print(blk)
		if ((i-1)==0):
			A=np.array(blk)
		else:
			A=np.concatenate((A,blk),0)
			
	U, s, vh = np.linalg.svd(A,True)
	#print(vh)
	H= np.zeros(9)
	i=0
	while (i<9):
		H[i]=H[i]+vh.T[i,8]
		i=i+1
	#print(A@H)
	H=np.reshape(H,(3,3))
	H33=H[2,2]
	H=H/H33
	print('H',H)
	


	'''
	# Images on the same window
	plt.subplot(121),plt.imshow(img),plt.title('Original')
	#plt.subplot(122),plt.imshow(mat_af),plt.title('Affine_transformation')
	plt.subplot(122),plt.imshow(mat_pr),plt.title('Perspective_transformation')
	'''


	'''
	#######Part A using required number of points #######
	# choose the point in the original image and choose corresponding point the transformation
	# go on for n such points
	n=8
	pointsls=plt.ginput(n,0)
	pointsar=np.array(pointsls)
	i=0
		
	while (i<8):
		i=i+2
		blk=np.array(blockfn(pointsar[i-2][0],pointsar[i-2][1],pointsar[i-1][0],pointsar[i-1][1]))
		#a=np.shape(blk)
		#print(a)
		if ((i-2)==0):
			A=np.array(blk)
		else:
			A=np.concatenate((A,blk),0)
			
	print(A)	
	'''


	'''
	#######Part B using more than required number of points #######
	# choose the point in the original image and choose corresponding point the transformation
	# go on for n such points
	n=8
	pointsls=plt.ginput(n,0)
	pointsar=np.array(pointsls)
	i=0
		
	while (i<8):
		i=i+2
		blk=np.array(blockfn(pointsar[i-2][0],pointsar[i-2][1],pointsar[i-1][0],pointsar[i-1][1]))
		#a=np.shape(blk)
		#print(a)
		if ((i-2)==0):
			A=np.array(blk)
		else:
			A=np.concatenate((A,blk),0)
			
	print(A)

	'''
	
	

	'''
	########part C- calculating the tf matrix by using in built functions
	#print(pointsls)
	pt1=[[pointsar[0][0],pointsar[0][1]],[pointsar[2][0],pointsar[2][1]],[pointsar[4][0],pointsar[4][1]],[pointsar[6][0],pointsar[6][1]]]
	pt2=[[pointsar[1][0],pointsar[1][1]],[pointsar[3][0],pointsar[3][1]],[pointsar[5][0],pointsar[4][1]],[pointsar[7][0],pointsar[7][1]]]
	pts1=np.float32(pt1)
	pts2=np.float32(pt2)
	M = cv2.getPerspectiveTransform(pts1,pts2)
	print('Matric calculated by in built function')
	print(M)	
	'''