import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math as m
import random
import matgen 
from matgen import *


if __name__ == '__main__':
	#reading the image
	img1=cv2.imread('totoro.jpg')
	img = img1[...,::-1].copy()
	rows,cols,ch = img.shape
	#setting random values 
	theta=random.random()
	tx=500*random.random()
	ty=500*random.random()	
	sx=1.5
	sy=0.9
	#shifting centre and bringing back
	px=0.5*rows
	py=0.5*cols
	tM=matgen.translmat(-px,-py)
	tiM=matgen.translmat(px,py)

################################ PART I ################################
	# choose the point in the original image and choose corresponding point the transformation
	# go on for n such points
	print("PART I:\n")
	print("Press esc to go to part B after new images show up")
	plt.subplot(121),plt.imshow(img),plt.title('Original')
	plt.suptitle('Choose four points and their intended locations alternatively')
	n=8
	pointsls=plt.ginput(n,0)
	plt.close( )

	#Saving the points in np array
	pointsar=np.array(pointsls)
	
	#generating the Affine transformation matrix - using A_inverse method 
	H_a=get_haff_h(6,pointsar)
	H_af=np.zeros((2,3))
	i=0
	while(i<2):
		j=0
		while (j<3):
			H_af[i,j]=H_a[i,j]
			j=j+1
		i=i+1	 
	print("\nThe first two rows of the Affine Transformation Matrix: ")
	print(H_af)

	#generating the Perspective transformation matrix - using A_inverse method 	
	H_per=get_hper_h(8,pointsar)
	print("\nThe Complete Perspective Transformation Matrix: ")
	print(H_per)
	
	img_pr=cv2.warpPerspective(img,H_per,(rows,cols))
	img_af=cv2.warpAffine(img,H_af,(rows,cols))

	
	fig=plt.figure( )
	gs=gridspec.GridSpec(2,2)
	ax1=fig.add_subplot(gs[0,:]),plt.imshow(img),plt.title('Original')
	ax2=fig.add_subplot(gs[1,0]),plt.imshow(img_pr),plt.title('Perspective')
	ax3=fig.add_subplot(gs[1,1]),plt.imshow(img_af),plt.title('Affine')	
	plt.draw()
	plt.waitforbuttonpress(0) # this will wait for indefinite time
	plt.close( )

############################# PART II ###############################
	print("\nPART II:\n")
	print("A)Using exact number of points:Non Homogenous method ")
	plt.subplot(121),plt.imshow(img),plt.title('Original')	
	plt.subplot(122),plt.imshow(img_af),plt.title('Affine')
	plt.suptitle('Choose three points and their corresponding locations alternatively on both images:')
	n=6
	points_aff=plt.ginput(n,0)
	plt.close( )

	plt.subplot(121),plt.imshow(img),plt.title('Original')	
	plt.subplot(122),plt.imshow(img_pr),plt.title('Perspective ')
	plt.suptitle('Choose four points and their corresponding locations alternatively on both images:')
	n=8
	points_per=plt.ginput(n,0)
	plt.close( )
	#generating the Perspective transformation matrix - using A_inverse method 	
	H_affine=get_haff_nh(points_aff)
	i=0
	j=0
	H_aff_nh=np.identity(3)
	while (i<2):
		j=0
		while (j<3):
			H_aff_nh[i][j]=H_affine[i][j]
			j=j+1
		i=i+1
	print("\nThe Affine Transformation Matrix:Non-Homogenous method")
	print(H_aff_nh)

	#generating the Perspective transformation matrix - using A_inverse method 	
	H_pers_nh=get_hper_nh(points_per)
	print("\nThe Perspective Transformation Matrix:Non-Homogenous method ")
	print(H_per_nh)

	print("\n \nB)Using excess number of points:Homogenous method ")

	plt.subplot(121),plt.imshow(img),plt.title('Original')	
	plt.subplot(122),plt.imshow(img_af),plt.title('Affine')
	plt.suptitle('Choose five points and their corresponding locations alternatively on both images:')
	n1=10
	points_aff=plt.ginput(n1,0)
	plt.close( )
	#generating the Affine transformation matrix - using Homogenous method
	H_aff_exs=get_haff_h(n1,points_aff)
	print("\nThe Affine Transformation Matrix:Homogenous method ")
	print(H_aff_exs)
	

	plt.subplot(121),plt.imshow(img),plt.title('Original')	
	plt.subplot(122),plt.imshow(img_pr),plt.title('Perspective ')
	plt.suptitle('Choose six points and their corresponding locations alternatively on both images:')
	n2=12
	points_per=plt.ginput(n2,0)
	plt.close( )
	#generating the Perspective transformation matrix - Homogenous method 	
	H_per_exs=get_hper_h(n2,points_per)
	print("\nThe Perspective Transformation Matrix:Homogenous method ")
	print(H_per_exs)


	print("\n \nC)Using Using inbuilt functions and Comparision:")

	'''
	A_inv=generateA_inv(8,pointsar)
	#print(A_inv.shape)

	i=0
	j=0

	#B=pointsar[1][0]
	while(i<4):
		j=0
		while(j<2):
			elem=ptsfin[i][j]
			if(i==0):
				B=[ptsfin[0][0],ptsfin[0][1]]
			else :
				B=np.append(B,elem)
			j=j+1	
		i=i+1	
	B=np.array(B)
	

	H1=A_inv@B
	H1=np.append(H1,[1])
	print(H1)
	H=H1.reshape((3,3))
	print(H)
	'''
	'''
	A_pr=generateA(8,pointsar)
	U, s, vh = np.linalg.svd(A_pr,True)
	#print(vh)
	H2_pr= np.zeros(9)
	i=0
	while (i<9):
		H2_pr[i]=H2_pr[i]+vh.T[i,8]
		i=i+1
	#print(A@H)
	H2_pr=np.reshape(H2_pr,(3,3))
	H33=H2_pr[2,2]
	H=H2_pr/H33
	print('H',H)
	'''
	
	
	'''
	plt.draw()
	plt.waitforbuttonpress(0) # this will wait for indefinite time
	plt.close( )

	plt.subplot(121),plt.imshow(img),plt.title('Original')
	plt.draw()
	plt.waitforbuttonpress(0) # this will wait for indefinite time
	plt.close( )
	'''


	'''


	H1=generateH(8,pointsar)
	#print(H1)
	H2=generateH(6,pointsar)
	#print(H2)
	'''






	
	#affine transformation matrix : tfmat_afs	
	tfmat1=matgen.translmat(tx,ty)@matgen.scalemat(sx,sy)@matgen.rotatmat(theta)
	tfmat_af=tiM@tfmat1@tM
	#print(tfmat_af)
	
	#perspective transformation matrix : tfmat_pr
	tfmat_pr=tfmat1
	tfmat_pr[2,0]=0.0008
	tfmat_pr[2,1]=0.0004
	tfmat_pr=tiM@tfmat_pr@tM
	tfmat_pr=tfmat_pr/tfmat_pr[2,2]
	#print('tfmat_pr:   ')
	#print(tfmat_pr)
	
	#applying the transformation to the image	
	mat_af=cv2.warpPerspective(img,tfmat_af,(rows,cols))
	#cv2.imshow('img',mat_af)
	#k= cv2.waitKey(0) 
	
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
	#print('H',H)

	#print(tfmat_pr-H)
	


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
	print('Matrix calculated by in built function')
	print(M)	
	'''