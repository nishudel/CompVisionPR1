import numpy as np
import pdb
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math as m
import random
import functionspr1b 
from functionspr1b import *


if __name__ == '__main__':
	#reading the image
	img1=cv2.imread('totoro.jpg')
	img = img1[...,::-1].copy()
	rows,cols,ch = img.shape

################################ PART I ################################
	# choose the point in the original image and choose corresponding point the transformation
	# go on for n such points
	print("\nPART I:\n")
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


	##### PART - A #####
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

	## Inbuilt function Affine- exact number of points
	pts1,pts2=formpoints(points_aff,6)
	#pdb.set_trace( )
	H_aff_in= cv2.getAffineTransform(np.float32(pts1),np.float32(pts2))
	i=0
	j=0
	H_aff_inb=np.identity(3)
	while (i<2):
		j=0
		while (j<3):
			H_aff_inb[i][j]=H_aff_in[i][j]
			j=j+1
		i=i+1
	print("\nThe Affine Transformation Matrix:Inbuilt Function-Exact Number of Points ")
	print(H_aff_inb)

	#generating the Perspective transformation matrix - using A_inverse method 	
	H_per_nh=get_hper_nh(points_per)
	print("\nThe Perspective Transformation Matrix:Non-Homogenous method ")
	print(H_per_nh)
	## Inbuilt function Perspective- exact number of points
	pts1,pts2=formpoints(points_per,8)
	H_per_inb= cv2.getPerspectiveTransform(np.float32(pts1),np.float32(pts2))
	print("\nThe Perspective Transformation Matrix:Inbuilt Function-Exact Number of Points ")
	print(H_per_inb)





	##### PART - B #####
	print("\n \nB)Using excess number of points:Homogenous method ")

	plt.subplot(121),plt.imshow(img),plt.title('Original')	
	plt.subplot(122),plt.imshow(img_af),plt.title('Affine')
	plt.suptitle('Choose five points and their corresponding locations alternatively on both images:')
	n1=10
	points_aff=plt.ginput(n1,0)
	plt.close( )
	#generating the Affine transformation matrix - using Homogenous method
	H_aff_h=get_haff_h(n1,points_aff)
	print("\nThe Affine Transformation Matrix:Homogenous method ")
	print(H_aff_h)

	## Inbuilt function Affine - excess number of points
	pts1,pts2=formpoints(points_aff,10)
	H_aff_in= cv2.estimateAffine2D(np.float32(pts1),np.float32(pts2))[0]
	i=0
	j=0
	H_aff_inb_exs=np.identity(3)
	while (i<2):
		j=0
		while (j<3):
			H_aff_inb_exs[i][j]=H_aff_in[i][j]
			j=j+1
		i=i+1
	print("\nThe Affine Transformation Matrix:Inbuilt Function-Excess Points ")
	print(H_aff_inb_exs)

	plt.subplot(121),plt.imshow(img),plt.title('Original')	
	plt.subplot(122),plt.imshow(img_pr),plt.title('Perspective ')
	plt.suptitle('Choose six points and their corresponding locations alternatively on both images:')
	n2=12
	points_per=plt.ginput(n2,0)
	plt.close( )
	#generating the Perspective transformation matrix - Homogenous method 	
	H_per_h=get_hper_h(n2,points_per)
	print("\nThe Perspective Transformation Matrix:Homogenous method ")
	print(H_per_h)
	## Inbuilt function Perspective- excess number of points
	pts1,pts2=formpoints(points_per,12)	
	H_per_inb_exs= cv2.findHomography(np.float32(pts1),np.float32(pts2))[0]
	print("\nThe Perspective Transformation Matrix:Inbuilt Function-Excess Points ")
	print(H_per_inb_exs)



	##### PART - C #####
	print("\n \nC)Using Using inbuilt functions and Comparision:")
	'''
	pt1_per=[[pt_per[0][0],pt_per[0][1]],[pt_per[2][0],pt_per[2][1]],[pt_per[4][0],pt_per[4][1]],[pt_per[6][0],pt_per[6][1]]]
	pt2_per=[[pt_per[1][0],pt_per[1][1]],[pt_per[3][0],pt_per[3][1]],[pt_per[5][0],pt_per[4][1]],[pt_per[7][0],pt_per[7][1]]]
	pts1=np.float32(pt1_per)
	pts2=np.float32(pt2_per)
	'''	

	D_per_nh_in=H_per_nh-H_per_inb
	D_aff_nh_in=H_aff_nh-H_aff_inb
	D_aff_h_in=H_aff_h-H_aff_inb_exs
	D_per_h_in=H_per_h-H_per_inb_exs

	print("\n Error Between Non-Homogenous method and Inbuilt functions:")
	print("\n Error in Perspective Transformation Matrix:")
	print(errorcalc(D_per_nh_in))
	print("\n Error in Affine Transformation Matrix:")
	print(errorcalc(D_aff_nh_in))
	print("\n Error Between Homogenous method and Inbuilt functions:")
	print("\n Error in Perspective Transformation Matrix:")
	print(errorcalc(D_per_h_in))
	print("\n Error in Affine Transformation Matrix:")
	print(errorcalc(D_aff_h_in))

