import numpy as np
import copy
import cv2
import pdb

import matplotlib.pyplot as plt
# Functions that support PR1A






######################## Convolve Function ######################## 

def Convolve(I,H):
	ri=I.shape[0]
	ci=I.shape[1]
	rh=H.shape[0]
	ch=H.shape[1]
	a=int((rh-1)/2)
	b=int((ch-1)/2)
	iconv=np.ones(I.shape)
	end=-1*(ch+1)	
	hconv=np.zeros(H.shape)

	
	for i in range(0,rh):
		hconv[i,:]=H[i,-1:end:-1]  #Rotating the kernel 180 for convolution

	          	    				   
	#Img=np.pad(I, ((a,b), (a,b)), 'constant')		 
	img=np.pad(I, ((a,b), (a,b),(0,0)), 'constant') #zero padding according to size of kernel
	
			     
	for i in range(a,a+ri):
		for j in range(b,b+ci):
			#I_conv[i-a,j-b]=np.sum(np.multiply(H_conv,Img[i-a:i+a+1,j-b:j+b+1]))    #Value at pixel after convolution 
			iconv[i-a,j-b,:]=np.sum(np.multiply(hconv,img[i-a:i-a+rh,j-b:j-b+ch,:]))
	'''
	n_img=np.uint8(iconv)
	new_img = Image.fromarray(n_img, 'RGB')		
	new_img.show()
	'''
	return img


######################## Image_crop Function ########################	
#This corps the images to an even shape 

def image_crop(img):
	sr=img.shape[0]
	sc=img.shape[1]

	if (sr%2==0 and sc%2==0 ):
		return img
	if (sr%2==0 and sc%2!=0 ):
		#new_img=np.zeros((sr,sc-1,3),np.uint8)
		new_img=img[:,0:sc-1,:]
		return new_img				

	if (sr%2!=0 and sc%2==0 ):
		#new_img=np.zeros((sr-1,sc,3),np.uint8)
		new_img=img[0:sr-1,:,:]
		return new_img				

	if (sr%2!=0 and sc%2!=0 ):
		#new_img=np.zeros((sr-1,sc-1,3),np.uint8)
		new_img=img[0:sr-1,0:sc-1,:]
		return new_img				

######################## Reduce Function ########################	

def Reduce(img):
	kernel=(1/16)*np.array([[1,2,1],[2,4,2],[1,2,1]])
	#dst = Convolve(img,kernel)
	dst = cv2.filter2D(img,-1,kernel)
	convolv_img=np.array(dst)
	nr=np.shape(convolv_img)[0]
	nc=np.shape(convolv_img)[1]
	#scaling_img=np.zeros((nr,nc,3))
	interp_img= np.zeros((int(nr/2),int(nc/2),3),np.uint8)
	
	
	for i in range(int(nr/2)):
		for j in range(int(nc/2)):
			#scaling_img[2*i,2*j,:]=convolv_img[2*i,2*j,:]/255
			interp_img[i,j,:]=convolv_img[2*i,2*j,:]

	new_img=np.uint8(interp_img)

	return new_img

########################  Function to add rows or columns ########################	

def modify_add(img,m,n,index):					#Adding empty(zero) pixels to  match sizes
	sr=img.shape[0]
	sc=img.shape[1]
	img_fin=np.zeros((m,n,3),np.uint8)
	if(index==1):								#First image for blending	
		for i in range(sr):
			for j in range(sc):
				img_fin[i,j,:]=img[i,j,:]
	
	if(index==2):								#second image for blending 
		for k in range(sr):
			for l in range(n-1,n-sc-1,-1):
				img_fin[k,l,:]=img[k,l-n+sc,:]

	return np.uint8(img_fin)



######################## Gaussian-Pyramid Function ########################
def makeven(x,y):
	if (x%2==0 and y%2==0 ):
		return x,y
	if (x%2==0 and y%2!=0 ):
		return x,(y+1)				
	if (x%2!=0 and y%2==0 ):
		return (x+1),y				
	if (x%2!=0 and y%2!=0 ):
		return (x+1),(y+1)				



def GaussianPyramid(image,n):
	pyramid=[image]
	img_new=image
	for i in range(1,n):
		img_new=Reduce(img_new)
		x1=img_new.shape[0]
		y1=img_new.shape[1]
		x,y=makeven(x1,y1)
		img_new=modify_add(img_new,x,y,1)
		pyramid.append(img_new)
	'''
	for j in range(n):
		print(pyramid[j].shape)
		pdb.set_trace()
		#cv2.imshow('pyramid',pyramid[j])		
		#cv2.waitKey(0)	
	'''

	return pyramid

######################## Expand Function ########################
def Expand(img):
	nr=np.shape(img)[0]
	nc=np.shape(img)[1]	
	new_img=np.zeros((2*nr,2*nc,3),np.uint8)

	for i in range(nr):
		for j in range(nc):
			new_img[2*i,2*j,:]=img[i,j,:]
			new_img[2*i+1,2*j+1,:]=img[i,j,:]
	'''
	cv2.imshow('exp',new_img)
	cv2.waitKey(0)		
	'''

	return new_img		


######################## Laplacian-Pyramid Function ########################
def LaplacianPyramids(img,n):
	images=GaussianPyramid(img,n)
	pyramid=[]
	exp_img1=np.zeros(img.shape)

	for i in range(n-1):
		exp_img1=Expand(images[i+1])
		exp_img=exp_img1[0:images[i].shape[0],0:images[i].shape[1],:]
		print(images[i].shape,exp_img.shape)
		pdb.set_trace()
		pyramid.append(images[i]-exp_img)

	pyramid.append(images[n-1])	

	'''
	for j in range(n):
		cv2.imshow('pyramid',pyramid[j])		
		cv2.waitKey(0)	
	'''
	
	return pyramid


######################## Reconstruction Function ########################
def Reconstruct(LI,n):
	
	reconstruct_img=[LI[n-1]]
	for i in range(n-2,-1,-1):
		exp_img1=Expand(reconstruct_img[n-2-i])
		exp_img=exp_img1[0:LI[i].shape[0],0:LI[i].shape[1],:]
		print(images[i].shape,exp_img.shape)
		pdb.set_trace()
		#pdb.set_trace()
		reconstruct_img.append(LI[i]+Expand(reconstruct_img[n-2-i]))

	
	for j in range(n):
		cv2.imshow('pyramid',reconstruct_img[j])		
		cv2.waitKey(0)	
	
	return reconstruct_img[n-1]

######################## Image stitching without Warping ########################



'''
def nowarpstitch(img1,img2):
	nr1=np.shape(img1)[0]
	nc1=np.shape(img1)[1]
	nr2=np.shape(img2)[0]
	nc2=np.shape(img2)[1]		
	bitmask=np.zeros((nr1+nr2,nc1+nc2),np.uint8)
	stich_img=np.zeros((nr1+nr2,nc1+nc2,3),np.uint8)
	plt.subplot(111),plt.imshow(bitmask),plt.title('Bitmask')
	plt.suptitle('Choose one point to divide the bitmask into two regions')
	boundary=plt.ginput(1,0)

	for i in range(boundary[0]):
		for j in range(boundary[1]):
			stich_img[i,j,:]=img1[i,j,:]

	for i in range(boundary[0],boundary[0]+):
		for j in range(boundary[1],stich_img.shape[1]):
			stich_img[i,j,:]=img2
	print(sr)
	pdb.set_trace()

	return boundary
'''



def normalblending(img1,img2):					# Both the images are of the same size
	sr=img1.shape[0]							# Assume equal size for both images (sr,sc)
	sc=img1.shape[1]
	new_size=np.array([sr,2*int(5*sc/6)])
	bitmask=np.zeros(new_size)			# Size of bitmask (sr,even(2*sc*5/6))
	plt.subplot(111),plt.imshow(bitmask),plt.title('Bitmask')
	plt.suptitle('Choose a point that divides Bitmask into left and right parts')
	boundary=plt.ginput(1,0)
	plt.close( )

	#img1_mod=np.zeros((sr,int(10*sc/6)))
	img1_mod=modify_add(img1,new_size[0],new_size[1],1)
	#cv2.imshow('img1_mod',img1_mod)	
	#img2_mod=np.zeros((sr,int(10*sc/6)))
	img2_mod=modify_add(img2,new_size[0],new_size[1],2)
	#cv2.imshow('img2_mod',img2_mod)
	#cv2.waitKey(0)


	n=4											#Number of levels in pyramid			
	LA=LaplacianPyramids(img1_mod,n)
	pdb.set_trace()
	LB=LaplacianPyramids(img2_mod,n)
	for i in range(sr):
		for j in range(boundary[1]):
			bitmask[i,j]=1						#Modifying Bitmask
	GS=GaussianPyramid(bitmask,n)
	imgpyramid=[]
	pdb.set_trace()
	for i in range(n):							#Applying Bitmask
		imgpyramid.append(GS[i]*LA[i]+(np.ones((GS[i].shape))-GS[i])*LB[i],uint8)

	Reconstruct(imgpyramid,n)

	return sr

	

