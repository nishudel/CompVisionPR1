import numpy as np
import pdb
import copy
import cv2

def convfn(I,H):
	ri=I.shape[0]
	ci=I.shape[1]
	rh=H.shape[0]
	ch=H.shape[1]
	a=int((rh-1)/2)
	b=int((ch-1)/2)
	iconv=np.ones(I.shape)
	img=np.pad(I, ((a,b), (a,b),(0,0)), 'constant')

	for i in range(a,a+ri):
		for j in range(b,b+ci):
			iconv[i-a,j-b,:]=np.sum(np.sum(np.multiply(H[:],img[i-a:i-a+rh,j-b:j-b+ch,:]),0),0) 



	return iconv

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

def GaussianPyramid(image,n):
	pyramid=[image]
	img_new=image
	for i in range(1,n):
		img_new=Reduce(img_new)
		pyramid.append(img_new)
	
	for j in range(n):
		#print(pyramid[j].shape)
		#pdb.set_trace()
		cv2.imshow('pyramid',pyramid[j])		
		cv2.waitKey(0)	
	

	return pyramid






if __name__ == '__main__':
	I=cv2.imread('totoro.jpg')
	"""
	H1=(1/16)*np.array([[1,2,1],[2,4,2],[1,2,1]])
	H=np.array([H1,H1,H1])
	#print(I[2,:,1])
	cv2.imshow('I',I)
	pdb.set_trace()
	A=convfn(I,H)
	B=np.uint8(np.clip(A,0,255))
	cv2.imshow('new_img',B)
	cv2.waitKey(0)

	"""
	img=GaussianPyramid(I,4)

