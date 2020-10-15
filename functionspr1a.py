import numpy as np


# Functions that support PR1A
def convolfn(I,H):
	siz_Ir=I.shape[0]
	siz_Ic=I.shape[1]
	siz_Hr=H.shape[0]
	siz_Hc=H.shape[1]
	H_conv=H
	I_conv=I
	end=-1*(siz_Hc+1)	
	for i in range(0,siz_Hr):
		H_conv[i,:]=H[i,-1:end:-1]           	      #Rotating the kernel 180 for convolution
	
	a=int((siz_Hr-1)/2)
	b=int((siz_Hc-1)/2)
	print(I.shape)
	Img=np.pad(I, ((a,b), (a,b)), 'constant')
	#Img=np.pad(I, ((a,b), (a,b),(0,0)), 'constant')     #zero padding according to size of kernel
	block= np.zeros((siz_Hr,siz_Hc))
	for i in range(a,a+siz_Ir):
		for j in range(b,b+siz_Ic):
			#print(i,j)
			#block=Img[i-a:i+a+1,j-b:j+b+1,0]
			I_conv[i-a,j-b]=np.sum(H_conv*Img[i-a:i+a+1,j-b:j+b+1])
			#I_conv[i-a,j-b,:]= np.sum(H_conv*block)     #Value at pixel after convolution 
	
	
	return I_conv