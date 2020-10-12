#matgen functions
import math as m
import numpy as np


#2D to 2D homogenous transformatio matrices

#translation by (tx,ty)

def translmat( tx,ty):
	tmat=np.float32([[1 ,0, tx],[0,1,ty],[0,0,1]])
	return tmat

def rotatmat( theta ):
	rmat=np.float32([[m.cos(theta),-1*m.sin(theta),0],[m.sin(theta),m.cos(theta),0],[0,0,1]])
	return rmat	

def scalemat( sx,sy ):
	scmat=np.float32([[sx,0,0],[0,sy,0],[0,0,1]])
	return scmat	

def xshear(shx):
	xshmat=np.float32([[1,sh,0],[0,1,0],[0,0,1]])	
	return xshmat

def yshear(shy):
	yshmat=np.float32([[1,0,0],[shy,1,0],[0,0,1]])	
	return yshmat	

#Affine Transformation-Non-Homogenous
def block_haff(x,y):
	block=[[x,y,1,0,0,0],[0,0,0,x,y,1]]
	return block

def get_haff_nh(pointsar):
	B=np.zeros((6,1))
	i=0
	j=1
	while(i<6):
		B[i]=B[i]+pointsar[j][0]
		i=i+1
		B[i]=B[i]+pointsar[j][1]
		j=j+2
		i=i+1

	i=0	
	while(i<5):
		blk=np.array(block_haff(pointsar[i][0],pointsar[i][1]))
		if (i==0):
			A=np.array(blk)
		else:
			A=np.concatenate((A,blk),0)		
		i=i+2

	H_aff=np.linalg.inv(A)@B
	H_aff=np.reshape(H_aff,(2,3))

	return H_aff	

#Perspective Transformation-Non-Homogenous
def block_hper(x,y,x_t,y_t):
	block=[[x,y,1,0,0,0,-x_t*x,-x_t*y],[0,0,0,x,y,1,-y_t*x,-y_t*y]]
	return block

def get_hper_nh(pointsar):
	B=np.zeros((8,1))
	i=0
	j=1
	while(i<8):
		B[i]=B[i]+pointsar[j][0]
		i=i+1
		B[i]=B[i]+pointsar[j][1]
		j=j+2
		i=i+1
	i=0	
	while(i<7):
		blk=np.array(block_hper(pointsar[i][0],pointsar[i][1],pointsar[i+1][0],pointsar[i+1][1]))
		if (i==0):
			A=np.array(blk)
		else:
			A=np.concatenate((A,blk),0)		
		i=i+2	
	
	H_1=np.linalg.inv(A)@B
	H_per=np.ones((9,1))
	k=0
	while(k<8):
		H_per[k]=H_1[k]
		k=k+1
	H_per=np.reshape(H_per,(3,3))	
	return H_per



#Affine Transformation Matrix : Homogenous method
def block_af_h(x,y,x_t,y_t):
	block=[[-x,-y,-1,0,0,0,x_t],[0,0,0,-x,-y,-1,y_t]]
	return block

def get_haff_h(n,pointsar):
	i=0
	while (i<n):
		blk=np.array(block_af_h(pointsar[i][0],pointsar[i][1],pointsar[i+1][0],pointsar[i+1][1]))
		
		if (i==0):
			A=np.array(blk)
		else:
			A=np.concatenate((A,blk),0)
		i=i+2

	U, s, vh = np.linalg.svd(A,True)	
	end=6
	vh1=vh.T/vh.T[end][end]
	H1= np.zeros(9)
	H1[8]=1
	i=0
	while (i<6):
		H1[i]=vh1[i][6]
		i=i+1
	#print(H1)
	H1=np.reshape(H1,(3,3))

	return H1

#Perspective Transformation Matrix : Homogenous method
def block_per_h(x,y,x_t,y_t):
	block=[[-x,-y,-1,0,0,0,x_t*x,x_t*y,x_t],[0,0,0,-x,-y,-1,y_t*x,y_t*y,y_t]]
	return block

def get_hper_h(n,pointsar):
	i=0
	while (i<n):
		blk=np.array(block_per_h(pointsar[i][0],pointsar[i][1],pointsar[i+1][0],pointsar[i+1][1]))
		
		if (i==0):
			A=np.array(blk)
		else:
			A=np.concatenate((A,blk),0)
		i=i+2

	U, s, vh = np.linalg.svd(A,True)	
	end=8
	vh1=vh.T/vh.T[end][end]
	H1= np.zeros(9)
	H1[8]=1
	i=0
	while (i<8):
		H1[i]=vh1[i][8]
		i=i+1
	H1=np.reshape(H1,(3,3))

	return H1

		











def blockfn_inv(x,y,x_t,y_t):
	block=[[x,y,1,0,0,0,-x_t*x,-x_t*y],[0,0,0,x,y,1,-y_t*x,-y_t*y]]
	return block

def generateA_inv(n,pointsar):
	i=0
	while (i<n):
		i=i+2
		blk=np.array(blockfn_inv(pointsar[i-2][0],pointsar[i-2][1],pointsar[i-1][0],pointsar[i-1][1]))
		
		if ((i-2)==0):
			A_inv=np.array(blk)
		else:
			A_inv=np.concatenate((A_inv,blk),0)
			
	return A_inv

def blockfn_pr(x,y,x_t,y_t):
	block=[[-x,-y,-1,0,0,0,x_t*x,x_t*y,x_t],[0,0,0,-x,-y,-1,y_t*x,y_t*y,y_t]]
	return block


def blockfn_af(x,y,x_t,y_t):
	block=[[-x,-y,-1,0,0,0],[0,0,0,-x,-y,-1]]
	return block

def generateA(n,pointsar):
	i=0
	if(n==8):
		while (i<n):
			i=i+2
			blk=np.array(blockfn_pr(pointsar[i-2][0],pointsar[i-2][1],pointsar[i-1][0],pointsar[i-1][1]))
			
			if ((i-2)==0):
				A=np.array(blk)
			else:
				A=np.concatenate((A,blk),0)
			#print(A.shape)		
	if(n==6):
		while (i<n):
			i=i+2
			blk=np.array(blockfn_af(pointsar[i-2][0],pointsar[i-2][1],pointsar[i-1][0],pointsar[i-1][1]))
			
			if ((i-2)==0):
				A=np.array(blk)
			else:
				A=np.concatenate((A,blk),0)
			
	return A

def generateH(n,pointsar):	
	A=generateA(n,pointsar)
	#print(A.shape)
	U, s, vh = np.linalg.svd(A,True)
	end=vh.shape[1]-1
	#print(end)
	vh=vh/vh[end,end]
	H2= np.zeros(9)
	i=0
	if(n==8):
		while (i<9):
			H2[i]=H2[i]+vh.T[i,8]
			i=i+1
		H2=np.reshape(H2,(3,3))
		H=H2
	else:
		while (i<6):
			H2[i]=H2[i]+vh.T[i,5]
			i=i+1
		H2=np.reshape(H2,(3,3))
		H2[2,2]=1
		H=H2

	return H
			