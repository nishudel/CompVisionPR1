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